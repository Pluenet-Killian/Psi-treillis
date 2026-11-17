// SPDX-License-Identifier: Apache-2.0
// codec_stream.cuh - the .psi file: its format, the encoder and the receiver.
//
// Contents:
//   - the format: the header (StreamHeader), the frame table (FrameRecord), the seal (StreamSeal), the structures of
//     the sparse fine level (FineInfo, FineTrailer), and the checks made when a reader opens a file (docs/FORMAT.md);
//   - the kernels specific to the standalone file chain: fingerprint, field error, coded key frames, sparse fine level;
//   - the rANS coder on the host side;
//   - the receiver (Receiver): it checks the payload of a frame against the geometry of the header and decodes it on
//     the GPU; state_hash and fine_hash give the fingerprint of the decoded state, to be compared with the frame table;
//   - the encoder (Sender, pack): each frame is projected onto the potential, quantised, predicted from the previous
//     frames and entropy-coded; with a tolerance, the sparse fine level is added and the error bound is verified on
//     all the voxels of every frame;
//   - the options of pack and play (SOptions) and the compile-time assertions that fix the layout of the format.
//
// The kernels of the codec are in codec_kernels.cuh. The measurement front end (the play loop and the command line)
// is in tools/psi_stream.cu; the library (src/psitreillis.cu) includes this file as well.
#ifndef PSIT_CODEC_STREAM_CUH
#define PSIT_CODEC_STREAM_CUH

#include "codec_kernels.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

// ---------------------------------------------------------------------------------------
// Format
// ---------------------------------------------------------------------------------------

struct StreamHeader {
    char magic[8];  // "PSISTRM2", or "PSISTRM3" if a sparse fine level follows (FineInfo); "PSISTRM1" (coarse levels bounded to 32767 >> fb; its reader clipped the lattice, this one does not: docs/FORMAT.md, section 8) is still readable
    int32_t n, degree, mf, levels, fb, key_interval, n_frames, first_frame;
    float delta, dt_frame, u_rms, re_lambda, kmax_eta, eps;
    int32_t coef[kMaxLevels][3];
    uint64_t index_offset;
    // reserved[0] = kClipMarker when the encoder has filled in its clipping counters; reserved[1 + s] = clippings of
    // site s (kClipKey to kClipLattice of codec_kernels.cuh), all levels and all frames; reserved[7] = clipped frames
    // (low-order 32 bits) and frames whose lattice wraps (high-order 32 bits: the reader then reads by differences).
    uint64_t reserved[8];
};
constexpr uint64_t kClipMarker = 0x3150494C43495350ull;  // "PSICLIP1"
struct FrameRecord {
    uint64_t offset, bytes, hash;
    int32_t type;      // 0: raw key; 1: innovation; 2: innovation after a key (prediction by persistence); 3: coded key
    float mean[3];     // mean velocity of the frame
    float field_err;   // relative field error (RMS over all the voxels) measured at encoding, negative if not measured
    float field_max;   // largest error of a component at voxel centres, divided by u'; zero if not measured
};
static inline bool is_key(int type) { return type == 0 || type == 3; }

// Seal of the file: 48 bytes after the frame table, except with pack --no-seal. The fingerprint of a frame covers only
// its quantised state: without the seal, an altered degree, fb, step or predictor in the header is read without error
// and gives another field. It is at the end of the file because the eight reserved words of the header are taken, and
// so that a sealed file is, to the byte, the file without a seal followed by 48 bytes: a reader from before the seal,
// which stops at the table, reads it without change, and the published fingerprints of files without a seal are
// recovered by removing these 48 bytes. Its size is that of a FrameRecord and its last sixteen bytes are zero: a tool
// that counts the frames from the size of the file sees one frame more in it, whose field_err and field_max are zero.
struct StreamSeal {
    char magic[8];         // "PSISEAL1"
    uint64_t header_hash;  // fingerprint of the 208 bytes of StreamHeader, as they are in the finished file
    uint64_t fine_hash;    // fingerprint of the 64 bytes of FineInfo (PSISTRM3); zero for the other files
    uint64_t index_hash;   // fingerprint of the frame table (48 bytes per frame)
    uint64_t reserved[2];  // zero; the reader refuses any other value
};
constexpr uint64_t kSealHeader = 0x3000ull, kSealFine = 0x3000ull + (1ull << 56), kSealIndex = 0x3000ull + (2ull << 56);  // salts

// Checks of the header of a .psi file, before any allocation: an absurd field would give a giant vector, or a file
// played with the kernels of another degree.
static void check_stream_header(const StreamHeader& sh) {
    CHK(memcmp(sh.magic, "PSISTRM1", 8) == 0 || memcmp(sh.magic, "PSISTRM2", 8) == 0 || memcmp(sh.magic, "PSISTRM3", 8) == 0, "invalid .psi file");
    CHK(sh.degree == 2 || sh.degree == 3, ".psi file: invalid degree");
    CHK(sh.n >= 16 && sh.n <= 8192 && (sh.n & (sh.n - 1)) == 0, ".psi file: invalid resolution");
    CHK(sh.n_frames >= 2 && sh.n_frames <= (1 << 24), ".psi file: invalid number of frames");
    CHK(sh.index_offset >= sizeof(StreamHeader), ".psi file: frame table missing (encoding interrupted?)");
}
// Margin left after the largest frame in the receive buffer on the GPU. An rANS block whose start is in the frame
// reads at most kRansBlockMax bytes (see rans_decode): with this margin, an altered file cannot make the decoder read
// beyond the buffer. What it reads after the end of the frame is the remainder of a previous frame.
constexpr size_t kPacketMargin = (kRansBlockMax + 7) & ~(size_t)7;
// Size of the receive buffer of a reader: the largest frame of the file, according to its table, and not the worst
// case of the coder (`worst`: 2056 bytes per block of 1024 symbols), four to five times larger. Rounded to 8 bytes,
// with the reading margin of a block.
static size_t packet_bytes(const std::vector<FrameRecord>& index, size_t worst) {
    size_t mx = 0;
    for (const FrameRecord& r : index) {
        CHK(r.bytes <= worst, "corrupt frame table: frame larger than the worst case of the coder");
        mx = std::max(mx, (size_t)r.bytes);
    }
    return ((mx + 7) & ~(size_t)7) + kPacketMargin;
}

static void seek64(FILE* f, uint64_t off) {
#ifdef _WIN32
    _fseeki64(f, (long long)off, SEEK_SET);
#else
    fseeko(f, (off_t)off, SEEK_SET);
#endif
}
static uint64_t file_size(FILE* f) {
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    return (uint64_t)_ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    return (uint64_t)ftello(f);
#endif
}

// Geometry of the pyramid and of the stream, common to the sender and the receiver.
struct Geo {
    int n = 0, D = 2, mf = 2, L = 3, fb = 2;
    int nl[kMaxLevels] = {};
    size_t nodes[kMaxLevels] = {}, spec[kMaxLevels] = {}, seg[kMaxLevels] = {}, koff[kMaxLevels] = {};
    double hl[kMaxLevels] = {}, x0[kMaxLevels] = {};
    size_t ecount = 0, units = 0, n_blocks = 0, key_bytes = 0, cap = 0, ex_cap = 0;
    unsigned long long lv_end[kMaxLevels] = {};
    // Coded key frame (type 3): one symbol of one byte per coefficient of the fine level, two (high-order, low-order)
    // per coefficient of the coarse levels. Segments ordered from the coarsest level to the finest, one rANS table
    // each. klo[l], khi[l]: start of the segments of level l; sym_bytes: size of the symbol buffer (keys and
    // innovations).
    size_t kcount = 0, n_kblocks = 0, klo[kMaxLevels] = {}, khi[kMaxLevels] = {}, sym_bytes = 0;
    unsigned long long kseg_end[2 * kMaxLevels] = {};
    int kseg = 0;

    void build(int n_, int D_, int mf_, int L_, int fb_) {
        n = n_;
        D = D_;
        mf = mf_;
        L = L_;
        fb = fb_;
        CHK(L >= 1 && L <= kMaxLevels && mf >= 1 && (mf & (mf - 1)) == 0 && (D == 2 || D == 3), "invalid stream geometry");
        ecount = 0;
        for (int l = 0; l < L; ++l) {
            nl[l] = (n / mf) >> l;
            CHK(nl[l] >= 8, "too many levels for this resolution");
            nodes[l] = (size_t)nl[l] * nl[l] * nl[l];
            spec[l] = (size_t)nl[l] * nl[l] * (nl[l] / 2 + 1);
            hl[l] = (double)kL / nl[l];
            x0[l] = D == 2 ? 0.5 * hl[l] : 0.0;
        }
        for (int l = L - 1; l >= 0; --l) {
            seg[l] = ecount;
            ecount += 3 * nodes[l];
        }
        check_codec_limits(D, L, fb, ecount);
        units = (ecount + 15) / 16;
        n_blocks = (units + 63) / 64;
        for (int j = 0; j < L; ++j) lv_end[j] = seg[L - 1 - j] + 3 * nodes[L - 1 - j];
        size_t o = (3 * nodes[0] + 7) & ~(size_t)7;  // key frame: level 0 (int8), then levels 1..L-1 (int16)
        koff[0] = 0;
        for (int l = 1; l < L; ++l) {
            koff[l] = o;
            o += 6 * nodes[l];
        }
        key_bytes = o;
        kcount = 0;
        kseg = 0;
        for (int l = L - 1; l >= 1; --l) {
            khi[l] = kcount;
            kcount += 3 * nodes[l];
            kseg_end[kseg++] = kcount;
            klo[l] = kcount;
            kcount += 3 * nodes[l];
            kseg_end[kseg++] = kcount;
        }
        klo[0] = kcount;
        kcount += 3 * nodes[0];
        kseg_end[kseg++] = kcount;
        n_kblocks = (kcount + 1023) / 1024;
        sym_bytes = std::max(units * 16, (kcount + 15) / 16 * 16);
        // A coded key is kept only if it does not exceed the raw key: `cap` does not depend on its worst case.
        ex_cap = std::max<size_t>(65536, ecount / 64);  // exceptions per frame, at most (see ExEntry in codec_kernels.cuh)
        cap = std::max(key_bytes, (size_t)L * 512 + n_blocks * 4 + n_blocks * 2056) + 8 + ex_cap * sizeof(ExEntry);
    }
    float field_scale(float delta) const { return (float)(delta / ((double)(1 << fb) * hl[0])); }
};

// ---------------------------------------------------------------------------------------
// Kernels specific to the standalone file chain
// ---------------------------------------------------------------------------------------

__host__ __device__ inline unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// Same fingerprint as hash_bytes, computed on the host: used for the seal, whose arrays are only a few kilobytes.
static uint64_t host_hash(const void* data, size_t count, uint64_t salt) {
    const unsigned char* a = (const unsigned char*)data;
    uint64_t h = 0;
    for (size_t i = 0; i < count; ++i) h ^= mix64(salt + (uint64_t)i * 256ull + a[i]);
    return h;
}
// Fingerprint of a byte array, independent of the order of execution of the threads: XOR, over all the bytes i, of
// mix64(salt + 256 i + a[i]). A thread processes 8 bytes, the warp reduces by XOR, then a single atomic per warp. XOR
// being associative and commutative, the value is that of one atomic per byte, without the contention on the output
// word. `a` must be aligned on 8 bytes (pointer returned by cudaMalloc) and the block must hold a whole number of
// warps; to be launched on (count + 7) / 8 threads. No thread exits before the reduction: the full mask of
// __shfl_xor_sync requires it.
__global__ void hash_bytes(const unsigned char* a, size_t count, unsigned long long salt, unsigned long long* out) {
    const size_t i0 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    unsigned long long h = 0;
    if (i0 + 8 <= count) {
        const unsigned long long w = *(const unsigned long long*)(a + i0);  // bytes i0 to i0 + 7, the first one low-order
        for (int j = 0; j < 8; ++j) h ^= mix64(salt + (unsigned long long)(i0 + j) * 256ull + ((w >> (8 * j)) & 0xFFull));
    } else {
        for (size_t i = i0; i < count; ++i) h ^= mix64(salt + (unsigned long long)i * 256ull + a[i]);
    }
    for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xFFFFFFFFu, h, o);
    if ((threadIdx.x & 31) == 0 && h != 0) atomicXor(out, h);
}
// Reconstruction error over the whole grid: at the centre of a voxel, the reference is the raw data itself.
// One thread per brick of 4^3 voxels; it returns the sum of the squares of the error and of the reference, and the
// largest error of a component in the brick.
template <class F>
__global__ void field_err_bricks(F f, const float* planar, int n, float* e2, float* r2, float* emax) {
    const int nb = n / 4;
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (size_t)nb * nb * nb) return;
    const int bx = (int)(b % nb), by = (int)((b / nb) % nb), bz = (int)(b / ((size_t)nb * nb));
    const size_t voxels = (size_t)n * n * n;
    const float h = kL / n;
    float se = 0.0f, sr = 0.0f, mx = 0.0f;
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) {
                const int ix = 4 * bx + i, iy = 4 * by + j, iz = 4 * bz + k;
                const size_t v = ((size_t)iz * n + iy) * n + ix;
                const V3 u = sample(f, V3{(ix + 0.5f) * h, (iy + 0.5f) * h, (iz + 0.5f) * h}, 0.0f);
                const float dx = u.x - planar[v], dy = u.y - planar[voxels + v], dz = u.z - planar[2 * voxels + v];
                se += dx * dx + dy * dy + dz * dz;
                sr += planar[v] * planar[v] + planar[voxels + v] * planar[voxels + v] + planar[2 * voxels + v] * planar[2 * voxels + v];
                mx = fmaxf(mx, fmaxf(fabsf(dx), fmaxf(fabsf(dy), fabsf(dz))));
            }
    e2[b] = se;
    r2[b] = sr;
    emax[b] = mx;
}
// Coded key frames (type 3). The residual is that of a 3D Lorenzo prediction of the state from its already decoded
// neighbours (x-1, y-1, z-1; zero outside the domain): the first difference of the state along the three axes. It is
// written modulo 2^8 (fine level) or 2^16 (coarse levels). Since the state fits in its type, three cumulative sums
// modulo the same power of two give it back exactly: neither overflow, nor exception list.
//   fine level   : one symbol, the residual on 8 bits;
//   coarse level : the residual on 16 bits is folded (0, -1, 1, -2... become 0, 1, 2, 3...) then split into two
//                  bytes, each with its table; the high-order byte is almost always zero.
template <class T>
__global__ void key_residual(const T* q, int n, unsigned char* lo, unsigned char* hi) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    const bool X = i % n != 0, Y = (i / n) % n != 0, Z = i / ((size_t)n * n) != 0;
    const ptrdiff_t sx = 3, sy = 3 * (ptrdiff_t)n, sz = 3 * (ptrdiff_t)n * n;
    for (int c = 0; c < 3; ++c) {
        const T* p = q + 3 * i + c;
        int r = p[0];
        if (X) r -= p[-sx];
        if (Y) r -= p[-sy];
        if (Z) r -= p[-sz];
        if (X && Y) r += p[-sx - sy];
        if (X && Z) r += p[-sx - sz];
        if (Y && Z) r += p[-sy - sz];
        if (X && Y && Z) r -= p[-sx - sy - sz];
        if (sizeof(T) == 1) {
            lo[3 * i + c] = (unsigned char)r;
        } else {
            const int r16 = (short)r;
            const uint32_t zz = (((uint32_t)r16 << 1) ^ (uint32_t)(r16 >> 31)) & 0xFFFFu;
            lo[3 * i + c] = (unsigned char)(zz & 0xFFu);
            hi[3 * i + c] = (unsigned char)(zz >> 8);
        }
    }
}
// Receiver, first cumulative sum (along x), read from the decoded symbols. One thread per line.
template <class T>
__global__ void key_scan_x(const unsigned char* lo, const unsigned char* hi, int n, T* q) {
    const size_t line = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= (size_t)n * n) return;
    const size_t j0 = 3 * line * n;
    int acc[3] = {0, 0, 0};
    for (int x = 0; x < n; ++x)
        for (int c = 0; c < 3; ++c) {
            const size_t j = j0 + 3 * (size_t)x + c;
            int r;
            if (sizeof(T) == 1) {
                r = (signed char)lo[j];
            } else {
                const uint32_t zz = (uint32_t)lo[j] | ((uint32_t)hi[j] << 8);
                r = (int)(zz >> 1) ^ -(int)(zz & 1u);
            }
            acc[c] += r;
            q[j] = (T)acc[c];
        }
}
// Receiver, following cumulative sums, in place: along y (axis 1) then along z (axis 2). One thread per line;
// neighbouring threads process neighbouring x, hence neighbouring addresses.
template <class T>
__global__ void key_scan(T* q, int n, int axis) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)n * n) return;
    const size_t a = t % n, b = t / n;
    const size_t first = axis == 1 ? b * n * n + a : b * n + a, stride = axis == 1 ? (size_t)n : (size_t)n * n;
    int acc[3] = {0, 0, 0};
    for (int k = 0; k < n; ++k) {
        T* p = q + 3 * (first + k * stride);
        for (int c = 0; c < 3; ++c) {
            acc[c] += p[c];
            p[c] = (T)acc[c];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Sparse fine level (PSISTRM3 files): state of a frame and structure read by the kernels
// ---------------------------------------------------------------------------------------
//
// A frame carries, for the fine level: the map of the bricks present (one bit per brick of 4^3 nodes), the mask of the
// 64 nodes of each brick present (one bit per retained node), and three coefficients on 8 bits per retained node, in
// the order of the bricks then of the bits. The fine level of a frame depends on no other frame. The kernels read the
// union of the two frames of the interval (see FineLevel in codec_kernels.cuh), rebuilt at each frame.

struct FineInfo {  // follows StreamHeader in a PSISTRM3 file
    float tolerance;                 // bound requested on the largest error of a component at voxel centres, as a fraction of u'
    float eps;                       // quantisation error of the fine level, as a fraction of u'
    float delta;                     // quantisation step of the fine potential
    float max_error;                 // largest error observed over the whole file, as a fraction of u'
    uint32_t max_bricks, max_nodes;  // maximum per frame: bricks present, nodes retained
    int32_t brick;                   // side of a brick, in nodes
    uint32_t frames_over;            // frames whose largest error exceeds the certification threshold (tolerance - margin / 2)
    // reserved[0]: arithmetic margin, as a fraction of u' (the bits of a float; zero for a file from before the evening
    // of 15 November, marked and certified at the tolerance itself). The other words are zero.
    uint32_t reserved[8];
};
static inline float fine_margin(const FineInfo& fi) {
    float m;
    memcpy(&m, &fi.reserved[0], sizeof(m));
    return m;
}
// Certification threshold, as a fraction of u': half of the margin covers the rounding difference between the marking
// kernel and the certification kernel, the other half is left to the other kernels of the reader.
static inline float fine_certified(float tolerance, float margin) { return tolerance - 0.5f * margin; }
struct FineTrailer {  // last sixteen bytes of the payload of a PSISTRM3 frame
    uint32_t n_bricks, n_nodes, bytes, reserved;  // bytes: size of the section of the fine level that precedes, a multiple of 8
};
// Maximum size of the section of the fine level of a frame (padding, rANS section in the worst case, frame trailer).
static size_t fine_section_cap(size_t bm_bytes, size_t max_bricks, size_t max_nodes) {
    const size_t count = bm_bytes + 8 * max_bricks + 3 * max_nodes, blocks = (count + 1023) / 1024;
    return 8 + 3 * 512 + blocks * (4 + 2056) + 8 + sizeof(FineTrailer);
}

// ---------------------------------------------------------------------------------------
// Opening of a file by a reader: size, frame table, seal
// ---------------------------------------------------------------------------------------

// What the opening establishes, on the host, before any allocation on the GPU.
struct StreamOpen {
    FineInfo fi{};
    bool has_fine = false;  // PSISTRM3 file
    bool sealed = false;    // seal present and verified
    std::vector<FrameRecord> index;
    uint64_t file_bytes = 0;
};
// To be called after check_stream_header. Reads the header of the fine level, the frame table and the seal, and
// refuses the file (message, code 1) if:
//   - its size is neither that of a file without a seal (it ends with the table), nor that of a sealed file;
//   - the seal is present and one of its three fingerprints does not match: the header, the header of the fine level
//     or the table have changed since writing;
//   - a frame of the table leaves the frame area, carries an unknown type, or the first one is not a key
//     (--seek goes back to a key).
// A file without a seal is accepted: `sealed` stays false and the reader says so.
static void read_stream_index(FILE* in, const StreamHeader& sh, StreamOpen& so) {
    so.has_fine = sh.magic[7] == '3';
    const uint64_t data0 = sizeof(StreamHeader) + (so.has_fine ? sizeof(FineInfo) : 0);
    if (so.has_fine) {
        seek64(in, sizeof(StreamHeader));
        CHK(fread(&so.fi, sizeof(so.fi), 1, in) == 1 && so.fi.brick == kFineSide, "header of the fine level unreadable");
    }
    so.file_bytes = file_size(in);
    const uint64_t table = (uint64_t)sh.n_frames * sizeof(FrameRecord);
    CHK(sh.index_offset >= data0 && sh.index_offset <= so.file_bytes && table <= so.file_bytes - sh.index_offset, "frame table unreadable");
    const uint64_t tail = so.file_bytes - sh.index_offset - table;
    static const char* const kTail = ".psi file: unexpected bytes after the frame table (neither end of file, nor seal)";
    CHK(tail == 0 || tail == sizeof(StreamSeal), kTail);
    so.index.resize(sh.n_frames);
    seek64(in, sh.index_offset);
    CHK(fread(so.index.data(), sizeof(FrameRecord), so.index.size(), in) == so.index.size(), "frame table unreadable");
    if (tail) {
        StreamSeal seal;
        CHK(fread(&seal, sizeof(seal), 1, in) == 1 && memcmp(seal.magic, "PSISEAL1", 8) == 0 && seal.reserved[0] == 0 && seal.reserved[1] == 0, kTail);
        CHK(seal.header_hash == host_hash(&sh, sizeof(sh), kSealHeader), "seal of the file: the header has been altered since writing");
        CHK(seal.fine_hash == (so.has_fine ? host_hash(&so.fi, sizeof(so.fi), kSealFine) : 0ull),
            "seal of the file: the header of the fine level has been altered since writing");
        CHK(seal.index_hash == host_hash(so.index.data(), so.index.size() * sizeof(FrameRecord), kSealIndex),
            "seal of the file: the frame table has been altered since writing");
        so.sealed = true;
    }
    const bool v1 = sh.magic[7] == '1';
    const uint64_t room = sh.index_offset - data0;  // frame area: from the end of the headers to the table
    for (const FrameRecord& r : so.index) {
        CHK(r.type >= 0 && r.type <= 3 && !(v1 && r.type == 3), "corrupt frame table: unknown frame type");
        CHK(r.bytes <= room && r.offset >= data0 && r.offset <= sh.index_offset - r.bytes, "corrupt frame table: frame outside the frame area");
    }
    CHK(is_key(so.index[0].type), "corrupt frame table: the first frame is not a key frame");
}

// Check, on the host, of an rANS section of `avail` bytes: S frequency tables, the offset table of n_blocks blocks,
// then the stream area. The frame is refused if the tables or the offsets leave the section, if a table does not sum
// to 4096, or if the offsets are not those of blocks that follow one another in the area: the first zero, each at
// least 4 bytes from the previous one (the initial state of a block), the last at least 4 bytes from the end. The
// start of each block is then in the frame: with the reading bound of rans_decode and kPacketMargin, no block makes
// the decoder read outside the receive buffer. The loop has no early exit: it goes once through the offset table
// (one word per block of 1024 symbols).
static void check_rans_section(const unsigned char* p, size_t avail, size_t S, size_t n_blocks, const char* msg) {
    const size_t head = S * 512 + n_blocks * 4;
    CHK(n_blocks >= 1 && head + 4 <= avail, msg);
    for (size_t s = 0; s < S; ++s) {
        const uint16_t* f = (const uint16_t*)(p + s * 512);
        unsigned sum = 0;
        for (int v = 0; v < 256; ++v) sum += f[v];
        CHK(sum == 4096, "corrupt frequency table");
    }
    const uint32_t* off = (const uint32_t*)(p + S * 512);
    uint64_t prev = off[0];
    bool bad = prev != 0;
    for (size_t b = 1; b < n_blocks; ++b) {
        const uint64_t o = off[b];
        bad |= o < prev + 4;
        prev = o;
    }
    CHK(!bad && prev + 4 <= avail - head, msg);
}

// PSISTRM1 files: their original reader clipped to +-32767 each partial sum of the pyramid and the sampling lattice;
// this one keeps the sums on 32 bits. Both give the same lattice if, and only if, no sum leaves +-32767: these kernels
// count those that leave it. Launched by play on PSISTRM1 files only, outside the measured times.
__global__ void count_beyond(const int* a, size_t count, int lim, unsigned* cnt) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (a[i] > lim || a[i] < -lim) atomicAdd(cnt, 1u);
}
template <int D>
__global__ void count_lattice_beyond(const signed char* c, const int* up, int n, int fb, int lim, unsigned* cnt) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    int q[3];
    fine_sum<D, signed char>(c, up, n, fb, (int)(i % n), (int)((i / n) % n), (int)(i / ((size_t)n * n)), q);
    const unsigned bad = (unsigned)(q[0] > lim || q[0] < -lim) + (unsigned)(q[1] > lim || q[1] < -lim) + (unsigned)(q[2] > lim || q[2] < -lim);
    if (bad) atomicAdd(cnt, bad);
}

__global__ void fine_popc32(const uint32_t* a, const uint32_t* b, size_t count, uint32_t* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (uint32_t)__popc(a[i] | (b ? b[i] : 0u));
}
__global__ void fine_popc64(const unsigned long long* a, size_t count, uint32_t* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (uint32_t)__popcll(a[i]);
}
__device__ inline bool fine_bit(const uint32_t* bm, size_t b) { return (bm[b >> 5] >> (b & 31)) & 1u; }
// Exclusive prefix sum of `count` entries, plus the total in out[count], without synchronisation or allocation: sums
// by blocks of 256, prefix of the blocks by a single thread, then rank of each entry in its block. a and out are
// distinct.
__global__ void scan_block_sums(const uint32_t* a, size_t count, uint32_t* blk) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x, i0 = b * 256;
    if (i0 >= count) return;
    const size_t i1 = i0 + 256 < count ? i0 + 256 : count;
    uint32_t sum = 0;
    for (size_t i = i0; i < i1; ++i) sum += a[i];
    blk[b] = sum;
}
__global__ void scan_block_prefix(uint32_t* blk, size_t blocks) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t run = 0;
    for (size_t b = 0; b < blocks; ++b) {
        const uint32_t v = blk[b];
        blk[b] = run;
        run += v;
    }
    blk[blocks] = run;
}
__global__ void scan_finish(const uint32_t* a, const uint32_t* blk, size_t count, uint32_t* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i > count) return;
    if (i == count) {
        out[i] = blk[(count + 255) / 256];
        return;
    }
    uint32_t sum = blk[i >> 8];
    for (size_t j = i & ~(size_t)255; j < i; ++j) sum += a[j];
    out[i] = sum;
}

struct FineSlots {
    const uint32_t *bm[2], *bbase[2];
    const unsigned long long* mask[2];
    const uint32_t* mbase[2];
    const signed char* coef[2];
    // Announced counts of each slot (bricks present, nodes retained) and number of blocks of `pool`. In a valid frame,
    // the map and the masks count exactly these numbers and the tests below change nothing. In an altered frame, the
    // decoded map can count more bricks, or the masks more nodes, than announced: what exceeds them is treated as
    // absent, instead of reading outside the masks and the coefficients or writing outside `pool`.
    uint32_t nbk[2], nnd[2], cap_pool;
};
// Union of the two slots: words of `tab` and of `occ`, and block of each brick present. One thread per brick.
__global__ void fine_fill(FineSlots s, const uint32_t* ubase, int nb, int D, int2* pool, uint32_t* tab, unsigned long long* occ) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (size_t)nb * nb * nb) return;
    const int bx = (int)(b % nb), by = (int)((b / nb) % nb), bz = (int)(b / ((size_t)nb * nb)), mb = nb - 1;
    uint32_t entry = 0;
    unsigned long long um[8];  // retained nodes of the eight bricks (b + {0,1}^3), the two slots merged
    for (int k = 0; k < 8; ++k) {
        const size_t o = ((size_t)((bz + (k >> 2)) & mb) * nb + ((by + ((k >> 1) & 1)) & mb)) * nb + ((bx + (k & 1)) & mb);
        const size_t w = o >> 5;
        const uint32_t bit = (uint32_t)(o & 31), below = (1u << bit) - 1u;
        um[k] = 0;
        for (int t = 0; t < 2; ++t)
            if ((s.bm[t][w] >> bit) & 1u) {
                const uint32_t rank = s.bbase[t][w] + (uint32_t)__popc(s.bm[t][w] & below);
                if (rank < s.nbk[t]) um[k] |= s.mask[t][rank];
            }
        if (fine_bit(s.bm[0], o) || fine_bit(s.bm[1], o)) entry |= 1u << k;
    }
    if (!entry) {
        tab[b] = 0;
        occ[b] = 0;
        return;
    }
    {  // occ: for each node of the brick, is there a retained node in the support of D + 1 nodes per axis that starts there?
        unsigned char rx[8][8];  // per row (Y, Z) of the cube of 8^3 nodes: bit x if there is a retained node in x .. x + D
        for (int Z = 0; Z < 8; ++Z)
            for (int Y = 0; Y < 8; ++Y) {
                const int kb = ((Z >> 2) << 2) | ((Y >> 2) << 1), sh = ((Z & 3) << 4) | ((Y & 3) << 2);
                const unsigned row = (unsigned)((um[kb] >> sh) & 0xFull) | ((unsigned)((um[kb | 1] >> sh) & 0xFull) << 4);
                unsigned r = row | (row >> 1) | (row >> 2);
                if (D == 3) r |= row >> 3;
                rx[Z][Y] = (unsigned char)(r & 0xFu);
            }
        unsigned long long oc = 0;
        for (int z = 0; z < 4; ++z)
            for (int y = 0; y < 4; ++y) {
                unsigned v = 0;
                for (int c = 0; c <= D; ++c)
                    for (int e = 0; e <= D; ++e) v |= rx[z + c][y + e];
                oc |= (unsigned long long)v << ((z << 4) | (y << 2));
            }
        occ[b] = oc;
    }
    if (entry & 1u) {
        const size_t w = b >> 5;
        const uint32_t bit = (uint32_t)(b & 31), below = (1u << bit) - 1u;
        const uint32_t ub = ubase[w] + (uint32_t)__popc((s.bm[0][w] | s.bm[1][w]) & below);
        if (ub >= s.cap_pool) {  // altered frame: more bricks than blocks; the brick stays without a block (block 0 for whoever would read it)
            tab[b] = entry & ~1u;
            return;
        }
        entry |= ub << 8;
        int2* blk = pool + (size_t)ub * kFineNodes;
        for (int k = 0; k < kFineNodes; ++k) blk[k] = make_int2(0, 0);
        for (int t = 0; t < 2; ++t) {
            if (!((s.bm[t][w] >> bit) & 1u)) continue;
            const uint32_t rank = s.bbase[t][w] + (uint32_t)__popc(s.bm[t][w] & below);
            if (rank >= s.nbk[t]) continue;
            const unsigned long long mk = s.mask[t][rank];
            if (s.mbase[t][rank] + (uint32_t)__popcll(mk) > s.nnd[t]) continue;
            const signed char* c = s.coef[t] + 3 * (size_t)s.mbase[t][rank];
            for (int k = 0, j = 0; k < kFineNodes; ++k) {
                if (!((mk >> k) & 1ull)) continue;
                const int wd = pack8(c[3 * j], c[3 * j + 1], c[3 * j + 2]);
                if (t) blk[k].y = wd;
                else blk[k].x = wd;
                ++j;
            }
        }
    }
    tab[b] = entry;
}
// Sender: mask of the retained nodes of each brick (one entry per brick) and brick map. `need`: one byte per node,
// non-zero if the node is retained; bm must be zeroed.
__global__ void fine_brick_masks(const unsigned char* need, int n, int nb, unsigned long long* emask, uint32_t* bm) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (size_t)nb * nb * nb) return;
    const int x0 = (int)(b % nb) << kFineLog, y0 = (int)((b / nb) % nb) << kFineLog, z0 = (int)(b / ((size_t)nb * nb)) << kFineLog;
    unsigned long long mk = 0;
    for (int k = 0; k < kFineNodes; ++k) {
        const int x = x0 + (k & (kFineSide - 1)), y = y0 + ((k >> kFineLog) & (kFineSide - 1)), z = z0 + (k >> (2 * kFineLog));
        if (need[((size_t)z * n + y) * n + x]) mk |= 1ull << k;
    }
    emask[b] = mk;
    if (mk) atomicOr(&bm[b >> 5], 1u << (b & 31));
}
// Sender: compaction of the masks, then of the coefficients of the retained nodes (q: three coefficients per node).
__global__ void fine_compact_masks(const unsigned long long* emask, const uint32_t* bm, const uint32_t* bbase, size_t bricks, unsigned long long* mask) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bricks || !fine_bit(bm, b)) return;
    mask[bbase[b >> 5] + (uint32_t)__popc(bm[b >> 5] & ((1u << (b & 31)) - 1u))] = emask[b];
}
__global__ void fine_compact_coefs(const signed char* q, const unsigned long long* emask, const uint32_t* bm, const uint32_t* bbase,
                                   const uint32_t* mbase, int n, int nb, signed char* coef) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (size_t)nb * nb * nb || !fine_bit(bm, b)) return;
    const int x0 = (int)(b % nb) << kFineLog, y0 = (int)((b / nb) % nb) << kFineLog, z0 = (int)(b / ((size_t)nb * nb)) << kFineLog;
    const unsigned long long mk = emask[b];
    signed char* c = coef + 3 * (size_t)mbase[bbase[b >> 5] + (uint32_t)__popc(bm[b >> 5] & ((1u << (b & 31)) - 1u))];
    for (int k = 0, j = 0; k < kFineNodes; ++k) {
        if (!((mk >> k) & 1ull)) continue;
        const int x = x0 + (k & (kFineSide - 1)), y = y0 + ((k >> kFineLog) & (kFineSide - 1)), z = z0 + (k >> (2 * kFineLog));
        const size_t node = ((size_t)z * n + y) * n + x;
        for (int d = 0; d < 3; ++d) c[3 * j + d] = q[3 * node + d];
        ++j;
    }
}

struct FineState {
    int n = 0, nb = 0;
    size_t bricks = 0, words = 0, cap_bricks = 0, cap_nodes = 0, cap_pool = 0;
    // Per frame slot: brick map, rank of the first brick of each word of the map, mask of the nodes of each brick
    // present, rank of the first retained node of each brick, coefficients of the retained nodes.
    uint32_t *bm[2] = {}, *bbase[2] = {}, *mbase[2] = {};
    unsigned long long* mask[2] = {};
    signed char* coef[2] = {};
    uint32_t n_bricks[2] = {}, n_nodes[2] = {};
    // Structure read by the kernels: union of the two slots.
    uint32_t *tab = nullptr, *ubase = nullptr;
    unsigned long long* occ = nullptr;
    int2* pool = nullptr;
    uint32_t *scnt = nullptr, *sblk = nullptr;  // work arrays of the prefix sums: counts, sums per block

    // out[0..count] = exclusive prefix sum of a[0..count-1]; out[count] is the total.
    void scan(const uint32_t* a, size_t count, uint32_t* out, cudaStream_t st) {
        if (!count) {
            CK(cudaMemsetAsync(out, 0, 4, st));
            return;
        }
        const size_t blocks = (count + 255) / 256;
        scan_block_sums<<<blocks_for(blocks), 256, 0, st>>>(a, count, sblk);
        scan_block_prefix<<<1, 32, 0, st>>>(sblk, blocks);
        scan_finish<<<blocks_for(count + 1), 256, 0, st>>>(a, sblk, count, out);
    }
    void init(int n_, size_t max_bricks, size_t max_nodes) {
        n = n_;
        nb = n >> kFineLog;
        CHK(n >= 32 && (n & (n - 1)) == 0, "fine level: resolution not supported");
        bricks = (size_t)nb * nb * nb;
        CHK(bricks <= ((size_t)1 << 24), "fine level: the block index of a brick fits in 24 bits (1024 fine nodes per axis at most)");
        words = bricks / 32;
        cap_bricks = std::max<size_t>(1, std::min(max_bricks, bricks));
        cap_nodes = std::max<size_t>(1, std::min(max_nodes, (size_t)n * n * n));
        cap_pool = std::min(2 * cap_bricks, bricks);
        for (int t = 0; t < 2; ++t) {
            CK(cudaMalloc(&bm[t], words * 4));
            CK(cudaMemset(bm[t], 0, words * 4));
            CK(cudaMalloc(&bbase[t], (words + 1) * 4));
            CK(cudaMemset(bbase[t], 0, (words + 1) * 4));
            CK(cudaMalloc(&mask[t], cap_bricks * 8));
            CK(cudaMalloc(&mbase[t], (cap_bricks + 1) * 4));
            CK(cudaMemset(mbase[t], 0, (cap_bricks + 1) * 4));
            CK(cudaMalloc(&coef[t], 3 * cap_nodes + 8));
        }
        CK(cudaMalloc(&tab, bricks * 4));
        CK(cudaMemset(tab, 0, bricks * 4));
        CK(cudaMalloc(&occ, bricks * 8));
        CK(cudaMemset(occ, 0, bricks * 8));
        CK(cudaMalloc(&ubase, (words + 1) * 4));
        const size_t longest = std::max(words, cap_bricks);
        CK(cudaMalloc(&scnt, longest * 4));
        CK(cudaMalloc(&sblk, ((longest + 255) / 256 + 1) * 4));
        CK(cudaMalloc(&pool, cap_pool * kFineNodes * sizeof(int2)));
    }
    size_t bm_bytes() const { return words * 4; }
    // Ranks of the frame of slot t, once its map, its masks and its counts are written.
    void index(int t, cudaStream_t st) {
        fine_popc32<<<blocks_for(words), 256, 0, st>>>(bm[t], nullptr, words, scnt);
        scan(scnt, words, bbase[t], st);
        if (n_bricks[t]) fine_popc64<<<blocks_for(n_bricks[t]), 256, 0, st>>>(mask[t], n_bricks[t], scnt);
        scan(scnt, n_bricks[t], mbase[t], st);
    }
    // Union of the two slots: brick tables and blocks of nodes. D: degree of the splines (size of the support).
    void rebuild(int D, cudaStream_t st) {
        fine_popc32<<<blocks_for(words), 256, 0, st>>>(bm[0], bm[1], words, scnt);
        scan(scnt, words, ubase, st);
        FineSlots s{};
        for (int t = 0; t < 2; ++t) {
            s.bm[t] = bm[t];
            s.bbase[t] = bbase[t];
            s.mask[t] = mask[t];
            s.mbase[t] = mbase[t];
            s.coef[t] = coef[t];
            s.nbk[t] = n_bricks[t];
            s.nnd[t] = n_nodes[t];
        }
        s.cap_pool = (uint32_t)cap_pool;
        fine_fill<<<blocks_for(bricks), 256, 0, st>>>(s, ubase, nb, D, pool, tab, occ);
    }
    // Sender: stores in slot t the nodes marked in `need`, with their coefficients q. emask: one entry per brick.
    // Returns false, with the slot emptied, if the capacity is exceeded.
    bool pack_dense(int t, const unsigned char* need, const signed char* q, unsigned long long* emask) {
        CK(cudaMemset(bm[t], 0, words * 4));
        fine_brick_masks<<<blocks_for(bricks), 256>>>(need, n, nb, emask, bm[t]);
        fine_popc32<<<blocks_for(words), 256>>>(bm[t], nullptr, words, scnt);
        scan(scnt, words, bbase[t], nullptr);
        uint32_t nbk = 0, nn = 0;
        CK(cudaMemcpy(&nbk, bbase[t] + words, 4, cudaMemcpyDeviceToHost));
        if (nbk <= cap_bricks) {
            n_bricks[t] = nbk;
            if (nbk) fine_compact_masks<<<blocks_for(bricks), 256>>>(emask, bm[t], bbase[t], bricks, mask[t]);
            if (nbk) fine_popc64<<<blocks_for(nbk), 256>>>(mask[t], nbk, scnt);
            scan(scnt, nbk, mbase[t], nullptr);
            CK(cudaMemcpy(&nn, mbase[t] + nbk, 4, cudaMemcpyDeviceToHost));
        }
        if (nbk > cap_bricks || nn > cap_nodes) {
            CK(cudaMemset(bm[t], 0, words * 4));
            n_bricks[t] = n_nodes[t] = 0;
            index(t, nullptr);
            CK(cudaDeviceSynchronize());
            return false;
        }
        n_nodes[t] = nn;
        if (nn) fine_compact_coefs<<<blocks_for(bricks), 256>>>(q, emask, bm[t], bbase[t], mbase[t], n, nb, coef[t]);
        return true;
    }
    FineLevel level(int D, float delta) const {
        FineLevel f{};
        f.occ = occ;
        f.tab = tab;
        f.pool = pool;
        f.n = n;
        f.nb = nb;
        f.inv_h = (float)((double)n / (double)kL);
        f.x0 = D == 2 ? (float)(0.5 * (double)kL / n) : 0.0f;
        f.s = (float)((double)delta * (double)n / (double)kL);
        return f;
    }
    size_t resident_bytes() const {
        const size_t longest = std::max(words, cap_bricks);
        return 2 * (words * 4 + (words + 1) * 4 + cap_bricks * 8 + (cap_bricks + 1) * 4 + 3 * cap_nodes + 8) + bricks * 12 + (words + 1) * 4 +
               cap_pool * kFineNodes * sizeof(int2) + longest * 4 + ((longest + 255) / 256 + 1) * 4;
    }
};

// Residual of the base at voxel centres, laid out for an in-place real-to-complex transform (lines of n + 2 floats).
template <class F>
__global__ void resid_padded(F f, const float* planar, int n, float* o0, float* o1, float* o2) {
    const size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t voxels = (size_t)n * n * n;
    if (v >= voxels) return;
    const int ix = (int)(v % n), iy = (int)((v / n) % n), iz = (int)(v / ((size_t)n * n));
    const float h = kL / n;
    const V3 u = sample(f, V3{(ix + 0.5f) * h, (iy + 0.5f) * h, (iz + 0.5f) * h}, 0.0f);
    const size_t j = ((size_t)iz * n + iy) * (n + 2) + ix;
    o0[j] = planar[v] - u.x;
    o1[j] = planar[voxels + v] - u.y;
    o2[j] = planar[2 * voxels + v] - u.z;
}
// Quantisation of the fine potential (in-place output of the inverse transform: lines of n + 2 floats).
__global__ void fine_quantize(const float* t0, const float* t1, const float* t2, int n, float inv_delta, signed char* q) {
    const size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= (size_t)n * n * n) return;
    const size_t j = (v / n) * (n + 2) + v % n;
    q[3 * v] = (signed char)clampi((int)rintf(t0[j] * inv_delta), -127, 127);
    q[3 * v + 1] = (signed char)clampi((int)rintf(t1[j] * inv_delta), -127, 127);
    q[3 * v + 2] = (signed char)clampi((int)rintf(t2[j] * inv_delta), -127, 127);
}
// Voxels whose error exceeds the tolerance: retains the fine nodes of their support (the same as those that fine_at
// reads at the centre of the voxel). cnt[0]: voxels outside the tolerance; cnt[1]: newly retained nodes (zero or
// not). b0, b1, b2 (optional): receive the evaluated field, as three planes of n^3 floats. At the first pass of a
// frame no fine node is retained yet: this field is that of the base, which the following passes read back
// (fine_mark_cached) instead of re-evaluating it. The base is stored by this kernel, and not by a separate kernel, to
// keep its arithmetic to the bit: evaluated elsewhere, it differed by a last bit in degree 2, and voxels at the
// limit of the tolerance were no longer marked.
template <int D, class F>
__global__ void fine_mark(F f, const float* planar, int n, float tol, unsigned char* need, unsigned* cnt, float* b0, float* b1, float* b2) {
    const size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t voxels = (size_t)n * n * n;
    if (v >= voxels) return;
    const int ix = (int)(v % n), iy = (int)((v / n) % n), iz = (int)(v / ((size_t)n * n)), m = n - 1;
    const float h = kL / n;
    const V3 u = sample(f, V3{(ix + 0.5f) * h, (iy + 0.5f) * h, (iz + 0.5f) * h}, 0.0f);
    if (b0) {
        b0[v] = u.x;
        b1[v] = u.y;
        b2[v] = u.z;
    }
    const float d = fmaxf(fabsf(planar[v] - u.x), fmaxf(fabsf(planar[voxels + v] - u.y), fabsf(planar[2 * voxels + v] - u.z)));
    if (!(d > tol)) return;
    int added = 0;
    for (int c = -1; c < D; ++c)
        for (int b = -1; b < D; ++b)
            for (int a = -1; a < D; ++a) {
                const size_t node = ((size_t)((iz + c) & m) * n + ((iy + b) & m)) * n + ((ix + a) & m);
                if (!need[node]) {
                    need[node] = 1;
                    ++added;
                }
            }
    atomicAdd(&cnt[0], 1u);
    if (added) atomicAdd(&cnt[1], (unsigned)added);
}
// Same marking as fine_mark, the base being read from the planes that its first pass filled instead of being
// re-evaluated. The complete field is formed as in sample(PsiFine): the value of the base, then the addition of
// fine_at.
template <int D>
__global__ void fine_mark_cached(FineLevel fl, int early, const float* b0, const float* b1, const float* b2, const float* planar, int n, float tol,
                                 unsigned char* need, unsigned* cnt) {
    const size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t voxels = (size_t)n * n * n;
    if (v >= voxels) return;
    const int ix = (int)(v % n), iy = (int)((v / n) % n), iz = (int)(v / ((size_t)n * n)), m = n - 1;
    const float h = kL / n;
    const V3 base = {b0[v], b1[v], b2[v]};
    const V3 u = base + fine_at<D>(fl, V3{(ix + 0.5f) * h, (iy + 0.5f) * h, (iz + 0.5f) * h}, 0.0f, early);
    const float d = fmaxf(fabsf(planar[v] - u.x), fmaxf(fabsf(planar[voxels + v] - u.y), fabsf(planar[2 * voxels + v] - u.z)));
    if (!(d > tol)) return;
    int added = 0;
    for (int c = -1; c < D; ++c)
        for (int b = -1; b < D; ++b)
            for (int a = -1; a < D; ++a) {
                const size_t node = ((size_t)((iz + c) & m) * n + ((iy + b) & m)) * n + ((ix + a) & m);
                if (!need[node]) {
                    need[node] = 1;
                    ++added;
                }
            }
    atomicAdd(&cnt[0], 1u);
    if (added) atomicAdd(&cnt[1], (unsigned)added);
}
// Probe: the field at given points, at the instant of the frame of slot `early`.
template <class F>
__global__ void probe_kernel(F f, const float* px, const float* py, const float* pz, uint32_t count, float* ox, float* oy, float* oz) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const V3 u = sample(f, V3{px[i], py[i], pz[i]}, 0.0f);
    ox[i] = u.x;
    oy[i] = u.y;
    oz[i] = u.z;
}
// Dumps for the probe: base lattice of a slot (three short integers per node) and dense fine level.
__global__ void lat_dump(const int4* lat, int slot, size_t nodes, short* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes) return;
    int q[3];
    lat_get(lat[i], slot, q);
    for (int c = 0; c < 3; ++c) out[3 * i + c] = (short)q[c];
}
__global__ void fine_dense(const unsigned char* need, signed char* q, size_t nodes) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes || need[i]) return;
    q[3 * i] = q[3 * i + 1] = q[3 * i + 2] = 0;
}

// ---------------------------------------------------------------------------------------
// rANS coder on the host side
// ---------------------------------------------------------------------------------------

// Writes into `out`: frequencies per segment (L x 256 uint16), offsets of the blocks, rANS streams. The `ecount`
// symbols are arranged in L consecutive segments, of ends lv_end[0..L-1], each with its table: the levels of an
// innovation frame, or the segments of a coded key frame. Returns the size written, or 0 (nothing written beyond the
// tables) if it would exceed `limit`.
static size_t rans_pack(const signed char* innov, size_t ecount, int L, const unsigned long long* lv_end, unsigned char* out, size_t limit) {
    const size_t n_blocks = (ecount + 1023) / 1024, block_cap = 2056;
    static std::vector<unsigned char> scratch;
    static std::vector<uint32_t> bsize;
    static std::vector<const unsigned char*> bptr;
    if (scratch.size() < n_blocks * block_cap) {
        scratch.resize(n_blocks * block_cap);
        bsize.resize(n_blocks);
        bptr.resize(n_blocks);
    }
    uint16_t* freq = (uint16_t*)out;
    std::vector<uint16_t> cum((size_t)L * 256);
    std::vector<uint8_t> lut((size_t)L * 4096);
    const unsigned T = std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::vector<size_t>> cnt(T, std::vector<size_t>((size_t)L * 256, 0));
    parallel_ranges(T, ecount, [&](unsigned t, size_t a, size_t b) {
        int j = 0;
        for (size_t i = a; i < b; ++i) {
            while (i >= lv_end[j]) ++j;
            ++cnt[t][(size_t)j * 256 + (unsigned char)innov[i]];
        }
    });
    for (int j = 0; j < L; ++j) {
        size_t c[256] = {};
        for (unsigned t = 0; t < T; ++t)
            for (int v = 0; v < 256; ++v) c[v] += cnt[t][(size_t)j * 256 + v];
        rans_build(c, freq + (size_t)j * 256, cum.data() + (size_t)j * 256, lut.data() + (size_t)j * 4096);
    }
    parallel_ranges(T, n_blocks, [&](unsigned, size_t a, size_t b) {
        for (size_t blk = a; blk < b; ++blk) {
            const size_t i0 = blk * 1024, i1 = std::min(i0 + 1024, ecount);
            unsigned char* end = scratch.data() + (blk + 1) * block_cap;
            unsigned char* ptr = end;
            uint32_t x = 1u << 23;
            int j = L - 1;
            for (size_t i = i1; i-- > i0;) {
                while (j > 0 && i < lv_end[j - 1]) --j;
                const unsigned char sym = (unsigned char)innov[i];
                const uint32_t f = freq[(size_t)j * 256 + sym], c = cum[(size_t)j * 256 + sym];
                const uint64_t x_max = (uint64_t)(1u << 19) * f;
                while (x >= x_max) {
                    *--ptr = (unsigned char)(x & 0xFF);
                    x >>= 8;
                }
                x = ((x / f) << 12) + (x % f) + c;
            }
            ptr -= 4;
            ptr[0] = (unsigned char)x;
            ptr[1] = (unsigned char)(x >> 8);
            ptr[2] = (unsigned char)(x >> 16);
            ptr[3] = (unsigned char)(x >> 24);
            bptr[blk] = ptr;
            bsize[blk] = (uint32_t)(end - ptr);
        }
    });
    size_t total = (size_t)L * 512 + n_blocks * 4;
    for (size_t blk = 0; blk < n_blocks; ++blk) total += bsize[blk];
    if (total > limit) return 0;
    uint32_t* off = (uint32_t*)(out + (size_t)L * 512);
    unsigned char* pay = out + (size_t)L * 512 + n_blocks * 4;
    size_t o = 0;
    for (size_t blk = 0; blk < n_blocks; ++blk) {
        off[blk] = (uint32_t)o;
        memcpy(pay + o, bptr[blk], bsize[blk]);
        o += bsize[blk];
    }
    return (size_t)L * 512 + n_blocks * 4 + o;
}
// Cumulative frequencies and inverse table of a level from its frequencies.
static void rans_tables(const uint16_t* freq, uint16_t* cum, uint8_t* lut) {
    int c = 0;
    for (int v = 0; v < 256; ++v) {
        cum[v] = (uint16_t)c;
        CHK(c + freq[v] <= 4096, "corrupt frequency table");
        for (int k = 0; k < freq[v]; ++k) lut[c + k] = (uint8_t)v;
        c += freq[v];
    }
    CHK(c == 4096, "corrupt frequency table");
}

static unsigned long long state_hash(const Geo& g, signed char* const q0[2], short* const q[kMaxLevels][2], int slot, unsigned long long* d_hash) {
    CK(cudaMemset(d_hash, 0, sizeof(unsigned long long)));
    hash_bytes<<<blocks_for((3 * g.nodes[0] + 7) / 8), 256>>>((const unsigned char*)q0[slot], 3 * g.nodes[0], 0x1000ull, d_hash);
    for (int l = 1; l < g.L; ++l)
        hash_bytes<<<blocks_for((6 * g.nodes[l] + 7) / 8), 256>>>((const unsigned char*)q[l][slot], 6 * g.nodes[l], 0x1000ull + ((unsigned long long)l << 56),
                                                                  d_hash);
    unsigned long long hv = 0;
    CK(cudaMemcpy(&hv, d_hash, sizeof(hv), cudaMemcpyDeviceToHost));
    return hv;
}
// Fingerprint of the fine level of slot t (map, masks, coefficients): it is combined by XOR with that of the levels.
static unsigned long long fine_hash(const FineState& fs, int t, unsigned long long* d_hash) {
    CK(cudaMemset(d_hash, 0, sizeof(unsigned long long)));
    hash_bytes<<<blocks_for((fs.bm_bytes() + 7) / 8), 256>>>((const unsigned char*)fs.bm[t], fs.bm_bytes(), 0x2000ull, d_hash);
    const size_t mb = 8 * (size_t)fs.n_bricks[t], cb = 3 * (size_t)fs.n_nodes[t];
    if (mb) hash_bytes<<<blocks_for((mb + 7) / 8), 256>>>((const unsigned char*)fs.mask[t], mb, 0x2000ull + (1ull << 56), d_hash);
    if (cb) hash_bytes<<<blocks_for((cb + 7) / 8), 256>>>((const unsigned char*)fs.coef[t], cb, 0x2000ull + (2ull << 56), d_hash);
    unsigned long long hv = 0;
    CK(cudaMemcpy(&hv, d_hash, sizeof(hv), cudaMemcpyDeviceToHost));
    return hv;
}

// ---------------------------------------------------------------------------------------
// Receiver: everything that resides on the GPU of the reader
// ---------------------------------------------------------------------------------------

template <int D>
struct Receiver {
    Geo g;
    float delta = 0.0f;
    Coef coef[kMaxLevels];
    signed char* q0[2] = {};
    short* q[kMaxLevels][2] = {};
    int* tmp[kMaxLevels] = {};  // partial sums on 32 bits
    int4* lat = nullptr;
    signed char* e = nullptr;
    unsigned char *packet = nullptr, *lut = nullptr;
    int coarse_lim = 32767;  // range of the coarse levels: 32767 >> fb for a PSISTRM1 file
    bool v1 = false;         // PSISTRM1 file: no exception list in the frames
    bool v3 = false;         // PSISTRM3 file: each frame ends with the section of the fine level, read or not (--no-fine)
    unsigned* d_legacy = nullptr;  // PSISTRM1 file: counter of the sums outside +-32767 (see legacy_beyond)
    // version: last character of the signature of the file ('1', '2' or '3').
    void set_format(char version, int fb) {
        v1 = version == '1';
        v3 = version == '3';
        coarse_lim = v1 ? 32767 >> fb : 32767;
        if (v1 && !d_legacy) CK(cudaMalloc(&d_legacy, sizeof(unsigned)));
    }
    // PSISTRM1 file: number of sums of the pyramid (partial sums of the coarse levels and sampling lattice) outside
    // +-32767 in the frame that has just been decoded into the slot `slot`. The original reader of this format clipped
    // them to this bound; if the number is zero, the lattice decoded here is that of the original reader.
    unsigned legacy_beyond(int slot) {
        CK(cudaMemset(d_legacy, 0, sizeof(unsigned)));
        for (int l = 1; l < g.L; ++l) count_beyond<<<blocks_for(3 * g.nodes[l]), 256>>>(tmp[l], 3 * g.nodes[l], 32767, d_legacy);
        count_lattice_beyond<D><<<blocks_for(g.nodes[0]), 256>>>(q0[slot], g.L > 1 ? tmp[1] : nullptr, g.nl[0], g.fb, 32767, d_legacy);
        unsigned v = 0;
        CK(cudaMemcpy(&v, d_legacy, sizeof(v), cudaMemcpyDeviceToHost));
        return v;
    }
    // Layout of the payload of a frame, established on the host by inspect() before any decoding kernel.
    struct Layout {
        const unsigned char* payload = nullptr;  // null: no checked frame waiting to be decoded
        size_t bytes = 0;
        int type = -1;
        uint32_t n_ex = 0, n_esc = 0;  // exceptions of the frame; escapes, at the head of the list
        size_t ex_off = 0;             // position of the exception list
        size_t tables = 0;             // position of the frequency tables (types 1, 2 and 3)
        size_t fine_off = 0;           // PSISTRM3: position of the section of the fine level
        FineTrailer tr{};
    } lay;
    unsigned short* cum = nullptr;
    unsigned long long *lvend = nullptr, *lvend_key = nullptr, *d_hash = nullptr;
    std::vector<uint8_t> h_lut;
    std::vector<uint16_t> h_cum;
    float mean[2][3] = {};
    // Sparse fine level (PSISTRM3).
    bool fine_on = false;
    FineState fs;
    float fine_delta = 0.0f;
    signed char* fsym = nullptr;  // decoded symbols of the section of the fine level: map, masks, coefficients
    // rANS tables of the section: cumulative frequencies (3 x 256 uint16), inverse tables (3 x 4096 bytes), ends of the
    // segments (3 x 8 bytes), in a single block of pinned host memory and a single transfer per frame.
    static constexpr size_t kFineCum = 0, kFineLut = 3 * 512, kFineEnd = 3 * 512 + 3 * 4096, kFineTab = 3 * 512 + 3 * 4096 + 24;
    unsigned char *h_ftab = nullptr, *d_ftab = nullptr;
    size_t fsym_bytes = 0;
    // The fine level of a frame does not depend on its base levels: it is decoded on its own GPU stream, at the same
    // time as them. The stream of the caller waits for the event fev before it reports itself finished.
    cudaStream_t fst = nullptr;
    cudaEvent_t fev = nullptr;

    // To be called after init. The capacity `cap` of the geometry must already include the section of the fine level.
    void init_fine(const FineInfo& fi) {
        fine_on = true;
        fine_delta = fi.delta;
        CHK(fi.brick == kFineSide, "fine level: brick size not supported");
        fs.init(g.n, fi.max_bricks, fi.max_nodes);
        fsym_bytes = fs.bm_bytes() + 8 * fs.cap_bricks + 3 * fs.cap_nodes + 1024;
        CK(cudaMalloc(&fsym, fsym_bytes));
        CK(cudaHostAlloc(&h_ftab, kFineTab, cudaHostAllocDefault));
        CK(cudaMalloc(&d_ftab, kFineTab));
        CK(cudaStreamCreate(&fst));
        CK(cudaEventCreateWithFlags(&fev, cudaEventDisableTiming));
    }
    // Check, on the host, of the payload of a frame against the geometry of the header. It reads only the host copy
    // and launches no kernel: a reader calls it while the payload crosses the bus. The frame is refused
    // (message, code 1) if:
    //   - PSISTRM3: the section of the fine level, found from the end of the frame, does not fit in it, or announces
    //     more bricks or nodes than the maxima of the header;
    //   - the exception list leaves the frame or exceeds its capacity;
    //   - raw key: the size is not exactly that of the states of the geometry, followed by the exception list;
    //   - other frames, and section of the fine level: see check_rans_section.
    // What it cannot see (bytes of a stream, value of an exception) falls under the fingerprint of the frame.
    void inspect(int type, const unsigned char* h, size_t bytes) {
        Layout y;
        y.payload = h;
        y.bytes = bytes;
        y.type = type;
        CHK(type >= 0 && type <= 3 && !(v1 && type == 3), "unknown frame type");
        size_t base_end = bytes;  // end of the base part: the whole frame, or the start of the section of the fine level
        if (v3) {
            CHK(bytes >= sizeof(FineTrailer) && bytes % 8 == 0, "frame without a section of the fine level");
            memcpy(&y.tr, h + bytes - sizeof(y.tr), sizeof(y.tr));
            CHK(y.tr.bytes % 8 == 0 && (size_t)y.tr.bytes + sizeof(y.tr) <= bytes, "corrupt section of the fine level");
            base_end = y.fine_off = bytes - sizeof(y.tr) - y.tr.bytes;
            if (fine_on) {
                CHK(y.tr.n_bricks <= fs.cap_bricks && y.tr.n_nodes <= fs.cap_nodes, "corrupt section of the fine level");
                const size_t count = fs.bm_bytes() + 8 * (size_t)y.tr.n_bricks + 3 * (size_t)y.tr.n_nodes;
                check_rans_section(h + y.fine_off, y.tr.bytes, 3, (count + 1023) / 1024, "corrupt section of the fine level");
            }
        }
        static const char* const kKey = "raw key frame: size inconsistent with the geometry of the header";
        if (type == 0) CHK(g.key_bytes <= base_end, kKey);
        // Exceptions (PSISTRM2): after the states of a raw key frame, at the head of the other frames.
        y.ex_off = type == 0 ? g.key_bytes : 0;
        size_t pos = y.ex_off;  // end of what precedes the frequency tables
        if (!v1) {
            CHK(y.ex_off + 8 <= base_end, "corrupt exception list");
            memcpy(&y.n_ex, h + y.ex_off, sizeof(y.n_ex));
            CHK(y.n_ex <= g.ex_cap && (size_t)y.n_ex * sizeof(ExEntry) <= base_end - y.ex_off - 8, "corrupt exception list");
            const ExEntry* hx = (const ExEntry*)(h + y.ex_off + 8);
            while (y.n_esc < y.n_ex && !(hx[y.n_esc].idx & kExFine)) ++y.n_esc;
            pos = y.ex_off + 8 + (size_t)y.n_ex * sizeof(ExEntry);
        }
        if (type == 0) {  // in PSISTRM3, the base part is padded to a multiple of 8 bytes
            CHK((v3 ? (pos + 7) & ~(size_t)7 : pos) == base_end, kKey);
        } else {
            y.tables = pos;
            check_rans_section(h + pos, base_end - pos, type == 3 ? (size_t)g.kseg : (size_t)g.L, type == 3 ? g.n_kblocks : g.n_blocks,
                               "corrupt frame: frequency tables or block offsets inconsistent with the geometry of the header");
        }
        lay = y;
    }
    // Section of the fine level, at the end of the payload: rANS symbols, then ranks and union of the two slots.
    void decode_fine(int t, const unsigned char* h_payload, cudaStream_t st) {
        const FineTrailer tr = lay.tr;
        const size_t off = lay.fine_off, bmb = fs.bm_bytes(), mb = 8 * (size_t)tr.n_bricks, cb = 3 * (size_t)tr.n_nodes;
        const size_t count = bmb + mb + cb, nblk = (count + 1023) / 1024;
        for (int s = 0; s < 3; ++s)
            rans_tables((const uint16_t*)(h_payload + off + (size_t)s * 512), (uint16_t*)(h_ftab + kFineCum) + (size_t)s * 256, h_ftab + kFineLut + (size_t)s * 4096);
        const unsigned long long ends[3] = {bmb, bmb + mb, count};
        memcpy(h_ftab + kFineEnd, ends, sizeof(ends));
        CK(cudaMemcpyAsync(d_ftab, h_ftab, kFineTab, cudaMemcpyHostToDevice, st));
        rans_decode<<<blocks_for(nblk), 256, 0, st>>>(packet + off + 3 * 512 + nblk * 4, (const uint32_t*)(packet + off + 3 * 512), d_ftab + kFineLut,
                                                     (const unsigned short*)(packet + off), (const unsigned short*)(d_ftab + kFineCum),
                                                     (const unsigned long long*)(d_ftab + kFineEnd), fsym, count);
        CK(cudaMemcpyAsync(fs.bm[t], fsym, bmb, cudaMemcpyDeviceToDevice, st));
        if (mb) CK(cudaMemcpyAsync(fs.mask[t], fsym + bmb, mb, cudaMemcpyDeviceToDevice, st));
        if (cb) CK(cudaMemcpyAsync(fs.coef[t], fsym + bmb + mb, cb, cudaMemcpyDeviceToDevice, st));
        fs.n_bricks[t] = tr.n_bricks;
        fs.n_nodes[t] = tr.n_nodes;
        fs.index(t, st);
        fs.rebuild(D, st);
    }

    void init(const Geo& geo, float delta_, const Coef* c) {
        g = geo;
        delta = delta_;
        for (int l = 0; l < g.L; ++l) coef[l] = c[l];
        // The sampling lattice is allocated first: it is the only array read in random access by the kernels.
        CK(cudaMalloc(&lat, g.nodes[0] * sizeof(int4)));
        CK(cudaMemset(lat, 0, g.nodes[0] * sizeof(int4)));
        for (int t = 0; t < 2; ++t) {
            CK(cudaMalloc(&q0[t], 3 * g.nodes[0]));
            CK(cudaMemset(q0[t], 0, 3 * g.nodes[0]));
            for (int l = 1; l < g.L; ++l) {
                CK(cudaMalloc(&q[l][t], 6 * g.nodes[l]));
                CK(cudaMemset(q[l][t], 0, 6 * g.nodes[l]));
            }
        }
        for (int l = 1; l < g.L; ++l) CK(cudaMalloc(&tmp[l], 12 * g.nodes[l]));
        CK(cudaMalloc(&e, g.sym_bytes));
        CK(cudaMemset(e, 0, g.sym_bytes));
        CK(cudaMalloc(&packet, g.cap));
        h_lut.resize((size_t)2 * kMaxLevels * 4096);  // one table per segment: up to 2 L - 1 for a coded key
        h_cum.resize((size_t)2 * kMaxLevels * 256);
        CK(cudaMalloc(&lut, h_lut.size()));
        CK(cudaMalloc(&cum, h_cum.size() * sizeof(uint16_t)));
        CK(cudaMalloc(&lvend, kMaxLevels * sizeof(unsigned long long)));
        CK(cudaMemcpy(lvend, g.lv_end, kMaxLevels * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&lvend_key, 2 * kMaxLevels * sizeof(unsigned long long)));
        CK(cudaMemcpy(lvend_key, g.kseg_end, 2 * kMaxLevels * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_hash, sizeof(unsigned long long)));
    }
    // The payload of the frame is already in `packet`; `h_payload` is its host copy (the frequency tables are read
    // from it to build the cumulative frequencies and the inverse table). No kernel is launched before inspect() has
    // accepted the frame: a reader calls it itself during the transfer, otherwise it is called here.
    void decode(int type, int slot_new, const float m[3], const unsigned char* h_payload, size_t bytes, cudaStream_t st) {
        if (lay.payload != h_payload || lay.bytes != bytes || lay.type != type) inspect(type, h_payload, bytes);
        if (fine_on) {  // the payload is already on the GPU: the fine level goes on its own stream, the levels on that of the caller
            decode_fine(slot_new, h_payload, fst);
            CK(cudaEventRecord(fev, fst));
        }
        decode_levels(type, slot_new, m, h_payload, st);
        if (fine_on) CK(cudaStreamWaitEvent(st, fev, 0));
        lay.payload = nullptr;  // the check holds only for this frame
    }
    void decode_levels(int type, int slot_new, const float m[3], const unsigned char* h_payload, cudaStream_t st) {
        const int L = g.L, slot_k = slot_new ^ 1;
        for (int c = 0; c < 3; ++c) mean[slot_new][c] = m[c];
        const size_t ex_off = lay.ex_off;
        const uint32_t n_ex = lay.n_ex, n_esc = lay.n_esc;
        const ExEntry* dx = (const ExEntry*)(packet + ex_off + 8);
        if (type == 0) {
            CK(cudaMemcpyAsync(q0[slot_new], packet + g.koff[0], 3 * g.nodes[0], cudaMemcpyDeviceToDevice, st));
            for (int l = 1; l < L; ++l) CK(cudaMemcpyAsync(q[l][slot_new], packet + g.koff[l], 6 * g.nodes[l], cudaMemcpyDeviceToDevice, st));
        } else if (type == 3) {  // coded key: rANS symbols, then three cumulative sums per level (see key_residual)
            const int S = g.kseg;
            const size_t base = lay.tables, tables = (size_t)S * 512;
            for (int s = 0; s < S; ++s)
                rans_tables((const uint16_t*)(h_payload + base + (size_t)s * 512), h_cum.data() + (size_t)s * 256, h_lut.data() + (size_t)s * 4096);
            CK(cudaMemcpyAsync(cum, h_cum.data(), (size_t)S * 256 * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(lut, h_lut.data(), (size_t)S * 4096, cudaMemcpyHostToDevice, st));
            rans_decode<<<blocks_for(g.n_kblocks), 256, 0, st>>>(packet + base + tables + g.n_kblocks * 4, (const uint32_t*)(packet + base + tables), lut,
                                                                (const unsigned short*)(packet + base), cum, lvend_key, e, g.kcount);
            const unsigned char* sym = (const unsigned char*)e;
            for (int l = L - 1; l >= 1; --l) {
                const size_t lines = (size_t)g.nl[l] * g.nl[l];
                key_scan_x<short><<<blocks_for(lines), 256, 0, st>>>(sym + g.klo[l], sym + g.khi[l], g.nl[l], q[l][slot_new]);
                key_scan<short><<<blocks_for(lines), 256, 0, st>>>(q[l][slot_new], g.nl[l], 1);
                key_scan<short><<<blocks_for(lines), 256, 0, st>>>(q[l][slot_new], g.nl[l], 2);
            }
            const size_t lines = (size_t)g.nl[0] * g.nl[0];
            key_scan_x<signed char><<<blocks_for(lines), 256, 0, st>>>(sym + g.klo[0], nullptr, g.nl[0], q0[slot_new]);
            key_scan<signed char><<<blocks_for(lines), 256, 0, st>>>(q0[slot_new], g.nl[0], 1);
            key_scan<signed char><<<blocks_for(lines), 256, 0, st>>>(q0[slot_new], g.nl[0], 2);
        } else {
            const size_t base = lay.tables;  // start of the frequency tables
            for (int j = 0; j < L; ++j)
                rans_tables((const uint16_t*)(h_payload + base + (size_t)j * 512), h_cum.data() + (size_t)j * 256, h_lut.data() + (size_t)j * 4096);
            CK(cudaMemcpyAsync(cum, h_cum.data(), (size_t)L * 256 * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(lut, h_lut.data(), (size_t)L * 4096, cudaMemcpyHostToDevice, st));
            rans_decode<<<blocks_for(g.n_blocks), 256, 0, st>>>(packet + base + (size_t)L * 512 + g.n_blocks * 4, (const uint32_t*)(packet + base + (size_t)L * 512),
                                                               lut, (const unsigned short*)(packet + base), cum, lvend, e, g.ecount);
            const Coef persist{1024, 0, 0};
            for (int l = L - 1; l >= 1; --l)
                level_apply<short><<<blocks_for(g.nodes[l]), 256, 0, st>>>(e + g.seg[l], q[l][slot_k], q[l][slot_new], g.nl[l], type == 2 ? persist : coef[l],
                                                                         coarse_lim, q[l][slot_new]);
            level_apply<signed char><<<blocks_for(g.nodes[0]), 256, 0, st>>>(e + g.seg[0], q0[slot_k], q0[slot_new], g.nl[0], type == 2 ? persist : coef[0], 127,
                                                                           q0[slot_new]);
            if (n_esc) {  // escapes: states that the 8-bit symbols did not carry
                LevelPtrs p{};
                p.q0 = q0[slot_new];
                p.L = L;
                for (int l = 0; l < L; ++l) {
                    p.q[l] = q[l][slot_new];
                    p.seg[l] = g.seg[l];
                    p.n3[l] = 3 * g.nodes[l];
                }
                apply_state_ex<<<blocks_for(n_esc), 256, 0, st>>>(dx, n_esc, p);
            }
        }
        for (int l = L - 1; l >= 1; --l)
            synth_level<D, short><<<blocks_for(g.nodes[l]), 256, 0, st>>>(q[l][slot_new], l + 1 < L ? tmp[l + 1] : nullptr, g.nl[l], g.fb, tmp[l], nullptr);
        synth_lattice<D, signed char><<<blocks_for(g.nodes[0]), 256, 0, st>>>(q0[slot_new], L > 1 ? tmp[1] : nullptr, g.nl[0], g.fb, lat, slot_new, nullptr, nullptr);
        if (n_ex > n_esc)  // overflows of the fine level
            apply_lattice_ex<<<blocks_for(n_ex - n_esc), 256, 0, st>>>(dx + n_esc, n_ex - n_esc, g.seg[0], g.nodes[0], lat, slot_new, g.fb, nullptr);
    }
    PsiField<D> field(int early) const {
        PsiField<D> f{};
        f.lat = lat;
        f.n = g.nl[0];
        f.inv_h = (float)(1.0 / g.hl[0]);
        f.x0 = (float)g.x0[0];
        f.s = g.field_scale(delta);
        f.early = early;
        for (int c = 0; c < 3; ++c) {
            f.m0[c] = mean[early][c];
            f.m1[c] = mean[early ^ 1][c];
        }
        return f;
    }
    size_t resident_bytes() const {
        size_t b = g.nodes[0] * sizeof(int4) + 2 * 3 * g.nodes[0] + g.sym_bytes + g.cap;
        for (int l = 1; l < g.L; ++l) b += (2 * 6 + 12) * g.nodes[l];
        if (fine_on) b += fs.resident_bytes() + fsym_bytes;
        return b;
    }
};

// ---------------------------------------------------------------------------------------
// Sender
// ---------------------------------------------------------------------------------------

template <int D>
struct Sender {
    Geo g;
    float delta = 0.0f;
    double ridge = 1e-5;
    Coef coef[kMaxLevels];
    size_t voxels = 0;
    cufftComplex *uh[3] = {}, *lh[3] = {}, *th[3] = {};
    float* lf[3] = {};
    double* a[kMaxLevels][3] = {};
    Cd *ta[kMaxLevels] = {}, *tc[kMaxLevels] = {};
    cufftHandle plan_n = 0, plan_r[kMaxLevels] = {}, plan_i[kMaxLevels] = {};
    signed char* q0[2] = {};
    short* q[kMaxLevels][2] = {};
    int* tmp[kMaxLevels] = {};  // partial sums on 32 bits
    signed char* e = nullptr;
    unsigned* cnt = nullptr;
    unsigned long long* d_hash = nullptr;
    int4* lat = nullptr;
    float *e2 = nullptr, *r2 = nullptr, *emax = nullptr;
    size_t bricks = 0;  // bricks of 4^3 voxels of the fidelity check
    std::vector<signed char> innov;
    float mean[2][3] = {};
    // Cumulative clippings: per level and per site, then per frame type (0 key, 1 innovation, 2 after a key) and per
    // site.
    long long clip[kMaxLevels][kClipSites] = {}, clip_type[3][kClipSites] = {};
    int frames_clipped = 0, frames_wrapped = 0;  // clipped frames; frames whose lattice leaves the range of a short
    ExEntry* ex_dev = nullptr;   // collection of the exceptions on the GPU
    std::vector<ExEntry> ex;     // exceptions of the frame that has just been encoded, sorted
    long long n_escape = 0, n_fine = 0;  // running totals: innovation escapes, overflows of the fine level
    bool raw_keys = false;       // write the key frames raw (type 0) rather than coded (type 3)
    // Sparse fine level (PSISTRM3): projection of the residual of the base at a step of 1 voxel, kept where the
    // tolerance requires it.
    bool fine_on = false;
    FineState fs;
    float tol_abs = 0.0f, fine_delta = 0.0f;  // tolerance in velocity units; quantisation step of the fine potential
    size_t spec_n = 0;
    cufftComplex* lhf[3] = {};
    double* af[3] = {};
    Cd* taf = nullptr;
    cufftHandle plan_fi = 0;
    signed char* fq = nullptr;          // quantised fine coefficients of all the nodes
    unsigned char* need = nullptr;      // one byte per fine node: retained or not
    unsigned long long* emask = nullptr;
    unsigned* fcnt = nullptr;
    std::vector<unsigned char> fsym;    // symbols of the section of the fine level, on the host side
    int fine_passes = 0;                // passes of the last frame
    unsigned fine_left = 0;             // voxels still outside the tolerance at the last pass of the last frame
    // Cap on the marking passes of a frame (pack --fine-passes). It is only a guard: the loop stops by itself when a
    // pass no longer adds a node. At 12, it caused the bound of 3 % to be missed at 512^3 in degree 3, a bound that is
    // held in 16 passes.
    int max_passes = 64;
    bool fine_capped = false;           // last frame: cap reached while the pass was still adding nodes

    void init_fine(float tol_abs_, float fine_delta_) {
        fine_on = true;
        tol_abs = tol_abs_;
        fine_delta = fine_delta_;
        const int n = g.n;
        // Capacity of the encoder: a quarter of the bricks and of the nodes. Beyond that, the tolerance asks for
        // refinement almost everywhere, and a lattice of step 1 voxel would be the right tool.
        fs.init(n, (size_t)(n / kFineSide) * (n / kFineSide) * (n / kFineSide) / 4, voxels / 4);
        for (int c = 0; c < 3; ++c) CK(cudaMalloc(&lhf[c], spec_n * sizeof(cufftComplex)));
        std::vector<double> as[3];
        alias_sums(n, D, as);
        for (int p = 0; p < 3; ++p) {
            CK(cudaMalloc(&af[p], as[p].size() * sizeof(double)));
            CK(cudaMemcpy(af[p], as[p].data(), as[p].size() * sizeof(double), cudaMemcpyHostToDevice));
        }
        const double h = (double)kL / n, x0 = D == 2 ? 0.5 * h : 0.0;
        std::vector<Cd> hta(n);
        for (int k = -n / 2; k < n / 2; ++k) {
            const double b = std::pow(sinc_pi((double)k / n), D + 1), pa = k * (x0 - 0.5 * h);
            hta[k + n / 2] = {b * std::cos(pa), b * std::sin(pa)};
        }
        CK(cudaMalloc(&taf, n * sizeof(Cd)));
        CK(cudaMemcpy(taf, hta.data(), n * sizeof(Cd), cudaMemcpyHostToDevice));
        CHK(cufftPlan3d(&plan_fi, n, n, n, CUFFT_C2R) == CUFFT_SUCCESS, "C2R plan of the fine level");
        CK(cudaMalloc(&fq, 3 * voxels));
        CK(cudaMalloc(&need, voxels));
        CK(cudaMalloc(&emask, fs.bricks * sizeof(unsigned long long)));
        CK(cudaMalloc(&fcnt, 2 * sizeof(unsigned)));
    }
    PsiField<D> base_field(int slot) const {
        PsiField<D> f{};
        f.lat = lat;
        f.n = g.nl[0];
        f.inv_h = (float)(1.0 / g.hl[0]);
        f.x0 = (float)g.x0[0];
        f.s = g.field_scale(delta);
        f.early = slot;
        for (int c = 0; c < 3; ++c) f.m0[c] = f.m1[c] = mean[slot][c];
        return f;
    }
    PsiFine<D, PsiField<D>> total_field(int slot) const {
        PsiFine<D, PsiField<D>> f{};
        f.base = base_field(slot);
        f.fine = fs.level(D, fine_delta);
        return f;
    }
    // Fine level of the frame that has just been encoded (its base lattice is synthesised). The residual of the base
    // at voxel centres is projected on a potential of step 1 voxel, like a level of the pyramid, then quantised. Only
    // the nodes on which a voxel outside the tolerance depends are kept: the complete field is re-evaluated, with the
    // kernel of the reader, as long as nodes are added. On exit, fine_left counts the voxels that the fine level does
    // not bring back under the tolerance. If the loop stops on its cap (max_passes) while it was still adding nodes,
    // fine_capped says so: a missed bound in that case does not prove a floor of the fine level.
    // The base field does not change from one pass to the next: the first pass stores it in the buffers lhf, free
    // after the quantisation, and the following ones read only the fine level (fine_mark_cached).
    void refine(const float* d_planar, int slot) {
        const int n = g.n;
        resid_padded<<<blocks_for(voxels), 256>>>(base_field(slot), d_planar, n, (float*)uh[0], (float*)uh[1], (float*)uh[2]);
        for (int c = 0; c < 3; ++c) CHK(cufftExecR2C(plan_n, (float*)uh[c], uh[c]) == CUFFT_SUCCESS, "R2C of the residual");
        fit_level<<<blocks_for(spec_n), 256>>>(uh[0], uh[1], uh[2], n, nullptr, nullptr, nullptr, 0, 0.0, n, ridge, af[0], af[1], af[2], taf, taf, lhf[0],
                                                lhf[1], lhf[2]);
        for (int c = 0; c < 3; ++c) CHK(cufftExecC2R(plan_fi, lhf[c], (float*)lhf[c]) == CUFFT_SUCCESS, "C2R of the fine level");
        fine_quantize<<<blocks_for(voxels), 256>>>((const float*)lhf[0], (const float*)lhf[1], (const float*)lhf[2], n, 1.0f / fine_delta, fq);
        CK(cudaMemset(need, 0, voxels));
        float* const bp[3] = {(float*)lhf[0], (float*)lhf[1], (float*)lhf[2]};  // free: fine_quantize has read them
        unsigned v[2] = {0, 0};
        fine_capped = false;
        for (fine_passes = 0;;) {
            CHK(fs.pack_dense(slot, need, fq, emask), "fine level: capacity exceeded, the tolerance asks to refine more than a quarter of the volume");
            fs.rebuild(D, nullptr);
            if (fine_passes == max_passes) {  // cap reached with nodes added: the structure contains them
                fine_capped = true;
                break;
            }
            ++fine_passes;
            CK(cudaMemset(fcnt, 0, 2 * sizeof(unsigned)));
            if (fine_passes == 1)  // no fine node retained: the evaluated field is that of the base, kept for the following passes
                fine_mark<D, PsiFine<D, PsiField<D>>><<<blocks_for(voxels), 256>>>(total_field(slot), d_planar, n, tol_abs, need, fcnt, bp[0], bp[1], bp[2]);
            else
                fine_mark_cached<D><<<blocks_for(voxels), 256>>>(fs.level(D, fine_delta), slot, bp[0], bp[1], bp[2], d_planar, n, tol_abs, need, fcnt);
            CK(cudaMemcpy(v, fcnt, sizeof(v), cudaMemcpyDeviceToHost));
            if (!v[1]) break;
        }
        fine_left = v[0];
    }
    // Appends to the payload (base_bytes bytes already written in `out`) the section of the fine level: padding to a
    // multiple of 8, tables and rANS streams of the three segments (map, masks, coefficients), padding, frame trailer.
    size_t append_fine(int slot, unsigned char* out, size_t base_bytes, size_t& section) {
        size_t pos = (base_bytes + 7) & ~(size_t)7;
        memset(out + base_bytes, 0, pos - base_bytes);
        const size_t bmb = fs.bm_bytes(), mb = 8 * (size_t)fs.n_bricks[slot], cb = 3 * (size_t)fs.n_nodes[slot];
        fsym.resize(bmb + mb + cb);
        CK(cudaMemcpy(fsym.data(), fs.bm[slot], bmb, cudaMemcpyDeviceToHost));
        if (mb) CK(cudaMemcpy(fsym.data() + bmb, fs.mask[slot], mb, cudaMemcpyDeviceToHost));
        if (cb) CK(cudaMemcpy(fsym.data() + bmb + mb, fs.coef[slot], cb, cudaMemcpyDeviceToHost));
        const unsigned long long ends[3] = {bmb, bmb + mb, bmb + mb + cb};
        const size_t coded = rans_pack((const signed char*)fsym.data(), fsym.size(), 3, ends, out + pos, (size_t)-1);
        const size_t padded = (coded + 7) & ~(size_t)7;
        memset(out + pos + coded, 0, padded - coded);
        const FineTrailer tr{fs.n_bricks[slot], fs.n_nodes[slot], (uint32_t)padded, 0u};
        memcpy(out + pos + padded, &tr, sizeof(tr));
        section = pos + padded + sizeof(tr) - base_bytes;
        return pos + padded + sizeof(tr);
    }
    // Probe: the decoded field at points off the grid, with the base lattice and the fine level of the frame, for an
    // independent verification in double precision. Each point is followed by its six neighbours at +-shift along the
    // axes.
    void write_probe(const char* path, int slot, int frame, float u_rms, bool wrapped) {
        const uint32_t base_points = 16384, count = 7 * base_points;
        const float shift = kL / g.n / 8.0f;
        std::vector<float> p(3 * (size_t)count), u(3 * (size_t)count);
        std::mt19937 rng(4321);
        std::uniform_real_distribution<float> U(0.0f, kL);
        for (uint32_t i = 0; i < base_points; ++i) {
            const float c[3] = {U(rng), U(rng), U(rng)};
            for (int k = 0; k < 7; ++k)
                for (int a = 0; a < 3; ++a) p[(size_t)a * count + 7 * i + k] = c[a] + ((k - 1) / 2 == a && k > 0 ? ((k & 1) ? shift : -shift) : 0.0f);
        }
        float *d_p = nullptr, *d_u = nullptr;
        CK(cudaMalloc(&d_p, p.size() * sizeof(float)));
        CK(cudaMalloc(&d_u, u.size() * sizeof(float)));
        CK(cudaMemcpy(d_p, p.data(), p.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (fine_on)
            probe_kernel<<<blocks_for(count), 256>>>(total_field(slot), d_p, d_p + count, d_p + 2 * (size_t)count, count, d_u, d_u + count, d_u + 2 * (size_t)count);
        else
            probe_kernel<<<blocks_for(count), 256>>>(base_field(slot), d_p, d_p + count, d_p + 2 * (size_t)count, count, d_u, d_u + count, d_u + 2 * (size_t)count);
        CK(cudaMemcpy(u.data(), d_u, u.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<short> hb(3 * g.nodes[0]);
        short* d_b = nullptr;
        CK(cudaMalloc(&d_b, hb.size() * sizeof(short)));
        lat_dump<<<blocks_for(g.nodes[0]), 256>>>(lat, slot, g.nodes[0], d_b);
        CK(cudaMemcpy(hb.data(), d_b, hb.size() * sizeof(short), cudaMemcpyDeviceToHost));
        std::vector<signed char> hf;
        if (fine_on) {
            fine_dense<<<blocks_for(voxels), 256>>>(need, fq, voxels);
            hf.resize(3 * voxels);
            CK(cudaMemcpy(hf.data(), fq, hf.size(), cudaMemcpyDeviceToHost));
        }
        CK(cudaFree(d_p));
        CK(cudaFree(d_u));
        CK(cudaFree(d_b));
        FILE* f = fopen(path, "wb");
        CHK(f, "cannot create the probe file");
        const int32_t hi[8] = {g.n, D, g.nl[0], frame, (int32_t)count, (int32_t)base_points, fine_on ? 1 : 0, wrapped ? 1 : 0};
        const float hf32[8] = {g.field_scale(delta), fine_on ? fs.level(D, fine_delta).s : 0.0f, shift, u_rms, tol_abs, mean[slot][0], mean[slot][1], mean[slot][2]};
        bool ok = fwrite("PSIPROBE", 1, 8, f) == 8 && fwrite(hi, sizeof(hi), 1, f) == 1 && fwrite(hf32, sizeof(hf32), 1, f) == 1;
        ok = ok && fwrite(hb.data(), sizeof(short), hb.size(), f) == hb.size();
        if (fine_on) ok = ok && fwrite(hf.data(), 1, hf.size(), f) == hf.size();
        ok = ok && fwrite(p.data(), sizeof(float), p.size(), f) == p.size() && fwrite(u.data(), sizeof(float), u.size(), f) == u.size();
        fclose(f);
        CHK(ok, "writing of the probe file");
    }
    long long clip_site(int site) const {
        long long t = 0;
        for (int l = 0; l < kMaxLevels; ++l) t += clip[l][site];
        return t;
    }
    long long clip_all() const {
        long long t = 0;
        for (int site = 0; site < kClipSites; ++site) t += clip_site(site);
        return t;
    }
    long long clipped() const { return clip_site(kClipKey) + clip_site(kClipTarget); }  // targets out of range: original definition
    void reset_clip() {
        memset(clip, 0, sizeof(clip));
        memset(clip_type, 0, sizeof(clip_type));
        frames_clipped = frames_wrapped = 0;
        n_escape = n_fine = 0;
    }

    void init(const Geo& geo, float delta_, double ridge_) {
        g = geo;
        delta = delta_;
        ridge = ridge_;
        const int n = g.n, L = g.L;
        voxels = (size_t)n * n * n;
        spec_n = (size_t)n * n * (n / 2 + 1);
        const float h = kL / n;
        for (int c = 0; c < 3; ++c) {
            CK(cudaMalloc(&uh[c], spec_n * sizeof(cufftComplex)));
            CK(cudaMalloc(&lh[c], g.spec[0] * sizeof(cufftComplex)));
            CK(cudaMalloc(&th[c], g.spec[L > 1 ? 1 : 0] * sizeof(cufftComplex)));
            CK(cudaMalloc(&lf[c], g.nodes[0] * sizeof(float)));
        }
        for (int l = 0; l < L; ++l) {
            std::vector<double> as[3];
            alias_sums(g.nl[l], D, as);
            for (int p = 0; p < 3; ++p) {
                CK(cudaMalloc(&a[l][p], as[p].size() * sizeof(double)));
                CK(cudaMemcpy(a[l][p], as[p].data(), as[p].size() * sizeof(double), cudaMemcpyHostToDevice));
            }
            std::vector<Cd> hta(n), htc(n);
            for (int k = -n / 2; k < n / 2; ++k) {
                const double b = std::pow(sinc_pi((double)k / g.nl[l]), D + 1), pa = k * (g.x0[l] - 0.5 * h);
                hta[k + n / 2] = {b * std::cos(pa), b * std::sin(pa)};
                if (l + 1 < L) {
                    const double bc = b * std::pow(sinc_pi((double)k / g.nl[l + 1]), D + 1), pc = k * (g.x0[l] - g.x0[l + 1]);
                    htc[k + n / 2] = {bc * std::cos(pc), bc * std::sin(pc)};
                } else {
                    htc[k + n / 2] = {0.0, 0.0};
                }
            }
            CK(cudaMalloc(&ta[l], n * sizeof(Cd)));
            CK(cudaMalloc(&tc[l], n * sizeof(Cd)));
            CK(cudaMemcpy(ta[l], hta.data(), n * sizeof(Cd), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(tc[l], htc.data(), n * sizeof(Cd), cudaMemcpyHostToDevice));
            CHK(cufftPlan3d(&plan_i[l], g.nl[l], g.nl[l], g.nl[l], CUFFT_C2R) == CUFFT_SUCCESS, "C2R plan");
            if (l > 0) CHK(cufftPlan3d(&plan_r[l], g.nl[l], g.nl[l], g.nl[l], CUFFT_R2C) == CUFFT_SUCCESS, "R2C plan of the level");
            coef[l] = l == 0 ? Coef{1024, 0, 0} : Coef{2048, -1024, 0};
        }
        CHK(cufftPlan3d(&plan_n, n, n, n, CUFFT_R2C) == CUFFT_SUCCESS, "R2C plan n");
        for (int t = 0; t < 2; ++t) {
            CK(cudaMalloc(&q0[t], 3 * g.nodes[0]));
            CK(cudaMemset(q0[t], 0, 3 * g.nodes[0]));
            for (int l = 1; l < L; ++l) {
                CK(cudaMalloc(&q[l][t], 6 * g.nodes[l]));
                CK(cudaMemset(q[l][t], 0, 6 * g.nodes[l]));
            }
        }
        for (int l = 1; l < L; ++l) CK(cudaMalloc(&tmp[l], 12 * g.nodes[l]));
        CK(cudaMalloc(&e, g.sym_bytes));
        CK(cudaMemset(e, 0, g.sym_bytes));
        // After the clipping counters: a word that flags a wrapped lattice, then the number of exceptions collected.
        CK(cudaMalloc(&cnt, (kMaxLevels * kClipSites + 2) * sizeof(unsigned)));
        CK(cudaMemset(cnt, 0, (kMaxLevels * kClipSites + 2) * sizeof(unsigned)));
        CK(cudaMalloc(&ex_dev, g.ex_cap * sizeof(ExEntry)));
        CK(cudaMalloc(&d_hash, sizeof(unsigned long long)));
        CK(cudaMalloc(&lat, g.nodes[0] * sizeof(int4)));
        CK(cudaMemset(lat, 0, g.nodes[0] * sizeof(int4)));
        CHK(n % 4 == 0, "the fidelity check by bricks requires a resolution that is a multiple of 4");
        bricks = voxels / 64;
        CK(cudaMalloc(&e2, bricks * sizeof(float)));
        CK(cudaMalloc(&r2, bricks * sizeof(float)));
        CK(cudaMalloc(&emax, bricks * sizeof(float)));
        innov.assign(g.sym_bytes, 0);
    }
    // Encodes the frame whose planes are in d_planar. type 0: key; 1: innovation against the fitted predictor;
    // 2: innovation against persistence.
    void encode(const float* d_planar, int slot, int type) {
        const int n = g.n, L = g.L;
        for (int c = 0; c < 3; ++c) CHK(cufftExecR2C(plan_n, (float*)d_planar + (size_t)c * voxels, uh[c]) == CUFFT_SUCCESS, "R2C");
        for (int c = 0; c < 3; ++c) {
            cufftComplex m0;
            CK(cudaMemcpy(&m0, uh[c], sizeof(cufftComplex), cudaMemcpyDeviceToHost));
            mean[slot][c] = m0.x / (float)voxels;
        }
        const float inv_delta = 1.0f / delta;
        const Coef persist{1024, 0, 0};
        const ExSink sink{ex_dev, cnt + kMaxLevels * kClipSites + 1, (unsigned)g.ex_cap};
        for (int l = L - 1; l >= 0; --l) {
            const bool top = l == L - 1;
            fit_level<<<blocks_for(g.spec[l]), 256>>>(uh[0], uh[1], uh[2], n, top ? nullptr : th[0], top ? nullptr : th[1], top ? nullptr : th[2],
                                                      top ? 0 : g.nl[l + 1], top ? 0.0 : (double)delta / ((double)(1 << g.fb) * (double)g.nodes[l + 1]),
                                                      g.nl[l], ridge, a[l][0], a[l][1], a[l][2], ta[l], tc[l], lh[0], lh[1], lh[2]);
            for (int c = 0; c < 3; ++c) CHK(cufftExecC2R(plan_i[l], lh[c], lf[c]) == CUFFT_SUCCESS, "C2R");
            const Coef k = type == 2 ? persist : coef[l];
            if (l == 0) {
                if (type == 0) level_key<signed char><<<blocks_for(g.nodes[0]), 256>>>(lf[0], lf[1], lf[2], inv_delta, 127, q0[slot], g.nodes[0], cnt, g.seg[0], sink);
                else
                    level_innov<signed char><<<blocks_for(g.nodes[0]), 256>>>(lf[0], lf[1], lf[2], inv_delta, q0[slot ^ 1], q0[slot], g.nl[0], k, 127,
                                                                            e + g.seg[0], q0[slot], cnt, g.seg[0], sink);
            } else {
                if (type == 0) level_key<short><<<blocks_for(g.nodes[l]), 256>>>(lf[0], lf[1], lf[2], inv_delta, 32767, q[l][slot], g.nodes[l], cnt + l * kClipSites, g.seg[l], sink);
                else
                    level_innov<short><<<blocks_for(g.nodes[l]), 256>>>(lf[0], lf[1], lf[2], inv_delta, q[l][slot ^ 1], q[l][slot], g.nl[l], k,
                                                                      32767, e + g.seg[l], q[l][slot], cnt + l * kClipSites, g.seg[l], sink);
                synth_level<D, short><<<blocks_for(g.nodes[l]), 256>>>(q[l][slot], l + 1 < L ? tmp[l + 1] : nullptr, g.nl[l], g.fb, tmp[l], cnt + l * kClipSites);
                int_to_planes<<<blocks_for(g.nodes[l]), 256>>>(tmp[l], lf[0], lf[1], lf[2], g.nodes[l]);
                for (int c = 0; c < 3; ++c) CHK(cufftExecR2C(plan_r[l], lf[c], th[c]) == CUFFT_SUCCESS, "R2C of the level");
            }
        }
        // Sampling lattice of the frame, as the receiver will synthesise it: its clippings are counted here, and it
        // is used for the fidelity check.
        synth_lattice<D, signed char><<<blocks_for(g.nodes[0]), 256>>>(q0[slot], L > 1 ? tmp[1] : nullptr, g.nl[0], g.fb, lat, slot, cnt, cnt + kMaxLevels * kClipSites);
        CK(cudaDeviceSynchronize());
        // Exceptions of the frame, sorted (innovation escapes, then overflows of the fine level) and kept for the
        // payload. The overflows are added to the lattice of the sender, as the receiver will do.
        unsigned n_ex = 0;
        CK(cudaMemcpy(&n_ex, cnt + kMaxLevels * kClipSites + 1, sizeof(n_ex), cudaMemcpyDeviceToHost));
        ex.resize(std::min<size_t>(n_ex, g.ex_cap));
        size_t n_esc = 0;
        if (!ex.empty()) {
            CK(cudaMemcpy(ex.data(), ex_dev, ex.size() * sizeof(ExEntry), cudaMemcpyDeviceToHost));
            std::sort(ex.begin(), ex.end(), [](const ExEntry& a, const ExEntry& b) { return a.idx < b.idx; });
            while (n_esc < ex.size() && !(ex[n_esc].idx & kExFine)) ++n_esc;
            if (n_esc < ex.size()) {
                CK(cudaMemcpy(ex_dev, ex.data(), ex.size() * sizeof(ExEntry), cudaMemcpyHostToDevice));
                apply_lattice_ex<<<blocks_for(ex.size() - n_esc), 256>>>(ex_dev + n_esc, (unsigned)(ex.size() - n_esc), g.seg[0], g.nodes[0], lat, slot,
                                                                         g.fb, cnt + kMaxLevels * kClipSites);
                CK(cudaDeviceSynchronize());
            }
        }
        n_escape += (long long)n_esc;
        n_fine += (long long)(ex.size() - n_esc);
        unsigned v[kMaxLevels * kClipSites + 2];
        CK(cudaMemcpy(v, cnt, sizeof(v), cudaMemcpyDeviceToHost));
        CK(cudaMemset(cnt, 0, sizeof(v)));
        frames_wrapped += v[kMaxLevels * kClipSites] != 0;
        bool any = false;
        for (int l = 0; l < kMaxLevels; ++l)
            for (int site = 0; site < kClipSites; ++site) {
                clip[l][site] += v[l * kClipSites + site];
                clip_type[type][site] += v[l * kClipSites + site];
                any = any || v[l * kClipSites + site] != 0;
            }
        frames_clipped += any;
    }
    // Payload of the frame that has just been encoded, written in `out` (host memory).
    // Writes the header of the exception list (count, then 4 reserved bytes) and the list; returns the bytes written.
    size_t write_ex(unsigned char* out) const {
        const uint32_t hdr[2] = {(uint32_t)ex.size(), 0u};
        memcpy(out, hdr, sizeof(hdr));
        if (!ex.empty()) memcpy(out + sizeof(hdr), ex.data(), ex.size() * sizeof(ExEntry));
        return sizeof(hdr) + ex.size() * sizeof(ExEntry);
    }
    // `stored` receives the type to record in the frame table: the one requested, or 3 for a coded key.
    size_t payload(int slot, int type, unsigned char* out, int& stored) {
        stored = type;
        if (type == 0 && !raw_keys) {  // coded key: exceptions, then 3D Lorenzo residual passed through rANS (see key_residual)
            unsigned char* sym = (unsigned char*)e;  // free: no innovation is produced for a key frame
            for (int l = g.L - 1; l >= 1; --l) key_residual<short><<<blocks_for(g.nodes[l]), 256>>>(q[l][slot], g.nl[l], sym + g.klo[l], sym + g.khi[l]);
            key_residual<signed char><<<blocks_for(g.nodes[0]), 256>>>(q0[slot], g.nl[0], sym + g.klo[0], nullptr);
            CK(cudaMemcpy(innov.data(), e, g.kcount, cudaMemcpyDeviceToHost));
            const size_t head = write_ex(out);
            const size_t coded = rans_pack(innov.data(), g.kcount, g.kseg, g.kseg_end, out + head, g.key_bytes);
            if (coded) {
                stored = 3;
                return head + coded;
            }
        }
        if (type == 0) {  // raw key: states of the levels, then exceptions (overflows of the fine level)
            CK(cudaMemcpy(out + g.koff[0], q0[slot], 3 * g.nodes[0], cudaMemcpyDeviceToHost));
            for (int l = 1; l < g.L; ++l) CK(cudaMemcpy(out + g.koff[l], q[l][slot], 6 * g.nodes[l], cudaMemcpyDeviceToHost));
            return g.key_bytes + write_ex(out + g.key_bytes);
        }
        const size_t head = write_ex(out);  // exceptions, then frequency tables, block offsets and rANS streams
        CK(cudaMemcpy(innov.data(), e, g.ecount, cudaMemcpyDeviceToHost));
        return head + rans_pack(innov.data(), g.ecount, g.L, g.lv_end, out + head, (size_t)-1);
    }
    // Field error of the encoded state against the raw frame, over all the voxels (the lattice has just been
    // synthesised by encode): relative RMS, and largest error of a component. dump (optional) receives this maximum
    // per brick.
    void field_error(const float* d_planar, int slot, double& rms, double& max_abs, std::vector<float>* dump) {
        if (fine_on) field_err_bricks<<<blocks_for(bricks), 256>>>(total_field(slot), d_planar, g.n, e2, r2, emax);
        else field_err_bricks<<<blocks_for(bricks), 256>>>(base_field(slot), d_planar, g.n, e2, r2, emax);
        const double se = thrust::reduce(thrust::device_pointer_cast(e2), thrust::device_pointer_cast(e2) + bricks, 0.0);
        const double sr = thrust::reduce(thrust::device_pointer_cast(r2), thrust::device_pointer_cast(r2) + bricks, 0.0);
        max_abs = thrust::reduce(thrust::device_pointer_cast(emax), thrust::device_pointer_cast(emax) + bricks, 0.0f, thrust::maximum<float>());
        rms = std::sqrt(se / sr);
        if (dump) {
            dump->resize(bricks);
            CK(cudaMemcpy(dump->data(), emax, bricks * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
};

// ---------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------

struct SOptions {
    std::string mode;           // pack | play
    std::string data = "data/turb256.seq";
    std::string file = "build/turb256.psi";
    int degree = 2, mf = 2, levels = 3, fb = 2;
    float eps = 0.005f;
    double ridge = 1e-5;
    int key_interval = 32;      // one key frame every K frames (0: the first only)
    int start = 16, frames = 33;  // pack: first frame and number of frames encoded
    int check_every = 1;        // pack: fidelity check over the whole grid every n frames (0: never)
    bool allow_clip = false;    // pack: do not exit with an error when a coefficient is clipped
    bool raw_keys = false;      // pack: raw key frames (type 0) instead of coded ones (type 3)
    // pack: the predictors are fitted on frames fit_at + 1 to fit_at + 4 of the sequence. -1: at the start of the
    // encoded segment (--start). The files written before this default existed were fitted on frames 1 to 4:
    // --fit-at 0.
    int fit_at = -1;
    bool legacy_step = false;   // pack: old formula of the quantisation step (see pack)
    // pack: bound on the largest error of a component at voxel centres, as a fraction of u' (0: no sparse fine
    // level). fine_eps: quantisation error of the fine level (0: tolerance / 8.5).
    float tolerance = 0.0f, fine_eps = 0.0f;
    // pack: cap on the marking passes of the fine level per frame. The loop stops by itself when a pass no longer
    // adds a node; pack says on how many frames it stopped on this cap. The files from before the evening of
    // 15 November were written with a cap of 12.
    int fine_passes = 64;
    // pack: arithmetic margin of the bound, as a fraction of u'. The fine level is marked at tolerance - margin and
    // certified at tolerance - margin / 2. Each of the two halves has its role:
    //  - the marking kernel and the certification kernel do not round in the same way (up to 2.3e-5 u' of difference
    //    measured, on one voxel, over 55 frames at 512^3): at equal thresholds, voxels judged within the tolerance by
    //    one leave it for the other, and the bound is declared missed by a few millionths of u';
    //  - the reader evaluates the field with yet other kernels (cache or not, direct reading or reading by
    //    differences), and the GPU kernel departs from the analytic field by at most 2e-5 u' (measured): the second
    //    half is left in reserve for them.
    // The default therefore leaves each half (5e-5 u') a little more than twice the largest difference measured.
    // 0: old behaviour (marking and certification at the tolerance itself, files from before the evening of
    // 15 November).
    float tolerance_margin = 1e-4f;
    std::string probe;          // pack: writes a probe of the decoded field; play: reads it back and compares its own field
    int probe_frame = -1;       // pack: probed frame, counted from --start (-1: the last)
    bool no_fine = false;       // play: ignores the sparse fine level of a PSISTRM3 file (measurement of its cost)
    // pack: writes the seal (StreamSeal) after the frame table. --no-seal: file without a seal, identical to the byte
    // to that of an encoder from before the seal (the sealed file is this file followed by 48 bytes).
    bool seal = true;
    bool require_seal = false;  // play: refuses a file without a seal instead of reporting it
    std::string dump_positions; // play: writes the final positions of the particles
    std::string brick_dump;     // pack: file receiving, per checked frame, the largest error per brick of 4^3 voxels
    int substeps = 2;           // play: RK4 steps per interval
    uint32_t particles = 1u << 20;
    unsigned seed = 1234;
    int resort = 0;             // play: Morton sort of the particles every n frames (0: never)
    bool sort0 = false;         // play: initial sort
    int raymarch = 0, ray_steps = 256;  // play: one image per frame
    int max_frames = 0;         // play: number of frames to read (0: all)
    int seek = -1;              // play: measures the time of direct access to this frame
    int block = 128;            // play: threads per block of rk4_psi and raymarch_psi
    int block_plain = 256;      // play: threads per block of the kernels without the support cache
    // play: support cache in registers. -1 = automatic. RK4 steps: yes in degree 2; in degree 3, where the cache of
    // 64 nodes overflows the registers, only if the particles are not sorted. Rays: yes in degree 2, no in degree 3.
    int cache = -1, ray_cache = -1;
    bool csv = false;
};

// ---- pack ------------------------------------------------------------------------------

// Where pack gets its frames from: a sequence file for the command line (pack(o), below), or what a calling program
// provides (library, src/psitreillis.cu). read(k, dst) writes into dst the frame k of the source: 3 n^3 floats, in
// three planes u, v, w. hd carries what pack reads from a sequence header (n, n_frames, dt_frame, u_rms...).
struct FrameSource {
    SequenceHeader hd{};
    std::function<void(int, float*)> read;
};

template <int D>
static int pack(const SOptions& o, const FrameSource& src) {
    const SequenceHeader hd = src.hd;
    const int n = hd.n;
    CHK((n & (n - 1)) == 0, "the resolution must be a power of two");
    CHK(o.start >= 0 && o.frames >= 2 && o.start + o.frames <= hd.n_frames && hd.n_frames >= 5, "frame range outside the sequence");
    const size_t voxels = (size_t)n * n * n;
    Geo g;
    g.build(n, D, o.mf, o.levels, o.fb);
    const double gain2 = D == 2 ? 2.0 * 1.0 * 0.55 * 0.55 : 2.0 * (2.0 / 3.0) * (2416.0 / 5040.0) * (2416.0 / 5040.0);
    double level_sum = 0.0;
    for (int l = 0; l < g.L; ++l) level_sum += std::pow(0.25, l);
    // Quantisation noise of the velocity: the rounding of the fine level and that of the refinement at fb bits. The
    // coarse levels add nothing to it: each level projects what the coarser levels, already quantised, leave.
    // --legacy-step keeps the old formula, which counted one rounding per level and therefore gave too small a step:
    // the quantisation error was 0.900 eps with three levels and fb = 2.
    const double noise = o.legacy_step ? level_sum : 1.0 + std::pow(0.25, g.fb);
    const float delta = (float)(std::sqrt(12.0) * o.eps * hd.u_rms * g.hl[0] / std::sqrt(gain2 * noise));
    printf("pack           : %s, %d^3, frames %d to %d -> %s\n", o.data.c_str(), n, o.start, o.start + o.frames - 1, o.file.c_str());
    printf("Codec          : degree %d, %d levels (%d^3 to %d^3), potential step %.3e, one key frame every %d frames\n", D, g.L, g.nl[0],
           g.nl[g.L - 1], delta, o.key_interval);
    printf("Step           : expected quantisation error %.3f eps, that is %.4e u' (%s formula)\n",
           std::sqrt((1.0 + std::pow(0.25, g.fb)) / noise), o.eps * std::sqrt((1.0 + std::pow(0.25, g.fb)) / noise), o.legacy_step ? "old" : "new");
    // Sparse fine level: its coefficient enters the field as it is (neither refinement, nor fractional bits), hence a
    // quantisation noise of a single rounding. Placed everywhere in degree 3, it leaves a largest error of about 6.7
    // times its quantisation error (measured at 256^3): tolerance / 8.5 keeps a margin.
    const bool fine = o.tolerance > 0.0f;
    const float fine_eps = !fine ? 0.0f : (o.fine_eps > 0.0f ? o.fine_eps : o.tolerance / 8.5f);
    const float fine_delta = (float)(std::sqrt(12.0) * fine_eps * hd.u_rms * ((double)kL / n) / std::sqrt(gain2));
    // Arithmetic margin (see SOptions::tolerance_margin): marking at tolerance - margin, certification at
    // tolerance - margin / 2. With a zero margin, both thresholds equal the tolerance, as before.
    const float margin = fine ? o.tolerance_margin : 0.0f;
    CHK(!fine || (margin >= 0.0f && margin <= 0.1f * o.tolerance), "--tolerance-margin: from 0 to one tenth of the tolerance");
    const float marked = o.tolerance - margin;        // marking threshold, as a fraction of u'
    const float certified = fine_certified(o.tolerance, margin);  // certification threshold, as a fraction of u'
    const float certified_abs = certified * hd.u_rms;
    if (fine) {
        printf("Fine level     : potential of step 1 voxel in bricks of %d^3 nodes; tolerance %.2f %% of u' on the largest error of a component "
               "at voxel centres; quantisation error %.3f %% of u', step %.3e\n",
               kFineSide, 100.0 * o.tolerance, 100.0 * fine_eps, fine_delta);
        printf("Certification  : marking at %.6f u', certification at %.6f u' (tolerance minus an arithmetic margin of %.1e u', then minus half of it); "
               "at most %d marking passes per frame\n",
               marked, certified, margin, o.fine_passes);
    }

    float* h_planar = nullptr;
    CK(cudaHostAlloc(&h_planar, 3 * voxels * sizeof(float), cudaHostAllocDefault));
    float* d_planar = nullptr;
    CK(cudaMalloc(&d_planar, 3 * voxels * sizeof(float)));
    double t_disk = 0.0;
    int n_disk = 0;
    auto load = [&](int k) {
        const auto t0 = std::chrono::steady_clock::now();
        src.read(k, h_planar);
        t_disk += seconds_since(t0);
        ++n_disk;
        CK(cudaMemcpy(d_planar, h_planar, 3 * voxels * sizeof(float), cudaMemcpyHostToDevice));
    };
    Sender<D> snd;
    snd.init(g, delta, o.ridge);
    snd.raw_keys = o.raw_keys;
    CHK(o.fine_passes >= 1 && o.fine_passes <= 1000, "--fine-passes: from 1 to 1000");
    snd.max_passes = o.fine_passes;
    if (fine) snd.init_fine(marked * hd.u_rms, fine_delta);

    // Fit of the predictors on four quantised frames, taken at the start of the encoded segment (except --fit-at).
    // Measured at 256^3: a fit made 90 frames earlier costs 0.8 % (degree 2) to 2.0 % (degree 3) on the innovations;
    // a fit redone at each key would regain only 0.1 to 0.8 % of it.
    const int fit_at = o.fit_at >= 0 ? o.fit_at : std::min(o.start, hd.n_frames - 5);
    {
        CHK(fit_at >= 0 && fit_at + 4 < hd.n_frames, "--fit-at: fitting frames outside the sequence");
        std::vector<std::vector<short>> hq[4];
        for (int t = 0; t < 4; ++t) {
            load(fit_at + t + 1);
            snd.encode(d_planar, t & 1, 0);
            hq[t].resize(g.L);
            for (int l = 0; l < g.L; ++l) {
                hq[t][l].resize(3 * g.nodes[l]);
                if (l == 0) {
                    std::vector<signed char> b(3 * g.nodes[0]);
                    CK(cudaMemcpy(b.data(), snd.q0[t & 1], b.size(), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < b.size(); ++i) hq[t][0][i] = b[i];
                } else {
                    CK(cudaMemcpy(hq[t][l].data(), snd.q[l][t & 1], 6 * g.nodes[l], cudaMemcpyDeviceToHost));
                }
            }
        }
        for (int l = 0; l < g.L; ++l) {
            const int nn = g.nl[l], m = nn - 1;
            const size_t plane = (size_t)nn * nn, stride = std::max<size_t>(1, g.nodes[l] / 400000);
            double G[3][3] = {}, b[3] = {};
            for (int t = 1; t <= 2; ++t)
                for (size_t i = 0; i < g.nodes[l]; i += stride) {
                    const int ix = (int)(i % nn), iy = (int)((i / nn) % nn), iz = (int)(i / plane);
                    const size_t nb[6] = {(size_t)iz * plane + (size_t)iy * nn + ((ix + 1) & m), (size_t)iz * plane + (size_t)iy * nn + ((ix - 1) & m),
                                          (size_t)iz * plane + (size_t)((iy + 1) & m) * nn + ix, (size_t)iz * plane + (size_t)((iy - 1) & m) * nn + ix,
                                          (size_t)((iz + 1) & m) * plane + (size_t)iy * nn + ix, (size_t)((iz - 1) & m) * plane + (size_t)iy * nn + ix};
                    for (int c = 0; c < 3; ++c) {
                        double z[3] = {(double)hq[t][l][3 * i + c], (double)hq[t - 1][l][3 * i + c], -6.0 * hq[t][l][3 * i + c]};
                        for (int j = 0; j < 6; ++j) z[2] += hq[t][l][3 * nb[j] + c];
                        const double y = hq[t + 1][l][3 * i + c];
                        for (int p = 0; p < 3; ++p) {
                            for (int d = 0; d < 3; ++d) G[p][d] += z[p] * z[d];
                            b[p] += z[p] * y;
                        }
                    }
                }
            double x[3] = {1.0, 0.0, 0.0};
            for (int p = 0; p < 3; ++p) G[p][p] *= 1.0 + 1e-9;
            CHK(solve3(G, b, x), "degenerate fit of the predictor");
            snd.coef[l] = Coef{(int)std::lround(1024.0 * x[0]), (int)std::lround(1024.0 * x[1]), (int)std::lround(1024.0 * x[2])};
        }
        snd.reset_clip();
    }

    FILE* out = fopen(o.file.c_str(), "wb");
    CHK(out, "cannot create the output file");
    StreamHeader sh{};
    memcpy(sh.magic, fine ? "PSISTRM3" : "PSISTRM2", 8);
    sh.n = n;
    sh.degree = D;
    sh.mf = o.mf;
    sh.levels = g.L;
    sh.fb = o.fb;
    sh.key_interval = o.key_interval;
    sh.n_frames = o.frames;
    sh.first_frame = o.start;
    sh.delta = delta;
    sh.dt_frame = hd.dt_frame;
    sh.u_rms = hd.u_rms;
    sh.re_lambda = hd.re_lambda;
    sh.kmax_eta = hd.kmax_eta;
    sh.eps = o.eps;
    for (int l = 0; l < g.L; ++l) {
        sh.coef[l][0] = snd.coef[l].a;
        sh.coef[l][1] = snd.coef[l].b;
        sh.coef[l][2] = snd.coef[l].c;
    }
    CHK(fwrite(&sh, sizeof(sh), 1, out) == 1, "writing of the header");

    FineInfo fi{};
    if (fine) {  // the header of the fine level follows; its maxima are rewritten at the end
        fi.tolerance = o.tolerance;
        fi.eps = fine_eps;
        fi.delta = fine_delta;
        fi.brick = kFineSide;
        memcpy(&fi.reserved[0], &margin, sizeof(margin));
        CHK(fwrite(&fi, sizeof(fi), 1, out) == 1, "writing of the header of the fine level");
    }
    std::vector<unsigned char> buf(g.cap + 64 + (fine ? fine_section_cap(snd.fs.bm_bytes(), snd.fs.cap_bricks, snd.fs.cap_nodes) : 0));
    std::vector<FrameRecord> index(o.frames);
    uint64_t pos = sizeof(sh) + (fine ? sizeof(fi) : 0);
    size_t fine_total = 0;
    double fine_bricks = 0.0, fine_nodes = 0.0, t_fine = 0.0;
    uint32_t max_bricks = 0, max_nodes = 0, frames_over = 0;
    int max_passes = 0, frames_capped = 0;  // largest number of passes; frames stopped on the cap of passes
    // Floor of the fine level: frames where, the marking loop having converged, voxels remain outside the marking
    // threshold although all their nodes are retained; largest number of such voxels in one frame.
    int frames_floor = 0;
    unsigned floor_voxels = 0;
    double t_enc = 0.0, err_sum = 0.0, err_max = 0.0, max_worst = 0.0;
    int err_n = 0;
    FILE* dump = nullptr;
    std::vector<float> brick;
    if (!o.brick_dump.empty()) {
        dump = fopen(o.brick_dump.c_str(), "wb");
        CHK(dump, "cannot create the per-brick log");
    }
    size_t key_total = 0, delta_total = 0, n_key = 0, n_key_coded = 0;
    double t_key = 0.0;
    for (int j = 0; j < o.frames; ++j) {
        const bool key = j == 0 || (o.key_interval > 0 && j % o.key_interval == 0);
        const bool after_key = !key && (j == 1 || (o.key_interval > 0 && j % o.key_interval == 1));
        const int type = key ? 0 : (after_key ? 2 : 1), slot = j & 1;
        load(o.start + j);
        const auto t0 = std::chrono::steady_clock::now();
        snd.encode(d_planar, slot, type);
        const auto tf = std::chrono::steady_clock::now();
        if (fine) snd.refine(d_planar, slot);
        t_fine += seconds_since(tf);
        int stored = type;
        size_t bytes = snd.payload(slot, type, buf.data(), stored);
        if (fine) {
            const auto ta = std::chrono::steady_clock::now();
            size_t section = 0;
            bytes = snd.append_fine(slot, buf.data(), bytes, section);
            t_fine += seconds_since(ta);
            fine_total += section;
            fine_bricks += snd.fs.n_bricks[slot];
            fine_nodes += snd.fs.n_nodes[slot];
            max_bricks = std::max(max_bricks, snd.fs.n_bricks[slot]);
            max_nodes = std::max(max_nodes, snd.fs.n_nodes[slot]);
            max_passes = std::max(max_passes, snd.fine_passes);
            frames_capped += snd.fine_capped;
            if (snd.fine_left && !snd.fine_capped) {
                ++frames_floor;
                floor_voxels = std::max(floor_voxels, snd.fine_left);
            }
        }
        t_enc += seconds_since(t0);
        if (key) t_key += seconds_since(t0);
        n_key_coded += stored == 3;
        FrameRecord& r = index[j];
        r.offset = pos;
        r.bytes = bytes;
        r.type = stored;
        r.hash = state_hash(g, snd.q0, snd.q, slot, snd.d_hash);
        if (fine) r.hash ^= fine_hash(snd.fs, slot, snd.d_hash);
        for (int c = 0; c < 3; ++c) r.mean[c] = snd.mean[slot][c];
        r.field_err = -1.0f;
        r.field_max = 0.0f;
        if (fine || (o.check_every > 0 && j % o.check_every == 0)) {  // with a tolerance, all the frames are checked
            double fe = 0.0, fm = 0.0;
            snd.field_error(d_planar, slot, fe, fm, dump ? &brick : nullptr);
            r.field_err = (float)fe;
            r.field_max = (float)(fm / hd.u_rms);
            err_sum += fe;
            err_max = std::max(err_max, fe);
            max_worst = std::max(max_worst, fm / hd.u_rms);
            ++err_n;
            if (fine && fm > (double)certified_abs) ++frames_over;
            if (dump) CHK(fwrite(brick.data(), sizeof(float), brick.size(), dump) == brick.size(), "writing of the per-brick log");
        }
        if (!o.probe.empty() && j == (o.probe_frame >= 0 ? o.probe_frame : o.frames - 1))
            snd.write_probe(o.probe.c_str(), slot, o.start + j, hd.u_rms, snd.frames_wrapped != 0);
        CHK(fwrite(buf.data(), 1, bytes, out) == bytes, "writing of a frame");
        pos += bytes;
        if (key) {
            key_total += bytes;
            ++n_key;
        } else {
            delta_total += bytes;
        }
    }
    sh.index_offset = pos;
    sh.reserved[0] = kClipMarker;
    for (int site = 0; site < kClipSites; ++site) sh.reserved[1 + site] = (uint64_t)snd.clip_site(site);
    sh.reserved[7] = (uint64_t)snd.frames_clipped | ((uint64_t)snd.frames_wrapped << 32);
    if (dump) fclose(dump);
    CHK(fwrite(index.data(), sizeof(FrameRecord), index.size(), out) == index.size(), "writing of the table");
    pos += index.size() * sizeof(FrameRecord);
    if (fine) {
        fi.max_error = (float)max_worst;
        fi.max_bricks = max_bricks;
        fi.max_nodes = max_nodes;
        fi.frames_over = frames_over;
    }
    if (o.seal) {  // fingerprints of the headers in their final state and of the table, following the table
        StreamSeal seal{};
        memcpy(seal.magic, "PSISEAL1", 8);
        seal.header_hash = host_hash(&sh, sizeof(sh), kSealHeader);
        seal.fine_hash = fine ? host_hash(&fi, sizeof(fi), kSealFine) : 0ull;
        seal.index_hash = host_hash(index.data(), index.size() * sizeof(FrameRecord), kSealIndex);
        CHK(fwrite(&seal, sizeof(seal), 1, out) == 1, "writing of the seal");
        pos += sizeof(seal);
    }
    seek64(out, 0);
    CHK(fwrite(&sh, sizeof(sh), 1, out) == 1, "writing of the header");
    if (fine) CHK(fwrite(&fi, sizeof(fi), 1, out) == 1, "writing of the header of the fine level");
    fclose(out);

    const double mib = 1048576.0, raw = 3.0 * voxels * sizeof(float) * o.frames / mib;
    const size_t n_delta = o.frames - n_key;
    printf("\nPredictors     : fitted on frames %d to %d;", fit_at + 1, fit_at + 4);
    for (int l = 0; l < g.L; ++l) printf(" level %d (%.3f, %.3f, %.3f)", l, snd.coef[l].a / 1024.0, snd.coef[l].b / 1024.0, snd.coef[l].c / 1024.0);
    printf("\nFile           : %.3f MiB for %d frames, against %.0f MiB raw: %.1fx\n", pos / mib, o.frames, raw, raw / (pos / mib));
    printf("Seal           : %s\n", o.seal ? "header and frame table sealed (48 bytes after the table)"
                                           : "none (--no-seal): the header and the frame table are covered by no fingerprint");
    printf("Frames         : %zu keys of %.3f MiB, %zu innovations of %.3f MiB on average\n", n_key, n_key ? key_total / mib / n_key : 0.0, n_delta,
           n_delta ? delta_total / mib / n_delta : 0.0);
    printf("Keys           : %zu coded out of %zu (3D Lorenzo residual, rANS), %.1fx less than a raw frame against %.1fx for a key written raw; "
           "encoding of a key %.2f ms\n",
           n_key_coded, n_key, n_key ? raw / o.frames / (key_total / mib / n_key) : 0.0, raw / o.frames / (g.key_bytes / mib), n_key ? 1e3 * t_key / n_key : 0.0);
    printf("Encoding       : %.2f ms per frame (projection, quantisation, entropy coder); raw disk read %.1f ms per frame\n",
           1e3 * t_enc / o.frames, 1e3 * t_disk / n_disk);
    if (err_n)
        printf("Fidelity       : relative field error %.4e on average, %.4e at worst; largest error of a component %.4f u'; %d frames "
               "checked, on all the voxels\n",
               err_sum / err_n, err_max, max_worst, err_n);
    printf("Saturations    : %lld coefficients\n", snd.clipped());
    printf("Exceptions     : %lld innovation escapes and %lld overflows of the fine level, that is %.4f MiB over the file\n", snd.n_escape, snd.n_fine,
           (snd.n_escape + snd.n_fine) * sizeof(ExEntry) / mib);
    if (snd.frames_wrapped)
        printf("Lattice        : %d frame(s) whose potential leaves the range of a short; the reader will read this file by differences\n", snd.frames_wrapped);
    if (fine) {
        const double base_total = (double)(key_total + delta_total) - (double)fine_total;
        printf("Fine level     : %.0f bricks and %.0f nodes per frame on average, that is %.3f %% of the nodes (at most %u bricks and %u nodes); "
               "%.4f MiB per frame, that is +%.2f %% in bytes; at most %d passes; %.1f ms per frame\n",
               fine_bricks / o.frames, fine_nodes / o.frames, 100.0 * fine_nodes / o.frames / (double)voxels, max_bricks, max_nodes,
               fine_total / mib / o.frames, 100.0 * fine_total / base_total, max_passes, 1e3 * t_fine / o.frames);
        printf("Passes         : cap of %d pass(es) per frame; %d frame(s) stopped on this cap while nodes were still being added\n",
               o.fine_passes, frames_capped);
        printf("Floor          : %d frame(s) where voxels remain outside the marking threshold once all their nodes are retained (at most %u voxels "
               "in one frame)\n",
               frames_floor, floor_voxels);
        printf("Bound          : largest error observed %.4f u' for a tolerance of %.4f u'; %u frame(s) above the certification threshold "
               "(%.6f u') out of %d checked\n",
               max_worst, o.tolerance, frames_over, certified, err_n);
    }
    const long long clip_all = snd.clip_all();
    printf("Clippings      : %lld in total, on %d frame(s)", clip_all, snd.frames_clipped);
    for (int site = 0; site < kClipSites; ++site)
        for (int l = 0; l < kMaxLevels; ++l)
            if (snd.clip[l][site]) printf("; %s, level %d: %lld", kClipName[site], l, snd.clip[l][site]);
    printf("\n");
    if (clip_all) {
        static const char* const kTypeName[3] = {"key frames", "innovations", "frames that follow a key"};
        printf("               : by frame type:");
        for (int t = 0; t < 3; ++t) {
            long long tt = 0;
            for (int site = 0; site < kClipSites; ++site) tt += snd.clip_type[t][site];
            printf(" %s %lld%s", kTypeName[t], tt, t < 2 ? ";" : "\n");
        }
    }
    if (o.csv)
        printf("CSV,pack,%d,%d,%d,%d,%.4g,%d,%.4f,%.4f,%.2f,%.4f,%.4f,%.3f,%.4e,%.4e,%lld,%.5f,%lld,%lld,%lld,%lld,%lld,%lld\n", n, D, g.L, o.frames, o.eps,
               o.key_interval, pos / mib, raw, raw / (pos / mib), n_key ? key_total / mib / n_key : 0.0, n_delta ? delta_total / mib / n_delta : 0.0,
               1e3 * t_enc / o.frames, err_n ? err_sum / err_n : 0.0, err_max, snd.clipped(), max_worst, snd.clip_site(0), snd.clip_site(1),
               snd.clip_site(2), snd.clip_site(3), snd.clip_site(4), snd.clip_site(5));
    if (o.csv && fine)
        printf("CSV,fine,%d,%d,%d,%.5g,%.5g,%.4f,%.6f,%.5f,%.6f,%u,%u,%u,%d,%.2f\n", n, D, o.frames, o.tolerance, fine_eps, fine_total / mib / o.frames,
               (double)fine_total / ((double)(key_total + delta_total) - (double)fine_total), max_worst, fine_nodes / o.frames / (double)voxels, max_bricks,
               max_nodes, frames_over, max_passes, 1e3 * t_fine / o.frames);
    // A separate line, so as not to shift the columns of "CSV,fine" that the scripts read: cap on the passes, largest
    // number of passes, frames stopped on the cap, arithmetic margin and certification threshold (fractions of u';
    // the marking threshold is tolerance - margin, the certification threshold tolerance - margin / 2), frames at the
    // floor of the fine level and largest number of voxels at the floor in one frame.
    if (o.csv && fine)
        printf("CSV,passes,%d,%d,%d,%d,%d,%d,%.3g,%.6f,%d,%u\n", n, D, o.frames, o.fine_passes, max_passes, frames_capped, margin, certified, frames_floor,
               floor_voxels);
    if (fine && frames_over) {
        fprintf(stderr, "Failure: bound not held. Largest error %.6f u' for a certification threshold of %.6f u' (tolerance %.4f u', margin %.1e u'), "
                        "on %u frame(s). The file is written.\n",
                max_worst, certified, o.tolerance, margin, frames_over);
        if (frames_capped)
            fprintf(stderr, "  The cap of %d pass(es) was reached on %d frame(s): rerun with a larger --fine-passes before concluding\n"
                            "  that the fine level has a floor.\n", o.fine_passes, frames_capped);
        else if (frames_floor)
            fprintf(stderr, "  The fine level cannot go below its own floor (%d frame(s) where voxels remain outside the threshold, all their\n"
                            "  nodes retained): relax --tolerance, or lower --fine-eps.\n", frames_floor);
        else
            fprintf(stderr, "  The marking had converged with no voxel outside the threshold: the certification kernel differs from the marking kernel by more\n"
                            "  than half a margin. Increase --tolerance-margin.\n");
        if (!(clip_all && !o.allow_clip)) return 4;
    }
    if (clip_all && !o.allow_clip) {
        fprintf(stderr, "Failure: %lld coefficient(s) clipped. The file is written, but it does not reproduce the requested projection.\n"
                        "  Increase --eps, or rerun with --allow-clipping to accept it.\n", clip_all);
        return 3;
    }
    return 0;
}

// pack from a sequence file (--data): the command line.
template <int D>
static int pack(const SOptions& o) {
    FILE* in = fopen(o.data.c_str(), "rb");
    CHK(in, "sequence not found");
    FrameSource src;
    src.hd = sequence_read_header(in);
    src.read = [&](int k, float* dst) { sequence_read_frame(in, src.hd, k, dst); };
    const int rc = pack<D>(o, src);
    fclose(in);
    return rc;
}

// The .psi format is that of these structures as they are in memory (docs/FORMAT.md). Their size and the offset of
// their fields are fixed here: if one of them moves, the compilation fails instead of writing unreadable files.
#include <cstddef>
static_assert(sizeof(StreamHeader) == 208 && offsetof(StreamHeader, coef) == 64 && offsetof(StreamHeader, index_offset) == 136 &&
                  offsetof(StreamHeader, reserved) == 144,
              "StreamHeader: layout of the .psi format");
static_assert(sizeof(FrameRecord) == 48 && offsetof(FrameRecord, type) == 24 && offsetof(FrameRecord, mean) == 28 &&
                  offsetof(FrameRecord, field_err) == 40 && offsetof(FrameRecord, field_max) == 44,
              "FrameRecord: layout of the .psi format");
static_assert(sizeof(FineInfo) == 64 && offsetof(FineInfo, brick) == 24 && offsetof(FineInfo, reserved) == 32, "FineInfo: layout of the .psi format");
static_assert(sizeof(FineTrailer) == 16, "FineTrailer: layout of the .psi format");
static_assert(sizeof(StreamSeal) == 48 && sizeof(StreamSeal) == sizeof(FrameRecord) && offsetof(StreamSeal, index_hash) == 24,
              "StreamSeal: layout of the .psi format");
static_assert(sizeof(ExEntry) == 8, "ExEntry: layout of the .psi format");
static_assert(sizeof(SequenceHeader) == 72, "SequenceHeader: layout of the sequence format");


#endif  // PSIT_CODEC_STREAM_CUH
