// SPDX-License-Identifier: Apache-2.0
// psi_ref: reference reader of .psi files, on CPU.
//
// Written from docs/FORMAT.md only, without reading the sources of the codec (src/): it puts the specification and the
// GPU reader to the test (claim REF-1 of docs/CLAIMS.md). C++17, with neither CUDA nor any external library.
//
//   psi_ref <file.psi> [--probe f.probe] [--state j out.bin] [--points f.txt --frame j] [--frames n]
//           [--threads n] [--csv]
//
//   --probe f.probe   compares, at the frame of the probe, the sampling lattice, the fine level and the velocity field
//                     with those recorded by `psi_stream pack --probe`
//   --state j f.bin   writes the decoded state of frame j (position in the file), in the layout of a raw key
//   --points f.txt    reads positions "x y z" (one per line, same units as the box of side 2 pi) and writes to the
//                     output the velocity field of the frame --frame j at these points
//   --frames n        reads only the first n frames
//   --threads n       number of threads (by default: that of the machine)
//   --csv             adds a line "CSV,ref,..."
//
// Exit code: 0 if the file is read and each recomputed fingerprint is that of the table; 1 if the file is refused or
// if a fingerprint differs; 2 for a usage error.
//
// The section numbers quoted in the comments are those of docs/FORMAT.md. What the specification does not say and had
// to be deduced or established by trial is marked "OUTSIDE THE SPECIFICATION"; the list is in
// benchmarks/independent_readers/cpp/README.md.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// The messages are in UTF-8: tell the Windows console so, without including <windows.h>.
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
#endif

namespace {

// --- Tools -----------------------------------------------------------------------------------------------------------

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const char* format, ...) {
    char text[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof text, format, args);
    va_end(args);
    throw Failure(text);
}

// The format is little-endian (section 1). These reads assume nothing about the host machine.
inline uint16_t read16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t read32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint64_t read64(const uint8_t* p) { return uint64_t(read32(p)) | (uint64_t(read32(p + 4)) << 32); }
inline float read_f32(const uint8_t* p) {
    uint32_t bits = read32(p);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int g_threads = 1;

// Runs f(start, end) on slices of [0, total), one per thread.
void in_parallel(uint64_t total, const std::function<void(uint64_t, uint64_t)>& f) {
    uint64_t threads = std::min<uint64_t>(uint64_t(g_threads), std::max<uint64_t>(total, 1));
    if (threads <= 1) {
        f(0, total);
        return;
    }
    std::vector<std::thread> team;
    std::vector<std::string> errors(threads);
    for (uint64_t t = 0; t < threads; ++t) {
        uint64_t start = total * t / threads, end = total * (t + 1) / threads;
        team.emplace_back([&, t, start, end] {
            try {
                f(start, end);
            } catch (const std::exception& e) {
                errors[t] = e.what();
            }
        });
    }
    for (auto& worker : team) worker.join();
    for (const auto& e : errors)
        if (!e.empty()) throw Failure(e);
}

// --- Fingerprints (section 7) ----------------------------------------------------------------------------------------

inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Fingerprint of a byte array: exclusive OR, over the positions i, of mix64(salt + 256 * i + a[i]).
uint64_t fingerprint(const uint8_t* a, uint64_t size, uint64_t salt) {
    uint64_t threads = std::min<uint64_t>(uint64_t(g_threads), size / 65536 + 1);
    std::vector<uint64_t> partials(threads, 0);
    in_parallel(threads, [&](uint64_t t0, uint64_t t1) {
        for (uint64_t t = t0; t < t1; ++t) {
            uint64_t h = 0;
            for (uint64_t i = size * t / threads, end = size * (t + 1) / threads; i < end; ++i)
                h ^= mix64(salt + 256 * i + a[i]);
            partials[t] = h;
        }
    });
    uint64_t h = 0;
    for (uint64_t p : partials) h ^= p;
    return h;
}

// --- Structures of the file (sections 2 to 4) ------------------------------------------------------------------------

constexpr uint64_t kHeaderSize = 208, kFineHeaderSize = 64, kFrameRecordSize = 48, kSealSize = 48;
constexpr uint64_t kClipMarker = 0x3150494C43495350ull;   // "PSICLIP1"
constexpr uint32_t kOverflow = 0x80000000u;                // bit 31 of ExEntry.idx
constexpr uint32_t kBlock = 1024;                          // symbols per rANS block
constexpr uint32_t kLowerBound = 1u << 23;                 // lower bound of the rANS state

struct Header {
    int version = 0;   // 1, 2 or 3
    int32_t n = 0, degree = 0, mf = 0, levels = 0, fb = 0, key_interval = 0, n_frames = 0, first_frame = 0;
    float delta = 0, dt_frame = 0, u_rms = 0, re_lambda = 0, kmax_eta = 0, eps = 0;
    int32_t coef[6][3] = {};
    uint64_t index_offset = 0;
    uint64_t reserved[8] = {};
};

struct FineHeader {
    float tolerance = 0, eps = 0, delta = 0, max_error = 0;
    uint32_t max_bricks = 0, max_nodes = 0;
    int32_t brick = 0;
    uint32_t frames_over = 0;
    float margin = 0;
};

struct FrameRecord {
    uint64_t offset = 0, bytes = 0, hash = 0;
    int32_t type = 0;
    float mean[3] = {};
    float field_err = 0, field_max = 0;
};

// Quantities derived from the header (section 5).
struct Geometry {
    int L = 0;
    uint64_t nl[6] = {}, nodes[6] = {}, seg[6] = {};
    uint64_t ecount = 0, n_blocks = 0, key_bytes = 0, n_kblocks = 0, ex_cap = 0;
    uint64_t nb = 0, bricks = 0;   // fine level: bricks per axis, bricks in all
};

struct PsiFile {
    std::string path;
    std::ifstream stream;
    uint64_t size = 0;
    Header e;
    FineHeader fine;
    Geometry g;
    std::vector<FrameRecord> table;
    bool sealed = false;
    uint64_t frames_start = 0;   // end of the headers
};

inline bool power_of_two(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

void read_bytes(PsiFile& f, uint64_t offset, uint8_t* buffer, uint64_t size, const char* what) {
    f.stream.clear();
    f.stream.seekg(std::streamoff(offset), std::ios::beg);
    f.stream.read(reinterpret_cast<char*>(buffer), std::streamsize(size));
    if (uint64_t(f.stream.gcount()) != size) fail("incomplete read (%s)", what);
}

// Opens the file and applies the opening checks of section 9.
void open_file(PsiFile& f, const std::string& path) {
    f.path = path;
    f.stream.open(path, std::ios::binary);
    if (!f.stream) fail("cannot open %s", path.c_str());
    f.stream.seekg(0, std::ios::end);
    f.size = uint64_t(f.stream.tellg());
    if (f.size < kHeaderSize) fail("file too short for a header (%llu bytes)", (unsigned long long)f.size);

    uint8_t h[kHeaderSize];
    read_bytes(f, 0, h, kHeaderSize, "header");
    Header& e = f.e;
    if (!std::memcmp(h, "PSISTRM1", 8)) e.version = 1;
    else if (!std::memcmp(h, "PSISTRM2", 8)) e.version = 2;
    else if (!std::memcmp(h, "PSISTRM3", 8)) e.version = 3;
    else fail("unknown signature: this is not a .psi file");
    int32_t* integers[8] = {&e.n, &e.degree, &e.mf, &e.levels, &e.fb, &e.key_interval, &e.n_frames, &e.first_frame};
    for (int i = 0; i < 8; ++i) *integers[i] = int32_t(read32(h + 8 + 4 * i));
    float* floats[6] = {&e.delta, &e.dt_frame, &e.u_rms, &e.re_lambda, &e.kmax_eta, &e.eps};
    for (int i = 0; i < 6; ++i) *floats[i] = read_f32(h + 40 + 4 * i);
    for (int l = 0; l < 6; ++l)
        for (int c = 0; c < 3; ++c) e.coef[l][c] = int32_t(read32(h + 64 + 12 * l + 4 * c));
    e.index_offset = read64(h + 136);
    for (int i = 0; i < 8; ++i) e.reserved[i] = read64(h + 144 + 8 * i);

    if (e.degree != 2 && e.degree != 3) fail("degree %d: only 2 and 3 exist", e.degree);
    if (!power_of_two(e.n) || e.n < 16 || e.n > 8192) fail("n = %d: a power of two from 16 to 8192 is expected", e.n);
    if (e.n_frames < 2 || e.n_frames > (1 << 24)) fail("n_frames = %d outside 2 to 2^24", e.n_frames);
    if (e.levels < 1 || e.levels > 6) fail("levels = %d outside 1 to 6", e.levels);
    if (!power_of_two(e.mf) || e.mf > e.n) fail("mf = %d: a power of two is expected", e.mf);
    if (e.fb < 0 || e.fb > 12) fail("fb = %d outside 0 to 12", e.fb);

    f.frames_start = kHeaderSize + (e.version == 3 ? kFineHeaderSize : 0);
    if (e.index_offset < f.frames_start)
        fail("index_offset = %llu: table missing (interrupted encoding?) or placed inside the headers",
             (unsigned long long)e.index_offset);
    uint64_t table_end = e.index_offset + kFrameRecordSize * uint64_t(e.n_frames);
    if (f.size == table_end) f.sealed = false;
    else if (f.size == table_end + kSealSize) f.sealed = true;
    else fail("file size %llu: %llu (without a seal) or %llu (with the seal) expected", (unsigned long long)f.size,
              (unsigned long long)table_end, (unsigned long long)(table_end + kSealSize));

    // Geometry (section 5).
    Geometry& g = f.g;
    g.L = e.levels;
    for (int l = 0; l < g.L; ++l) {
        g.nl[l] = uint64_t(e.n / e.mf) >> l;
        if (g.nl[l] < 8) fail("level %d: %llu nodes per axis, at least 8 expected", l, (unsigned long long)g.nl[l]);
        g.nodes[l] = g.nl[l] * g.nl[l] * g.nl[l];
        g.ecount += 3 * g.nodes[l];
    }
    if (g.ecount >= (1ull << 31)) fail("ecount = %llu: it must fit in 31 bits", (unsigned long long)g.ecount);
    g.seg[g.L - 1] = 0;
    for (int l = g.L - 2; l >= 0; --l) g.seg[l] = g.seg[l + 1] + 3 * g.nodes[l + 1];
    g.n_blocks = (g.ecount + kBlock - 1) / kBlock;
    g.key_bytes = 3 * g.nodes[0];
    for (int l = 1; l < g.L; ++l) g.key_bytes += 6 * g.nodes[l];
    g.n_kblocks = (g.key_bytes + kBlock - 1) / kBlock;
    g.ex_cap = std::max<uint64_t>(65536, g.ecount / 64);
    uint64_t ceiling = uint64_t(g.L - 1) * (32768ull << e.fb);
    if (ceiling >= (e.degree == 2 ? (1ull << 25) : (1ull << 22)))
        fail("fb = %d with %d levels in degree %d: the integer refinement would overflow", e.fb, g.L, e.degree);

    // Header of the fine level (section 3).
    if (e.version == 3) {
        uint8_t b[kFineHeaderSize];
        read_bytes(f, kHeaderSize, b, kFineHeaderSize, "header of the fine level");
        f.fine.tolerance = read_f32(b);
        f.fine.eps = read_f32(b + 4);
        f.fine.delta = read_f32(b + 8);
        f.fine.max_error = read_f32(b + 12);
        f.fine.max_bricks = read32(b + 16);
        f.fine.max_nodes = read32(b + 20);
        f.fine.brick = int32_t(read32(b + 24));
        f.fine.frames_over = read32(b + 28);
        f.fine.margin = read_f32(b + 32);
        if (f.fine.brick != 4) fail("fine level: bricks of side %d, only 4 exists", f.fine.brick);
        if (e.n < 32 || e.n > 1024) fail("fine level: n = %d outside 32 to 1024", e.n);
        g.nb = uint64_t(e.n) / 4;
        g.bricks = g.nb * g.nb * g.nb;
    }

    // Frame table (section 4).
    std::vector<uint8_t> t(kFrameRecordSize * uint64_t(e.n_frames));
    read_bytes(f, e.index_offset, t.data(), t.size(), "frame table");
    f.table.resize(size_t(e.n_frames));
    for (int j = 0; j < e.n_frames; ++j) {
        const uint8_t* p = t.data() + kFrameRecordSize * uint64_t(j);
        FrameRecord& r = f.table[size_t(j)];
        r.offset = read64(p);
        r.bytes = read64(p + 8);
        r.hash = read64(p + 16);
        r.type = int32_t(read32(p + 24));
        for (int c = 0; c < 3; ++c) r.mean[c] = read_f32(p + 28 + 4 * c);
        r.field_err = read_f32(p + 40);
        r.field_max = read_f32(p + 44);
        if (r.type < 0 || r.type > 3) fail("frame %d: unknown type %d", j, r.type);
        if (r.type == 3 && e.version == 1) fail("frame %d: coded key in a PSISTRM1 file", j);
        if (r.offset < f.frames_start || r.offset > e.index_offset || r.bytes > e.index_offset - r.offset)
            fail("frame %d: offset %llu and size %llu outside the frame area", j, (unsigned long long)r.offset,
                 (unsigned long long)r.bytes);
    }
    if (f.table[0].type != 0 && f.table[0].type != 3) fail("frame 0 is not a key frame");

    // Seal (section 4.1).
    if (f.sealed) {
        uint8_t s[kSealSize];
        read_bytes(f, table_end, s, kSealSize, "seal");
        if (std::memcmp(s, "PSISEAL1", 8) != 0) fail("the 48 bytes that follow the table are not a seal");
        if (read64(s + 32) != 0 || read64(s + 40) != 0) fail("seal: reserved words not zero");
        if (read64(s + 8) != fingerprint(h, kHeaderSize, 0x3000)) fail("seal: the header has changed since the encoding");
        uint64_t expected_fine = 0;
        if (e.version == 3) {
            uint8_t b[kFineHeaderSize];
            read_bytes(f, kHeaderSize, b, kFineHeaderSize, "header of the fine level");
            expected_fine = fingerprint(b, kFineHeaderSize, 0x3000 + (1ull << 56));
        }
        if (read64(s + 16) != expected_fine) fail("seal: the header of the fine level has changed since the encoding");
        if (read64(s + 24) != fingerprint(t.data(), t.size(), 0x3000 + (2ull << 56)))
            fail("seal: the frame table has changed since the encoding");
    }
}

// --- rANS (section 6.1) ----------------------------------------------------------------------------------------------

struct RansTable {
    uint16_t freq[256];
    uint16_t cum[256];
    uint8_t sym[4096];
};

void build_table(const uint8_t* p, RansTable& t, int frame, const char* what, int index) {
    uint32_t sum = 0;
    for (int v = 0; v < 256; ++v) {
        t.freq[v] = read16(p + 2 * v);
        sum += t.freq[v];
    }
    if (sum != 4096) fail("frame %d: frequency table %d (%s) of sum %u instead of 4096", frame, index, what, sum);
    uint32_t c = 0;
    for (int v = 0; v < 256; ++v) {
        t.cum[v] = uint16_t(c);
        for (uint32_t k = 0; k < t.freq[v]; ++k) t.sym[c + k] = uint8_t(v);
        c += t.freq[v];
    }
}

// Decodes an rANS section: `ends.size()` tables of 512 bytes starting at `offset`, the offset table, then the streams,
// all of it within [offset, zone_end) of the payload `d`. `ends[k]` is the position (exclusive) of the last symbol of
// segment k; the last one equals the number of symbols. Returns the symbols in `out`.
void decode_rans(const std::vector<uint8_t>& d, uint64_t offset, uint64_t zone_end, const std::vector<uint64_t>& ends,
                 std::vector<uint8_t>& out, int frame, const char* what) {
    uint64_t total = ends.back();
    uint64_t n_tables = ends.size();
    uint64_t n_blocks = (total + kBlock - 1) / kBlock;
    if (zone_end > d.size() || offset > zone_end || 512 * n_tables + 4 * n_blocks > zone_end - offset)
        fail("frame %d: the tables and the offsets (%s) leave the frame", frame, what);
    std::vector<RansTable> tables(n_tables);
    for (uint64_t k = 0; k < n_tables; ++k) build_table(d.data() + offset + 512 * k, tables[k], frame, what, int(k));
    const uint8_t* offsets = d.data() + offset + 512 * n_tables;
    uint64_t zone = offset + 512 * n_tables + 4 * n_blocks;   // start of the stream area
    uint64_t zone_size = zone_end - zone;
    uint64_t previous = 0;
    for (uint64_t b = 0; b < n_blocks; ++b) {
        uint64_t o = read32(offsets + 4 * b);
        if (b == 0 ? o != 0 : o < previous + 4)
            fail("frame %d: offset of block %llu (%s) inconsistent", frame, (unsigned long long)b, what);
        previous = o;
    }
    if (n_blocks && (zone_size < 4 || previous > zone_size - 4))
        fail("frame %d: the last block (%s) starts outside the stream area", frame, what);

    out.assign(total, 0);
    const uint8_t* base = d.data();
    const uint64_t limit = d.size();   // a valid stream does not read beyond its area; beyond the frame, zeros are read
    in_parallel(n_blocks, [&](uint64_t b0, uint64_t b1) {
        for (uint64_t b = b0; b < b1; ++b) {
            uint64_t i = b * kBlock, end = std::min<uint64_t>(i + kBlock, total);
            uint64_t k = 0;
            while (ends[k] <= i) ++k;
            uint64_t p = zone + read32(offsets + 4 * b);
            uint32_t x = read32(base + p);
            p += 4;
            if (x < kLowerBound) x = kLowerBound;   // altered file only
            uint64_t n_read = 0;
            while (i < end) {
                const RansTable& t = tables[k];
                uint64_t segment_end = std::min<uint64_t>(end, ends[k]);
                for (; i < segment_end; ++i) {
                    uint32_t s = x & 4095;
                    uint8_t v = t.sym[s];
                    out[i] = v;
                    x = t.freq[v] * (x >> 12) + s - t.cum[v];
                    while (x < kLowerBound) {
                        x = (x << 8) | (p < limit ? base[p] : 0);
                        ++p;
                        if (++n_read > 2048) break;   // reading bound of a block: 2052 bytes, state included
                    }
                }
                ++k;
            }
        }
    });
}

// --- State of a frame ------------------------------------------------------------------------------------------------

struct Exception {
    uint32_t idx;
    int32_t val;
};

struct FineLevel {
    uint32_t n_bricks = 0, n_nodes = 0;
    std::vector<uint8_t> brick_map;   // bricks / 8 bytes
    std::vector<uint8_t> masks;       // 8 bytes per brick present
    std::vector<uint8_t> coefs;       // 3 bytes per retained node
};

struct State {
    std::vector<int16_t> q[6];          // q[l][3 * i + c], i = (z * nl + y) * nl + x
    std::vector<Exception> overflows;   // exceptions of the frame whose bit 31 is 1
    FineLevel fine;
    bool valid = false;
};

struct Counters {
    uint64_t raw_keys = 0, coded_keys = 0, innovations = 0, after_key = 0;
    uint64_t escapes = 0, overflows = 0, ignored_overflows = 0;
    uint64_t bricks = 0, nodes = 0;
};

// Exception list (section 6.3) at offset `p` of the payload. Returns the offset that follows it.
uint64_t read_exceptions(const std::vector<uint8_t>& d, uint64_t p, uint64_t zone_end, const Geometry& g, int frame,
                         std::vector<Exception>& list) {
    if (zone_end < 8 || p > zone_end - 8) fail("frame %d: the exception list leaves the frame", frame);
    uint64_t n_ex = read32(d.data() + p);
    if (n_ex > g.ex_cap) fail("frame %d: %llu exceptions, %llu at most", frame, (unsigned long long)n_ex, (unsigned long long)g.ex_cap);
    if (8 * n_ex > zone_end - p - 8) fail("frame %d: the exception list leaves the frame", frame);
    list.resize(n_ex);
    for (uint64_t k = 0; k < n_ex; ++k) {
        list[k].idx = read32(d.data() + p + 8 + 8 * k);
        list[k].val = int32_t(read32(d.data() + p + 12 + 8 * k));
    }
    return p + 8 + 8 * n_ex;
}

// Stores the overflows of the list in the state. Returns the number of escapes: the first entries whose bit 31 is
// zero.
uint64_t sort_exceptions(const std::vector<Exception>& list, const Geometry& g, State& state, Counters& c) {
    uint64_t escapes = 0;
    while (escapes < list.size() && !(list[escapes].idx & kOverflow)) ++escapes;
    state.overflows.clear();
    for (uint64_t k = escapes; k < list.size(); ++k) {
        if (!(list[k].idx & kOverflow)) continue;   // an escape after an overflow: outside the order of the format
        uint64_t position = list[k].idx & ~kOverflow;
        if (position < g.seg[0] || position >= g.ecount) {   // not a coefficient of level 0: ignored
            ++c.ignored_overflows;
            continue;
        }
        state.overflows.push_back({uint32_t(position), list[k].val});
    }
    c.overflows += state.overflows.size();
    return escapes;
}

inline int16_t clamp_to(int64_t v, int64_t bound) { return int16_t(v < -bound ? -bound : (v > bound ? bound : v)); }

// Raw key (section 5, type 0).
void decode_raw_key(const std::vector<uint8_t>& d, uint64_t base_end, const PsiFile& f, int frame, State& state, Counters& c) {
    const Geometry& g = f.g;
    uint64_t expected = g.key_bytes;
    std::vector<Exception> list;
    if (base_end < g.key_bytes) fail("frame %d: raw key of %llu bytes, at least %llu expected", frame,
                                    (unsigned long long)base_end, (unsigned long long)g.key_bytes);
    if (f.e.version >= 2) expected = read_exceptions(d, g.key_bytes, base_end, g, frame, list);
    if (f.e.version == 3) expected = (expected + 7) & ~7ull;
    if (expected != base_end)
        fail("frame %d: raw key of %llu bytes, %llu expected", frame, (unsigned long long)base_end, (unsigned long long)expected);
    const uint8_t* p = d.data();
    for (uint64_t i = 0; i < 3 * g.nodes[0]; ++i) state.q[0][i] = int8_t(p[i]);
    p += 3 * g.nodes[0];
    for (int l = 1; l < g.L; ++l) {
        for (uint64_t i = 0; i < 3 * g.nodes[l]; ++i) state.q[l][i] = int16_t(read16(p + 2 * i));
        p += 6 * g.nodes[l];
    }
    sort_exceptions(list, g, state, c);   // a key carries only overflows (6.3)
    ++c.raw_keys;
}

// Cumulative sums along x, then y, then z (6.4), modulo 2^16; `mask` brings level 0 back to 8 bits.
void cumulative_sums(std::vector<uint16_t>& r, uint64_t nl) {
    in_parallel(nl, [&](uint64_t z0, uint64_t z1) {
        for (uint64_t z = z0; z < z1; ++z) {
            uint16_t* plane = r.data() + 3 * z * nl * nl;
            for (uint64_t y = 0; y < nl; ++y) {
                uint16_t* row = plane + 3 * y * nl;
                for (uint64_t i = 3; i < 3 * nl; ++i) row[i] = uint16_t(row[i] + row[i - 3]);
            }
            for (uint64_t y = 1; y < nl; ++y) {
                uint16_t* row = plane + 3 * y * nl;
                for (uint64_t i = 0; i < 3 * nl; ++i) row[i] = uint16_t(row[i] + row[i - 3 * nl]);
            }
        }
    });
    uint64_t step = 3 * nl * nl;
    in_parallel(step, [&](uint64_t i0, uint64_t i1) {
        for (uint64_t z = 1; z < nl; ++z) {
            uint16_t* plane = r.data() + z * step;
            for (uint64_t i = i0; i < i1; ++i) plane[i] = uint16_t(plane[i] + plane[i - step]);
        }
    });
}

// Coded key (section 5, type 3; section 6.4).
void decode_coded_key(const std::vector<uint8_t>& d, uint64_t base_end, const PsiFile& f, int frame, State& state, Counters& c) {
    const Geometry& g = f.g;
    std::vector<Exception> list;
    uint64_t p = read_exceptions(d, 0, base_end, g, frame, list);
    // 2L - 1 segments: for l from L-1 to 1, high-order bytes then low-order bytes of level l; lastly level 0.
    std::vector<uint64_t> ends;
    std::vector<uint64_t> starts;
    uint64_t position = 0;
    for (int l = g.L - 1; l >= 1; --l)
        for (int half = 0; half < 2; ++half) {
            starts.push_back(position);
            position += 3 * g.nodes[l];
            ends.push_back(position);
        }
    starts.push_back(position);
    position += 3 * g.nodes[0];
    ends.push_back(position);
    std::vector<uint8_t> sym;
    decode_rans(d, p, base_end, ends, sym, frame, "coded key");

    std::vector<uint16_t> r;
    uint64_t k = 0;
    for (int l = g.L - 1; l >= 1; --l, k += 2) {
        uint64_t m = 3 * g.nodes[l];
        const uint8_t* high = sym.data() + starts[k];
        const uint8_t* low = sym.data() + starts[k + 1];
        r.resize(m);
        for (uint64_t i = 0; i < m; ++i) {
            uint16_t folded = uint16_t((high[i] << 8) | low[i]);
            r[i] = uint16_t((folded >> 1) ^ uint16_t(-(folded & 1)));   // 0, 1, 2, 3 become 0, -1, 1, -2 again
        }
        cumulative_sums(r, g.nl[l]);
        for (uint64_t i = 0; i < m; ++i) state.q[l][i] = int16_t(r[i]);
    }
    uint64_t m = 3 * g.nodes[0];
    const uint8_t* s0 = sym.data() + starts[k];
    r.resize(m);
    for (uint64_t i = 0; i < m; ++i) r[i] = s0[i];
    cumulative_sums(r, g.nl[0]);
    for (uint64_t i = 0; i < m; ++i) state.q[0][i] = int8_t(uint8_t(r[i]));   // modulo 2^8
    sort_exceptions(list, g, state, c);
    ++c.coded_keys;
}

// Innovation (section 5, types 1 and 2; section 6.2). `q1` is the previous frame, `q2` the one before it.
void decode_innovation(const std::vector<uint8_t>& d, uint64_t base_end, const PsiFile& f, int frame, int type,
                       const State& q1, const State& q2, State& state, Counters& c) {
    const Geometry& g = f.g;
    std::vector<Exception> list;
    uint64_t p = 0;
    if (f.e.version >= 2) p = read_exceptions(d, 0, base_end, g, frame, list);
    std::vector<uint64_t> ends;   // one table per level, from the coarsest to the finest
    for (int l = g.L - 1; l >= 0; --l) ends.push_back(g.seg[l] + 3 * g.nodes[l]);
    std::vector<uint8_t> sym;
    decode_rans(d, p, base_end, ends, sym, frame, "innovations");

    for (int l = 0; l < g.L; ++l) {
        const int64_t a = type == 2 ? 1024 : f.e.coef[l][0];
        const int64_t b = type == 2 ? 0 : f.e.coef[l][1];
        const int64_t cl = type == 2 ? 0 : f.e.coef[l][2];
        const int64_t bound = l == 0 ? 127 : (f.e.version == 1 ? (32767 >> f.e.fb) : 32767);
        const uint64_t nl = g.nl[l];
        const int16_t* p1 = q1.q[l].data();
        const int16_t* p2 = q2.q[l].data();
        int16_t* out = state.q[l].data();
        const uint8_t* s = sym.data() + g.seg[l];
        in_parallel(nl, [&](uint64_t z0, uint64_t z1) {
            for (uint64_t z = z0; z < z1; ++z) {
                uint64_t zm = (z + nl - 1) % nl, zp = (z + 1) % nl;
                for (uint64_t y = 0; y < nl; ++y) {
                    uint64_t ym = (y + nl - 1) % nl, yp = (y + 1) % nl;
                    for (uint64_t x = 0; x < nl; ++x) {
                        uint64_t xm = (x + nl - 1) % nl, xp = (x + 1) % nl;
                        uint64_t i = 3 * ((z * nl + y) * nl + x);
                        uint64_t ixm = 3 * ((z * nl + y) * nl + xm), ixp = 3 * ((z * nl + y) * nl + xp);
                        uint64_t iym = 3 * ((z * nl + ym) * nl + x), iyp = 3 * ((z * nl + yp) * nl + x);
                        uint64_t izm = 3 * ((zm * nl + y) * nl + x), izp = 3 * ((zp * nl + y) * nl + x);
                        for (int k = 0; k < 3; ++k) {
                            int64_t centre = p1[i + k];
                            int64_t prediction = a * centre + b * p2[i + k] + 512;
                            if (cl) {
                                int64_t lap = int64_t(p1[ixm + k]) + p1[ixp + k] + p1[iym + k] + p1[iyp + k] + p1[izm + k] + p1[izp + k] - 6 * centre;
                                prediction += cl * lap;
                            }
                            prediction >>= 10;   // arithmetic shift: rounds towards minus infinity
                            out[i + k] = clamp_to(prediction + int8_t(s[i + k]), bound);
                        }
                    }
                }
            }
        });
    }
    // Escapes: the state is set as it is.
    uint64_t escapes = sort_exceptions(list, g, state, c);
    for (uint64_t k = 0; k < escapes; ++k) {
        uint64_t position = list[k].idx;
        if (position >= g.ecount) fail("frame %d: escape at position %llu, outside the %llu coefficients", frame,
                                       (unsigned long long)position, (unsigned long long)g.ecount);
        int l = g.L - 1;
        while (l > 0 && position >= g.seg[l] + 3 * g.nodes[l]) --l;
        state.q[l][position - g.seg[l]] = int16_t(list[k].val);
    }
    c.escapes += escapes;
    if (type == 2) ++c.after_key;
    else ++c.innovations;
}

// Section of the fine level (sections 5 and 6.6). Returns the offset of its start in the payload.
uint64_t decode_fine_level(const std::vector<uint8_t>& d, const PsiFile& f, int frame, FineLevel& fine, Counters& c) {
    const Geometry& g = f.g;
    uint64_t size = d.size();
    if (size < 16 || (size & 7)) fail("frame %d: payload of %llu bytes, a multiple of 8 of at least 16 expected", frame, (unsigned long long)size);
    const uint8_t* trailer = d.data() + size - 16;
    fine.n_bricks = read32(trailer);
    fine.n_nodes = read32(trailer + 4);
    uint64_t bytes = read32(trailer + 8);
    if (fine.n_bricks > f.fine.max_bricks || fine.n_nodes > f.fine.max_nodes)
        fail("frame %d: %u bricks and %u nodes, beyond the maxima of the header (%u and %u)", frame, fine.n_bricks, fine.n_nodes,
             f.fine.max_bricks, f.fine.max_nodes);
    if ((bytes & 7) || bytes > size - 16) fail("frame %d: section of the fine level of %llu bytes, outside the frame", frame, (unsigned long long)bytes);
    uint64_t start = size - 16 - bytes;
    std::vector<uint64_t> ends = {g.bricks / 8, g.bricks / 8 + 8ull * fine.n_bricks, g.bricks / 8 + 8ull * fine.n_bricks + 3ull * fine.n_nodes};
    std::vector<uint8_t> sym;
    decode_rans(d, start, size - 16, ends, sym, frame, "fine level");
    fine.brick_map.assign(sym.begin(), sym.begin() + std::ptrdiff_t(ends[0]));
    fine.masks.assign(sym.begin() + std::ptrdiff_t(ends[0]), sym.begin() + std::ptrdiff_t(ends[1]));
    fine.coefs.assign(sym.begin() + std::ptrdiff_t(ends[1]), sym.end());
    c.bricks += fine.n_bricks;
    c.nodes += fine.n_nodes;
    return start;
}

// Decodes frame `j` into `state`, from the two previous states.
void decode_frame(PsiFile& f, int j, const State& q1, const State& q2, State& state, std::vector<uint8_t>& d, Counters& c) {
    const FrameRecord& r = f.table[size_t(j)];
    d.resize(r.bytes);
    read_bytes(f, r.offset, d.data(), r.bytes, "payload");
    for (int l = 0; l < f.g.L; ++l) state.q[l].resize(3 * f.g.nodes[l]);
    uint64_t base_end = r.bytes;
    state.fine = FineLevel();
    if (f.e.version == 3) base_end = decode_fine_level(d, f, j, state.fine, c);
    if (r.type == 0) decode_raw_key(d, base_end, f, j, state, c);
    else if (r.type == 3) decode_coded_key(d, base_end, f, j, state, c);
    else {
        if (!q1.valid) fail("frame %d: innovation without a previous frame", j);
        decode_innovation(d, base_end, f, j, r.type, q1, q2.valid ? q2 : q1, state, c);
    }
    state.valid = true;
}

// Fingerprint of the state of a frame (section 7).
uint64_t state_fingerprint(const PsiFile& f, const State& state) {
    const Geometry& g = f.g;
    std::vector<uint8_t> bytes(3 * g.nodes[0]);
    for (uint64_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(state.q[0][i]);
    uint64_t h = fingerprint(bytes.data(), bytes.size(), 0x1000);
    for (int l = 1; l < g.L; ++l) {
        bytes.resize(6 * g.nodes[l]);
        for (uint64_t i = 0; i < 3 * g.nodes[l]; ++i) {
            uint16_t v = uint16_t(state.q[l][i]);
            bytes[2 * i] = uint8_t(v);
            bytes[2 * i + 1] = uint8_t(v >> 8);
        }
        h ^= fingerprint(bytes.data(), bytes.size(), 0x1000 + (uint64_t(l) << 56));
    }
    if (f.e.version == 3) {
        h ^= fingerprint(state.fine.brick_map.data(), state.fine.brick_map.size(), 0x2000);
        h ^= fingerprint(state.fine.masks.data(), state.fine.masks.size(), 0x2000 + (1ull << 56));
        h ^= fingerprint(state.fine.coefs.data(), state.fine.coefs.size(), 0x2000 + (2ull << 56));
    }
    return h;
}

void write_state(const PsiFile& f, const State& state, const std::string& path) {
    std::ofstream s(path, std::ios::binary);
    if (!s) fail("cannot write %s", path.c_str());
    std::vector<uint8_t> bytes(3 * f.g.nodes[0]);
    for (uint64_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(state.q[0][i]);
    s.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    for (int l = 1; l < f.g.L; ++l) {
        bytes.resize(6 * f.g.nodes[l]);
        for (uint64_t i = 0; i < 3 * f.g.nodes[l]; ++i) {
            uint16_t v = uint16_t(state.q[l][i]);
            bytes[2 * i] = uint8_t(v);
            bytes[2 * i + 1] = uint8_t(v >> 8);
        }
        s.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    }
}

// --- Sampling lattice (section 6.5) ----------------------------------------------------------------------------------

inline uint64_t modulo(int64_t i, uint64_t n) {
    int64_t r = i % int64_t(n);
    return uint64_t(r < 0 ? r + int64_t(n) : r);
}

// Refines the sum `coarse` (ng nodes per axis, 32 bits) to a lattice twice as fine.
// OUTSIDE THE SPECIFICATION: section 6.5 says "integer masks of the B-splines" without giving them. Established by
// trial against the lattice recorded by `pack --probe`: two-scale relation of the B-spline, per axis
//   degree 2: fine[2i] = 3 g[i] + g[i-1], fine[2i+1] = 3 g[i] + g[i+1]             (sum of the weights: 4)
//   degree 3: fine[2i] = g[i-1] + 6 g[i] + g[i+1], fine[2i+1] = 4 g[i] + 4 g[i+1]  (sum of the weights: 8)
// as a tensor product over the three axes, with no intermediate division, then a single division by 64 (degree 2) or
// 512 (degree 3), rounded to nearest: (sum + half) >> 6 or 9. The indices are periodic.
void refine(const std::vector<int32_t>& coarse, uint64_t ng, int degree, std::vector<int32_t>& fine) {
    const uint64_t nf = 2 * ng;
    fine.assign(3 * nf * nf * nf, 0);
    const int bits = degree == 2 ? 6 : 9;
    const int64_t half = int64_t(1) << (bits - 1);
    // Per parity: offsets of the coarse nodes and weights.
    int n_taps[2], off[2][3], weights[2][3];
    if (degree == 2) {
        n_taps[0] = 2; off[0][0] = -1; weights[0][0] = 1; off[0][1] = 0; weights[0][1] = 3;
        n_taps[1] = 2; off[1][0] = 0; weights[1][0] = 3; off[1][1] = 1; weights[1][1] = 1;
    } else {
        n_taps[0] = 3; off[0][0] = -1; weights[0][0] = 1; off[0][1] = 0; weights[0][1] = 6; off[0][2] = 1; weights[0][2] = 1;
        n_taps[1] = 2; off[1][0] = 0; weights[1][0] = 4; off[1][1] = 1; weights[1][1] = 4;
    }
    in_parallel(nf, [&](uint64_t z0, uint64_t z1) {
        for (uint64_t Z = z0; Z < z1; ++Z) {
            const int pz = int(Z & 1);
            for (uint64_t Y = 0; Y < nf; ++Y) {
                const int py = int(Y & 1);
                for (uint64_t X = 0; X < nf; ++X) {
                    const int px = int(X & 1);
                    int64_t s[3] = {0, 0, 0};
                    for (int a = 0; a < n_taps[pz]; ++a) {
                        uint64_t gz = modulo(int64_t(Z >> 1) + off[pz][a], ng);
                        for (int b = 0; b < n_taps[py]; ++b) {
                            uint64_t gy = modulo(int64_t(Y >> 1) + off[py][b], ng);
                            int64_t wzy = int64_t(weights[pz][a]) * weights[py][b];
                            for (int c = 0; c < n_taps[px]; ++c) {
                                uint64_t gx = modulo(int64_t(X >> 1) + off[px][c], ng);
                                const int32_t* v = coarse.data() + 3 * ((gz * ng + gy) * ng + gx);
                                int64_t w = wzy * weights[px][c];
                                s[0] += w * v[0];
                                s[1] += w * v[1];
                                s[2] += w * v[2];
                            }
                        }
                    }
                    int32_t* out = fine.data() + 3 * ((Z * nf + Y) * nf + X);
                    for (int k = 0; k < 3; ++k) out[k] = int32_t((s[k] + half) >> bits);
                }
            }
        }
    });
}

// Velocity field of a frame: sampling lattice, fine level, scales.
struct Field {
    int n = 0, degree = 0;
    uint64_t nl0 = 0;
    bool direct = true;             // direct reading of the lattice, or reading by differences (6.5)
    std::vector<int16_t> lattice;   // sum of the levels modulo 2^16, 3 per node
    uint64_t out_of_range = 0;      // values of the sum outside the range of a 16-bit integer
    double scale = 0, fine_scale = 0, mean[3] = {0, 0, 0};
    bool has_fine_level = false;
    uint64_t nb = 0;
    std::vector<int32_t> brick_rank;   // position of the brick among those present, or -1
    std::vector<int8_t> blocks;        // 192 bytes per brick present: 3 coefficients for each of the 64 nodes
    uint64_t fine_nodes = 0;
};

void build_field(const PsiFile& f, const State& state, const FrameRecord& r, Field& field) {
    const Geometry& g = f.g;
    const Header& e = f.e;
    field.n = e.n;
    field.degree = e.degree;
    field.nl0 = g.nl[0];
    // Direct reading if the file is PSISTRM1, or if the marker is there and no frame is wrapped.
    field.direct = e.version == 1 || (e.reserved[0] == kClipMarker && (e.reserved[7] >> 32) == 0);

    // sum = (state << fb) + refinement of the sum of the coarser level, from the coarsest level to level 0.
    std::vector<int32_t> sum(3 * g.nodes[g.L - 1]), next;
    for (uint64_t i = 0; i < sum.size(); ++i) sum[i] = int32_t(state.q[g.L - 1][i]) * (1 << e.fb);
    for (int l = g.L - 2; l >= 0; --l) {
        refine(sum, g.nl[l + 1], e.degree, next);
        const int16_t* q = state.q[l].data();
        for (uint64_t i = 0; i < next.size(); ++i) next[i] += int32_t(q[i]) * (1 << e.fb);
        sum.swap(next);
    }
    // Overflows of level 0: val << fb, added after the synthesis (6.3).
    for (const Exception& x : state.overflows) sum[x.idx - g.seg[0]] += x.val * (1 << e.fb);
    field.lattice.resize(sum.size());
    field.out_of_range = 0;
    for (uint64_t i = 0; i < sum.size(); ++i) {
        if (sum[i] < -32768 || sum[i] > 32767) ++field.out_of_range;
        field.lattice[i] = int16_t(uint16_t(uint32_t(sum[i])));   // modulo 2^16
    }

    // OUTSIDE THE SPECIFICATION, resolved by deduction: section 2 gives delta / (2^fb * h0), h0 = 2 pi / nl[0]; the
    // order of the single-precision operations of the GPU reader is not described, the computation is done here in
    // double precision.
    const double two_pi = 6.283185307179586476925286766559;
    field.scale = double(e.delta) / (double(1 << e.fb) * (two_pi / double(g.nl[0])));
    for (int k = 0; k < 3; ++k) field.mean[k] = double(r.mean[k]);

    // Sparse fine level (6.6).
    field.has_fine_level = e.version == 3;
    field.fine_nodes = 0;
    if (field.has_fine_level) {
        field.nb = g.nb;
        field.fine_scale = double(f.fine.delta) * double(e.n) / two_pi;
        field.brick_rank.assign(g.bricks, -1);
        field.blocks.assign(192ull * state.fine.n_bricks, 0);
        uint64_t brick = 0, node = 0;
        for (uint64_t b = 0; b < g.bricks && brick < state.fine.n_bricks; ++b) {
            if (!((state.fine.brick_map[b >> 3] >> (b & 7)) & 1)) continue;
            uint64_t mask = read64(state.fine.masks.data() + 8 * brick);
            int8_t* block = field.blocks.data() + 192 * brick;
            for (int k = 0; k < 64 && node < state.fine.n_nodes; ++k) {
                if (!((mask >> k) & 1)) continue;
                for (int c = 0; c < 3; ++c) block[3 * k + c] = int8_t(state.fine.coefs[3 * node + c]);
                ++node;
            }
            field.brick_rank[b] = int32_t(brick);
            ++brick;
        }
        field.fine_nodes = node;
    }
}

// Coefficient of the fine level at node (x, y, z) of the grid of step one voxel; zero if it is not retained.
inline const int8_t* fine_node(const Field& field, uint64_t x, uint64_t y, uint64_t z) {
    static const int8_t zero[3] = {0, 0, 0};
    int32_t brick = field.brick_rank[((z >> 2) * field.nb + (y >> 2)) * field.nb + (x >> 2)];
    if (brick < 0) return zero;
    return field.blocks.data() + 192ull * uint64_t(brick) + 3 * (((z & 3) * 4 + (y & 3)) * 4 + (x & 3));
}

// Weights of the B-spline and of its derivative for the coordinate u, counted in steps of the lattice from node 0.
// Returns the index of the first node of the support.
// OUTSIDE THE SPECIFICATION, established by trial against the field recorded by `pack --probe`: cardinal B-spline
// centred on its node; node i is at (i + 1/2) steps in degree 2 and at i steps in degree 3, for the lattice as for the
// fine level (section 6.6 says so only for the fine level).
inline int64_t spline_weights(double u, int degree, double* w, double* d) {
    if (degree == 2) {
        double m = std::floor(u + 0.5), r = u - m;
        w[0] = 0.5 * (0.5 - r) * (0.5 - r);
        w[1] = 0.75 - r * r;
        w[2] = 0.5 * (0.5 + r) * (0.5 + r);
        d[0] = -(0.5 - r);
        d[1] = -2.0 * r;
        d[2] = 0.5 + r;
        return int64_t(m) - 1;
    }
    double i = std::floor(u), t = u - i;
    w[0] = (1 - t) * (1 - t) * (1 - t) / 6.0;
    w[1] = (3 * t * t * t - 6 * t * t + 4) / 6.0;
    w[2] = (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6.0;
    w[3] = t * t * t / 6.0;
    d[0] = -(1 - t) * (1 - t) / 2.0;
    d[1] = (3 * t * t - 4 * t) / 2.0;
    d[2] = (-3 * t * t + 2 * t + 1) / 2.0;
    d[3] = t * t / 2.0;
    return int64_t(i) - 1;
}

// Velocity at point (x, y, z) of the periodic box of side 2 pi: mean + scale x curl of the potential.
// OUTSIDE THE SPECIFICATION, resolved by deduction then by trial: u = (dPsi_z/dy - dPsi_y/dz, dPsi_x/dz - dPsi_z/dx,
// dPsi_y/dx - dPsi_x/dy), the derivatives being taken with respect to the coordinate counted in steps of the lattice.
void velocity(const Field& field, const double* pos, double* u) {
    const double two_pi = 6.283185307179586476925286766559;
    const int m = field.degree + 1;
    double g[3][3] = {};   // g[c][axis] = dPsi_c / d axis, in units of the lattice
    auto accumulate = [&](double step, uint64_t N, bool fine, double scale) {
        const double shift = field.degree == 2 ? 0.5 : 0.0;
        double w[3][4], d[3][4];
        int64_t i0[3];
        for (int a = 0; a < 3; ++a) i0[a] = spline_weights(pos[a] / step - shift, field.degree, w[a], d[a]);
        int16_t ref[3] = {0, 0, 0};
        bool first = true;
        double s[3][3] = {};
        for (int a = 0; a < m; ++a) {
            uint64_t z = modulo(i0[2] + a, N);
            for (int b = 0; b < m; ++b) {
                uint64_t y = modulo(i0[1] + b, N);
                for (int c = 0; c < m; ++c) {
                    uint64_t x = modulo(i0[0] + c, N);
                    double v[3];
                    if (fine) {
                        const int8_t* p = fine_node(field, x, y, z);
                        for (int k = 0; k < 3; ++k) v[k] = double(p[k]);
                    } else {
                        const int16_t* p = field.lattice.data() + 3 * ((z * N + y) * N + x);
                        if (field.direct) {
                            for (int k = 0; k < 3; ++k) v[k] = double(p[k]);
                        } else {   // by differences modulo 2^16 with the first node of the support
                            if (first) {
                                for (int k = 0; k < 3; ++k) ref[k] = p[k];
                                first = false;
                            }
                            for (int k = 0; k < 3; ++k) v[k] = double(int16_t(uint16_t(uint16_t(p[k]) - uint16_t(ref[k]))));
                        }
                    }
                    double wx = d[0][c] * w[1][b] * w[2][a], wy = w[0][c] * d[1][b] * w[2][a], wz = w[0][c] * w[1][b] * d[2][a];
                    for (int k = 0; k < 3; ++k) {
                        s[k][0] += v[k] * wx;
                        s[k][1] += v[k] * wy;
                        s[k][2] += v[k] * wz;
                    }
                }
            }
        }
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) g[k][a] += scale * s[k][a];
    };
    accumulate(two_pi / double(field.nl0), field.nl0, false, field.scale);
    if (field.has_fine_level) accumulate(two_pi / double(field.n), uint64_t(field.n), true, field.fine_scale);
    u[0] = field.mean[0] + g[2][1] - g[1][2];
    u[1] = field.mean[1] + g[0][2] - g[2][0];
    u[2] = field.mean[2] + g[1][0] - g[0][1];
}

// --- Probe written by `psi_stream pack --probe` ----------------------------------------------------------------------
//
// OUTSIDE THE SPECIFICATION: docs/FORMAT.md does not describe this file. Layout established as a black box, from the
// size and the content of probe files at 64^3, 256^3 and 512^3 (little-endian):
//    0  char[8]   "PSIPROBE"
//    8  uint32    n; 12 degree; 16 nl[0]; 20 probed frame, counted in the source sequence (first_frame + position)
//   24  uint32    number of points P; 28 number of base points (P / 7); 32 fine level present (0 or 1); 36 zero
//   40  float32   scale of the lattice delta / (2^fb * h0); 44 scale of the fine level; 48 one eighth of a voxel
//   52  float32   u'; 56 marking threshold in velocity units (zero without a fine level); 60 mean velocity (3 values)
//   72  int16     sampling lattice of the encoder: 3 * nl[0]^3 values, in the order of section 1
//       int8      if the fine level is present: 3 * n^3 coefficients of the fine level, dense, same order
//       float32   positions: P x coordinates, then P y coordinates, then P z coordinates
//       float32   velocities of the encoder: P x components, then P y components, then P z components
// Each base point is followed by its six neighbours at +-1/8 of a voxel along x, y, then z.
struct Probe {
    uint32_t n = 0, degree = 0, nl0 = 0, frame = 0, points = 0, bases = 0, fine = 0;
    float scale = 0, fine_scale = 0, step = 0, u_rms = 0, threshold = 0, mean[3] = {};
    std::vector<uint8_t> bytes;
    uint64_t lattice_offset = 0, fine_offset = 0, positions_offset = 0, velocities_offset = 0;
};

void read_probe(const std::string& path, Probe& s) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) fail("cannot open the probe %s", path.c_str());
    stream.seekg(0, std::ios::end);
    uint64_t size = uint64_t(stream.tellg());
    if (size < 72) fail("probe too short (%llu bytes)", (unsigned long long)size);
    s.bytes.resize(size);
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(s.bytes.data()), std::streamsize(size));
    if (uint64_t(stream.gcount()) != size) fail("incomplete read of the probe");
    const uint8_t* p = s.bytes.data();
    if (std::memcmp(p, "PSIPROBE", 8) != 0) fail("%s is not a probe file", path.c_str());
    s.n = read32(p + 8);
    s.degree = read32(p + 12);
    s.nl0 = read32(p + 16);
    s.frame = read32(p + 20);
    s.points = read32(p + 24);
    s.bases = read32(p + 28);
    s.fine = read32(p + 32);
    s.scale = read_f32(p + 40);
    s.fine_scale = read_f32(p + 44);
    s.step = read_f32(p + 48);
    s.u_rms = read_f32(p + 52);
    s.threshold = read_f32(p + 56);
    for (int k = 0; k < 3; ++k) s.mean[k] = read_f32(p + 60 + 4 * k);
    if (s.n > 8192 || s.nl0 > s.n || s.fine > 1) fail("inconsistent probe header");
    s.lattice_offset = 72;
    s.fine_offset = s.lattice_offset + 6ull * s.nl0 * s.nl0 * s.nl0;
    s.positions_offset = s.fine_offset + (s.fine ? 3ull * s.n * s.n * s.n : 0);
    s.velocities_offset = s.positions_offset + 12ull * s.points;
    if (s.velocities_offset + 12ull * s.points != size)
        fail("probe of %llu bytes, %llu expected from its header", (unsigned long long)size,
             (unsigned long long)(s.velocities_offset + 12ull * s.points));
}

struct ProbeReport {
    uint64_t lattice_nodes = 0, fine_nodes = 0;                // nodes that differ from those of the encoder
    double max_difference = 0, rms_difference = 0;             // in velocity units
    double scale_difference = 0, fine_scale_difference = 0;    // relative difference of the scales with those of the probe
};

void compare_probe(const PsiFile& f, const Field& field, const Probe& s, ProbeReport& b) {
    if (int(s.n) != f.e.n || int(s.degree) != f.e.degree || s.nl0 != f.g.nl[0])
        fail("the probe (%u^3, degree %u, %u nodes per axis) is not that of this file", s.n, s.degree, s.nl0);
    if (bool(s.fine) != field.has_fine_level) fail("the probe and the file do not agree on the presence of the fine level");
    const uint8_t* p = s.bytes.data();
    uint64_t nodes = f.g.nodes[0];
    std::vector<uint64_t> partials(nodes / 4096 + 1, 0);
    in_parallel(partials.size(), [&](uint64_t t0, uint64_t t1) {
        for (uint64_t t = t0; t < t1; ++t) {
            uint64_t different = 0;
            for (uint64_t i = t * 4096, end = std::min<uint64_t>(nodes, i + 4096); i < end; ++i) {
                bool equal = true;
                for (int k = 0; k < 3; ++k) equal = equal && int16_t(read16(p + s.lattice_offset + 2 * (3 * i + k))) == field.lattice[3 * i + k];
                different += !equal;
            }
            partials[t] = different;
        }
    });
    for (uint64_t d : partials) b.lattice_nodes += d;
    if (s.fine) {
        const uint64_t n = s.n;
        std::vector<uint64_t> per_plane(n, 0);
        in_parallel(n, [&](uint64_t z0, uint64_t z1) {
            for (uint64_t z = z0; z < z1; ++z) {
                uint64_t different = 0;
                for (uint64_t y = 0; y < n; ++y)
                    for (uint64_t x = 0; x < n; ++x) {
                        const int8_t* mine = fine_node(field, x, y, z);
                        const uint8_t* theirs = p + s.fine_offset + 3 * ((z * n + y) * n + x);
                        different += int8_t(theirs[0]) != mine[0] || int8_t(theirs[1]) != mine[1] || int8_t(theirs[2]) != mine[2];
                    }
                per_plane[z] = different;
            }
        });
        for (uint64_t d : per_plane) b.fine_nodes += d;
    }
    b.scale_difference = std::fabs(field.scale - double(s.scale)) / std::fabs(double(s.scale));
    if (s.fine) b.fine_scale_difference = std::fabs(field.fine_scale - double(s.fine_scale)) / std::fabs(double(s.fine_scale));
    const uint64_t P = s.points;
    std::vector<double> maxima(P / 1024 + 1, 0.0), squares(P / 1024 + 1, 0.0);
    in_parallel(maxima.size(), [&](uint64_t t0, uint64_t t1) {
        for (uint64_t t = t0; t < t1; ++t) {
            double worst = 0, sum = 0;
            for (uint64_t i = t * 1024, end = std::min<uint64_t>(P, i + 1024); i < end; ++i) {
                double pos[3], u[3];
                for (uint64_t k = 0; k < 3; ++k) pos[k] = double(read_f32(p + s.positions_offset + 4 * (k * P + i)));
                velocity(field, pos, u);
                for (uint64_t k = 0; k < 3; ++k) {
                    double e = std::fabs(u[k] - double(read_f32(p + s.velocities_offset + 4 * (k * P + i))));
                    worst = std::max(worst, e);
                    sum += e * e;
                }
            }
            maxima[t] = worst;
            squares[t] = sum;
        }
    });
    double sum = 0;
    for (uint64_t t = 0; t < maxima.size(); ++t) {
        b.max_difference = std::max(b.max_difference, maxima[t]);
        sum += squares[t];
    }
    b.rms_difference = P ? std::sqrt(sum / double(3 * P)) : 0.0;
}

// Writes to the output the field of the frame at the positions of the text file `path` ("x y z" per line).
void write_points(const Field& field, const std::string& path) {
    std::ifstream stream(path);
    if (!stream) fail("cannot open %s", path.c_str());
    double pos[3], u[3];
    while (stream >> pos[0] >> pos[1] >> pos[2]) {
        velocity(field, pos, u);
        std::printf("POINT %.9g %.9g %.9g %.9g %.9g %.9g\n", pos[0], pos[1], pos[2], u[0], u[1], u[2]);
    }
}

}   // namespace

int main(int argc, char** argv) {
    std::string path, probe, state_out, points;
    int state_frame = -1, points_frame = -1, max_frames = -1;
    bool csv = false;
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    g_threads = int(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Failure: missing value after %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--probe") probe = value();
        else if (a == "--state") { state_frame = std::atoi(value()); state_out = value(); }
        else if (a == "--points") points = value();
        else if (a == "--frame") points_frame = std::atoi(value());
        else if (a == "--frames") max_frames = std::atoi(value());
        else if (a == "--threads") g_threads = std::max(1, std::atoi(value()));
        else if (a == "--csv") csv = true;
        else if (!a.empty() && a[0] != '-' && path.empty()) path = a;
        else {
            std::fprintf(stderr, "Usage: psi_ref <file.psi> [--probe f.probe] [--state j out.bin] [--points f.txt --frame j]\n"
                                 "               [--frames n] [--threads n] [--csv]\n");
            return 2;
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "Usage: psi_ref <file.psi> [--probe f.probe] [--state j out.bin] [--points f.txt --frame j]\n"
                             "               [--frames n] [--threads n] [--csv]\n");
        return 2;
    }

    try {
        PsiFile f;
        open_file(f, path);
        const Header& e = f.e;
        std::printf("File        : %s, PSISTRM%d, %d^3, degree %d, %d levels, %d frames (%d to %d), %s\n", path.c_str(), e.version, e.n,
                    e.degree, e.levels, e.n_frames, e.first_frame, e.first_frame + e.n_frames - 1, f.sealed ? "seal verified" : "no seal");
        int to_read = max_frames > 0 ? std::min(max_frames, e.n_frames) : e.n_frames;
        Probe s;
        int probe_position = -1;
        if (!probe.empty()) {
            read_probe(probe, s);
            probe_position = int(s.frame) - e.first_frame;
            if (probe_position < 0 || probe_position >= to_read)
                fail("the probe is for frame %u of the sequence, outside the frames read (%d to %d)", s.frame, e.first_frame,
                     e.first_frame + to_read - 1);
        }
        if (!points.empty() && (points_frame < 0 || points_frame >= to_read))
            fail("--points requires --frame j, with j from 0 to %d", to_read - 1);
        State states[3];
        std::vector<uint8_t> payload;
        Counters c;
        ProbeReport report;
        uint64_t probe_out_of_range = 0;
        bool direct = true;
        int differing = 0, first = -1;
        for (int j = 0; j < to_read; ++j) {
            State& state = states[j % 3];
            const State& q1 = states[(j + 2) % 3];
            const State& q2 = states[(j + 1) % 3];
            decode_frame(f, j, q1, q2, state, payload, c);
            uint64_t h = state_fingerprint(f, state);
            if (h != f.table[size_t(j)].hash) {
                if (first < 0) first = j;
                ++differing;
            }
            if (j == state_frame) write_state(f, state, state_out);
            if (j == probe_position || (!points.empty() && j == points_frame)) {
                Field field;
                build_field(f, state, f.table[size_t(j)], field);
                direct = field.direct;
                if (j == probe_position) {
                    compare_probe(f, field, s, report);
                    probe_out_of_range = field.out_of_range;
                }
                if (!points.empty() && j == points_frame) write_points(field, points);
            }
        }
        std::printf("Frames      : %d read out of %d; %llu coded key(s), %llu raw key(s), %llu innovation(s) after a key, %llu innovation(s)\n",
                    to_read, e.n_frames, (unsigned long long)c.coded_keys, (unsigned long long)c.raw_keys,
                    (unsigned long long)c.after_key, (unsigned long long)c.innovations);
        std::printf("Exceptions  : %llu escape(s), %llu overflow(s) of level 0, %llu overflow(s) ignored\n",
                    (unsigned long long)c.escapes, (unsigned long long)c.overflows, (unsigned long long)c.ignored_overflows);
        if (e.version == 3)
            std::printf("Fine level  : %llu brick(s) and %llu node(s) over the frames read\n", (unsigned long long)c.bricks, (unsigned long long)c.nodes);
        if (differing)
            std::printf("Fingerprints: %d frame(s) whose fingerprint differs from that of the table, the first at position %d\n", differing, first);
        else
            std::printf("Fingerprints: 0 frame(s) whose fingerprint differs from that of the table\n");
        if (probe_position >= 0) {
            const double u = double(e.u_rms);
            std::printf("Probe       : frame %u (position %d), %u points; lattice read %s, %llu value(s) outside the 16-bit range\n",
                        s.frame, probe_position, s.points, direct ? "directly" : "by differences", (unsigned long long)probe_out_of_range);
            std::printf("Lattice     : %llu node(s) out of %llu differ from those of the encoder; fine level: %llu node(s) out of %llu\n",
                        (unsigned long long)report.lattice_nodes, (unsigned long long)f.g.nodes[0], (unsigned long long)report.fine_nodes,
                        (unsigned long long)(s.fine ? uint64_t(e.n) * e.n * e.n : 0));
            std::printf("Velocity    : largest difference from the field of the probe %.3e u', RMS difference %.3e u' (u' = %.6f); scales to within %.1e and %.1e\n",
                        report.max_difference / u, report.rms_difference / u, u, report.scale_difference, report.fine_scale_difference);
        }
        if (csv) {
            // n, degree, version, frames read, frames of the file, seal, differing fingerprints, escapes, overflows
            std::printf("CSV,ref,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n", e.n, e.degree, e.version, to_read, e.n_frames, f.sealed ? 1 : 0, differing,
                        (unsigned long long)c.escapes, (unsigned long long)c.overflows);
            // frame, points, differing lattice nodes, differing fine nodes, largest and RMS difference in u', direct
            // reading
            if (probe_position >= 0)
                std::printf("CSV,probe,%u,%u,%llu,%llu,%.4e,%.4e,%d\n", s.frame, s.points, (unsigned long long)report.lattice_nodes,
                            (unsigned long long)report.fine_nodes, report.max_difference / double(e.u_rms), report.rms_difference / double(e.u_rms), direct ? 1 : 0);
        }
        return differing ? 1 : 0;
    } catch (const Failure& x) {
        std::fprintf(stderr, "Failure: %s\n", x.what());
        return 1;
    } catch (const std::exception& x) {
        std::fprintf(stderr, "Failure: %s\n", x.what());
        return 1;
    }
}
