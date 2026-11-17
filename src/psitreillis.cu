// SPDX-License-Identifier: Apache-2.0
// psitreillis.cu - the library: public C API (include/psitreillis.h) on top of the codec kernels of codec_stream.cuh.
//
// The codec sources are written as programs: on an error they print a message and exit, and they never release what
// they allocate on the GPU. This file turns them into a library without changing their behaviour:
//   - CK and CHK (codec_kernels.cuh) and sequence_fail (sequence.h) are redefined to throw; every API function catches
//     and returns PSIT_ERROR with a message (psit_last_error);
//   - every GPU allocation, stream, event, FFT plan and file opened by the codec is recorded in an arena, released
//     when a reader is closed or when a compression ends, whether it succeeded or not;
//   - the encoder's report on stdout is silenced unless psit_options::verbose is set.
// The standard, CUDA and Thrust headers are included first, so that the redefinitions below apply to the codec
// sources only.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// windows.h, pulled in by Thrust, defines `near` and `far` as empty macros; the codec uses `near` as a variable name.
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

#define PSIT_BUILD
#include "psitreillis.h"

// ---- Errors -----------------------------------------------------------------------------------------------------

namespace {
struct Failure {
    std::string msg;
};
thread_local std::string g_error;
[[noreturn]] void fail(const std::string& msg) { throw Failure{msg}; }
}  // namespace

// ---- Resources opened by the codec ------------------------------------------------------------------------------

namespace psit_track {
struct Arena {
    std::vector<void*> dev, host;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    std::vector<cufftHandle> plans;
    std::vector<FILE*> files;
    void release() {
        for (cufftHandle p : plans) ::cufftDestroy(p);
        for (cudaEvent_t e : events) ::cudaEventDestroy(e);
        for (cudaStream_t s : streams) ::cudaStreamDestroy(s);
        for (void* p : dev) ::cudaFree(p);
        for (void* p : host) ::cudaFreeHost(p);
        for (FILE* f : files) ::fclose(f);
        plans.clear();
        events.clear();
        streams.clear();
        dev.clear();
        host.clear();
        files.clear();
    }
};
thread_local Arena* current = nullptr;
struct Scope {
    Arena* previous;
    explicit Scope(Arena* a) : previous(current) { current = a; }
    ~Scope() { current = previous; }
};
template <class T>
inline void forget(std::vector<T>& v, T p) {
    for (size_t i = v.size(); i-- > 0;)
        if (v[i] == p) {
            v.erase(v.begin() + i);
            return;
        }
}
template <class T>
inline cudaError_t malloc_dev(T** p, size_t bytes) {
    const cudaError_t e = ::cudaMalloc((void**)p, bytes);
    if (e == cudaSuccess && current) current->dev.push_back((void*)*p);
    return e;
}
template <class T>
inline cudaError_t malloc_host(T** p, size_t bytes, unsigned flags) {
    const cudaError_t e = ::cudaHostAlloc((void**)p, bytes, flags);
    if (e == cudaSuccess && current) current->host.push_back((void*)*p);
    return e;
}
inline cudaError_t free_dev(void* p) {
    if (current) forget(current->dev, p);
    return ::cudaFree(p);
}
inline cudaError_t free_host(void* p) {
    if (current) forget(current->host, p);
    return ::cudaFreeHost(p);
}
inline cudaError_t stream_create(cudaStream_t* s) {
    const cudaError_t e = ::cudaStreamCreate(s);
    if (e == cudaSuccess && current) current->streams.push_back(*s);
    return e;
}
inline cudaError_t event_create(cudaEvent_t* ev, unsigned flags) {
    const cudaError_t e = ::cudaEventCreateWithFlags(ev, flags);
    if (e == cudaSuccess && current) current->events.push_back(*ev);
    return e;
}
inline cufftResult plan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
    const cufftResult r = ::cufftPlan3d(plan, nx, ny, nz, type);
    if (r == CUFFT_SUCCESS && current) current->plans.push_back(*plan);
    return r;
}
inline cufftResult plan_destroy(cufftHandle plan) {
    if (current) forget(current->plans, plan);
    return ::cufftDestroy(plan);
}
inline FILE* file_open(const char* path, const char* mode) {
    FILE* f = ::fopen(path, mode);
    if (f && current) current->files.push_back(f);
    return f;
}
inline int file_close(FILE* f) {
    if (current) forget(current->files, f);
    return ::fclose(f);
}
}  // namespace psit_track

#define CK(call)                                                                        \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) fail(std::string("CUDA: ") + cudaGetErrorString(e_));    \
    } while (0)
#define CHK(cond, msg)             \
    do {                           \
        if (!(cond)) fail(msg);    \
    } while (0)
#define PSI_SEQUENCE_FAIL(msg) fail(msg)

#define cudaMalloc(p, bytes) psit_track::malloc_dev((p), (bytes))
#define cudaHostAlloc(p, bytes, flags) psit_track::malloc_host((p), (bytes), (flags))
#define cudaFree(p) psit_track::free_dev((p))
#define cudaFreeHost(p) psit_track::free_host((p))
#define cudaStreamCreate(s) psit_track::stream_create((s))
#define cudaEventCreateWithFlags(e, flags) psit_track::event_create((e), (flags))
#define cufftPlan3d(plan, nx, ny, nz, type) psit_track::plan3d((plan), (nx), (ny), (nz), (type))
#define cufftDestroy(plan) psit_track::plan_destroy((plan))
#define fopen(path, mode) psit_track::file_open((path), (mode))
#define fclose(f) psit_track::file_close((f))

#include "codec_stream.cuh"

// ---- Kernels ----------------------------------------------------------------------------------------------------

// The decoded field at voxel centres, for `count` voxels starting at voxel `first`. The position of a voxel centre is
// written as the encoder writes it when it measures its error (field_err_bricks).
template <class F>
__global__ void psit_grid_kernel(F f, int n, size_t first, uint32_t count, float* ou, float* ov, float* ow) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const size_t v = first + i;
    const int ix = (int)(v % n), iy = (int)((v / n) % n), iz = (int)(v / ((size_t)n * n));
    const float h = kL / n;
    const V3 u = sample(f, V3{(ix + 0.5f) * h, (iy + 0.5f) * h, (iz + 0.5f) * h}, 0.0f);
    ou[i] = u.x;
    ov[i] = u.y;
    ow[i] = u.z;
}

// ---- Reader -----------------------------------------------------------------------------------------------------

struct psit_reader {
    psit_track::Arena arena;
    FILE* in = nullptr;
    StreamHeader sh{};
    StreamOpen so;
    int cur = -1;  // frame held by the decoder, -1 if none
    virtual ~psit_reader() {}
    virtual void init() = 0;
    virtual void grid(int frame, float* out) = 0;
    virtual void points(int frame, const float* x, const float* y, const float* z, size_t count, float* u, float* v, float* w) = 0;
};

namespace {

constexpr size_t kGridChunk = (size_t)1 << 25;   // voxels decoded per kernel launch
constexpr size_t kPointChunk = (size_t)1 << 22;  // points sampled per kernel launch

template <int D>
struct ReaderT final : psit_reader {
    Geo g;
    Receiver<D> rx;
    cudaStream_t stream = nullptr;
    unsigned char* h_buf = nullptr;
    bool fine = false, plain = false;
    float *d_out = nullptr, *d_pos = nullptr;
    size_t out_cap = 0, pos_cap = 0;

    // Same steps as `psi_stream play`.
    void init() override {
        g.build(sh.n, D, sh.mf, sh.levels, sh.fb);
        fine = so.has_fine;
        if (fine) {
            const size_t nb = (size_t)(sh.n >> kFineLog);
            g.cap += fine_section_cap(nb * nb * nb / 8, so.fi.max_bricks, so.fi.max_nodes);
        }
        plain = sh.magic[7] == '1' || (sh.reserved[0] == kClipMarker && (sh.reserved[7] >> 32) == 0);
        g.cap = packet_bytes(so.index, g.cap);
        CK(cudaStreamCreate(&stream));
        Coef coef[kMaxLevels];
        for (int l = 0; l < g.L; ++l) coef[l] = Coef{sh.coef[l][0], sh.coef[l][1], sh.coef[l][2]};
        rx.init(g, sh.delta, coef);
        rx.set_format(sh.magic[7], sh.fb);
        if (fine) rx.init_fine(so.fi);
        CK(cudaHostAlloc(&h_buf, g.cap, cudaHostAllocDefault));
    }

    // Reads and decodes frame j into slot j & 1, and checks the decoded state against the fingerprint of the file.
    void ingest(int j) {
        const FrameRecord& r = so.index[j];
        CHK(r.bytes <= g.cap, "frame larger than expected");
        seek64(in, r.offset);
        CHK(fread(h_buf, 1, r.bytes, in) == r.bytes, "unreadable frame");
        CK(cudaMemcpyAsync(rx.packet, h_buf, r.bytes, cudaMemcpyHostToDevice, stream));
        rx.inspect(r.type, h_buf, r.bytes);
        CK(cudaStreamSynchronize(stream));
        rx.decode(r.type, j & 1, r.mean, h_buf, r.bytes, stream);
        CK(cudaStreamSynchronize(stream));
        unsigned long long hv = state_hash(g, rx.q0, rx.q, j & 1, rx.d_hash);
        if (fine) hv ^= fine_hash(rx.fs, j & 1, rx.d_hash);
        if (hv != r.hash) fail("frame " + std::to_string(j) + ": the decoded state differs from the encoder's (altered file)");
    }

    // Brings the decoder to frame j: the next frame, a later frame of the same group, or from the previous key frame.
    void ensure(int j) {
        if (j == cur) return;
        int key = j;
        while (!is_key(so.index[key].type)) --key;
        const int first = (cur >= key && cur < j) ? cur + 1 : key;
        cur = -1;
        for (int k = first; k <= j; ++k) ingest(k);
        cur = j;
    }

    PsiPlain<D> plain_field(int slot) const {
        PsiPlain<D> f{};
        static_cast<PsiField<D>&>(f) = rx.field(slot);
        return f;
    }
    PsiFine<D, PsiPlain<D>> fine_plain(int slot) const {
        PsiFine<D, PsiPlain<D>> f{};
        f.base = plain_field(slot);
        f.fine = rx.fs.level(D, rx.fine_delta);
        return f;
    }
    PsiFine<D, PsiField<D>> fine_field(int slot) const {
        PsiFine<D, PsiField<D>> f{};
        f.base = rx.field(slot);
        f.fine = rx.fs.level(D, rx.fine_delta);
        return f;
    }

    void reserve(float** buffer, size_t* capacity, size_t floats) {
        if (*capacity >= floats) return;
        if (*buffer) CK(cudaFree(*buffer));
        *buffer = nullptr;
        *capacity = 0;
        CK(cudaMalloc(buffer, floats * sizeof(float)));
        *capacity = floats;
    }

    template <class F>
    void launch_grid(const F& f, size_t first, uint32_t count) {
        psit_grid_kernel<<<blocks_for(count), 256, 0, stream>>>(f, sh.n, first, count, d_out, d_out + count, d_out + 2 * (size_t)count);
    }
    template <class F>
    void launch_points(const F& f, uint32_t count) {
        probe_kernel<<<blocks_for(count), 256, 0, stream>>>(f, d_pos, d_pos + count, d_pos + 2 * (size_t)count, count, d_out, d_out + count,
                                                            d_out + 2 * (size_t)count);
    }

    void grid(int frame, float* out) override {
        ensure(frame);
        const int slot = frame & 1;
        const size_t voxels = (size_t)sh.n * sh.n * sh.n, chunk = std::min(voxels, kGridChunk);
        reserve(&d_out, &out_cap, 3 * chunk);
        for (size_t first = 0; first < voxels; first += chunk) {
            const uint32_t count = (uint32_t)std::min(chunk, voxels - first);
            if (fine && plain) launch_grid(fine_plain(slot), first, count);
            else if (fine) launch_grid(fine_field(slot), first, count);
            else if (plain) launch_grid(plain_field(slot), first, count);
            else launch_grid(rx.field(slot), first, count);
            CK(cudaStreamSynchronize(stream));
            for (int c = 0; c < 3; ++c)
                CK(cudaMemcpy(out + (size_t)c * voxels + first, d_out + (size_t)c * count, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CK(cudaGetLastError());
    }

    void points(int frame, const float* x, const float* y, const float* z, size_t total, float* u, float* v, float* w) override {
        ensure(frame);
        const int slot = frame & 1;
        const size_t chunk = std::min(total, kPointChunk);
        if (!chunk) return;
        reserve(&d_out, &out_cap, 3 * chunk);
        reserve(&d_pos, &pos_cap, 3 * chunk);
        const float n = (float)sh.n, h = kL / sh.n;
        std::vector<float> pos(3 * chunk);
        const float* in_xyz[3] = {x, y, z};
        float* out_uvw[3] = {u, v, w};
        for (size_t first = 0; first < total; first += chunk) {
            const uint32_t count = (uint32_t)std::min(chunk, total - first);
            for (int c = 0; c < 3; ++c)
                for (uint32_t i = 0; i < count; ++i) {
                    const float a = in_xyz[c][first + i];
                    float wrapped = a - n * floorf(a / n);  // periodic, in [0, n)
                    if (!(wrapped >= 0.0f && wrapped < n)) wrapped = 0.0f;
                    pos[(size_t)c * count + i] = (wrapped + 0.5f) * h;
                }
            CK(cudaMemcpy(d_pos, pos.data(), 3 * (size_t)count * sizeof(float), cudaMemcpyHostToDevice));
            if (fine && plain) launch_points(fine_plain(slot), count);
            else if (fine) launch_points(fine_field(slot), count);
            else if (plain) launch_points(plain_field(slot), count);
            else launch_points(rx.field(slot), count);
            CK(cudaStreamSynchronize(stream));
            for (int c = 0; c < 3; ++c)
                CK(cudaMemcpy(out_uvw[c] + first, d_out + (size_t)c * count, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CK(cudaGetLastError());
    }
};

// The codec core words its refusals for its own command-line programs. The most common ones are reworded here for a
// caller of the library, followed by the original message; the others are passed as they are. Each entry is found by
// a fragment of the original message.
std::string for_caller(const std::string& msg) {
    static const struct {
        const char* fragment;
        const char* reworded;
    } table[] = {
        {"capacity exceeded", "the error bound is too tight for this eps: more than a quarter of the volume would need refinement (raise the tolerance or lower eps)"},
        {"the header of the fine level has been altered", "seal mismatch: the header of the error-bound level was altered after the file was written"},
        {"the header has been altered", "seal mismatch: the file header was altered after the file was written"},
        {"the frame table has been altered", "seal mismatch: the frame table was altered after the file was written"},
        {"unexpected bytes after the frame table", "unexpected bytes after the frame table (neither the end of the file nor a seal)"},
        {"invalid .psi file", "not a valid .psi file"},
        {"invalid sequence file", "not a valid .seq file"},
        {"incomplete frame read", "truncated sequence: a frame could not be read in full"},
        {"cannot create the output file", "cannot create the output file"},
        {"unreadable frame", "a frame could not be read from the file (truncated file)"},
        {"degenerate fit of the predictor", "the temporal predictor could not be fitted (identical or empty frames?)"},
    };
    for (const auto& t : table)
        if (msg.find(t.fragment) != std::string::npos) return std::string(t.reworded) + " [" + msg + "]";
    return msg;
}

// Runs f, turns a failure into PSIT_ERROR and a message.
template <class F>
int guarded(F&& f) {
    try {
        f();
        g_error.clear();
        return PSIT_OK;
    } catch (const Failure& e) {
        g_error = for_caller(e.msg);
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return PSIT_ERROR;
}

void check_frame(const psit_reader* r, int frame) {
    if (!r) fail("null reader");
    if (frame < 0 || frame >= r->sh.n_frames) fail("frame " + std::to_string(frame) + " is outside the file (0 to " + std::to_string(r->sh.n_frames - 1) + ")");
}

// Silences stdout while the encoder runs.
struct Silence {
    int saved = -1;
    explicit Silence(bool on) {
        if (!on) return;
        fflush(stdout);
#ifdef _WIN32
        saved = _dup(_fileno(stdout));
        FILE* sink = ::fopen("NUL", "w");
        if (sink) {
            _dup2(_fileno(sink), _fileno(stdout));
            ::fclose(sink);
        }
#else
        saved = dup(fileno(stdout));
        FILE* sink = ::fopen("/dev/null", "w");
        if (sink) {
            dup2(fileno(sink), fileno(stdout));
            ::fclose(sink);
        }
#endif
    }
    ~Silence() {
        if (saved < 0) return;
        fflush(stdout);
#ifdef _WIN32
        _dup2(saved, _fileno(stdout));
        _close(saved);
#else
        dup2(saved, fileno(stdout));
        close(saved);
#endif
    }
};

// RMS of one velocity component over the frames to compress: the mean, over the frames, of the RMS of the three
// components about their mean.
float measure_u_rms(const FrameSource& src, int first, int count) {
    const size_t voxels = (size_t)src.hd.n * src.hd.n * src.hd.n;
    std::vector<float> frame(3 * voxels);
    double total = 0.0;
    for (int k = first; k < first + count; ++k) {
        src.read(k, frame.data());
        double variance = 0.0;
        for (int c = 0; c < 3; ++c) {
            const float* a = frame.data() + (size_t)c * voxels;
            double s1 = 0.0, s2 = 0.0;
            for (size_t i = 0; i < voxels; ++i) {
                s1 += a[i];
                s2 += (double)a[i] * a[i];
            }
            const double mean = s1 / voxels;
            variance += s2 / voxels - mean * mean;
        }
        total += std::sqrt(std::max(variance, 0.0) / 3.0);
    }
    return (float)(total / count);
}

// Common path of the three compression entry points. Returns the code of `pack`: 0, 3 (clipped) or 4 (bound missed).
int compress_source(FrameSource& src, const char* psi_path, const psit_options* options) {
    if (!psi_path) fail("null output path");
    psit_options opt;
    if (options) opt = *options;
    else psit_default_options(&opt);
    const int n = src.hd.n, available = src.hd.n_frames;
    if (n < 64 || (n & (n - 1)) != 0) fail("the grid size must be a power of two, at least 64 (got " + std::to_string(n) + ")");
    if (available < 5) fail("the source must hold at least 5 frames (got " + std::to_string(available) + ")");
    if (opt.degree != 2 && opt.degree != 3) fail("degree must be 2 or 3");
    if (!(opt.eps > 0.0f && opt.eps < 1.0f)) fail("eps must be between 0 and 1 (a fraction of u_rms)");
    if (opt.tolerance < 0.0f || opt.abs_tolerance < 0.0f || opt.tolerance >= 1.0f) fail("tolerance must be between 0 and 1, abs_tolerance must not be negative");
    if (opt.key_interval < 0) fail("key_interval must not be negative");
    const int first = opt.first_frame, count = opt.frame_count > 0 ? opt.frame_count : available - first;
    if (first < 0 || count < 2 || first + count > available)
        fail("frames " + std::to_string(first) + " to " + std::to_string(first + count - 1) + " are not all in the source (" + std::to_string(available) +
             " frames; at least 2 must be compressed)");
    if (!(src.hd.dt_frame > 0.0f)) fail("dt must be positive");
    if (opt.u_rms > 0.0f) src.hd.u_rms = opt.u_rms;
    else if (!(src.hd.u_rms > 0.0f)) src.hd.u_rms = measure_u_rms(src, first, count);
    if (!(src.hd.u_rms > 0.0f) || !std::isfinite(src.hd.u_rms)) fail("the velocity has no fluctuation (u_rms is zero or not finite)");

    SOptions o;
    o.mode = "pack";
    o.data = "(source)";
    o.file = psi_path;
    o.degree = opt.degree;
    o.eps = opt.eps;
    o.key_interval = opt.key_interval;
    o.start = first;
    o.frames = count;
    o.tolerance = opt.tolerance > 0.0f ? opt.tolerance : opt.abs_tolerance / src.hd.u_rms;
    o.seal = opt.seal != 0;

    psit_track::Arena arena;
    int rc = 0;
    try {
        psit_track::Scope scope(&arena);
        Silence quiet(!opt.verbose);
        rc = o.degree == 3 ? pack<3>(o, src) : pack<2>(o, src);
    } catch (...) {
        arena.release();
        throw;
    }
    arena.release();
    return rc;
}

// The three outcomes of `pack` that leave a file, and the failures.
template <class F>
int compress_guarded(F&& run) {
    int rc = PSIT_ERROR;
    try {
        rc = run();
        g_error.clear();
        if (rc == PSIT_CLIPPED) g_error = "the file is written, but some coefficients were clipped: it does not reproduce the requested projection (raise eps)";
        if (rc == PSIT_BOUND_MISSED) g_error = "the file is written, but the requested error bound is not certified on every frame (relax the tolerance)";
        return rc;
    } catch (const Failure& e) {
        g_error = for_caller(e.msg);
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return PSIT_ERROR;
}

}  // namespace

// ---- Public API -------------------------------------------------------------------------------------------------

extern "C" {

const char* psit_version(void) { return PSIT_VERSION; }
const char* psit_last_error(void) { return g_error.c_str(); }

int psit_open(const char* path, psit_reader** reader) {
    return guarded([&] {
        if (!path || !reader) fail("psit_open: null argument");
        *reader = nullptr;
        StreamHeader sh{};
        {
            FILE* probe = ::fopen(path, "rb");
            if (!probe) fail(std::string("cannot open ") + path);
            const bool ok = fread(&sh, sizeof(sh), 1, probe) == 1;
            ::fclose(probe);
            if (!ok) fail(std::string(path) + " is not a .psi file (too short)");
        }
        check_stream_header(sh);
        psit_reader* r = sh.degree == 3 ? static_cast<psit_reader*>(new ReaderT<3>()) : static_cast<psit_reader*>(new ReaderT<2>());
        r->sh = sh;
        try {
            psit_track::Scope scope(&r->arena);
            r->in = fopen(path, "rb");
            CHK(r->in, ".psi file not found");
            StreamHeader again{};
            CHK(fread(&again, sizeof(again), 1, r->in) == 1, "invalid .psi file");
            read_stream_index(r->in, r->sh, r->so);
            r->init();
        } catch (...) {
            r->arena.release();
            delete r;
            throw;
        }
        *reader = r;
    });
}

void psit_close(psit_reader* reader) {
    if (!reader) return;
    reader->arena.release();
    delete reader;
}

int psit_get_info(const psit_reader* r, psit_info* info) {
    return guarded([&] {
        if (!r || !info) fail("psit_get_info: null argument");
        psit_info i{};
        i.n = r->sh.n;
        i.frames = r->sh.n_frames;
        i.degree = r->sh.degree;
        i.first_frame = r->sh.first_frame;
        i.key_interval = r->sh.key_interval;
        i.format = r->sh.magic[7] - '0';
        i.sealed = r->so.sealed ? 1 : 0;
        i.has_bound = r->so.has_fine ? 1 : 0;
        i.bound_held = r->so.has_fine && r->so.fi.frames_over == 0 ? 1 : 0;
        i.dt = r->sh.dt_frame;
        i.u_rms = r->sh.u_rms;
        i.eps = r->sh.eps;
        i.tolerance = r->so.has_fine ? r->so.fi.tolerance : 0.0f;
        double sum = 0.0;
        int measured = 0;
        for (const FrameRecord& f : r->so.index) {
            i.max_error = std::max(i.max_error, f.field_max);
            if (f.field_err >= 0.0f) {
                sum += f.field_err;
                ++measured;
            }
        }
        i.rms_error = measured ? (float)(sum / measured) : -1.0f;
        i.file_bytes = r->so.file_bytes;
        i.raw_bytes = (uint64_t)r->sh.n_frames * 3ull * (uint64_t)r->sh.n * r->sh.n * r->sh.n * sizeof(float);
        *info = i;
    });
}

int psit_get_frame_info(const psit_reader* r, int frame, psit_frame_info* info) {
    return guarded([&] {
        if (!info) fail("psit_get_frame_info: null argument");
        check_frame(r, frame);
        const FrameRecord& f = r->so.index[frame];
        psit_frame_info i{};
        i.is_key = is_key(f.type) ? 1 : 0;
        i.bytes = f.bytes;
        i.rms_error = f.field_err;
        i.max_error = f.field_max;
        for (int c = 0; c < 3; ++c) i.mean[c] = f.mean[c];
        *info = i;
    });
}

int psit_decode(psit_reader* r, int frame, float* out) {
    const int rc = guarded([&] {
        check_frame(r, frame);
        if (!out) fail("psit_decode: null output");
        psit_track::Scope scope(&r->arena);
        r->grid(frame, out);
    });
    if (rc != PSIT_OK && r) r->cur = -1;
    return rc;
}

int psit_sample(psit_reader* r, int frame, const float* x, const float* y, const float* z, size_t count, float* u, float* v, float* w) {
    const int rc = guarded([&] {
        check_frame(r, frame);
        if (count && (!x || !y || !z || !u || !v || !w)) fail("psit_sample: null array");
        psit_track::Scope scope(&r->arena);
        r->points(frame, x, y, z, count, u, v, w);
    });
    if (rc != PSIT_OK && r) r->cur = -1;
    return rc;
}

void psit_default_options(psit_options* options) {
    if (!options) return;
    psit_options o{};
    o.degree = 3;
    o.eps = 0.005f;
    o.key_interval = 32;
    o.seal = 1;
    *options = o;
}

int psit_compress(psit_read_frame read, void* user, int n, int frames, float dt, const char* psi_path, const psit_options* options) {
    return compress_guarded([&]() -> int {
        if (!read) fail("psit_compress: null callback");
        FrameSource src;
        memcpy(src.hd.magic, "TLSEQ01", 8);
        src.hd.n = n;
        src.hd.n_frames = frames;
        src.hd.dt_frame = dt;
        src.read = [&](int k, float* dst) {
            if (read(user, k, dst) != 0) fail("the frame callback failed on frame " + std::to_string(k));
        };
        return compress_source(src, psi_path, options);
    });
}

int psit_compress_memory(const float* data, int n, int frames, float dt, const char* psi_path, const psit_options* options) {
    return compress_guarded([&]() -> int {
        if (!data) fail("psit_compress_memory: null data");
        FrameSource src;
        memcpy(src.hd.magic, "TLSEQ01", 8);
        src.hd.n = n;
        src.hd.n_frames = frames;
        src.hd.dt_frame = dt;
        const size_t floats = 3 * (size_t)n * n * n;
        src.read = [=](int k, float* dst) { memcpy(dst, data + (size_t)k * floats, floats * sizeof(float)); };
        return compress_source(src, psi_path, options);
    });
}

int psit_compress_file(const char* seq_path, const char* psi_path, const psit_options* options) {
    return compress_guarded([&]() -> int {
        if (!seq_path) fail("psit_compress_file: null input path");
        FILE* in = ::fopen(seq_path, "rb");
        if (!in) fail(std::string("cannot open ") + seq_path);
        FrameSource src;
        int rc = PSIT_ERROR;
        try {
            src.hd = sequence_read_header(in);
            src.read = [&](int k, float* dst) { sequence_read_frame(in, src.hd, k, dst); };
            rc = compress_source(src, psi_path, options);
        } catch (...) {
            ::fclose(in);
            throw;
        }
        ::fclose(in);
        return rc;
    });
}

}  // extern "C"
