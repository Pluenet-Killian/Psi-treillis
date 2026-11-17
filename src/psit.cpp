// SPDX-License-Identifier: Apache-2.0
// psit - command-line front end of the psitreillis library.
//
//   psit info FILE.psi
//   psit compress INPUT OUTPUT.psi [options]
//   psit decompress FILE.psi OUTPUT [--first K] [--count C]
//   psit sample FILE.psi --frame K --at X,Y,Z [--at X,Y,Z ...]
//   psit convert INPUT OUTPUT [--n N] [--dt DT]
//
// Field files are recognised by their extension:
//   .npy   NumPy array, little-endian float32, C order, shape (frames, 3, n, n, n)
//   .seq   the project's own sequence format (72-byte header, then the frames)
//   other  raw little-endian float32, frame after frame; needs --n
// In the three formats a frame is three planes u, v, w of n^3 values, each indexed (z * n + y) * n + x.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "psitreillis.h"

namespace {

[[noreturn]] void die(const std::string& msg, int code = 1) {
    fprintf(stderr, "psit: %s\n", msg.c_str());
    exit(code);
}

void seek64(FILE* f, uint64_t offset) {
#ifdef _WIN32
    if (_fseeki64(f, (long long)offset, SEEK_SET) != 0) die("seek failed");
#else
    if (fseeko(f, (off_t)offset, SEEK_SET) != 0) die("seek failed");
#endif
}
uint64_t file_size(FILE* f) {
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    const uint64_t s = (uint64_t)_ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    const uint64_t s = (uint64_t)ftello(f);
#endif
    seek64(f, 0);
    return s;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = strlen(suffix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        char a = s[s.size() - n + i], b = suffix[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

// Header of a .seq file (src/sequence.h).
struct SeqHeader {
    char magic[8];  // "TLSEQ01\0"
    int32_t n, n_frames;
    float dt_frame, nu, u_rms, kmax_eta, re_lambda;
    float reserved[9];
};
static_assert(sizeof(SeqHeader) == 72, "layout of a .seq header");

// A field file opened for reading: where its frames start, how many there are.
struct FieldFile {
    FILE* f = nullptr;
    std::string path;
    uint64_t offset = 0;
    int n = 0, frames = 0;
    float dt = 1.0f, u_rms = 0.0f;
    size_t frame_bytes() const { return 3 * (size_t)n * n * n * sizeof(float); }
    void read(int k, float* dst) const {
        seek64(f, offset + (uint64_t)k * frame_bytes());
        if (fread(dst, 1, frame_bytes(), f) != frame_bytes()) die(path + ": frame " + std::to_string(k) + " is truncated");
    }
};

// Parses the dictionary of a .npy header: descr, fortran_order, shape.
void parse_npy(const std::string& dict, const std::string& path, std::vector<long long>& shape) {
    if (dict.find("'descr': '<f4'") == std::string::npos && dict.find("'descr':'<f4'") == std::string::npos)
        die(path + ": the array must be little-endian float32 ('<f4')");
    if (dict.find("'fortran_order': False") == std::string::npos && dict.find("'fortran_order':False") == std::string::npos)
        die(path + ": the array must be in C order");
    const size_t open = dict.find('(', dict.find("'shape'")), close = dict.find(')', open);
    if (open == std::string::npos || close == std::string::npos) die(path + ": no shape in the .npy header");
    const std::string inside = dict.substr(open + 1, close - open - 1);
    const char* p = inside.c_str();
    while (*p) {
        char* end = nullptr;
        const long long v = strtoll(p, &end, 10);
        if (end == p) {
            ++p;
            continue;
        }
        shape.push_back(v);
        p = end;
    }
}

FieldFile open_field(const std::string& path, int n_option, float dt_option) {
    FieldFile ff;
    ff.path = path;
    ff.f = fopen(path.c_str(), "rb");
    if (!ff.f) die("cannot open " + path);
    const uint64_t size = file_size(ff.f);
    if (ends_with(path, ".seq")) {
        SeqHeader h{};
        if (fread(&h, sizeof(h), 1, ff.f) != 1 || memcmp(h.magic, "TLSEQ01", 7) != 0) die(path + ": not a .seq file");
        ff.n = h.n;
        ff.frames = h.n_frames;
        ff.dt = h.dt_frame;
        ff.u_rms = h.u_rms;
        ff.offset = sizeof(h);
    } else if (ends_with(path, ".npy")) {
        unsigned char head[12];
        if (fread(head, 1, 10, ff.f) != 10 || memcmp(head, "\x93NUMPY", 6) != 0) die(path + ": not a .npy file");
        size_t len = head[8] | ((size_t)head[9] << 8), start = 10;
        if (head[6] >= 2) {
            if (fread(head + 10, 1, 2, ff.f) != 2) die(path + ": truncated .npy header");
            len |= ((size_t)head[10] << 16) | ((size_t)head[11] << 24);
            start = 12;
        }
        std::string dict(len, '\0');
        if (fread(&dict[0], 1, len, ff.f) != len) die(path + ": truncated .npy header");
        std::vector<long long> shape;
        parse_npy(dict, path, shape);
        if (shape.size() == 4) shape.insert(shape.begin(), 1);
        if (shape.size() != 5 || shape[1] != 3 || shape[2] != shape[3] || shape[3] != shape[4])
            die(path + ": the array must have the shape (frames, 3, n, n, n)");
        ff.frames = (int)shape[0];
        ff.n = (int)shape[2];
        ff.offset = start + len;
    } else {
        if (n_option <= 0) die(path + " is read as raw float32: give the grid size with --n");
        ff.n = n_option;
        if (size == 0 || size % ff.frame_bytes() != 0) die(path + ": the size is not a whole number of frames of " + std::to_string(ff.n) + "^3 x 3 float32");
        ff.frames = (int)(size / ff.frame_bytes());
    }
    if (dt_option > 0.0f) ff.dt = dt_option;
    if (ff.n <= 0 || ff.frames <= 0) die(path + ": empty field");
    if (ff.offset + (uint64_t)ff.frames * ff.frame_bytes() > size) die(path + ": shorter than its header says");
    return ff;
}

// A field file opened for writing, in the format of its extension.
FILE* create_field(const std::string& path, int n, int frames, float dt, float u_rms) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) die("cannot create " + path);
    if (ends_with(path, ".seq")) {
        SeqHeader h{};
        memcpy(h.magic, "TLSEQ01", 8);
        h.n = n;
        h.n_frames = frames;
        h.dt_frame = dt;
        h.u_rms = u_rms;
        if (fwrite(&h, sizeof(h), 1, f) != 1) die("write failed: " + path);
    } else if (ends_with(path, ".npy")) {
        char dict[160];
        snprintf(dict, sizeof(dict), "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, 3, %d, %d, %d), }", frames, n, n, n);
        std::string head(dict);
        while ((10 + head.size() + 1) % 64 != 0) head += ' ';
        head += '\n';
        const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)(head.size() & 0xFF), (unsigned char)(head.size() >> 8)};
        if (fwrite(magic, 1, 10, f) != 10 || fwrite(head.data(), 1, head.size(), f) != head.size()) die("write failed: " + path);
    }
    return f;
}

void check(int rc, const char* what) {
    if (rc != PSIT_OK) die(std::string(what) + ": " + psit_last_error());
}

void print_info(const std::string& path, const psit_info& i) {
    const double mib = 1048576.0;
    printf("file        : %s (format %d, %s)\n", path.c_str(), i.format, i.sealed ? "sealed" : "no seal");
    printf("grid        : %d^3, %d frames (from frame %d of the source), dt %g\n", i.n, i.frames, i.first_frame, i.dt);
    printf("codec       : degree %d, one key frame every %d frames, eps %g, u_rms %g\n", i.degree, i.key_interval, i.eps, i.u_rms);
    printf("size        : %.3f MiB, %.1f times smaller than raw float32 (%.1f MiB)\n", i.file_bytes / mib, (double)i.raw_bytes / (double)i.file_bytes,
           i.raw_bytes / mib);
    if (i.rms_error >= 0.0f) printf("rms error   : %.4f %% of the field, mean over the frames (measured by the encoder)\n", 100.0 * i.rms_error);
    else printf("rms error   : not measured\n");
    if (i.max_error > 0.0f) printf("max error   : %.4f %% of u_rms, largest component error at voxel centres\n", 100.0 * i.max_error);
    if (!i.has_bound) printf("error bound : none requested\n");
    else printf("error bound : %.4f %% of u_rms requested, %s\n", 100.0 * i.tolerance, i.bound_held ? "certified on every voxel of every frame" : "NOT certified");
}

int cmd_info(const std::string& path) {
    psit_reader* r = nullptr;
    check(psit_open(path.c_str(), &r), "open");
    psit_info i;
    check(psit_get_info(r, &i), "info");
    print_info(path, i);
    psit_close(r);
    return 0;
}

struct Args {
    std::vector<std::string> positional;
    std::vector<std::string> at;
    int n = 0, first = 0, count = 0, frame = 0, degree = 3, key_interval = 32;
    float dt = 0.0f, eps = 0.005f, tolerance = 0.0f, abs_tolerance = 0.0f, u_rms = 0.0f;
    bool seal = true, verbose = false, quiet = false;
};

int read_callback(void* user, int frame, float* dst) {
    static_cast<const FieldFile*>(user)->read(frame, dst);
    return 0;
}

int cmd_compress(const Args& a) {
    if (a.positional.size() != 2) die("usage: psit compress INPUT OUTPUT.psi [options]", 2);
    const FieldFile in = open_field(a.positional[0], a.n, a.dt);
    psit_options o;
    psit_default_options(&o);
    o.degree = a.degree;
    o.eps = a.eps;
    o.tolerance = a.tolerance;
    o.abs_tolerance = a.abs_tolerance;
    o.key_interval = a.key_interval;
    o.first_frame = a.first;
    o.frame_count = a.count;
    o.u_rms = a.u_rms > 0.0f ? a.u_rms : in.u_rms;
    o.seal = a.seal ? 1 : 0;
    o.verbose = a.verbose ? 1 : 0;
    // A .seq file goes through psit_compress_file, which keeps the whole header of the sequence: the .psi file is then
    // the one `psi_stream pack` writes, byte for byte.
    const bool seq = ends_with(a.positional[0], ".seq") && a.dt <= 0.0f;
    if (seq) o.u_rms = a.u_rms;
    const int rc = seq ? psit_compress_file(a.positional[0].c_str(), a.positional[1].c_str(), &o)
                       : psit_compress(read_callback, (void*)&in, in.n, in.frames, in.dt, a.positional[1].c_str(), &o);
    fclose(in.f);
    if (rc == PSIT_ERROR) die(std::string("compress: ") + psit_last_error());
    if (rc != PSIT_OK) fprintf(stderr, "psit: warning: %s\n", psit_last_error());
    if (!a.quiet) {
        psit_reader* r = nullptr;
        check(psit_open(a.positional[1].c_str(), &r), "open");
        psit_info i;
        check(psit_get_info(r, &i), "info");
        print_info(a.positional[1], i);
        psit_close(r);
    }
    return rc;
}

int cmd_decompress(const Args& a) {
    if (a.positional.size() != 2) die("usage: psit decompress FILE.psi OUTPUT [--first K] [--count C]", 2);
    psit_reader* r = nullptr;
    check(psit_open(a.positional[0].c_str(), &r), "open");
    psit_info i;
    check(psit_get_info(r, &i), "info");
    const int first = a.first, count = a.count > 0 ? a.count : i.frames - first;
    if (first < 0 || count < 1 || first + count > i.frames) die("frames out of range: the file holds " + std::to_string(i.frames) + " frames");
    FILE* out = create_field(a.positional[1], i.n, count, i.dt, i.u_rms);
    std::vector<float> frame(3 * (size_t)i.n * i.n * i.n);
    for (int k = first; k < first + count; ++k) {
        check(psit_decode(r, k, frame.data()), "decode");
        if (fwrite(frame.data(), sizeof(float), frame.size(), out) != frame.size()) die("write failed: " + a.positional[1]);
    }
    fclose(out);
    psit_close(r);
    if (!a.quiet) printf("%d frame(s) of %d^3 written to %s\n", count, i.n, a.positional[1].c_str());
    return 0;
}

int cmd_sample(const Args& a) {
    if (a.positional.size() != 1 || a.at.empty()) die("usage: psit sample FILE.psi --frame K --at X,Y,Z [--at X,Y,Z ...]", 2);
    std::vector<float> x, y, z;
    for (const std::string& s : a.at) {
        float p[3];
        if (sscanf(s.c_str(), "%f,%f,%f", &p[0], &p[1], &p[2]) != 3) die("--at expects X,Y,Z in grid-index units, got " + s, 2);
        x.push_back(p[0]);
        y.push_back(p[1]);
        z.push_back(p[2]);
    }
    psit_reader* r = nullptr;
    check(psit_open(a.positional[0].c_str(), &r), "open");
    std::vector<float> u(x.size()), v(x.size()), w(x.size());
    check(psit_sample(r, a.frame, x.data(), y.data(), z.data(), x.size(), u.data(), v.data(), w.data()), "sample");
    for (size_t i = 0; i < x.size(); ++i) printf("%g %g %g  %.9g %.9g %.9g\n", x[i], y[i], z[i], u[i], v[i], w[i]);
    psit_close(r);
    return 0;
}

int cmd_convert(const Args& a) {
    if (a.positional.size() != 2) die("usage: psit convert INPUT OUTPUT [--n N] [--dt DT] [--first K] [--count C]", 2);
    const FieldFile in = open_field(a.positional[0], a.n, a.dt);
    const int first = a.first, count = a.count > 0 ? a.count : in.frames - first;
    if (first < 0 || count < 1 || first + count > in.frames) die("frames out of range: the input holds " + std::to_string(in.frames) + " frames");
    FILE* out = create_field(a.positional[1], in.n, count, in.dt, a.u_rms > 0.0f ? a.u_rms : in.u_rms);
    std::vector<float> frame(3 * (size_t)in.n * in.n * in.n);
    for (int k = first; k < first + count; ++k) {
        in.read(k, frame.data());
        if (fwrite(frame.data(), sizeof(float), frame.size(), out) != frame.size()) die("write failed: " + a.positional[1]);
    }
    fclose(out);
    fclose(in.f);
    if (!a.quiet) printf("%d frame(s) of %d^3 written to %s\n", count, in.n, a.positional[1].c_str());
    return 0;
}

void usage() {
    printf("psit %s - Psi-treillis: divergence-free compression of periodic velocity fields (GPU)\n\n"
           "  psit info FILE.psi\n"
           "  psit compress INPUT OUTPUT.psi [--degree 2|3] [--eps 0.005] [--tolerance T | --abs-tolerance A]\n"
           "                [--key-interval 32] [--first K] [--count C] [--n N] [--dt DT] [--u-rms U]\n"
           "                [--no-seal] [--verbose] [--quiet]\n"
           "  psit decompress FILE.psi OUTPUT [--first K] [--count C] [--quiet]\n"
           "  psit sample FILE.psi --frame K --at X,Y,Z [--at X,Y,Z ...]\n"
           "  psit convert INPUT OUTPUT [--n N] [--dt DT] [--first K] [--count C]\n\n"
           "Field files, by extension: .npy (float32, shape (frames, 3, n, n, n)), .seq (project format),\n"
           "anything else is raw float32 and needs --n. A frame is three planes u, v, w of n^3 values.\n\n"
           "  --eps            target RMS quantisation error, as a fraction of u_rms (default 0.005)\n"
           "  --tolerance      certified bound on the largest component error, as a fraction of u_rms\n"
           "  --abs-tolerance  the same bound in the units of the data\n"
           "  --at X,Y,Z       a point in grid-index units: (i, j, k) is the centre of voxel x=i, y=j, z=k\n\n"
           "Exit codes: 0 success; 1 error; 3 file written but coefficients clipped; 4 file written but the\n"
           "error bound is not certified.\n",
           psit_version());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "help")) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    const std::string cmd = argv[1];
    if (cmd == "version" || cmd == "--version") {
        printf("%s\n", psit_version());
        return 0;
    }
    Args a;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) die("missing value for " + s, 2);
            return argv[++i];
        };
        if (s == "--n") a.n = atoi(next());
        else if (s == "--dt") a.dt = (float)atof(next());
        else if (s == "--degree") a.degree = atoi(next());
        else if (s == "--eps") a.eps = (float)atof(next());
        else if (s == "--tolerance") a.tolerance = (float)atof(next());
        else if (s == "--abs-tolerance") a.abs_tolerance = (float)atof(next());
        else if (s == "--key-interval") a.key_interval = atoi(next());
        else if (s == "--first") a.first = atoi(next());
        else if (s == "--count") a.count = atoi(next());
        else if (s == "--frame") a.frame = atoi(next());
        else if (s == "--u-rms") a.u_rms = (float)atof(next());
        else if (s == "--at") a.at.push_back(next());
        else if (s == "--no-seal") a.seal = false;
        else if (s == "--verbose") a.verbose = true;
        else if (s == "--quiet") a.quiet = true;
        else if (s.rfind("--", 0) == 0) die("unknown option " + s + " (psit --help)", 2);
        else a.positional.push_back(s);
    }
    if (cmd == "info") {
        if (a.positional.size() != 1) die("usage: psit info FILE.psi", 2);
        return cmd_info(a.positional[0]);
    }
    if (cmd == "compress") return cmd_compress(a);
    if (cmd == "decompress") return cmd_decompress(a);
    if (cmd == "sample") return cmd_sample(a);
    if (cmd == "convert") return cmd_convert(a);
    die("unknown command " + cmd + " (psit --help)", 2);
}
