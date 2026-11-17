// SPDX-License-Identifier: Apache-2.0
// psi_stream.cu - measurement front end of the codec: the command line of its two modes, pack and play.
//
//   pack  encodes frames of a sequence file (.seq) into a .psi file. This file only parses the options: the encoder
//         (pack<D>), the file format and the receiver are in src/codec_stream.cuh. Exit code 0; 3 if coefficients were
//         clipped (unless --allow-clipping); otherwise 4 if the bound of --tolerance was missed. The file is written
//         in the three cases.
//   play  reads a .psi file alone, without the sequence it was encoded from. Each frame is read from the disk, sent
//         over the bus and decoded on the GPU; particles are advected in the decoded field by RK4 steps and, with
//         --raymarch, one image per frame is ray-marched. play times each of these stages, recomputes the fingerprint
//         of the state after each frame and compares it with that of the frame table: exit code 1 if a frame differs.
//         The kernels of the advection and of the ray marching are in playback_kernels.cuh.
//
// The lines printed by the two modes are read back by the test suite (tests/test_codec.py), by tools/claims.py and by
// the measurement scripts of benchmarks/.
//
//   psi_stream pack [--data f.seq] [--file f.psi] [--degree 2|3] [--mf 2] [--levels 3] [--fb 2] [--eps 0.005]
//                   [--ridge 1e-5] [--key-interval 32] [--start 16] [--frames 33] [--check-every 1] [--allow-clipping]
//                   [--raw-keys] [--legacy-step] [--fit-at frame] [--tolerance 0.05] [--tolerance-margin 1e-4]
//                   [--fine-eps e] [--fine-passes 64] [--probe f.probe] [--probe-frame j] [--no-seal]
//                   [--brick-dump f.bin] [--csv]
//
//     --data, --file       sequence to encode, .psi file to write
//     --degree             degree of the B-splines, 2 or 3 (default 2)
//     --mf, --levels       step of level 0 in voxels (2), number of levels (3)
//     --fb                 fractional bits of the sum of the levels (2)
//     --eps                target quantisation error, as a fraction of u' (0.005)
//     --ridge              ridge regularisation of the fit of the levels (1e-5)
//     --key-interval K     one key frame every K frames (32; 0: the first only)
//     --start, --frames    first frame and number of frames encoded (16, 33)
//     --check-every n      fidelity check on the whole grid every n frames (1; 0: never)
//     --allow-clipping     do not exit with an error when a coefficient is clipped
//     --raw-keys           raw key frames (type 0) instead of coded ones (type 3)
//     --legacy-step        former formula of the quantisation step
//     --fit-at f           the predictors are fitted on frames f + 1 to f + 4 of the sequence (-1, the default: at the
//                          start of the encoded segment)
//     --tolerance t        bound on the largest error of a component at voxel centres, as a fraction of u' (0, the
//                          default: no sparse fine level)
//     --tolerance-margin   arithmetic margin of the bound, as a fraction of u' (1e-4)
//     --fine-eps           quantisation error of the fine level (0, the default: tolerance / 8.5)
//     --fine-passes        cap on the marking passes of the fine level per frame (64)
//     --probe f            writes a probe of the decoded field; --probe-frame: the probed frame, counted from --start
//                          (-1, the default: the last)
//     --no-seal            file without a seal
//     --brick-dump f       file that receives, per checked frame, the largest error per brick of 4^3 voxels
//     --csv                also prints the "CSV," lines
//
//   psi_stream play [--file f.psi] [--substeps 2] [--particles 1048576] [--seed 1234] [--sort] [--resort n]
//                   [--raymarch 1024] [--ray-steps 256] [--max-frames n] [--seek frame] [--block 128]
//                   [--block-plain 256] [--cache -1|0|1] [--ray-cache -1|0|1] [--no-fine] [--probe f.probe]
//                   [--require-seal] [--dump-positions f.bin] [--csv]
//
//     --file               .psi file to read
//     --substeps           RK4 steps per frame interval (2)
//     --particles, --seed  number of particles (1048576) and seed of their initial positions (1234)
//     --sort, --resort n   Morton sort of the particles: initial, and every n frames (0, the default: never)
//     --raymarch R         one image of R^2 rays per frame (0, the default: none); --ray-steps: samples per ray (256)
//     --max-frames n       number of frames to read (0, the default: all)
//     --seek j             measures the direct-access time to frame j
//     --block              threads per block of the kernels with a support cache (128); --block-plain: of the kernels
//                          without one (256)
//     --cache, --ray-cache support cache in registers, for the RK4 step and for the rays: 1, 0, or -1 (automatic)
//     --no-fine            ignores the sparse fine level of a PSISTRM3 file (measurement of its cost)
//     --probe f            reads back the probe written by pack and compares its own field with it
//     --require-seal       refuses a file without a seal instead of reporting it
//     --dump-positions f   writes the final positions of the particles
//     --csv                also prints the "CSV,play" line

#include "playback_kernels.cuh"


static SOptions sparse(int argc, char** argv) {
    SOptions o;
    CHK(argc >= 2, "usage: psi_stream pack|play [options]");
    o.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--data") o.data = next();
        else if (a == "--file") o.file = next();
        else if (a == "--degree") o.degree = atoi(next());
        else if (a == "--mf") o.mf = atoi(next());
        else if (a == "--levels") o.levels = atoi(next());
        else if (a == "--fb") o.fb = atoi(next());
        else if (a == "--eps") o.eps = (float)atof(next());
        else if (a == "--ridge") o.ridge = atof(next());
        else if (a == "--key-interval") o.key_interval = atoi(next());
        else if (a == "--start") o.start = atoi(next());
        else if (a == "--frames") o.frames = atoi(next());
        else if (a == "--check-every") o.check_every = atoi(next());
        else if (a == "--allow-clipping") o.allow_clip = true;
        else if (a == "--raw-keys") o.raw_keys = true;
        else if (a == "--fit-at") o.fit_at = atoi(next());
        else if (a == "--legacy-step") o.legacy_step = true;
        else if (a == "--tolerance") o.tolerance = (float)atof(next());
        else if (a == "--fine-eps") o.fine_eps = (float)atof(next());
        else if (a == "--fine-passes") o.fine_passes = atoi(next());
        else if (a == "--tolerance-margin") o.tolerance_margin = (float)atof(next());
        else if (a == "--probe") o.probe = next();
        else if (a == "--probe-frame") o.probe_frame = atoi(next());
        else if (a == "--no-fine") o.no_fine = true;
        else if (a == "--no-seal") o.seal = false;
        else if (a == "--require-seal") o.require_seal = true;
        else if (a == "--dump-positions") o.dump_positions = next();
        else if (a == "--brick-dump") o.brick_dump = next();
        else if (a == "--substeps") o.substeps = atoi(next());
        else if (a == "--particles") o.particles = (uint32_t)atoi(next());
        else if (a == "--seed") o.seed = (unsigned)atoi(next());
        else if (a == "--resort") o.resort = atoi(next());
        else if (a == "--sort") o.sort0 = true;
        else if (a == "--raymarch") o.raymarch = atoi(next());
        else if (a == "--ray-steps") o.ray_steps = atoi(next());
        else if (a == "--max-frames") o.max_frames = atoi(next());
        else if (a == "--seek") o.seek = atoi(next());
        else if (a == "--block") o.block = atoi(next());
        else if (a == "--block-plain") o.block_plain = atoi(next());
        else if (a == "--cache") o.cache = atoi(next());
        else if (a == "--ray-cache") o.ray_cache = atoi(next());
        else if (a == "--csv") o.csv = true;
        else {
            fprintf(stderr, "Usage: psi_stream pack [--data f.seq] [--file f.psi] [--degree 2|3] [--mf 2] [--levels 3] [--eps 0.005]\n"
                            "                       [--key-interval 32] [--start 16] [--frames 33] [--check-every 1]\n"
                            "                       [--allow-clipping] [--brick-dump f.bin] [--raw-keys] [--legacy-step] [--fit-at frame]\n"
                            "                       [--tolerance 0.05] [--tolerance-margin 1e-4] [--fine-eps e] [--fine-passes 64]\n"
                            "                       [--probe f.probe] [--probe-frame j] [--no-seal]\n"
                            "       psi_stream play [--file f.psi] [--substeps 2] [--particles 1048576] [--sort] [--resort n]\n"
                            "                       [--raymarch 1024] [--ray-steps 256] [--max-frames n] [--seek frame] [--block 128] [--csv]\n"
                            "                       [--cache -1|0|1] [--ray-cache -1|0|1] [--block-plain 256] [--no-fine] [--probe f.probe]\n"
                            "                       [--require-seal]\n");
            exit(2);
        }
    }
    return o;
}

// ---- play ------------------------------------------------------------------------------

template <int D>
static int play(const SOptions& o, const StreamHeader& sh, const StreamOpen& so, FILE* in) {
    Geo g;
    g.build(sh.n, D, sh.mf, sh.levels, sh.fb);
    // Sparse fine level (PSISTRM3): its header follows that of the stream; --no-fine ignores it, to measure what it
    // costs.
    const bool has_fine = so.has_fine, fine = has_fine && !o.no_fine;
    const bool legacy = sh.magic[7] == '1';  // PSISTRM1: the sums outside +-32767 are counted (Receiver::legacy_beyond)
    const FineInfo& fi = so.fi;
    if (has_fine) {
        const size_t nb = (size_t)(sh.n >> kFineLog);
        g.cap += fine_section_cap(nb * nb * nb / 8, fi.max_bricks, fi.max_nodes);
    }
    const int n_frames = o.max_frames > 0 ? std::min(o.max_frames, sh.n_frames) : sh.n_frames;
    CHK(n_frames >= 2, "at least two frames are needed");
    CHK(o.block >= 32 && o.block <= 1024 && o.block % 32 == 0, "--block: multiple of 32, from 32 to 1024");
    CHK(o.block_plain >= 32 && o.block_plain <= 1024 && o.block_plain % 32 == 0, "--block-plain: multiple of 32, from 32 to 1024");
    const bool sorted = o.sort0 || o.resort > 0;
    const bool use_cache = o.cache < 0 ? (D == 2 || !sorted) : o.cache != 0;
    const bool ray_cache = o.ray_cache < 0 ? D == 2 : o.ray_cache != 0;
    // Direct reading of the lattice as long as no frame wraps (PSISTRM1 files: never), by differences otherwise.
    const bool plain = sh.magic[7] == '1' || (sh.reserved[0] == kClipMarker && (sh.reserved[7] >> 32) == 0);
    const std::vector<FrameRecord>& index = so.index;  // read and checked at opening (read_stream_index)
    const uint64_t file_bytes = so.file_bytes;
    // Receive buffers (device and host) at the size of the largest frame of the file, and not at the worst case of the
    // coder.
    g.cap = packet_bytes(index, g.cap);

    size_t free0 = 0, free1 = 0, total = 0;
    CK(cudaMemGetInfo(&free0, &total));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    Coef coef[kMaxLevels];
    for (int l = 0; l < g.L; ++l) coef[l] = Coef{sh.coef[l][0], sh.coef[l][1], sh.coef[l][2]};
    Receiver<D> rx;
    rx.init(g, sh.delta, coef);
    rx.set_format(sh.magic[7], sh.fb);
    if (fine) rx.init_fine(fi);
    auto plain_field = [&](int early) {
        PsiPlain<D> f{};
        static_cast<PsiField<D>&>(f) = rx.field(early);
        return f;
    };
    // Complete field: base and fine level, for the kernels without a support cache (rays, and the RK4 step when it has
    // no cache).
    auto fine_plain = [&](int early) {
        PsiFine<D, PsiPlain<D>> f{};
        f.base = plain_field(early);
        f.fine = rx.fs.level(D, rx.fine_delta);
        return f;
    };
    auto fine_field = [&](int early) {
        PsiFine<D, PsiField<D>> f{};
        f.base = rx.field(early);
        f.fine = rx.fs.level(D, rx.fine_delta);
        return f;
    };
    // Probe written by pack: same points, same frame; the field of the reader must be that of the encoder.
    std::vector<float> probe_p, probe_u;
    int probe_frame = -1;
    uint32_t probe_count = 0;
    double probe_diff = -1.0;
    float *d_pp = nullptr, *d_pu = nullptr;
    if (!o.probe.empty()) {
        FILE* pf = fopen(o.probe.c_str(), "rb");
        CHK(pf, "probe file not found");
        char magic[8];
        int32_t hi[8];
        float hf[8];
        CHK(fread(magic, 1, 8, pf) == 8 && memcmp(magic, "PSIPROBE", 8) == 0 && fread(hi, sizeof(hi), 1, pf) == 1 && fread(hf, sizeof(hf), 1, pf) == 1,
            "invalid probe file");
        CHK(hi[0] == sh.n && hi[1] == D, "the probe does not come from this file");
        probe_frame = hi[3] - sh.first_frame;
        probe_count = (uint32_t)hi[4];
        const uint64_t skip = (uint64_t)3 * hi[2] * hi[2] * hi[2] * sizeof(short) + (hi[6] ? (uint64_t)3 * sh.n * sh.n * sh.n : 0);
        seek64(pf, 8 + sizeof(hi) + sizeof(hf) + skip);
        probe_p.resize(3 * (size_t)probe_count);
        probe_u.resize(3 * (size_t)probe_count);
        CHK(fread(probe_p.data(), sizeof(float), probe_p.size(), pf) == probe_p.size() && fread(probe_u.data(), sizeof(float), probe_u.size(), pf) == probe_u.size(),
            "truncated probe file");
        fclose(pf);
        CK(cudaMalloc(&d_pp, probe_p.size() * sizeof(float)));
        CK(cudaMalloc(&d_pu, probe_u.size() * sizeof(float)));
        CK(cudaMemcpy(d_pp, probe_p.data(), probe_p.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    unsigned char* h_buf = nullptr;
    CK(cudaHostAlloc(&h_buf, g.cap, cudaHostAllocDefault));

    const uint32_t np = o.particles;
    float *d_p[3], *d_q[3];
    uint32_t *d_key = nullptr, *d_idx = nullptr;
    {
        std::vector<float> pos((size_t)np * 3);
        std::mt19937 rng(o.seed);
        std::uniform_real_distribution<float> U(0.0f, kL);
        for (auto& v : pos) v = U(rng);
        for (int c = 0; c < 3; ++c) {
            CK(cudaMalloc(&d_p[c], (size_t)np * sizeof(float)));
            CK(cudaMalloc(&d_q[c], (size_t)np * sizeof(float)));
            CK(cudaMemcpy(d_p[c], pos.data() + (size_t)c * np, (size_t)np * sizeof(float), cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d_key, (size_t)np * sizeof(uint32_t)));
        CK(cudaMalloc(&d_idx, (size_t)np * sizeof(uint32_t)));
    }
    float* d_img = nullptr;
    if (o.raymarch > 0) CK(cudaMalloc(&d_img, (size_t)o.raymarch * o.raymarch * sizeof(float)));
    CK(cudaMemGetInfo(&free1, &total));

    double t_disk = 0.0, t_bus = 0.0, t_dec = 0.0, t_adv = 0.0, t_sort = 0.0, t_ray = 0.0;
    double tk_disk = 0.0, tk_bus = 0.0, tk_dec = 0.0;  // share of the key frames read in steady state
    uint64_t bytes_read = 0, key_bytes_read = 0;
    int n_key_read = 0;
    int n_sort = 0, bad_hash = 0;
    long long legacy_sums = 0;  // PSISTRM1: sums of the pyramid outside +-32767, and frames that count some
    int legacy_frames = 0;
    auto sort_particles = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        morton_keys<<<blocks_for(np), 256, 0, stream>>>(d_p[0], d_p[1], d_p[2], np, d_key, d_idx);
        CK(cudaStreamSynchronize(stream));
        thrust::sort_by_key(thrust::device_pointer_cast(d_key), thrust::device_pointer_cast(d_key) + np, thrust::device_pointer_cast(d_idx));
        gather3<<<blocks_for(np), 256, 0, stream>>>(d_p[0], d_p[1], d_p[2], d_idx, np, d_q[0], d_q[1], d_q[2]);
        CK(cudaStreamSynchronize(stream));
        for (int c = 0; c < 3; ++c) std::swap(d_p[c], d_q[c]);
        t_sort += seconds_since(t0);
        ++n_sort;
    };
    // Reads, transfers and decodes frame j into slot j & 1.
    auto ingest = [&](int j, bool timed) {
        const FrameRecord& r = index[j];
        CHK(r.bytes <= g.cap, "frame larger than expected");
        auto t0 = std::chrono::steady_clock::now();
        seek64(in, r.offset);
        CHK(fread(h_buf, 1, r.bytes, in) == r.bytes, "unreadable frame");
        const double dd = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        CK(cudaMemcpyAsync(rx.packet, h_buf, r.bytes, cudaMemcpyHostToDevice, stream));
        rx.inspect(r.type, h_buf, r.bytes);  // check of the payload on the host, while it crosses the bus
        CK(cudaStreamSynchronize(stream));
        const double db = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        rx.decode(r.type, j & 1, r.mean, h_buf, r.bytes, stream);
        CK(cudaStreamSynchronize(stream));
        const double dc = seconds_since(t0);
        if (legacy) {  // outside the measured times
            const unsigned beyond = rx.legacy_beyond(j & 1);
            legacy_sums += beyond;
            legacy_frames += beyond != 0;
        }
        if (timed) {
            t_disk += dd;
            t_bus += db;
            t_dec += dc;
            bytes_read += r.bytes;
            if (is_key(r.type)) {
                tk_disk += dd;
                tk_bus += db;
                tk_dec += dc;
                key_bytes_read += r.bytes;
                ++n_key_read;
            }
        }
        // The fingerprint of a PSISTRM3 frame also covers its fine level: without it (--no-fine), it is not checked.
        if (!has_fine || fine) {
            unsigned long long hv = state_hash(g, rx.q0, rx.q, j & 1, rx.d_hash);
            if (fine) hv ^= fine_hash(rx.fs, j & 1, rx.d_hash);
            if (hv != r.hash) ++bad_hash;
        }
        if (j == probe_frame) {
            const size_t c = probe_count;
            if (fine && plain) probe_kernel<<<blocks_for(c), 256, 0, stream>>>(fine_plain(j & 1), d_pp, d_pp + c, d_pp + 2 * c, probe_count, d_pu, d_pu + c, d_pu + 2 * c);
            else if (fine) probe_kernel<<<blocks_for(c), 256, 0, stream>>>(fine_field(j & 1), d_pp, d_pp + c, d_pp + 2 * c, probe_count, d_pu, d_pu + c, d_pu + 2 * c);
            else if (plain) probe_kernel<<<blocks_for(c), 256, 0, stream>>>(plain_field(j & 1), d_pp, d_pp + c, d_pp + 2 * c, probe_count, d_pu, d_pu + c, d_pu + 2 * c);
            else probe_kernel<<<blocks_for(c), 256, 0, stream>>>(rx.field(j & 1), d_pp, d_pp + c, d_pp + 2 * c, probe_count, d_pu, d_pu + c, d_pu + 2 * c);
            std::vector<float> got(probe_u.size());
            CK(cudaMemcpy(got.data(), d_pu, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            probe_diff = 0.0;
            for (size_t i = 0; i < got.size(); ++i) probe_diff = std::max(probe_diff, (double)fabsf(got[i] - probe_u[i]));
        }
    };

    // Direct access: time to reach a given frame from the key frame that precedes it.
    double seek_ms = 0.0;
    int seek_decoded = 0;
    if (o.seek >= 0) {
        CHK(o.seek < sh.n_frames, "direct-access frame outside the file");
        int k0 = o.seek;
        while (!is_key(index[k0].type)) --k0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int j = k0; j <= o.seek; ++j) {
            ingest(j, false);
            ++seek_decoded;
        }
        seek_ms = 1e3 * seconds_since(t0);
    }

    const float dt = sh.dt_frame / o.substeps, dw = 1.0f / o.substeps;
    const float sigma = 2.0f / (3.0f * sh.u_rms * sh.u_rms * kL);
    if (o.sort0) {
        sort_particles();
        t_sort = 0.0;
        n_sort = 0;
    }
    bad_hash = 0;
    legacy_sums = 0;
    legacy_frames = 0;
    ingest(0, false);  // first key frame: initial state, outside the steady state
    const auto t_all = std::chrono::steady_clock::now();
    for (int j = 0; j + 1 < n_frames; ++j) {
        ingest(j + 1, true);
        if (o.resort > 0 && j > 0 && j % o.resort == 0) sort_particles();
        auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < o.substeps; ++s)
            if (fine && use_cache && plain)
                rk4_psi_fine<D, true><<<blocks_for(np, o.block), o.block, 0, stream>>>(rx.field(j & 1), rx.fs.level(D, rx.fine_delta), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (fine && use_cache)
                rk4_psi_fine<D, false><<<blocks_for(np, o.block), o.block, 0, stream>>>(rx.field(j & 1), rx.fs.level(D, rx.fine_delta), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (fine && plain)
                rk4_kernel<PsiFine<D, PsiPlain<D>>><<<blocks_for(np, o.block_plain), o.block_plain, 0, stream>>>(fine_plain(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (fine)
                rk4_kernel<PsiFine<D, PsiField<D>>><<<blocks_for(np, o.block_plain), o.block_plain, 0, stream>>>(fine_field(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (use_cache && plain) rk4_psi_plain<D><<<blocks_for(np, o.block), o.block, 0, stream>>>(rx.field(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (use_cache) rk4_psi<D><<<blocks_for(np, o.block), o.block, 0, stream>>>(rx.field(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else if (plain)
                rk4_kernel<PsiPlain<D>><<<blocks_for(np, o.block_plain), o.block_plain, 0, stream>>>(plain_field(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
            else rk4_kernel<PsiField<D>><<<blocks_for(np, o.block_plain), o.block_plain, 0, stream>>>(rx.field(j & 1), d_p[0], d_p[1], d_p[2], np, s * dw, dw, dt);
        CK(cudaStreamSynchronize(stream));
        t_adv += seconds_since(t0);
        if (o.raymarch > 0) {
            t0 = std::chrono::steady_clock::now();
            const size_t npx = (size_t)o.raymarch * o.raymarch;
            if (fine && plain)
                raymarch_kernel<PsiFine<D, PsiPlain<D>>><<<blocks_for(npx, o.block_plain), o.block_plain, 0, stream>>>(fine_plain(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            else if (fine)
                raymarch_kernel<PsiFine<D, PsiField<D>>><<<blocks_for(npx, o.block_plain), o.block_plain, 0, stream>>>(fine_field(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            else if (ray_cache && plain) raymarch_psi_plain<D><<<blocks_for(npx, o.block), o.block, 0, stream>>>(rx.field(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            else if (ray_cache) raymarch_psi<D><<<blocks_for(npx, o.block), o.block, 0, stream>>>(rx.field(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            else if (plain)
                raymarch_kernel<PsiPlain<D>><<<blocks_for(npx, o.block_plain), o.block_plain, 0, stream>>>(plain_field(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            else raymarch_kernel<PsiField<D>><<<blocks_for(npx, o.block_plain), o.block_plain, 0, stream>>>(rx.field(j & 1), d_img, o.raymarch, o.ray_steps, 0.5f, sigma);
            CK(cudaStreamSynchronize(stream));
            t_ray += seconds_since(t0);
        }
    }
    const double wall = seconds_since(t_all);
    CK(cudaGetLastError());

    unsigned long long pos_hash = 0;
    CK(cudaMemset(rx.d_hash, 0, sizeof(unsigned long long)));
    for (int c = 0; c < 3; ++c) hash_floats<<<blocks_for(np), 256>>>(d_p[c], np, (unsigned long long)(c + 1) << 60, rx.d_hash);
    CK(cudaMemcpy(&pos_hash, rx.d_hash, sizeof(pos_hash), cudaMemcpyDeviceToHost));
    if (!o.dump_positions.empty()) {  // final positions, three planes of np floats: to compare two kernels
        std::vector<float> hp((size_t)np);
        FILE* df = fopen(o.dump_positions.c_str(), "wb");
        CHK(df, "cannot create the positions file");
        for (int c = 0; c < 3; ++c) {
            CK(cudaMemcpy(hp.data(), d_p[c], (size_t)np * sizeof(float), cudaMemcpyDeviceToHost));
            CHK(fwrite(hp.data(), sizeof(float), hp.size(), df) == hp.size(), "writing of the positions");
        }
        fclose(df);
    }

    const int nf = n_frames - 1;
    const double mib = 1048576.0, steps = (double)nf * o.substeps;
    const double raw_frame = 3.0 * (double)sh.n * sh.n * sh.n * sizeof(float) / mib;
    const double disk_ms = 1e3 * t_disk / nf, bus_ms = 1e3 * t_bus / nf, dec_ms = 1e3 * t_dec / nf, step_ms = 1e3 * t_adv / steps;
    const double sort_ms = 1e3 * t_sort / nf, ray_ms = o.raymarch > 0 ? 1e3 * t_ray / nf : 0.0;
    const double frame_ms = disk_ms + bus_ms + dec_ms + sort_ms + o.substeps * step_ms + ray_ms;
    printf("play           : %s, %d^3, degree %d, %d frames read out of %d, %u particles, %d RK4 steps per frame\n", o.file.c_str(), sh.n, D, n_frames,
           sh.n_frames, np, o.substeps);
    if (sh.reserved[0] == kClipMarker) {
        unsigned long long tot = 0;
        for (int site = 0; site < kClipSites; ++site) tot += sh.reserved[1 + site];
        if (tot)
            printf("Warning        : the encoder clipped %llu coefficient(s) on %llu frame(s); this file does not reproduce the requested projection\n", tot,
                   (unsigned long long)(sh.reserved[7] & 0xFFFFFFFFull));
    }
    if (has_fine)
        printf("Fine level     : %s; tolerance %.2f %% of u', largest error observed at encoding %.2f %%; at most %u bricks and %u nodes per frame; "
               "%.1f MiB resident\n",
               fine ? "read" : "ignored (--no-fine): integrity not checked", 100.0 * fi.tolerance, 100.0 * fi.max_error,
               fi.max_bricks, fi.max_nodes, fine ? (rx.fs.resident_bytes() + rx.fsym_bytes) / 1048576.0 : 0.0);
    if (has_fine) {
        if (fi.frames_over)
            printf("Warning        : the encoder missed the bound on %u frame(s); largest error %.6f u' for a certification threshold of %.6f u' "
                   "(tolerance %.5f u')\n",
                   fi.frames_over, fi.max_error, fine_certified(fi.tolerance, fine_margin(fi)), fi.tolerance);
        else if (fine_margin(fi) > 0.0f)
            printf("Certification  : bound held at encoding; largest error %.6f u' for a certification threshold of %.6f u' (tolerance %.5f u', "
                   "arithmetic margin %.1e u')\n",
                   fi.max_error, fine_certified(fi.tolerance, fine_margin(fi)), fi.tolerance, fine_margin(fi));
        else
            printf("Certification  : bound held at encoding, with no arithmetic margin (file from before the evening of 15 November, or --tolerance-margin 0)\n");
    }
    if (legacy) {
        if (legacy_sums)
            printf("Warning        : PSISTRM1 file: %lld sum(s) of the pyramid outside +-32767 on %d frame(s); the original reader of this format clipped "
                   "them, the field rendered here differs from its own\n",
                   legacy_sums, legacy_frames);
        else
            printf("PSISTRM1       : no sum of the pyramid outside +-32767 on the %d frames read: the lattice is the one that the original reader of "
                   "this format rendered, which clipped at this bound\n",
                   n_frames);
    }
    if (probe_frame >= 0)
        printf("Probe          : frame %d, %u points; largest difference between the field of the reader and that of the encoder: %.3e%s\n", probe_frame + sh.first_frame,
               probe_count, probe_diff, probe_diff < 0.0 ? " (frame not read)" : "");
    printf("Kernels        : RK4 step %s support cache (blocks of %d threads)%s; lattice read %s\n", use_cache ? "with" : "without",
           use_cache ? o.block : o.block_plain, o.raymarch > 0 ? (ray_cache ? "; rays with cache" : "; rays without cache") : "",
           plain ? "directly" : "by differences");
    printf("File           : %.3f MiB; read in steady state: %.4f MiB per frame, that is %.0fx less than a raw frame (%.0f MiB)\n", file_bytes / mib,
           bytes_read / mib / nf, raw_frame / (bytes_read / mib / nf), raw_frame);
    printf("Per frame      : disk %.3f ms, bus %.3f ms, decoding %.3f ms, sort %.3f ms, %d RK4 steps of %.3f ms%s\n", disk_ms, bus_ms, dec_ms, sort_ms,
           o.substeps, step_ms, o.raymarch > 0 ? ", one image" : "");
    const double kn = n_key_read ? (double)n_key_read : 1.0;
    if (n_key_read)
        printf("Key frames     : %d read in steady state (%s), %.3f MiB each: disk %.3f ms, bus %.3f ms, decoding %.3f ms\n", n_key_read,
               index[0].type == 3 ? "coded" : "raw", key_bytes_read / mib / kn, 1e3 * tk_disk / kn, 1e3 * tk_bus / kn, 1e3 * tk_dec / kn);
    if (o.raymarch > 0) printf("Ray marching   : %.3f ms per image (%d^2 rays x %d samples)\n", ray_ms, o.raymarch, o.ray_steps);
    printf("Full budget    : %.3f ms per frame, that is %.1f frames per second (wall clock: %.3f ms per frame)\n", frame_ms, 1e3 / frame_ms,
           1e3 * wall / nf);
    printf("Sort           : %d sorts during the reading, %.3f ms each\n", n_sort, n_sort ? 1e3 * t_sort / n_sort : 0.0);
    printf("GPU memory     : %.1f MiB allocated by the reader, of which %.1f MiB of receiver state; lattice at %p (remainder modulo 2 MiB: %llu KiB)\n",
           (free0 - free1) / mib, rx.resident_bytes() / mib, (void*)rx.lat, (unsigned long long)((uintptr_t)rx.lat % (2u << 20)) / 1024ull);
    print_seal(so);
    printf("Integrity      : %d frame(s) whose state differs from that of the sender; fingerprint of the final positions %016llx\n", bad_hash, pos_hash);
    if (o.seek >= 0) printf("Direct access  : frame %d reached in %.2f ms (%d frames decoded from the key)\n", o.seek, seek_ms, seek_decoded);
    if (o.csv)
        printf("CSV,play,%d,%d,%u,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%d,%016llx,%d,%.4f,%.4f,%.4f,%.4f\n", sh.n, D, np, o.substeps,
               o.sort0 ? 1 : 0, o.resort, bytes_read / mib / nf, disk_ms, bus_ms, dec_ms, sort_ms, step_ms, ray_ms, frame_ms, 1e3 * wall / nf,
               (free0 - free1) / mib, rx.resident_bytes() / mib, bad_hash, pos_hash, n_key_read, key_bytes_read / mib / kn, 1e3 * tk_disk / kn,
               1e3 * tk_bus / kn, 1e3 * tk_dec / kn);
    return bad_hash ? 1 : 0;
}

int main(int argc, char** argv) {
    const SOptions o = sparse(argc, argv);
    if (o.mode == "pack") {
        CHK(o.degree == 2 || o.degree == 3, "--degree is 2 or 3");
        return o.degree == 3 ? pack<3>(o) : pack<2>(o);
    }
    CHK(o.mode == "play", "unknown mode (pack|play)");
    FILE* in = fopen(o.file.c_str(), "rb");
    CHK(in, ".psi file not found");
    StreamHeader sh{};
    CHK(fread(&sh, sizeof(sh), 1, in) == 1, "invalid .psi file");
    check_stream_header(sh);
    StreamOpen so;
    read_stream_index(in, sh, so);  // header of the fine level, frame table, seal
    CHK(so.sealed || !o.require_seal, "file without a seal, refused by --require-seal");
    const int rc = sh.degree == 3 ? play<3>(o, sh, so, in) : play<2>(o, sh, so, in);
    fclose(in);
    return rc;
}
