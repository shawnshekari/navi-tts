// Frame kernel gate: prefill, then greedy frames against tests/reference/greedy_codes.npy
// (argmax, repetition penalty 1.05, EOS suppressed for 2 frames - the reference's settings).
// Reports how many leading frames match all 16 codes and the per-frame device time.
// Also compares the code predictor's 16 inputs for frame 0 against cp_inputs.npy: on a
// model whose talker and predictor widths differ that is the only float-level check of
// the small_to_mtp_projection stage, and matching codes do not stand in for it (a missing
// bias or a modest projection error can leave every argmax winner intact).
// Usage: test_frame <model.navi> <reference dir>

#include "model/qwen3tts/frame.h"
#include "model/qwen3tts/params.h"
#include "model/qwen3tts/talker.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Same shape of report as test_prefill: max|d| against the reference's own scale, and
// the relative RMS that the gate is set on. `finite` covers both sides and the metrics:
// a non-finite value anywhere makes rel_rms a NaN, and every comparison against a NaN
// is false, so a row like that would never be picked as the worst one and would pass
// a bare threshold test unnoticed.
struct Cmp { double max_abs = 0, ref_max = 0, rel_rms = 0; bool finite = true; };

static Cmp compare(const float * a, const std::vector<float> & b, std::size_t off, std::size_t n) {
    Cmp c;
    double se = 0, ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = a[i], y = b[off + i];
        if (!std::isfinite(x) || !std::isfinite(y)) c.finite = false;
        const double d = std::fabs(x - y);
        c.max_abs = std::max(c.max_abs, d);
        c.ref_max = std::max(c.ref_max, std::fabs(y));
        se += d * d;
        ss += y * y;
    }
    c.rel_rms = std::sqrt(se / (ss > 0 ? ss : 1e-30));
    if (!std::isfinite(c.rel_rms) || !std::isfinite(c.max_abs)) c.finite = false;
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        const auto ids = navi::Npy::load(ref + "/text_ids.npy").as_i32();
        const auto spk = navi::Npy::load(ref + "/speaker_embedding.npy").as_f32();
        const auto greedy_npy = navi::Npy::load(ref + "/greedy_codes.npy");
        const auto greedy = greedy_npy.as_i32();
        const int n_ref = static_cast<int>(greedy_npy.shape[0]);
        const auto cp_in_npy = navi::Npy::load(ref + "/cp_inputs.npy");
        const auto want_cp_in = cp_in_npy.as_f32();

        navi::Device dev = navi::Device::open();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) {
            return t.name.rfind("talker.", 0) == 0;
        }, navi::qwen3tts::Talker::fused_groups(params));
        file.close();
        auto talker = navi::qwen3tts::Talker::create(dev, w, params);
        auto frame = navi::qwen3tts::Frame::create(dev, w, params);
        std::printf("frame kernel: grid %d blocks (occupancy %d/MP, %d MPs), %zu B LDS/block\n", frame->grid_blocks(),
                    frame->occupancy_per_mp(), dev.info().multiprocessors, frame->lds_bytes());
        const bool projects = params.cp.hidden != params.talker.hidden;
        std::printf("code predictor: hidden %d, inputs stored at the talker's %d -> %s\n", params.cp.hidden,
                    params.talker.hidden, projects ? "small_to_mtp_projection ACTIVE" : "widths equal, no projection");
        if (cp_in_npy.shape.size() != 2 || cp_in_npy.shape[0] != 16 ||
            cp_in_npy.shape[1] != static_cast<std::size_t>(frame->cp_hidden())) {
            std::printf("FAIL: cp_inputs.npy is [%zu, %zu], want [16, %d]\n",
                        cp_in_npy.shape.size() > 0 ? cp_in_npy.shape[0] : 0,
                        cp_in_npy.shape.size() > 1 ? cp_in_npy.shape[1] : 0, frame->cp_hidden());
            return 1;
        }
        frame->enable_cp_input_capture();

        const auto & pr = talker->build_prompt(ids, "english", spk);
        talker->prefill(pr);
        const int H = params.talker.hidden;

        navi::qwen3tts::Sampling s;
        s.temperature = 0.f; s.cp_temperature = 0.f; s.repetition_penalty = 1.05f; s.min_frames = 2;
        frame->reset();

        int matched_frames = 0, first_bad = -1;
        double total_us = 0, talker_us = 0;
        std::vector<int> per_frame_matches;
        Cmp cp_worst;
        int cp_worst_pos = 0;
        bool cp_rows_ok = true;   // every row, not just the one reported as worst
        for (int i = 0; i < n_ref; ++i) {
            const float * trailing = i < pr.n_trailing ? pr.trailing + static_cast<std::size_t>(i) * H : pr.tts_pad;
            navi::qwen3tts::FrameResult r = i == 0
                ? frame->run_first(talker->d_hidden(), talker->d_logits(), trailing, s, 0, i)
                : frame->run(talker->kv(), pr.n_prefill + i - 1, trailing, s, 0, i);
            if (!r.ok) { std::printf("frame %d FAILED: %s\n", i, r.error.c_str()); return 1; }
            if (i == 0) {
                // Straight after frame 0: every later frame overwrites the capture.
                const int CH = frame->cp_hidden();
                std::vector<float> got(static_cast<std::size_t>(16) * CH);
                if (hipMemcpy(got.data(), frame->d_cp_inputs(), got.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
                    std::printf("FAIL: could not read the cp input capture\n");
                    return 1;
                }
                for (int pos = 0; pos < 16; ++pos) {
                    const std::size_t off = static_cast<std::size_t>(pos) * CH;
                    const Cmp c = compare(got.data() + off, want_cp_in, off, static_cast<std::size_t>(CH));
                    const bool row_ok = c.finite && c.rel_rms < 5e-3;
                    if (!row_ok) {
                        cp_rows_ok = false;
                        std::printf("cp inputs, position %d: rel-rms %.2e%s  OVER THE GATE\n", pos,
                                    c.rel_rms, c.finite ? "" : " (NON-FINITE)");
                    }
                    // a non-finite row outranks any finite one, and is never displaced by one
                    const bool worse = !c.finite || (cp_worst.finite && c.rel_rms > cp_worst.rel_rms);
                    if (worse) { cp_worst = c; cp_worst_pos = pos; }
                }
                std::printf("cp inputs, frame 0, worst of 16 positions (pos %d): max|d| %.3e (ref max %.3e, rel %.2e)"
                            "  rel-rms %.2e%s\n", cp_worst_pos, cp_worst.max_abs, cp_worst.ref_max,
                            cp_worst.ref_max > 0 ? cp_worst.max_abs / cp_worst.ref_max : 0.0, cp_worst.rel_rms,
                            cp_worst.finite ? "" : "  NON-FINITE");
            }
            total_us += r.us;
            if (i > 0) talker_us += r.us;
            int m = 0;
            for (int c = 0; c < 16; ++c) m += r.codes[c] == greedy[static_cast<std::size_t>(i) * 16 + c];
            per_frame_matches.push_back(m);
            if (m == 16 && first_bad < 0) matched_frames = i + 1;
            else if (first_bad < 0) first_bad = i;
            if (i < 4 || m != 16) {
                std::printf("frame %2d: %2d/16 match  %.0f us  cb0 %d (ref %d)%s\n", i, m, r.us, r.codes[0],
                            greedy[static_cast<std::size_t>(i) * 16], r.eos ? "  EOS" : "");
            }
            if (r.eos) break;
        }
        std::printf("leading frames fully matched: %d of %d (first divergence at frame %d)\n", matched_frames, n_ref, first_bad);
        std::printf("frame time: %.0f us mean over %zu frames (frames with a talker step: %.0f us mean)\n",
                    total_us / per_frame_matches.size(), per_frame_matches.size(),
                    per_frame_matches.size() > 1 ? talker_us / (per_frame_matches.size() - 1) : 0.0);
        // Sampled path, the model's defaults: same seed twice must be bit-identical
        // (DESIGN 1 non-goals / 5.1), a different seed must differ.
        auto sampled_run = [&](std::uint64_t seed, int max_frames) {
            std::vector<std::int32_t> out;
            talker->prefill(pr);
            frame->reset();
            navi::qwen3tts::Sampling d;   // defaults = generation_config.json
            for (int i = 0; i < max_frames; ++i) {
                const float * trailing = i < pr.n_trailing ? pr.trailing + static_cast<std::size_t>(i) * H : pr.tts_pad;
                navi::qwen3tts::FrameResult r = i == 0
                    ? frame->run_first(talker->d_hidden(), talker->d_logits(), trailing, d, seed, i)
                    : frame->run(talker->kv(), pr.n_prefill + i - 1, trailing, d, seed, i);
                if (!r.ok) { std::printf("sampled frame %d FAILED: %s\n", i, r.error.c_str()); std::exit(1); }
                if (r.eos) break;
                out.insert(out.end(), r.codes, r.codes + 16);
            }
            return out;
        };
        const auto a = sampled_run(2, 200), b = sampled_run(2, 200), c = sampled_run(3, 200);
        std::printf("sampled seed 2: %zu frames, run-to-run %s; seed 3: %zu frames, %s\n", a.size() / 16,
                    a == b ? "bit-identical" : "DIFFERENT", c.size() / 16, a == c ? "identical to seed 2 (!)" : "differs");
        const bool det_ok = a == b && a != c && !a.empty();

        // The barrier's deadline (a cooperative kernel must never be able
        // to hang the box): with a spin cap far below a frame's ~700 barriers' worth
        // of waiting, the frame must come back as a failure, promptly, and a normal
        // Frame must still work afterwards.
        bool timeout_ok = false;
        {
            navi::qwen3tts::FrameOptions tight;
            tight.spin_cap = 1;
            auto tf = navi::qwen3tts::Frame::create(dev, w, params, tight);
            talker->prefill(pr);
            tf->reset();
            const auto t0 = std::chrono::steady_clock::now();
            navi::qwen3tts::FrameResult r = tf->run_first(talker->d_hidden(), talker->d_logits(), pr.trailing, s, 0, 0);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            timeout_ok = !r.ok && r.error.find("timed out") != std::string::npos && ms < 2000.0;
            std::printf("spin cap 1: %s in %.0f ms (%s)\n", r.ok ? "frame SUCCEEDED (!)" : "failed cleanly", ms, r.error.c_str());
            talker->prefill(pr);
            frame->reset();
            r = frame->run_first(talker->d_hidden(), talker->d_logits(), pr.trailing, s, 0, 0);
            if (!r.ok) { std::printf("frame after the timeout FAILED: %s\n", r.error.c_str()); timeout_ok = false; }
        }

        // The LDS score vector bounds the position as well as the KV cache does: attend()
        // keeps one score per key position, so a position past the vector would write off
        // its end. Neither bound is reachable through Talker's default max_positions, so
        // exercise it with a cache that claims more room than the kernel's vector has.
        // Both must be refused before anything is launched or written.
        bool bound_ok = false;
        {
            navi::qwen3tts::KvCache fake = talker->kv();   // same device buffers, a bigger claim
            fake.max_pos = frame->max_position() + 16;
            const auto over = frame->run(fake, frame->max_position(), pr.trailing, s, 0, 1);
            const auto neg = frame->run(fake, -1, pr.trailing, s, 0, 1);
            const auto has = [](const navi::qwen3tts::FrameResult & r) {
                return !r.ok && r.error.find("score vector") != std::string::npos;
            };
            bound_ok = has(over) && has(neg);
            std::printf("position bound (vector %d): at the limit %s, negative %s\n", frame->max_position(),
                        has(over) ? "refused" : "ACCEPTED (!)", has(neg) ? "refused" : "ACCEPTED (!)");
            talker->prefill(pr);
            frame->reset();
            const auto after = frame->run_first(talker->d_hidden(), talker->d_logits(), pr.trailing, s, 0, 0);
            if (!after.ok) { std::printf("frame after the bound checks FAILED: %s\n", after.error.c_str()); bound_ok = false; }
        }

        // gate (DESIGN 6): the first frames match code-for-code; a later divergence in a greedy
        // chain is a near-tie flipping under f16 KV / accumulation order, not a bug by itself.
        // cp inputs: one f16 embedding row (or the f32 hidden at position 0) through one f16
        // GEMV and an f32 bias against an f32 reference, so the error class is the same weight
        // quantisation the prefill gate allows 5e-3 rel-rms for (test_prefill.cpp "prefill embeds").
        const bool cp_ok = cp_rows_ok && cp_worst.finite && cp_worst.rel_rms < 5e-3;
        if (!cp_ok) std::printf("cp inputs FAILED the 5e-3 rel-rms gate\n");
        const bool ok = matched_frames >= 2 && det_ok && timeout_ok && cp_ok && bound_ok;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
