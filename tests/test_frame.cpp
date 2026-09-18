// Frame kernel gate: prefill, then greedy frames against tests/reference/greedy_codes.npy
// (argmax, repetition penalty 1.05, EOS suppressed for 2 frames - the reference's settings).
// Reports how many leading frames match all 16 codes and the per-frame device time.
// Usage: test_frame <model.navi> <reference dir>

#include "model/qwen3tts/frame.h"
#include "model/qwen3tts/params.h"
#include "model/qwen3tts/talker.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        const auto ids = navi::Npy::load(ref + "/text_ids.npy").as_i32();
        const auto spk = navi::Npy::load(ref + "/speaker_embedding.npy").as_f32();
        const auto greedy_npy = navi::Npy::load(ref + "/greedy_codes.npy");
        const auto greedy = greedy_npy.as_i32();
        const int n_ref = static_cast<int>(greedy_npy.shape[0]);

        navi::Device dev = navi::Device::open();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) {
            return t.name.rfind("talker.", 0) == 0;
        }, navi::qwen3tts::Talker::fused_groups(params));
        file.close();
        auto talker = navi::qwen3tts::Talker::create(dev, w, params);
        auto frame = navi::qwen3tts::Frame::create(dev, w, params);
        std::printf("frame kernel: grid %d blocks (occupancy %d/MP, %d MPs)\n", frame->grid_blocks(),
                    frame->occupancy_per_mp(), dev.info().multiprocessors);

        const auto & pr = talker->build_prompt(ids, "english", spk);
        talker->prefill(pr);
        const int H = params.talker.hidden;

        navi::qwen3tts::Sampling s;
        s.temperature = 0.f; s.cp_temperature = 0.f; s.repetition_penalty = 1.05f; s.min_frames = 2;
        frame->reset();

        int matched_frames = 0, first_bad = -1;
        double total_us = 0, talker_us = 0;
        std::vector<int> per_frame_matches;
        for (int i = 0; i < n_ref; ++i) {
            const float * trailing = i < pr.n_trailing ? pr.trailing + static_cast<std::size_t>(i) * H : pr.tts_pad;
            navi::qwen3tts::FrameResult r = i == 0
                ? frame->run_first(talker->d_hidden(), talker->d_logits(), trailing, s, 0, i)
                : frame->run(talker->kv(), pr.n_prefill + i - 1, trailing, s, 0, i);
            if (!r.ok) { std::printf("frame %d FAILED: %s\n", i, r.error.c_str()); return 1; }
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

        // The barrier's deadline (CLAUDE.md: a cooperative kernel must never be able
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

        // gate (DESIGN 6): the first frames match code-for-code; a later divergence in a greedy
        // chain is a near-tie flipping under f16 KV / accumulation order, not a bug by itself.
        const bool ok = matched_frames >= 2 && det_ok && timeout_ok;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
