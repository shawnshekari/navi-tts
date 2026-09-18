// Prompt + prefill gate against tests/reference: prefill_embeds, trailing_text,
// tts_pad_embed, prefill_kv (all layers), prefill_hidden, prefill_logits.
// Usage: test_prefill <model.navi> <reference dir>

#include "model/qwen3tts/params.h"
#include "model/qwen3tts/talker.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct Cmp { double max_abs = 0, ref_max = 0, rel_rms = 0; std::size_t argmax = 0; };

static Cmp compare(const std::vector<float> & a, const std::vector<float> & b) {
    Cmp c;
    double se = 0, ss = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > c.max_abs) { c.max_abs = d; c.argmax = i; }
        c.ref_max = std::max(c.ref_max, static_cast<double>(std::fabs(b[i])));
        se += d * d;
        ss += static_cast<double>(b[i]) * b[i];
    }
    c.rel_rms = std::sqrt(se / (ss > 0 ? ss : 1e-30));
    return c;
}

static std::vector<float> download(const float * d, std::size_t n) {
    std::vector<float> h(n);
    if (hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost) != hipSuccess) { std::fprintf(stderr, "hipMemcpy failed\n"); std::exit(2); }
    return h;
}

static std::vector<float> download_f16(const unsigned short * d, std::size_t n) {
    std::vector<unsigned short> h(n);
    if (hipMemcpy(h.data(), d, n * 2, hipMemcpyDeviceToHost) != hipSuccess) { std::fprintf(stderr, "hipMemcpy failed\n"); std::exit(2); }
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        _Float16 f;
        __builtin_memcpy(&f, &h[i], 2);
        out[i] = static_cast<float>(f);
    }
    return out;
}

static void report(const char * what, const Cmp & c) {
    std::printf("  %-16s max|d| %.3e (ref max %.3e, rel %.2e)  rel-rms %.2e\n", what, c.max_abs, c.ref_max,
                c.ref_max > 0 ? c.max_abs / c.ref_max : 0.0, c.rel_rms);
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        const auto ids = navi::Npy::load(ref + "/text_ids.npy").as_i32();
        const auto spk = navi::Npy::load(ref + "/speaker_embedding.npy").as_f32();
        const auto want_prefill = navi::Npy::load(ref + "/prefill_embeds.npy");
        const auto want_trailing = navi::Npy::load(ref + "/trailing_text.npy");
        const auto want_pad = navi::Npy::load(ref + "/tts_pad_embed.npy").as_f32();
        const auto want_hidden = navi::Npy::load(ref + "/prefill_hidden.npy").as_f32();
        const auto want_logits = navi::Npy::load(ref + "/prefill_logits.npy").as_f32();
        const auto want_kv = navi::Npz::load(ref + "/prefill_kv.npz");
        const auto frame0 = navi::Npy::load(ref + "/frame0_codes.npy").as_i32();

        navi::Device dev = navi::Device::open();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) {
            return t.name.rfind("talker.model.", 0) == 0 || t.name.rfind("talker.text_projection.", 0) == 0 ||
                   t.name == "talker.codec_head.weight";
        }, navi::qwen3tts::Talker::fused_groups(params));
        file.close();
        auto talker = navi::qwen3tts::Talker::create(dev, w, params);

        const auto & pr = talker->build_prompt(ids, "english", spk);
        const int H = params.talker.hidden;
        std::printf("prompt: %d prefill rows, %d trailing rows (reference %zu / %zu)\n", pr.n_prefill, pr.n_trailing,
                    want_prefill.shape[0], want_trailing.shape[0]);
        if (pr.n_prefill != static_cast<int>(want_prefill.shape[0]) || pr.n_trailing != static_cast<int>(want_trailing.shape[0])) {
            std::printf("FAIL: prompt shape\n");
            return 1;
        }
        bool ok = true;
        {
            const Cmp a = compare(download(pr.prefill, static_cast<std::size_t>(pr.n_prefill) * H), want_prefill.as_f32());
            const Cmp b = compare(download(pr.trailing, static_cast<std::size_t>(pr.n_trailing) * H), want_trailing.as_f32());
            const Cmp c = compare(download(pr.tts_pad, static_cast<std::size_t>(H)), want_pad);
            report("prefill embeds", a); report("trailing text", b); report("tts_pad", c);
            ok = ok && a.rel_rms < 5e-3 && b.rel_rms < 5e-3 && c.rel_rms < 5e-3;
        }

        talker->prefill(pr);
        talker->prefill(pr);   // second run for timing without first-launch cost
        std::printf("prefill: %.2f ms for %d positions\n", talker->last_ms(), pr.n_prefill);

        const auto & kv = talker->kv();
        const std::size_t kv_row = static_cast<std::size_t>(kv.kv_dim);
        Cmp worst_k, worst_v;
        int worst_kl = 0, worst_vl = 0;
        for (int l = 0; l < kv.n_layer; ++l) {
            const auto & wk = want_kv.at("k_" + std::to_string(l));
            const auto & wv = want_kv.at("v_" + std::to_string(l));
            const std::size_t n = static_cast<std::size_t>(pr.n_prefill) * kv_row;
            const Cmp ck = compare(download_f16(kv.k[static_cast<std::size_t>(l)], n), wk.as_f32());
            const Cmp cv = compare(download_f16(kv.v[static_cast<std::size_t>(l)], n), wv.as_f32());
            if (ck.rel_rms > worst_k.rel_rms) { worst_k = ck; worst_kl = l; }
            if (cv.rel_rms > worst_v.rel_rms) { worst_v = cv; worst_vl = l; }
        }
        std::printf("kv cache, worst layers: K layer %d, V layer %d\n", worst_kl, worst_vl);
        report("K", worst_k); report("V", worst_v);
        const Cmp ch = compare(download(talker->d_hidden(), static_cast<std::size_t>(H)), want_hidden);
        const auto logits = download(talker->d_logits(), static_cast<std::size_t>(params.talker.vocab));
        const Cmp cl = compare(logits, want_logits);
        report("hidden", ch); report("logits", cl);

        // greedy cb0 with the suppression window (ids >= vocab-1024 except eos)
        int best = -1;
        for (int i = 0; i < params.talker.vocab; ++i) {
            if (i >= params.talker.vocab - 1024 && i != params.codec_eos_id) continue;
            if (best < 0 || logits[static_cast<std::size_t>(i)] > logits[static_cast<std::size_t>(best)]) best = i;
        }
        std::printf("  greedy cb0       %d (reference %d)\n", best, frame0[0]);

        // Talker weights are exact (bf16 is a subset of f16), so the only error sources are
        // the f16 KV store (~3e-4 rel-rms by itself) and accumulation order. Gates: KV 1e-3,
        // hidden and logits 5e-4 rel-rms (DESIGN 6: 0.02% / 0.05% of peak), greedy cb0 exact.
        ok = ok && worst_k.rel_rms < 1e-3 && worst_v.rel_rms < 1e-3 && ch.rel_rms < 5e-4 && cl.rel_rms < 5e-4 &&
             ch.max_abs / ch.ref_max < 2e-4 && cl.max_abs / cl.ref_max < 5e-4 && best == frame0[0];
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
