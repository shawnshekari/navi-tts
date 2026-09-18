// Vocoder gate: decode tests/reference/sampled_codes.npy and compare with the
// PyTorch float32 PCM (sampled_pcm.npy); also chunked (8 frames) vs one-shot.
// Usage: test_vocoder <model.navi> <reference dir> [out.wav]

#include "model/qwen3tts/params.h"
#include "model/qwen3tts/vocoder.h"
#include "runtime/audio/wav.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

struct Cmp { double max_abs = 0, rms_err = 0, snr_db = 0; std::size_t argmax = 0; };

static Cmp compare(const std::vector<float> & a, const std::vector<float> & b) {
    Cmp c;
    double se = 0, ss = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > c.max_abs) { c.max_abs = d; c.argmax = i; }
        se += d * d;
        ss += static_cast<double>(b[i]) * b[i];
    }
    c.rms_err = std::sqrt(se / n);
    c.snr_db = 10 * std::log10(ss / (se > 0 ? se : 1e-30));
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir [out.wav]\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        const auto codes_npy = navi::Npy::load(ref + "/sampled_codes.npy");
        const auto pcm_npy = navi::Npy::load(ref + "/sampled_pcm.npy");
        const auto codes = codes_npy.as_i32();
        const auto want = pcm_npy.as_f32();
        std::vector<float> want16;   // filled once the blob's codec dtype is known
        const int n_frames = static_cast<int>(codes_npy.shape[0]);
        std::printf("reference: %d frames, %zu samples\n", n_frames, want.size());

        navi::Device dev = navi::Device::open();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        // kernel exactness is judged against torch run at the blob's own weight precision
        const bool codec_f32 = file.has_kv("general.codec_dtype") && file.kv_str("general.codec_dtype") == "f32";
        want16 = codec_f32 ? want : navi::Npy::load(ref + "/sampled_pcm_f16w.npy").as_f32();
        std::printf("codec weights: %s\n", codec_f32 ? "f32" : "f16");
        const std::string dec = "codec.decoder.";
        auto select = [&](const navi::TensorInfo & t) {
            return t.name.compare(0, dec.size(), dec) == 0 && !navi::qwen3tts::vocoder_owns_tensor(t.name);
        };
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, select);
        auto voc = navi::qwen3tts::Vocoder::create(dev, file, w, params, /*max_chunk*/ 128);
        auto voc8 = navi::qwen3tts::Vocoder::create(dev, file, w, params, /*max_chunk*/ 8);
        file.close();

        if (const char * dir = std::getenv("NAVI_VOCODER_STAGES")) {
            // one-shot only, every stage to <dir>/<name>.npy for tools-side comparison
            std::string d(dir);
            voc->set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
                std::vector<float> h(static_cast<std::size_t>(rows) * cols);
                (void) hipMemcpy(h.data(), buf, h.size() * 4, hipMemcpyDeviceToHost);
                navi::Npy::save_f32(d + "/" + name + ".npy", h.data(), {static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)});
            });
            std::vector<float> tmp;
            voc->decode(codes, n_frames, tmp);
            voc->set_stage_hook(nullptr);
            voc->reset();
            std::printf("stages written to %s\n", dir);
        }
        std::vector<float> one, chunked;
        voc->decode(codes, n_frames, one);        // warm-up + one shot
        voc->reset(); one.clear();
        voc->decode(codes, n_frames, one);
        const double ms_one = voc->last_ms();
        voc8->decode(codes, n_frames, chunked);
        const double ms_chunk = voc8->last_ms();
        if (one.size() != want.size() || chunked.size() != want.size()) {
            std::printf("FAIL: got %zu / %zu samples, want %zu\n", one.size(), chunked.size(), want.size());
            return 1;
        }
        const Cmp a = compare(one, want), k = compare(one, want16), b = compare(chunked, want), c = compare(chunked, one);
        std::printf("one-shot vs torch %s: max|d| %.3e at %zu, rms %.3e, SNR %.1f dB   <- kernel exactness\n",
                    codec_f32 ? "f32 " : "f16w", k.max_abs, k.argmax, k.rms_err, k.snr_db);
        std::printf("one-shot vs torch f32 : max|d| %.3e at %zu, rms %.3e, SNR %.1f dB   <- weight budget  (%.2f ms, %.3f ms/frame)\n",
                    a.max_abs, a.argmax, a.rms_err, a.snr_db, ms_one, ms_one / n_frames);
        std::printf("chunked8 vs torch f32 : max|d| %.3e, rms %.3e, SNR %.1f dB   (%.2f ms, %.3f ms/frame)\n",
                    b.max_abs, b.rms_err, b.snr_db, ms_chunk, ms_chunk / n_frames);
        std::printf("chunked8 vs one-shot  : max|d| %.3e, rms %.3e\n", c.max_abs, c.rms_err);
        if (argc > 3) { navi::audio::write_wav(argv[3], one, params.codec_sample_rate); std::printf("wrote %s\n", argv[3]); }

        // gates (DESIGN 6): kernels exact against the same-precision reference
        // (2e-5), chunked == one-shot (1e-5), and the f16-weight budget vs f32
        // held at >= 60 dB SNR - the 1e-4 absolute figure needs f32 weights.
        const bool ok_kernel = k.max_abs <= 2e-5;
        const bool ok_chunk = c.max_abs <= 1e-5;
        const bool ok_budget = a.snr_db >= 60;
        const bool ok = ok_kernel && ok_chunk && ok_budget;
        std::printf("%s (kernel %s, chunked %s, budget %s)\n", ok ? "PASS" : "FAIL", ok_kernel ? "ok" : "FAIL",
                    ok_chunk ? "ok" : "FAIL", ok_budget ? "ok" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
