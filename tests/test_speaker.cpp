// Speaker encoder gate: voice_1.wav -> mel (vs speaker_mels.npy) -> ECAPA (vs speaker_embedding.npy).
// Usage: test_speaker <model.navi> <reference dir>

#include "model/qwen3tts/params.h"
#include "model/qwen3tts/speaker.h"
#include "runtime/audio/wav.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct Cmp { double max_abs = 0, rel_rms = 0, cosine = 0; };
static Cmp compare(const std::vector<float> & a, const std::vector<float> & b) {
    Cmp c; double se = 0, ss = 0, dot = 0, aa = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double d = a[i] - b[i];
        c.max_abs = std::max(c.max_abs, std::fabs(d)); se += d * d; ss += static_cast<double>(b[i]) * b[i];
        dot += static_cast<double>(a[i]) * b[i]; aa += static_cast<double>(a[i]) * a[i];
    }
    c.rel_rms = std::sqrt(se / ss); c.cosine = dot / std::sqrt(aa * ss);
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        const auto wav = navi::audio::read_wav(ref + "/voice_1.wav");
        const auto mels_npy = navi::Npy::load(ref + "/speaker_mels.npy");
        const auto want_mel = mels_npy.as_f32();
        const auto want_emb = navi::Npy::load(ref + "/speaker_embedding.npy").as_f32();
        const int T_ref = static_cast<int>(mels_npy.shape[1]);

        navi::Device dev = navi::Device::open();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        if (wav.sample_rate != params.spk_sample_rate) { std::printf("FAIL: voice_1.wav is %d Hz\n", wav.sample_rate); return 1; }
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) { return t.name.rfind("speaker_encoder.", 0) == 0; });
        file.close();
        auto enc = navi::qwen3tts::SpeakerEncoder::create(dev, w, params);

        int T = 0;
        const auto mel = navi::qwen3tts::SpeakerEncoder::mel(wav.pcm, params, &T);
        std::printf("mel: %d frames (reference %d), %zu samples in\n", T, T_ref, wav.pcm.size());
        if (T != T_ref) { std::printf("FAIL: frame count\n"); return 1; }
        const Cmp cm = compare(mel, want_mel);
        std::printf("  log-mel vs torch    max|d| %.3e  rel-rms %.2e\n", cm.max_abs, cm.rel_rms);

        const auto emb_ref_mel = enc->embed_mel(want_mel, T_ref);
        const double ms1 = enc->last_ms();
        const auto emb_ref_mel2 = enc->embed_mel(want_mel, T_ref);
        const Cmp ce = compare(emb_ref_mel, want_emb);
        std::printf("  ECAPA on torch mel  max|d| %.3e  rel-rms %.2e  cosine %.6f   (%.1f ms, then %.1f ms)\n",
                    ce.max_abs, ce.rel_rms, ce.cosine, ms1, enc->last_ms());
        const auto emb = enc->embed(wav.pcm);
        const Cmp cf = compare(emb, want_emb);
        std::printf("  full path           max|d| %.3e  rel-rms %.2e  cosine %.6f\n", cf.max_abs, cf.rel_rms, cf.cosine);
        (void) emb_ref_mel2;

        const bool ok = cm.rel_rms < 1e-4 && ce.rel_rms < 2e-3 && cf.rel_rms < 2e-3 && cf.cosine > 0.99999;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
