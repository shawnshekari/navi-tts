// In-context cloning through the whole graph, with STORED codes: the path a voice cloned
// on one box and served on another takes. Nothing here touches the codec encoder, so it
// runs on a blob converted without one - that independence is the point.
//
// Also the generation-position policy: the prompt plus the generated frames have to fit the
// KV cache, and synth() clamps max_frames to what is left rather than failing after
// streaming audio. The cap and the refusal are both unreachable at the shipped sizes, so
// this builds graphs with a small cache to reach them, and derives the clamp from the
// ENGINE's own frame count at that seed - not from PyTorch's, whose sampler is not ours
// (DESIGN 1).
//
// Usage: test_icl_synth <model.navi> <reference dir>

#include "model/qwen3tts/graph.h"
#include "runtime/common/error.h"
#include "runtime/common/json.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string model = argv[1], ref = argv[2];
        const navi::Json manifest = navi::Json::parse(read_file(ref + "/manifest.json"));
        const std::string ref_text = manifest.str_or("icl_ref_text", "");
        if (ref_text.empty()) { std::printf("no icl_ref_text in the manifest: nothing to run\n"); return 0; }
        const auto codes_npy = navi::Npy::load(ref + "/ref_codes.npy");
        const auto codes = codes_npy.as_i32();
        const int ref_frames = static_cast<int>(codes_npy.shape[0]);
        const auto spk = navi::Npy::load(ref + "/speaker_embedding.npy").as_f32();
        const std::string text = manifest.str_or("text", "The quick brown fox jumps over the lazy dog.");

        navi::Device dev = navi::Device::open();
        bool ok = true;

        // --- the baseline: ICL through the serving path, stored codes ----------
        int base_frames = 0, prefill_rows = 0;
        bool base_eos = false;
        std::string base_sha;
        double base_rtf = 0;
        {
            navi::qwen3tts::GraphOptions gopt;
            gopt.max_ref_frames = ref_frames;
            auto graph = navi::qwen3tts::Graph::load(dev, model, {}, 32, gopt);
            navi::qwen3tts::SynthRequest sr;
            sr.text = text;
            sr.speaker = spk;
            sr.ref_text = ref_text;
            sr.ref_codes = codes;
            sr.ref_frames = ref_frames;
            sr.seed = 2;
            std::vector<float> pcm, pcm2;
            const auto st = graph->synth(sr, pcm);
            const auto st2 = graph->synth(sr, pcm2);
            base_frames = st.n_frames;
            base_eos = st.eos;
            prefill_rows = st.prefill_rows;
            base_rtf = st.rtf;
            const bool same = pcm == pcm2 && st2.n_frames == st.n_frames;
            std::printf("icl synth: %d prefill rows (%d reference frames), %d frames (%s), %.2f s audio, "
                        "RTF %.3f, run to run %s\n",
                        st.prefill_rows, ref_frames, st.n_frames, st.eos ? "eos" : "cap", st.audio_s, st.rtf,
                        same ? "bit-identical" : "DIFFERS");
            ok = ok && st.icl && st.prefill_rows == 10 + ref_frames && st.n_frames > 0 && same;

            // the same request without the reference: embedding-only, for the contrast
            navi::qwen3tts::SynthRequest eo = sr;
            eo.ref_text.clear();
            eo.ref_codes = {};
            eo.ref_frames = 0;
            std::vector<float> eo_pcm;
            const auto eo_st = graph->synth(eo, eo_pcm);
            std::printf("  embedding-only for contrast: %d prefill rows, %d frames, %.2f s audio, RTF %.3f\n",
                        eo_st.prefill_rows, eo_st.n_frames, eo_st.audio_s, eo_st.rtf);
            ok = ok && !eo_st.icl && eo_st.prefill_rows == 10;

            // half a reference is refused rather than silently downgraded
            for (int half = 0; half < 2; ++half) {
                navi::qwen3tts::SynthRequest bad = sr;
                if (half == 0) bad.ref_text.clear();
                else { bad.ref_codes = {}; bad.ref_frames = 0; }
                std::vector<float> junk;
                bool refused = false;
                try {
                    graph->synth(bad, junk);
                } catch (const std::exception & ex) {
                    refused = true;
                    std::printf("  refused: %s\n", ex.what());
                }
                ok = ok && refused;
            }
        }
        if (base_frames < 4) {
            std::printf("the baseline is only %d frames: too short to place a clamp below it, so the clamp "
                        "cases are skipped and this run proves only the stored-codes path\n", base_frames);
            std::printf("%s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        // --- the clamp, derived from that baseline -----------------------------
        // A cache that leaves room for half the baseline's frames: generation must stop at
        // the clamp, with hit_cap set and no error, even though the request asks for more.
        {
            const int budget = base_frames / 2;
            navi::qwen3tts::GraphOptions gopt;
            gopt.max_ref_frames = ref_frames;
            gopt.max_positions = prefill_rows + budget;
            auto graph = navi::qwen3tts::Graph::load(dev, model, {}, 32, gopt);
            navi::qwen3tts::SynthRequest sr;
            sr.text = text;
            sr.speaker = spk;
            sr.ref_text = ref_text;
            sr.ref_codes = codes;
            sr.ref_frames = ref_frames;
            sr.seed = 2;
            sr.max_frames = 4096;            // far more than the cache can hold
            std::vector<float> pcm;
            const auto st = graph->synth(sr, pcm);
            const bool clamped = st.frame_cap == budget && st.n_frames == budget && st.hit_cap && !st.eos;
            std::printf("clamp: a cache of %d positions after a %d-row prompt leaves %d frames; asked for 4096, "
                        "got %d (%s)  %s\n", gopt.max_positions, prefill_rows, budget, st.n_frames,
                        st.hit_cap ? "hit_cap" : "no cap flag", clamped ? "ok" : "WRONG");
            ok = ok && clamped;
        }

        // --- and the refusal, when the prompt itself does not fit --------------
        {
            navi::qwen3tts::GraphOptions gopt;
            gopt.max_ref_frames = ref_frames;
            gopt.max_positions = prefill_rows / 2;
            auto graph = navi::qwen3tts::Graph::load(dev, model, {}, 32, gopt);
            navi::qwen3tts::SynthRequest sr;
            sr.text = text;
            sr.speaker = spk;
            sr.ref_text = ref_text;
            sr.ref_codes = codes;
            sr.ref_frames = ref_frames;
            sr.seed = 2;
            std::vector<float> pcm;
            bool refused = false;
            try {
                graph->synth(sr, pcm);
            } catch (const std::exception & ex) {
                refused = true;
                std::printf("refused: %s\n", ex.what());
            }
            if (!refused) std::printf("ACCEPTED (!): a %d-row prompt in a %d-position cache\n",
                                      prefill_rows, gopt.max_positions);
            ok = ok && refused;
        }

        (void) base_eos;
        (void) base_rtf;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
