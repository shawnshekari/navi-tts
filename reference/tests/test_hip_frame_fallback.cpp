// Fallback test for the single fused talker+cp frame kernel
// (docs/talker_fusion_handoff.md, "THE NEXT TASK").
//
// When the fused-frame cooperative launch fails mid-generation (barrier
// timeout on a busy GPU), generate() must fall back to the three-launch
// CHAINED path (NOT ggml) and complete the request.
//
// The real trigger is a mid-generation barrier timeout: the fused frame
// runs for a few frames, then the GPU gets busy and the cooperative grid
// can no longer run to completion. We simulate that deterministically
// with the QWEN3_TTS_TEST_FRAME_FAIL_AFTER hook in HipFrameFusion::run_frame
// (fails the Nth call), so the test needs no GPU hog and is not
// scheduler-dependent.
//
// Pass requires:
//   - generate() returns true and produces frames
//   - the fused frame was initialized ("HIP fused frame:" present)
//   - the frame-fusion fallback fired exactly once ("falling back to
//     chained path") — the latch held, no retry
//   - the frames after the fallback still completed (via the chained path)
//
// usage: test_hip_frame_fallback --model <tts.gguf> [--frames N] [--fail-after N]

#include "tts_transformer.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qwen3_tts;

static std::string read_stream(FILE * f) {
    std::string s;
    char buf[4096];
    size_t n;
    fseek(f, 0, SEEK_SET);
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    return s;
}

static size_t count_occurrences(const std::string & hay, const std::string & needle) {
    size_t n = 0;
    for (size_t pos = 0; (pos = hay.find(needle, pos)) != std::string::npos; pos += 1) n++;
    return n;
}

int main(int argc, char ** argv) {
    std::string model;
    int frames = 24;
    int fail_after = 2;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fail-after") && i + 1 < argc) fail_after = atoi(argv[++i]);
    }
    if (model.empty()) {
        fprintf(stderr, "usage: %s --model <tts.gguf> [--frames N] [--fail-after N]\n", argv[0]);
        return 2;
    }
    if (fail_after >= frames) {
        fprintf(stderr, "FAIL: --fail-after must be < --frames\n");
        return 2;
    }

    // Fused talker + fused cp + frame fusion, with the fused-frame launch
    // forced to fail after `fail_after` successful frames.
    setenv("QWEN3_TTS_USE_HIP_TALKER", "1", 1);
    setenv("QWEN3_TTS_USE_HIP_CODE_PRED", "1", 1);
    setenv("QWEN3_TTS_USE_HIP_FRAME_FUSION", "1", 1);
    setenv("QWEN3_TTS_TEST_FRAME_FAIL_AFTER", std::to_string(fail_after).c_str(), 1);

    TTSTransformer tf;
    if (!tf.load_model(model)) {
        fprintf(stderr, "FAIL: load_model: %s\n", tf.get_error().c_str());
        return 1;
    }
    const tts_transformer_config cfg = tf.get_config();

    std::vector<int32_t> text = {14990, 4790, 323, 4667, 9745};
    std::vector<float> spk(cfg.hidden_size, 0.0f);
    std::vector<int32_t> out;
    // Fixed seed: the test samples at temp 0.9, so without a seed the
    // sampled cb0 can hit EOS before `frames` and the run is
    // nondeterministic. This seed produces a >= `frames`-length output.
    tf.set_seed(1234);

    fflush(stderr);
    int saved_err = dup(STDERR_FILENO);
    FILE * cap = tmpfile();
    if (!cap) { perror("tmpfile"); return 1; }
    dup2(fileno(cap), STDERR_FILENO);

    const bool gen_ok = tf.generate(text.data(), (int) text.size(), spk.data(), frames, out,
                                   2050, 1.05f, 0.9f, 50);

    fflush(stderr);
    dup2(saved_err, STDERR_FILENO);
    close(saved_err);
    const std::string log = read_stream(cap);
    fclose(cap);

    const size_t n_frames = out.size() / (size_t) cfg.n_codebooks;
    const bool attempted = log.find("HIP fused frame:") != std::string::npos;
    const size_t chained_fallbacks = count_occurrences(log, "falling back to chained path");

    fprintf(stderr, "generate ok=%d frames=%zu fused_attempted=%d chained_fallbacks=%zu\n",
            (int) gen_ok, n_frames, (int) attempted, chained_fallbacks);
    if (!gen_ok) {
        fprintf(stderr, "FAIL: generate failed: %s\n", tf.get_error().c_str());
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (!attempted) {
        fprintf(stderr, "FAIL: fused frame was never attempted (no init line)\n");
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (chained_fallbacks == 0) {
        fprintf(stderr, "FAIL: fused-frame fallback never exercised\n");
        return 1;
    }
    if (chained_fallbacks > 1) {
        fprintf(stderr, "FAIL: latch did not hold — fused frame retried after first failure\n");
        return 1;
    }
    if ((int) n_frames < frames) {
        fprintf(stderr, "FAIL: only %zu/%d frames produced after fallback\n", n_frames, frames);
        return 1;
    }
    printf("PASS: request completed via chained-path fallback after mid-generation fused-frame failure; latch held\n");
    return 0;
}
