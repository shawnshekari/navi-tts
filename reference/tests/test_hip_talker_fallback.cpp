// Gate 1 test for the fused HIP talker (docs/talker_fusion_handoff.md):
// when the cooperative grid cannot run to completion because a second
// process saturates the GPU, generate() must fall back to the ggml
// talker mid-generation and complete the request instead of erroring.
//
// The test forks a GPU-hog child (tests/gpu_saturate.hip, same binary
// via --saturate), waits until its first kernel is in flight, then runs
// a short fused-talker generation with stderr captured. Pass requires:
//   - generate() returns true and produces frames
//   - the fused talker was initialized (init line present)
//   - the fallback fired at least once ("falling back to GGML talker")
//
// usage: test_hip_talker_fallback --model <tts.gguf> [--frames N]
//        test_hip_talker_fallback --saturate <seconds>   (internal hog mode)

#include "tts_transformer.h"
#include "gpu_saturate.h"

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
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

static void reap(pid_t child) {
    if (child > 0) {
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
    }
}

int main(int argc, char ** argv) {
    std::string model;
    int frames = 24;
    int saturate_secs = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--saturate") && i + 1 < argc) saturate_secs = atoi(argv[++i]);
    }

    // Internal GPU-hog mode: run as the forked child, never returns early.
    if (saturate_secs > 0) {
        bool ok = gpu_saturate_run(saturate_secs, STDOUT_FILENO);
        _exit(ok ? 0 : 1);
    }

    if (model.empty()) {
        fprintf(stderr, "usage: %s --model <tts.gguf> [--frames N]\n", argv[0]);
        return 2;
    }

    // --- saturating second process (fork before any HIP init here) ---
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return 1; }
    pid_t child = fork();
    if (child < 0) { perror("fork"); return 1; }
    if (child == 0) {
        close(pipefd[0]);
        bool ok = gpu_saturate_run(120, pipefd[1]);
        _exit(ok ? 0 : 1);
    }
    close(pipefd[1]);
    bool saturated = false;
    struct pollfd pfd = { pipefd[0], POLLIN, 0 };
    if (poll(&pfd, 1, 60000) > 0) {
        char r = 0;
        if (read(pipefd[0], &r, 1) == 1 && r == 'R') saturated = true;
    }
    close(pipefd[0]);
    if (!saturated) {
        fprintf(stderr, "FAIL: saturator process did not come up\n");
        reap(child);
        return 1;
    }
    fprintf(stderr, "saturator pid %d is holding the GPU\n", (int) child);

    // --- fused-talker generation under contention ---
    // The GPU time-slices between processes, so with the default ~0.5 s
    // barrier cap a 2.4 ms cooperative step can still complete inside
    // one slice of our process even with the hog running. Tighten the
    // cap so the barrier cannot wait out full-grid residency under the
    // second process — the gate-1 failure becomes deterministic instead
    // of scheduler-dependent.
    setenv("QWEN3_TTS_USE_HIP_TALKER", "1", 1);
    setenv("QWEN3_TTS_HIP_BARRIER_SPINS", "1", 1);
    TTSTransformer tf;
    if (!tf.load_model(model)) {
        fprintf(stderr, "FAIL: load_model: %s\n", tf.get_error().c_str());
        reap(child);
        return 1;
    }
    const tts_transformer_config cfg = tf.get_config();

    std::vector<int32_t> text = {14990, 4790, 323, 4667, 9745};
    std::vector<float> spk(cfg.hidden_size, 0.0f);
    std::vector<int32_t> out;

    fflush(stderr);
    int saved_err = dup(STDERR_FILENO);
    FILE * cap = tmpfile();
    if (!cap) { perror("tmpfile"); reap(child); return 1; }
    dup2(fileno(cap), STDERR_FILENO);

    const bool gen_ok = tf.generate(text.data(), (int) text.size(), spk.data(), frames, out,
                                   2050, 1.05f, 0.9f, 50);

    // Second request in the same process: with self-heal the latch is no
    // longer permanent — the second request re-probes the fused talker
    // (which fails again under the still-running hog, then re-latches).
    std::vector<int32_t> out2;
    const bool gen2_ok = tf.generate(text.data(), (int) text.size(), spk.data(), frames, out2,
                                    2050, 1.05f, 0.9f, 50);

    fflush(stderr);
    dup2(saved_err, STDERR_FILENO);
    close(saved_err);
    const std::string log = read_stream(cap);
    fclose(cap);
    reap(child);

    const size_t n_frames = out.size() / (size_t) cfg.n_codebooks;
    const size_t n_frames2 = out2.size() / (size_t) cfg.n_codebooks;
    const bool attempted = log.find("HIP fused talker:") != std::string::npos;
    size_t fallbacks = 0;
    for (size_t pos = 0; (pos = log.find("falling back to GGML talker", pos)) != std::string::npos; pos += 1) {
        fallbacks++;
    }

    fprintf(stderr, "generate ok=%d/%d frames=%zu/%zu fused_attempted=%d fallbacks=%zu\n",
            (int) gen_ok, (int) gen2_ok, n_frames, n_frames2, (int) attempted, fallbacks);
    if (!gen_ok || !gen2_ok) {
        fprintf(stderr, "FAIL: generate failed: %s\n", tf.get_error().c_str());
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (!attempted) {
        fprintf(stderr, "FAIL: fused talker was never attempted (no init line)\n");
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (fallbacks == 0) {
        fprintf(stderr, "FAIL: fused talker completed under saturation — fallback never exercised\n");
        return 1;
    }
    // Self-heal contract (replaces the old "latch never retries"): the
    // second request must re-probe the latched fused talker, the probe
    // must fail under the still-running hog, and it must re-latch.
    const bool reprobe = log.find("self-heal: probing fused talker") != std::string::npos;
    const bool relatch = log.find("self-heal: fused talker probe failed") != std::string::npos;
    if (!reprobe) {
        fprintf(stderr, "FAIL: second request did not re-probe the latched fused talker\n");
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (!relatch) {
        fprintf(stderr, "FAIL: re-probe did not re-latch after failing under saturation\n");
        fputs(log.c_str(), stderr);
        return 1;
    }
    if (n_frames == 0 || n_frames2 == 0) {
        fprintf(stderr, "FAIL: no frames produced via fallback\n");
        return 1;
    }
    printf("PASS: requests completed via ggml fallback under GPU saturation; latch re-armed and re-probed on the second request\n");
    return 0;
}
