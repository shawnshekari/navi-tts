#pragma once

// The controlled A/B harness (DESIGN 7): same code path as serving, no HTTP,
// fixed text/voice/seed, one JSON line to append to bench/results.jsonl.
// Load and upload, then the utterance through Graph::synth (the serving
// path): stage times, TTFA, RTF and the WAV's sha256. talker_ms / cp_ms stay
// null - the frame kernel is one launch and does not split them.

#include "runtime/device/device.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace navi {

struct BenchOptions {
    std::string model_path;
    std::string text = "The quick brown fox jumps over the lazy dog, and the bench text stays fixed.";
    std::string voice_wav = "tests/reference/voice_1.wav";   // the reference voice (manifest.json)
    std::uint64_t seed = 2;
    int repeats = 1;
    int warmup = 1;                                          // untimed utterances first (DESIGN 2)
    // In-context cloning: the reference's transcript. With it the harness encodes voice_wav
    // to reference codes and benches the ICL path, whose prompt is 10 + ref_frames rows
    // instead of 10 - which costs prefill once AND adds that many keys to every frame's
    // attention, so both are reported.
    std::string ref_text;
    int vocoder_batch = 0;                                   // 0: the streaming configuration (STREAM_FIRST_BATCH, then STREAM_BATCH); N: N throughout
};

struct BenchResult {
    // provenance
    std::string git_hash, gfx, rocm, hip_runtime, device_name, model_name, model_dtype;
    std::string timestamp_utc;
    int multiprocessors = 0;
    // load
    double load_map_ms = 0, load_upload_ms = 0, upload_gbps = 0;
    std::uint64_t weight_bytes = 0;
    // stages (M1+)
    std::optional<double> prefill_ms, frame_ms, talker_ms, cp_ms, vocoder_ms_per_frame, ttfa_ms, rtf;
    std::optional<int> n_frames;
    // in-context cloning
    bool icl = false;
    std::optional<int> prefill_rows, ref_frames;
    // Per-position frame times, for comparing the two conditioning modes without the
    // confound of utterances of different lengths. A position the run never reached is
    // absent, not zero.
    std::vector<std::pair<int, double>> frame_ms_at;
    std::optional<std::string> wav_sha256;
    std::string text;
    std::string voice;          // the reference WAV's basename: a run with another voice is another measurement
    std::uint64_t seed = 0;

    std::string to_json() const;
};

BenchResult run_bench(const Device & dev, const BenchOptions & opt);

} // namespace navi
