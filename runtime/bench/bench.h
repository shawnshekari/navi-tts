#pragma once

// The controlled A/B harness (DESIGN 7): same code path as serving, no HTTP,
// fixed text/voice/seed, one JSON line to append to bench/results.jsonl.
// M0 measures what exists: device, load and upload. Frame/vocoder/TTFA/RTF
// fields are null until the stages exist, so the JSON shape is fixed now.

#include "runtime/device/device.h"

#include <cstdint>
#include <optional>
#include <string>

namespace navi {

struct BenchOptions {
    std::string model_path;
    std::string text = "The quick brown fox jumps over the lazy dog, and the bench text stays fixed.";
    std::string voice = "voice_1";
    std::uint64_t seed = 2;
    int repeats = 1;
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
    std::optional<std::string> wav_sha256;
    std::string text;
    std::uint64_t seed = 0;

    std::string to_json() const;
};

BenchResult run_bench(const Device & dev, const BenchOptions & opt);

} // namespace navi
