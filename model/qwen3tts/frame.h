#pragma once

// The persistent cooperative frame kernel (DESIGN 5.1): one launch per frame
// does the talker step at position n_past, samples codebook 0 on device, runs
// the 15 code-predictor steps sampling each, and assembles the next frame's
// step embedding. Hidden state never leaves the device between phases.
// Every grid barrier has a spin cap and an abort flag: a timeout fails the
// frame (FrameResult::ok == false) and nothing is latched for the next request.
//
// Frame 0 is special: the prefill already produced the hidden and logits of
// the last prompt position, so run_first() skips the talker phases.

#include "model/qwen3tts/params.h"
#include "model/qwen3tts/talker.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"

#include <cstdint>
#include <memory>
#include <string>

namespace navi::qwen3tts {

struct Sampling {
    float temperature = 0.9f;      // <= 0: greedy
    int   top_k = 50;
    float repetition_penalty = 1.05f;   // <= 0: off; over codebook-0 tokens seen this request
    float cp_temperature = 0.9f;
    int   cp_top_k = 50;
    int   min_frames = 2;          // EOS suppressed before this many frames (HF min_new_tokens)
};

struct FrameOptions {
    int blocks_per_mp = 1;                       // residency beats throughput (DESIGN 2)
    unsigned long long spin_cap = 50000000ull;   // ~0.5 s; tightened from p99 at M4
};

struct FrameResult {
    bool ok = false;               // false: barrier timeout (or launch failure); the request fails
    bool eos = false;              // codes[0] == codec_eos: the frame is not audio
    std::int32_t codes[16] = {};
    double us = 0;                 // device time of the launch
    std::string error;
};

class Frame {
public:
    static std::unique_ptr<Frame> create(const Device & dev, const DeviceWeights & w, const Params & p,
                                         const FrameOptions & opt = {});
    ~Frame();

    // Per request: clears the repetition set.
    void reset();

    // Frame 0 from the prefill's post-norm hidden and logits (device pointers).
    FrameResult run_first(const float * d_hidden, const float * d_logits, const float * d_trailing,
                          const Sampling & s, std::uint64_t seed, int frame_index);
    // Frames 1..: talker step on the previous frame's step embedding at position n_past.
    FrameResult run(KvCache & kv, int n_past, const float * d_trailing, const Sampling & s, std::uint64_t seed,
                    int frame_index);

    const float * d_step_out() const;   // next frame's step embedding
    int grid_blocks() const;
    int occupancy_per_mp() const;       // hipOccupancyMaxActiveBlocksPerMultiprocessor for the kernel

    struct Impl;

private:
    Frame() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
