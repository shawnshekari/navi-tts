#pragma once

// Single fused talker+code-predictor frame kernel
// (docs/talker_fusion_handoff.md, "THE NEXT TASK").
//
// One cooperative launch per frame does the whole frame on-device:
//   1. talker 28-layer step (reads the previous frame's step embedding,
//      writes cb0 + hidden + the F16 KV cache row at n_past),
//   2. cp 16-position loop (position 0 = talker hidden, position 1 =
//      codec_embd(cb0), positions 2..15 = code_pred_embd rows of the
//      sampled codes) producing the 15 codes,
//   3. the step_embd assembly for the NEXT frame (the k_assemble_step_embd
//      math) — so the host only reads back the 16 codes (cb0 + 15) and
//      never touches hidden or step_embd.
//
// The talker's ~170 barriers/step and the cp's 16-position x ~8-phase
// barriers interleave in ONE grid sharing ONE deadline-aware
// grid_barrier buffer and ONE spin cap (hip/hip_dev.h).
//
// Residency is the risk on a busy GPU: init() measures
// hipOccupancyMaxActiveBlocksPerMultiprocessor for the fused kernel and
// sizes the grid accordingly. On barrier timeout run_frame() returns
// false and the caller must fall back to the existing three-launch
// chained path (HipTalker::run_device + HipCodePredictor::run_device +
// assemble_step_embd), NOT to ggml, so the win degrades gracefully.
//
// Bit-identical parity vs the chained path: the phase code is the same
// device code with the same summation order and the same per-position
// sampling RNG seeds.
//
// Plain C++ interface; HIP types stay inside the .hip file.

#include "hip/talker_hip.h"
#include "hip/code_pred_hip.h"

#include <cstdint>
#include <string>

namespace qwen3_tts {

class HipFrameFusion {
public:
    HipFrameFusion();
    ~HipFrameFusion();

    // Allocates work buffers on the device. `d_seen` is the talker's
    // repetition-penalty seen set (HipTalker::device_seen()) so the
    // penalty state is shared across the fused/chained boundary.
    bool init(const tl_params & tp, const tl_weights & tw,
              const cp_params & cpp, const cp_weights & cw,
              uint8_t * d_seen, std::string * err);
    void release();

    // Runs one whole frame. `d_step_embd` is a DEVICE pointer to this
    // frame's step embedding (the chained assembly output). `trailing_row`
    // is the NEXT frame's trailing text row (host f32[hidden], staged
    // H2D before the launch). `talker_seed` / `cp_seed` are the seeds the
    // chained path would have used for this frame's talker step and the
    // following cp pass, so sampling is bit-identical.
    // Returns false on barrier timeout (the cooperative grid could not
    // run to completion) — fall back to the chained path.
    bool run_frame(const float * d_step_embd, int32_t n_past, const tl_kv & kv,
                  const float * trailing_row,
                  float temperature, int32_t top_k, float repetition_penalty,
                  int32_t eos_id, uint64_t talker_seed, uint64_t cp_seed,
                  std::string * err);

    // Host readback after a successful run_frame: out16[0] = cb0,
    // out16[1..15] = the 15 codes.
    bool get_codes(int32_t * out16, std::string * err);

    // Device pointer to the assembled NEXT-frame step embedding (input
    // for the next run_frame, or for a chained HipTalker::run_device
    // after a fallback).
    float * device_step_embd() { return d_step_embd_; }

    // Wall time of the last run_frame in microseconds (device-side,
    // event timed).
    float last_run_us() const { return last_run_us_; }

    int grid_blocks() const { return grid_blocks_; }
    int occupancy_per_cu() const { return occ_per_cu_; }

    // Self-heal probe mode: tighten the cooperative barrier spin cap so a
    // probe of a latched fused-frame path gives up fast on a busy GPU.
    void set_probe_mode(bool tight);

private:
    struct impl;
    impl * p_ = nullptr;
    int grid_blocks_ = 0;
    int occ_per_cu_ = 0;
    float last_run_us_ = 0;
    float * d_step_embd_ = nullptr;
};

} // namespace qwen3_tts
