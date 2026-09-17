#pragma once

// GPU hog for the gate-1 fused-talker fallback test
// (docs/talker_fusion_handoff.md). Simulates a co-resident process
// (llama-server & friends) so the fused cooperative grid cannot get full
// residency and its deadline barrier times out.

namespace qwen3_tts {

// Saturates the GPU from the calling process: repeatedly launches a
// full-grid compute-spin kernel (4 blocks/CU) until `seconds` elapse,
// writing 'R' to `ready_fd` once the first kernel is in flight.
// Returns false on a HIP error.
bool gpu_saturate_run(int seconds, int ready_fd);

} // namespace qwen3_tts
