#pragma once

// Fused HIP code predictor (docs/code_predictor_plan.md, Phase 2).
//
// Runs the whole code-predictor pass for one talker frame — 16 positions
// (talker hidden, codebook-0 embedding, then one embedding per predicted
// code) through 5 transformer layers, with the 15 output heads — either as
// one cooperative kernel launch with grid-wide barriers between phases, or
// as one launch per phase (same device code; for debugging).
//
// Plain C++ interface; HIP types stay inside the .hip file.

#include <cstdint>
#include <cstddef>
#include <string>

namespace qwen3_tts {

constexpr int CP_MAX_LAYERS = 8;
constexpr int CP_MAX_POS    = 16;   // 2 prefill + 14 steps
constexpr int CP_N_HEADS_OUT = 15;  // output heads / codes per frame

struct cp_params {
    int hidden    = 1024;
    int n_head    = 16;
    int n_kv_head = 8;
    int head_dim  = 128;
    int ff        = 3072;
    int vocab     = 2048;
    int n_layer   = 5;
    float eps        = 1e-6f;
    float rope_theta = 1000000.0f;
};

// Device pointers to the weights, F16 matrices in ggml row-major layout
// (row = output index, contiguous over the input index) and F32 norms.
struct cp_weights {
    const uint16_t * attn_norm[CP_MAX_LAYERS];   // [hidden] f32, stored as raw bits
    const uint16_t * q_norm[CP_MAX_LAYERS];      // [head_dim] f32
    const uint16_t * k_norm[CP_MAX_LAYERS];      // [head_dim] f32
    const uint16_t * ffn_norm[CP_MAX_LAYERS];    // [hidden] f32
    const uint16_t * wq[CP_MAX_LAYERS];          // [n_head*head_dim, hidden] f16
    const uint16_t * wk[CP_MAX_LAYERS];          // [n_kv_head*head_dim, hidden]
    const uint16_t * wv[CP_MAX_LAYERS];
    const uint16_t * wo[CP_MAX_LAYERS];          // [hidden, n_head*head_dim]
    const uint16_t * w_gate[CP_MAX_LAYERS];      // [ff, hidden]
    const uint16_t * w_up[CP_MAX_LAYERS];
    const uint16_t * w_down[CP_MAX_LAYERS];      // [hidden, ff]
    const uint16_t * output_norm;                // [hidden] f32
    const uint16_t * talker_codec_embd;          // [talker_vocab, hidden] f16, row for cb0
    const uint16_t * codec_embd[CP_N_HEADS_OUT]; // [vocab, hidden] f16, one per step
    const uint16_t * lm_head[CP_N_HEADS_OUT];    // [vocab, hidden] f16
};

class HipCodePredictor {
public:
    HipCodePredictor();
    ~HipCodePredictor();

    // Allocates work buffers on the device for the given shapes.
    bool init(const cp_params & params, const cp_weights & weights, std::string * err);
    void release();

    // Runs one frame. `hidden` is the talker's hidden state (host, f32),
    // `cb0_token` the codebook-0 token (position 1's embedding input).
    // The 15 codes are sampled ON DEVICE (greedy when temperature <= 0,
    // else temperature + top-k multinomial with a PCG32 seeded from
    // `seed` per position) and returned in `codes_out` [15]; the raw
    // logits come back in `logits` (host, f32 [15][vocab]) unless
    // `logits` is nullptr, in which case the 123 KB readback is skipped.
    // fused=false runs one launch per phase.
    bool run(const float * hidden, int32_t cb0_token, float temperature,
             int32_t top_k, uint64_t seed, int32_t * codes_out, float * logits,
             bool fused, std::string * err);

    // Device-resident variant: `d_hidden` is a DEVICE pointer to the
    // talker's post-norm hidden (e.g. HipTalker::device_hidden()), so
    // the 4 KB host->device copy is skipped and the kernel reads the
    // talker's buffer in place. The caller must ensure the producing
    // kernel has completed before this runs (the synchronous generate()
    // loop guarantees it) and must not overwrite `d_hidden` until the
    // returned codes have been read back.
    bool run_device(const float * d_hidden, int32_t cb0_token, float temperature,
                    int32_t top_k, uint64_t seed, int32_t * codes_out, float * logits,
                    bool fused, std::string * err);

    // Assemble the next talker step embedding on-device from the codes
    // sampled by the last run()/run_device() plus `trailing_row` (host,
    // f32 [hidden], this frame's trailing text row). Result is available
    // via device_step_embd() for a chained HipTalker::run_device() call,
    // removing the 16 embedding-row D2H + host adds from the critical
    // path. Bit-identical to the host assembly (same summation order).
    bool assemble_step_embd(const float * trailing_row, std::string * err);

    // Device pointer to the last assembled step embedding.
    float * device_step_embd();

    // Wall time of the last run() in microseconds (device-side, event timed).
    float last_run_us() const { return last_run_us_; }

    int grid_blocks() const { return grid_blocks_; }

    // Self-heal probe mode: tighten the cooperative barrier spin cap so a
    // probe of a latched path gives up fast on a still-busy GPU.
    void set_probe_mode(bool tight);

private:
    struct impl;
    impl * p_ = nullptr;
    int grid_blocks_ = 0;
    float last_run_us_ = 0;
};

} // namespace qwen3_tts
