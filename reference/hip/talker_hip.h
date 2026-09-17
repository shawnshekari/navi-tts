#pragma once

// Fused HIP talker (docs/code_predictor_plan.md: "the talker fusion is
// the next big lever").
//
// Runs one talker decode step — the per-frame step embedding through 28
// Qwen2 layers with attention over the growing KV cache, the output norm,
// the codec head, and on-device sampling of the codebook-0 token — either
// as one cooperative kernel launch with grid-wide barriers between
// phases, or as one launch per phase (same device code; for debugging).
//
// The KV cache is external: the fused kernel reads and writes the same
// F16 cache the ggml prefill produced (per-layer device pointers, one
// row of n_kv_head*head_dim per position), so the prefill stays on the
// ggml backend and only the decode steps are fused.
//
// Sampling mirrors the host generate() loop: tokens in
// [codec_vocab - 1024, codec_vocab) are suppressed except `eos_id`,
// a HuggingFace-style repetition penalty is applied to every token seen
// in a previous frame (tracked in a device-side byte set, reset per
// request), then greedy argmax or temperature + top-k multinomial with a
// PCG32 seeded per call.
//
// Plain C++ interface; HIP types stay inside the .hip file.

#include <cstdint>
#include <cstddef>
#include <string>

namespace qwen3_tts {

constexpr int TL_MAX_LAYERS = 32;
constexpr int TL_MAX_CTX    = 3072;  // attention window (scores live in LDS)

struct tl_params {
    int hidden    = 1024;
    int n_head    = 16;
    int n_kv_head = 8;
    int head_dim  = 128;
    int ff        = 3072;
    int vocab     = 3072;   // talker codec vocab (codec_head rows)
    int n_layer   = 28;
    float eps        = 1e-6f;
    float rope_theta = 1000000.0f;
};

// Device pointers to the weights, F16 matrices in ggml row-major layout
// (row = output index, contiguous over the input index) and F32 norms.
struct tl_weights {
    const uint16_t * attn_norm[TL_MAX_LAYERS];   // [hidden] f32, stored as raw bits
    const uint16_t * q_norm[TL_MAX_LAYERS];      // [head_dim] f32
    const uint16_t * k_norm[TL_MAX_LAYERS];      // [head_dim] f32
    const uint16_t * ffn_norm[TL_MAX_LAYERS];    // [hidden] f32
    const uint16_t * wq[TL_MAX_LAYERS];          // [n_head*head_dim, hidden] f16
    const uint16_t * wk[TL_MAX_LAYERS];          // [n_kv_head*head_dim, hidden]
    const uint16_t * wv[TL_MAX_LAYERS];
    const uint16_t * wo[TL_MAX_LAYERS];          // [hidden, n_head*head_dim]
    const uint16_t * w_gate[TL_MAX_LAYERS];      // [ff, hidden]
    const uint16_t * w_up[TL_MAX_LAYERS];
    const uint16_t * w_down[TL_MAX_LAYERS];      // [hidden, ff]
    const uint16_t * output_norm;                // [hidden] f32
    const uint16_t * codec_head;                 // [vocab, hidden] f16
};

// Per-layer device pointers into the F16 KV cache. Row (position) stride
// is n_kv_head*head_dim elements, matching the ggml k_cache/v_cache
// tensors' [head_dim, n_kv_head, n_ctx] contiguous layout.
struct tl_kv {
    const uint16_t * k[TL_MAX_LAYERS];
    const uint16_t * v[TL_MAX_LAYERS];
};

class HipTalker {
public:
    HipTalker();
    ~HipTalker();

    // Allocates work buffers on the device for the given shapes.
    bool init(const tl_params & params, const tl_weights & weights, std::string * err);
    void release();

    // Runs one decode step. `step_embd` is the host F32 [hidden] step
    // embedding (codec_embd(cb0) + code_pred embd sum + trailing text
    // row, as built by generate()). `n_past` positions are already in
    // `kv`; this step reads them and writes its own K/V at position
    // n_past. cb0 is sampled on device and returned in `cb0_out`; the
    // post-norm hidden and raw logits come back in `hidden_out` /
    // `logits_out` unless those are nullptr. `repetition_penalty` <= 0
    // disables the penalty. fused=false runs one launch per phase.
    bool run(const float * step_embd, int32_t n_past, const tl_kv & kv,
             float temperature, int32_t top_k, float repetition_penalty,
             int32_t eos_id, uint64_t seed,
             int32_t * cb0_out, float * hidden_out, float * logits_out,
             bool fused, std::string * err);

    // Device-resident variant: `d_step_embd` is a DEVICE pointer to the
    // step embedding (e.g. HipCodePredictor::device_step_embd()), so the
    // host->device copy is replaced by a device-to-device copy into the
    // residual buffer. Same semantics as run() otherwise.
    bool run_device(const float * d_step_embd, int32_t n_past, const tl_kv & kv,
                    float temperature, int32_t top_k, float repetition_penalty,
                    int32_t eos_id, uint64_t seed,
                    int32_t * cb0_out, float * hidden_out, float * logits_out,
                    bool fused, std::string * err);

    // Mark a token as previously generated (seeds the repetition-penalty
    // set with tokens sampled outside the fused path, e.g. frame 0 from
    // the prefill logits).
    bool mark_seen(int32_t token, std::string * err);
    void reset_repetition();

    // Device-side hidden buffer (for chaining into the fused code
    // predictor without a host round-trip).
    float * device_hidden() { return d_hidden_; }

    // Copy the last run's post-norm hidden (P.hidden floats) to the host.
    // run()/run_device() skip this copy when chaining into the fused cp;
    // the ggml cp needs it if it has to take over mid-request.
    bool fetch_hidden(float * host_out, std::string * err);

    // Device-side repetition-penalty seen set (vocab bytes). Shared with
    // the fused-frame kernel so the penalty state survives across the
    // fused/chained boundary; reset per request via reset_repetition().
    uint8_t * device_seen();

    // Wall time of the last run() in microseconds (device-side, event timed).
    float last_run_us() const { return last_run_us_; }

    int grid_blocks() const { return grid_blocks_; }

    // Self-heal probe mode: tighten the cooperative barrier spin cap so a
    // probe of a latched path gives up fast on a still-busy GPU instead
    // of stalling the full normal deadline. tight=true for a probe
    // request, false to restore the normal (generous) cap.
    void set_probe_mode(bool tight);

private:
    bool run_impl(const float * step_embd, bool step_is_device, int32_t n_past, const tl_kv & kv,
                 float temperature, int32_t top_k, float repetition_penalty,
                 int32_t eos_id, uint64_t seed,
                 int32_t * cb0_out, float * hidden_out, float * logits_out,
                 bool fused, std::string * err);

    struct impl;
    impl * p_ = nullptr;
    int grid_blocks_ = 0;
    float last_run_us_ = 0;
    float * d_hidden_ = nullptr;
};

} // namespace qwen3_tts
