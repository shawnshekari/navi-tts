#pragma once

// The talker's prompt assembly and prefill (docs/model.md "Prompt layout",
// "Talker"). Prefill runs the 28 layers over the whole prompt at once and
// leaves the f16 KV cache in the layout the frame kernel reads:
// [layer][position][n_kv_head * head_dim].

#include "model/qwen3tts/params.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace navi::qwen3tts {

struct KvCache {
    int n_layer = 0, max_pos = 0, kv_dim = 0;
    std::vector<unsigned short *> k, v;   // per layer, [max_pos][kv_dim] f16
    int n_used = 0;                       // positions filled
};

// An in-context cloning reference: the codes a codec encoder made from the reference
// audio, and the tokenised reference transcript. Both are needed - ICL conditions on the
// pair, and the transcript tells the model what the codes say.
struct IclRef {
    std::span<const std::int32_t> ids;     // the transcript wrapped as its own assistant turn
    std::span<const std::int32_t> codes;   // [n_frames][n_code_groups], row-major
    int n_frames = 0;
};

// Device buffers describing one request's prompt.
struct Prompt {
    float * prefill = nullptr;    // [n_prefill][hidden]
    float * trailing = nullptr;   // [n_trailing][hidden]: text tokens[1:] then tts_eos
    float * tts_pad = nullptr;    // [hidden]: the row used once trailing text is exhausted
    int n_prefill = 0, n_trailing = 0;
};

class Talker {
public:
    // `max_ref_frames` sizes the in-context cloning buffers; 0 disables ICL and allocates
    // none of them. It is independent of whether the blob has a codec encoder: a voice
    // cloned on one box can be served by a blob that cannot make codes itself. A non-zero
    // value also requires the code predictor's embedding tables to have been uploaded -
    // the ICL prompt sums them per reference frame.
    static std::unique_ptr<Talker> create(const Device & dev, const DeviceWeights & w, const Params & p,
                                          int max_prompt_tokens = 1024, int max_positions = 2048,
                                          int max_ref_frames = 0);
    ~Talker();

    // Tensors the prefill wants adjacent in the arena so that q/k/v and
    // gate/up run as one GEMM each (DeviceWeights::upload groups). create()
    // checks the adjacency and uses separate GEMMs where it does not hold.
    static DeviceWeights::Groups fused_groups(const Params & p);

    // Streaming text mode (docs/model.md "Prompt layout"). `ids` is the tokenized
    // "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"; `language` an
    // entry of Params::language_ids or "" for auto; `speaker` a host f32 [hidden]
    // embedding. Buffers stay valid until the next call.
    //
    // `icl` null is speaker-embedding cloning, which is what the engine has always done.
    // With a reference, the prompt gains one row per reference frame - the 16 code
    // embeddings summed at the talker's width - with the reference transcript, the target
    // text and tts_eos laid along them on the text side and tts_pad filling the rest; the
    // text that does not fit becomes the trailing rows. The prefix is the same 9 rows
    // either way. Both forms are refused rather than truncated if they exceed the sizes
    // create() was given.
    const Prompt & build_prompt(std::span<const std::int32_t> ids, const std::string & language,
                                std::span<const float> speaker, const IclRef * icl = nullptr);

    // Runs the prompt through the talker; positions 0..n_prefill-1 land in the
    // KV cache (n_used = n_prefill). Post-norm hidden and codec logits of the
    // last position stay on device.
    void prefill(const Prompt & prompt);

    KvCache & kv();
    const float * d_hidden() const;     // [hidden]
    const float * d_logits() const;     // [codec vocab]
    double last_ms() const;

    struct Impl;

private:
    Talker() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
