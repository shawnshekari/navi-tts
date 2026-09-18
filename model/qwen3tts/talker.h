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

// Device buffers describing one request's prompt.
struct Prompt {
    float * prefill = nullptr;    // [n_prefill][hidden]
    float * trailing = nullptr;   // [n_trailing][hidden]: text tokens[1:] then tts_eos
    float * tts_pad = nullptr;    // [hidden]: the row used once trailing text is exhausted
    int n_prefill = 0, n_trailing = 0;
};

class Talker {
public:
    static std::unique_ptr<Talker> create(const Device & dev, const DeviceWeights & w, const Params & p,
                                          int max_prompt_tokens = 1024, int max_positions = 2048);
    ~Talker();

    // Speaker-embedding cloning, streaming text mode (docs/model.md). `ids` is the
    // tokenized "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n";
    // `language` an entry of Params::language_ids or "" for auto; `speaker` a
    // host f32 [hidden] embedding. Buffers stay valid until the next call.
    const Prompt & build_prompt(std::span<const std::int32_t> ids, const std::string & language,
                                std::span<const float> speaker);

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
