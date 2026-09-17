#pragma once

// The Qwen3 text tokenizer: added-token splitting, NFC, the Qwen2 fast
// pre-tokenizer pattern, byte-level BPE. Exact against HF's Qwen2TokenizerFast
// (tests/reference/tokenizer_cases.json).

#include "runtime/text/bpe.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace navi {
class NaviFile;
}

namespace navi::qwen3tts {

class Tokenizer {
public:
    // From the tokenizer.* byte tensors in a .navi (before the file is closed).
    static Tokenizer from_navi(const NaviFile & f);
    static Tokenizer from_files(std::string_view vocab_json, std::string_view merges_txt,
                                std::string_view tokenizer_config_json);

    std::vector<std::int32_t> encode(std::string_view text) const;
    std::string decode(std::span<const std::int32_t> ids) const;

    // id of an added token by its text (e.g. "<|im_start|>"), -1 if none
    std::int32_t added_id(std::string_view content) const;
    std::int32_t n_vocab() const { return n_vocab_; }

    // The pre-tokenizer on its own, for tests: byte ranges [begin, end) of each piece.
    static std::vector<std::pair<std::size_t, std::size_t>> pretokenize(std::u32string_view s);

private:
    struct Added { std::string content; std::int32_t id; bool special; };
    text::ByteBpe      bpe_;
    std::vector<Added> added_;       // longest content first
    std::int32_t       n_vocab_ = 0;

    void encode_segment(std::string_view text, std::vector<std::int32_t> & out) const;
};

} // namespace navi::qwen3tts
