#pragma once

// Byte-level BPE (GPT-2 / Qwen2 style): a vocab of byte strings written in the
// bytes-to-unicode alphabet, and ranked merges. Encodes one pre-token's bytes.
// Merge order follows HF tokenizers exactly: one pair at a time from a heap
// keyed by (rank, position), re-examining neighbours after every merge.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace navi::text {

class ByteBpe {
public:
    void load(std::string_view vocab_json, std::string_view merges_txt);

    std::vector<std::int32_t> encode(std::string_view bytes) const;

    std::int32_t        id_of(std::string_view bytes) const;      // -1 if absent
    const std::string & bytes_of(std::int32_t id) const;
    std::int32_t        n_vocab() const { return static_cast<std::int32_t>(id_to_bytes_.size()); }

private:
    static std::uint64_t key(std::int32_t a, std::int32_t b) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
    }
    struct Merge { std::int32_t rank, id; };
    std::unordered_map<std::string, std::int32_t> vocab_;
    std::vector<std::string> id_to_bytes_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    std::int32_t byte_id_[256] = {};
};

// GPT-2 bytes_to_unicode: byte -> code point and back.
char32_t byte_to_unicode(unsigned char b);
int      unicode_to_byte(char32_t cp);   // -1 if not in the alphabet

} // namespace navi::text
