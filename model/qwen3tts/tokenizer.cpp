#include "model/qwen3tts/tokenizer.h"

#include "runtime/common/error.h"
#include "runtime/common/json.h"
#include "runtime/text/unicode.h"
#include "runtime/weights/navi_file.h"

#include <algorithm>

namespace navi::qwen3tts {

using text::Cat;

namespace {

std::string_view tensor_text(const NaviFile & f, const char * name) {
    const TensorInfo & t = f.get(name);
    if (t.dtype != DType::U8) fail(std::string(name) + " is not a byte tensor");
    return std::string_view(static_cast<const char *>(f.data(t)), t.nbytes);
}

// Character classes of the pattern. A = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], B = [\p{Ll}\p{Lm}\p{Lo}\p{M}].
struct Cls {
    bool letter, number, mark, a, b, space, crlf;
};

Cls classify(char32_t cp) {
    const Cat c = text::category(cp);
    Cls k{};
    k.letter = text::is_letter(c);
    k.number = c == Cat::N;
    k.mark   = c == Cat::M;
    k.a = c == Cat::Lu || c == Cat::Lt || c == Cat::Lm || c == Cat::Lo || k.mark;
    k.b = c == Cat::Ll || c == Cat::Lm || c == Cat::Lo || k.mark;
    k.space = text::is_white_space(cp);
    k.crlf  = cp == '\r' || cp == '\n';
    return k;
}

} // namespace

// The Qwen2 fast pre-tokenizer (tokenizers' Split, behaviour Isolated):
//
//   [^\r\n\p{L}\p{N}]?[A]*[B]+ | [^\r\n\p{L}\p{N}]?[A]+[B]* | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Alternation is ordered (first alternative that matches wins, not the
// longest) and each alternative backtracks the way a PCRE engine would; the
// cases below spell that out per alternative. Returns pieces as index ranges
// into the code point string.
std::vector<std::pair<std::size_t, std::size_t>> Tokenizer::pretokenize(std::u32string_view s) {
    const std::size_t n = s.size();
    std::vector<Cls> k(n);
    for (std::size_t i = 0; i < n; ++i) k[i] = classify(s[i]);

    // [A]*[B]+ from i: A* is greedy, then backs off one char at a time until a
    // B-run of length >= 1 follows. Returns end or i on failure.
    auto a_star_b_plus = [&](std::size_t i) -> std::size_t {
        std::size_t ka = i;
        while (ka < n && k[ka].a) ++ka;
        for (std::size_t split = ka + 1; split-- > i;) {
            if (split < n && k[split].b) {
                std::size_t e = split;
                while (e < n && k[e].b) ++e;
                return e;
            }
        }
        return i;
    };
    // [A]+[B]* from i
    auto a_plus_b_star = [&](std::size_t i) -> std::size_t {
        std::size_t e = i;
        while (e < n && k[e].a) ++e;
        if (e == i) return i;
        while (e < n && k[e].b) ++e;
        return e;
    };
    auto prefix_ok = [&](std::size_t i) { return !k[i].crlf && !k[i].letter && !k[i].number; };

    std::vector<std::pair<std::size_t, std::size_t>> out;
    std::size_t i = 0;
    while (i < n) {
        std::size_t e = i;
        // 1. [^\r\n\p{L}\p{N}]?[A]*[B]+   (optional prefix tried first, then without)
        if (prefix_ok(i) && i + 1 < n) e = a_star_b_plus(i + 1) > i + 1 ? a_star_b_plus(i + 1) : i;
        if (e == i) e = a_star_b_plus(i);
        // 2. [^\r\n\p{L}\p{N}]?[A]+[B]*
        if (e == i && prefix_ok(i) && i + 1 < n) e = a_plus_b_star(i + 1) > i + 1 ? a_plus_b_star(i + 1) : i;
        if (e == i) e = a_plus_b_star(i);
        // 3. \p{N}
        if (e == i && k[i].number) e = i + 1;
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
        if (e == i) {
            std::size_t j = i;
            if (s[j] == ' ' && j + 1 < n) ++j;
            std::size_t p = j;
            while (p < n && !k[p].space && !k[p].letter && !k[p].number) ++p;
            if (p > j) {
                while (p < n && (k[p].crlf || s[p] == '/')) ++p;
                e = p;
            }
        }
        // 5. \s*[\r\n]+   : the whitespace run must contain a CR/LF; ends after the last one
        // 6. \s+(?!\S)    : the run minus its last char, unless the run ends the string
        // 7. \s+
        if (e == i && k[i].space) {
            std::size_t r = i;
            while (r < n && k[r].space) ++r;
            std::size_t last_crlf = i;
            bool have = false;
            for (std::size_t p = i; p < r; ++p) if (k[p].crlf) { last_crlf = p; have = true; }
            if (have) e = last_crlf + 1;
            else if (r == n) e = r;
            else if (r - i >= 2) e = r - 1;
            else e = r;
        }
        if (e == i) {
            // Nothing matched (a lone letterless mark, an unpaired char, ...): a
            // regex Split with Isolated behaviour leaves such a char as its own piece.
            e = i + 1;
        }
        out.emplace_back(i, e);
        i = e;
    }
    return out;
}

Tokenizer Tokenizer::from_navi(const NaviFile & f) {
    return from_files(tensor_text(f, "tokenizer.vocab.json"), tensor_text(f, "tokenizer.merges.txt"),
                      tensor_text(f, "tokenizer.config.json"));
}

Tokenizer Tokenizer::from_files(std::string_view vocab_json, std::string_view merges_txt,
                                std::string_view tokenizer_config_json) {
    Tokenizer t;
    t.bpe_.load(vocab_json, merges_txt);
    t.n_vocab_ = t.bpe_.n_vocab();
    const Json cfg = Json::parse(tokenizer_config_json);
    for (const auto & [id_str, tok] : cfg.get("added_tokens_decoder").as_object()) {
        Added a;
        a.id = static_cast<std::int32_t>(std::stol(id_str));
        a.content = tok.get("content").as_string();
        a.special = tok.bool_or("special", false);
        if (tok.bool_or("lstrip", false) || tok.bool_or("rstrip", false) || tok.bool_or("single_word", false) ||
            tok.bool_or("normalized", false)) {
            fail("tokenizer: added token '" + a.content + "' uses lstrip/rstrip/single_word/normalized, not supported");
        }
        t.added_.push_back(std::move(a));
        t.n_vocab_ = std::max(t.n_vocab_, t.added_.back().id + 1);
    }
    std::sort(t.added_.begin(), t.added_.end(),
              [](const Added & x, const Added & y) { return x.content.size() > y.content.size(); });
    return t;
}

std::int32_t Tokenizer::added_id(std::string_view content) const {
    for (const Added & a : added_) if (a.content == content) return a.id;
    return -1;
}

void Tokenizer::encode_segment(std::string_view text, std::vector<std::int32_t> & out) const {
    if (text.empty()) return;
    const std::u32string cps = text::nfc(text::utf8_decode(text));
    for (const auto & [b, e] : pretokenize(cps)) {
        const std::string piece = text::utf8_encode(std::u32string_view(cps).substr(b, e - b));
        const auto ids = bpe_.encode(piece);
        out.insert(out.end(), ids.begin(), ids.end());
    }
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> out;
    // added tokens are matched on the raw text, leftmost, longest first
    std::size_t seg = 0;
    for (std::size_t i = 0; i < text.size();) {
        const Added * hit = nullptr;
        for (const Added & a : added_) {
            if (text.compare(i, a.content.size(), a.content) == 0) { hit = &a; break; }
        }
        if (!hit) { ++i; continue; }
        encode_segment(text.substr(seg, i - seg), out);
        out.push_back(hit->id);
        i += hit->content.size();
        seg = i;
    }
    encode_segment(text.substr(seg), out);
    return out;
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids) const {
    std::string out;
    for (const std::int32_t id : ids) {
        bool added = false;
        for (const Added & a : added_) if (a.id == id) { out += a.content; added = true; break; }
        if (!added) out += bpe_.bytes_of(id);
    }
    return out;
}

} // namespace navi::qwen3tts
