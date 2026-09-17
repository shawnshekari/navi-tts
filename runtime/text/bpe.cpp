#include "runtime/text/bpe.h"

#include "runtime/common/error.h"
#include "runtime/common/json.h"
#include "runtime/text/unicode.h"

#include <queue>

namespace navi::text {

namespace {

struct ByteAlphabet {
    char32_t to_cp[256];
    std::unordered_map<char32_t, int> to_byte;
    ByteAlphabet() {
        // printable ranges map to themselves; the rest to 256, 257, ...
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            to_cp[b] = keep ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + n++);
            to_byte[to_cp[b]] = b;
        }
    }
};

const ByteAlphabet & alphabet() {
    static const ByteAlphabet a;
    return a;
}

// vocab / merges entries are in the unicode alphabet; turn them into raw bytes
std::string unmap(const std::string & token, const char * what) {
    std::string out;
    for (const char32_t cp : utf8_decode(token)) {
        const int b = unicode_to_byte(cp);
        if (b < 0) fail(std::string("bpe: ") + what + " entry '" + token + "' is outside the byte alphabet");
        out += static_cast<char>(b);
    }
    return out;
}

} // namespace

char32_t byte_to_unicode(unsigned char b) { return alphabet().to_cp[b]; }

int unicode_to_byte(char32_t cp) {
    const auto & m = alphabet().to_byte;
    const auto it = m.find(cp);
    return it == m.end() ? -1 : it->second;
}

void ByteBpe::load(std::string_view vocab_json, std::string_view merges_txt) {
    const Json v = Json::parse(vocab_json);
    const auto & obj = v.as_object();
    vocab_.clear();
    vocab_.reserve(obj.size() * 2);
    std::int32_t max_id = -1;
    for (const auto & [tok, id] : obj) max_id = std::max(max_id, static_cast<std::int32_t>(id.as_int()));
    id_to_bytes_.assign(static_cast<std::size_t>(max_id + 1), std::string());
    for (const auto & [tok, idj] : obj) {
        const auto id = static_cast<std::int32_t>(idj.as_int());
        std::string bytes = unmap(tok, "vocab");
        id_to_bytes_[static_cast<std::size_t>(id)] = bytes;
        vocab_.emplace(std::move(bytes), id);
    }
    for (int b = 0; b < 256; ++b) {
        const auto it = vocab_.find(std::string(1, static_cast<char>(b)));
        if (it == vocab_.end()) fail("bpe: vocab has no token for byte " + std::to_string(b));
        byte_id_[b] = it->second;
    }

    merges_.clear();
    std::int32_t rank = 0;
    std::size_t pos = 0;
    while (pos < merges_txt.size()) {
        std::size_t eol = merges_txt.find('\n', pos);
        if (eol == std::string_view::npos) eol = merges_txt.size();
        std::string_view line = merges_txt.substr(pos, eol - pos);
        pos = eol + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.empty() || line[0] == '#') continue;
        const std::size_t sp = line.find(' ');
        if (sp == std::string_view::npos) fail("bpe: malformed merge line '" + std::string(line) + "'");
        const std::string a = unmap(std::string(line.substr(0, sp)), "merges");
        const std::string b = unmap(std::string(line.substr(sp + 1)), "merges");
        const std::int32_t ia = id_of(a), ib = id_of(b), iab = id_of(a + b);
        if (ia < 0 || ib < 0 || iab < 0) fail("bpe: merge '" + std::string(line) + "' names tokens not in the vocab");
        merges_.emplace(key(ia, ib), Merge{rank++, iab});
    }
}

std::int32_t ByteBpe::id_of(std::string_view bytes) const {
    const auto it = vocab_.find(std::string(bytes));
    return it == vocab_.end() ? -1 : it->second;
}

const std::string & ByteBpe::bytes_of(std::int32_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_bytes_.size()) fail("bpe: id out of range");
    return id_to_bytes_[static_cast<std::size_t>(id)];
}

std::vector<std::int32_t> ByteBpe::encode(std::string_view bytes) const {
    if (bytes.empty()) return {};
    // No "whole piece is in the vocab" shortcut: without ignore_merges the
    // merge path decides, and HF tokenizers takes the same path.

    struct Sym { std::int32_t id; int prev, next; bool alive; };
    std::vector<Sym> syms(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        syms[i] = {byte_id_[static_cast<unsigned char>(bytes[i])], static_cast<int>(i) - 1,
                   i + 1 < bytes.size() ? static_cast<int>(i) + 1 : -1, true};
    }

    struct Cand { std::int32_t rank; int pos; std::int32_t id; };
    auto cmp = [](const Cand & a, const Cand & b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> heap(cmp);
    auto push = [&](int pos) {
        const int nx = syms[pos].next;
        if (nx < 0) return;
        const auto it = merges_.find(key(syms[pos].id, syms[nx].id));
        if (it != merges_.end()) heap.push({it->second.rank, pos, it->second.id});
    };
    for (std::size_t i = 0; i + 1 < syms.size(); ++i) push(static_cast<int>(i));

    while (!heap.empty()) {
        const Cand c = heap.top();
        heap.pop();
        Sym & left = syms[c.pos];
        if (!left.alive || left.next < 0) continue;
        Sym & right = syms[left.next];
        // stale entry: the pair at this position changed since it was queued
        const auto it = merges_.find(key(left.id, right.id));
        if (it == merges_.end() || it->second.id != c.id) continue;
        left.id = c.id;
        right.alive = false;
        left.next = right.next;
        if (right.next >= 0) syms[right.next].prev = c.pos;
        if (left.prev >= 0) push(left.prev);
        push(c.pos);
    }

    std::vector<std::int32_t> out;
    for (int i = 0; i >= 0; i = syms[i].next) out.push_back(syms[i].id);
    return out;
}

} // namespace navi::text
