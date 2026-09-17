#include "runtime/text/unicode.h"

#include <algorithm>

namespace navi::text {

namespace {

struct CatRange { char32_t first, last; std::uint8_t cls; };
struct CpRange  { char32_t first, last; };
struct Decomp   { char32_t cp, a, b; };
struct Ccc      { char32_t cp; std::uint8_t ccc; };
struct Comp     { char32_t a, b, composed; };

#include "runtime/text/unicode_tables.inc"

template <class R, std::size_t N>
const R * find_range(const R (&table)[N], char32_t cp) {
    const R * it = std::upper_bound(std::begin(table), std::end(table), cp,
                                    [](char32_t c, const R & r) { return c < r.first; });
    if (it == std::begin(table)) return nullptr;
    --it;
    return cp <= it->last ? it : nullptr;
}

std::uint8_t ccc(char32_t cp) {
    const Ccc * it = std::lower_bound(std::begin(CCCS), std::end(CCCS), cp,
                                      [](const Ccc & c, char32_t x) { return c.cp < x; });
    return (it != std::end(CCCS) && it->cp == cp) ? it->ccc : 0;
}

const Decomp * decomp(char32_t cp) {
    const Decomp * it = std::lower_bound(std::begin(DECOMPS), std::end(DECOMPS), cp,
                                         [](const Decomp & d, char32_t x) { return d.cp < x; });
    return (it != std::end(DECOMPS) && it->cp == cp) ? it : nullptr;
}

char32_t compose_pair(char32_t a, char32_t b) {
    const Comp * it = std::lower_bound(std::begin(COMPS), std::end(COMPS), std::pair{a, b},
                                       [](const Comp & c, const std::pair<char32_t, char32_t> & p) {
                                           return c.a != p.first ? c.a < p.first : c.b < p.second;
                                       });
    return (it != std::end(COMPS) && it->a == a && it->b == b) ? it->composed : 0;
}

// Hangul (Unicode 3.12)
constexpr char32_t S_BASE = 0xAC00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11A7;
constexpr int L_COUNT = 19, V_COUNT = 21, T_COUNT = 28, N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT;

char32_t compose_any(char32_t a, char32_t b) {
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT) {
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    }
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE && b < T_BASE + T_COUNT) {
        return a + (b - T_BASE);
    }
    return compose_pair(a, b);
}

void decompose_into(char32_t cp, std::u32string & out) {
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        const int s = static_cast<int>(cp - S_BASE);
        out.push_back(L_BASE + s / N_COUNT);
        out.push_back(V_BASE + (s % N_COUNT) / T_COUNT);
        if (s % T_COUNT) out.push_back(T_BASE + s % T_COUNT);
        return;
    }
    if (const Decomp * d = decomp(cp)) {
        decompose_into(d->a, out);
        if (d->b) decompose_into(d->b, out);
        return;
    }
    out.push_back(cp);
}

} // namespace

Cat category(char32_t cp) {
    const CatRange * r = find_range(CAT_RANGES, cp);
    return r ? static_cast<Cat>(r->cls) : Cat::Other;
}

bool is_white_space(char32_t cp) { return find_range(WHITE_SPACE, cp) != nullptr; }

std::u32string nfc(std::u32string_view s) {
    // fast path: all ASCII (or no decomposable / combining chars) is already NFC
    bool trivial = true;
    for (const char32_t c : s) {
        if (c >= 0x300 && (ccc(c) || decomp(c) || (c >= S_BASE && c < S_BASE + S_COUNT) ||
                           (c >= L_BASE && c < L_BASE + L_COUNT) || (c >= V_BASE && c < V_BASE + V_COUNT))) {
            trivial = false;
            break;
        }
    }
    if (trivial) return std::u32string(s);

    std::u32string d;
    d.reserve(s.size() + 8);
    for (const char32_t c : s) decompose_into(c, d);

    // canonical ordering: stable sort each run of non-starters by ccc
    for (std::size_t i = 0; i < d.size();) {
        if (ccc(d[i]) == 0) { ++i; continue; }
        std::size_t j = i;
        while (j < d.size() && ccc(d[j]) != 0) ++j;
        std::stable_sort(d.begin() + i, d.begin() + j, [](char32_t x, char32_t y) { return ccc(x) < ccc(y); });
        i = j;
    }

    // canonical composition (UAX #15 sample algorithm): a char composes with the
    // last starter unless something between them has ccc 0 or ccc >= its own.
    std::u32string out;
    out.reserve(d.size());
    std::size_t starter = std::u32string::npos;
    int last_ccc = 0;
    for (const char32_t c : d) {
        const int cc = ccc(c);
        if (starter != std::u32string::npos && (last_ccc < cc || last_ccc == 0)) {
            if (const char32_t comp = compose_any(out[starter], c)) {
                out[starter] = comp;
                continue;
            }
        }
        out.push_back(c);
        if (cc == 0) starter = out.size() - 1;
        last_ccc = cc;
    }
    return out;
}

std::u32string utf8_decode(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    const auto * p = reinterpret_cast<const unsigned char *>(s.data());
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned c = p[i];
        int len = 0;
        char32_t cp = 0;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + len > n) { out.push_back(0xFFFD); ++i; continue; }
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        // overlong, surrogate and out-of-range forms are invalid
        if (ok && ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
                   cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = false;
        if (!ok) { out.push_back(0xFFFD); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void utf8_append(std::string & out, char32_t cp) {
    if (cp < 0x80) { out += static_cast<char>(cp); }
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

std::string utf8_encode(std::u32string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char32_t c : s) utf8_append(out, c);
    return out;
}

} // namespace navi::text
