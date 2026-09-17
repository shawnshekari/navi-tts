// Tokenizer gate: every string in tests/reference/tokenizer_cases.json must
// encode to exactly the ids HF's Qwen2TokenizerFast produced, and decode back
// to the input. Usage: test_tokenizer <model.navi> <tokenizer_cases.json>

#include "model/qwen3tts/tokenizer.h"
#include "runtime/common/json.h"
#include "runtime/text/unicode.h"
#include "runtime/weights/navi_file.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string show(const std::string & s) {
    std::string o;
    for (const char32_t c : navi::text::utf8_decode(s)) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20 || (c >= 0x7F && c < 0xA0)) { char b[16]; std::snprintf(b, sizeof b, "\\x%02x", static_cast<unsigned>(c)); o += b; }
        else navi::text::utf8_append(o, c);
    }
    return o.size() > 80 ? o.substr(0, 77) + "..." : o;
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s model.navi tokenizer_cases.json\n", argv[0]); return 2; }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto tok = navi::qwen3tts::Tokenizer::from_navi(file);
        const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const navi::Json cases = navi::Json::parse(read_file(argv[2]));

        int n = 0, bad = 0;
        for (const auto & [text, want_j] : cases.as_object()) {
            ++n;
            std::vector<std::int32_t> want;
            for (const auto & v : want_j.as_array()) want.push_back(static_cast<std::int32_t>(v.as_int()));
            const auto got = tok.encode(text);
            // decode returns the NFC form of the input (the normalizer ran before BPE)
            const std::string back = tok.decode(got);
            const std::string expect_back = navi::text::utf8_encode(navi::text::nfc(navi::text::utf8_decode(text)));
            const bool ok = got == want && back == expect_back;
            if (!ok) {
                ++bad;
                std::printf("FAIL  %s\n", show(text).c_str());
                std::printf("  want:");
                for (auto id : want) std::printf(" %d", id);
                std::printf("\n  got: ");
                for (auto id : got) std::printf(" %d", id);
                std::printf("\n  pieces:");
                const auto cps = navi::text::nfc(navi::text::utf8_decode(text));
                for (const auto & [b, e] : navi::qwen3tts::Tokenizer::pretokenize(cps)) {
                    std::printf(" [%s]", show(navi::text::utf8_encode(std::u32string_view(cps).substr(b, e - b))).c_str());
                }
                std::printf("\n");
                if (back != expect_back) std::printf("  decode differs: %s\n", show(back).c_str());
            }
        }
        std::printf("tokenizer: %d/%d cases exact (load %.0f ms, vocab %d)\n", n - bad, n, load_ms, tok.n_vocab());
        return bad ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
