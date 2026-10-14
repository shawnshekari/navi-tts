// Voice store gate: the manifest's payload hashes and what they are for. No GPU, no model
// - the store is host code, and these are the states a crash or an edit can leave behind.
//
// The store writes its payloads first and its manifest last, so a crash in between leaves
// a manifest describing the OLD payloads. That is detectable because the manifest hashes
// them, and detection is all it claims (store.h): this test pins the four outcomes -
// embedding mismatch skips the voice, codes mismatch demotes it to embedding-only, a
// manifest from an older build is accepted and upgraded before its payloads are replaced,
// and an unreadable embedding stops a replacement before it writes anything.
//
// It also pins the rule that reference codes are never thrown away for audio that has not
// changed, since they depend on the audio alone and a model without a codec encoder cannot
// make them again.
//
// Usage: test_voice_store [scratch dir]

#include "runtime/common/json.h"
#include "runtime/voices/store.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    std::printf("  %-58s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

std::string read_file(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const fs::path & p, const std::string & bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A voice with everything a real one has, at toy sizes.
navi::voices::Voice make_voice(const std::string & name, const std::string & ref_text, int frames) {
    navi::voices::Voice v;
    v.id = navi::voices::Store::id_for(name);
    v.name = name;
    v.ref_text = ref_text;
    v.model = "test-model";
    v.embedding.assign(8, 0.5f);
    for (int i = 0; i < 8; ++i) v.embedding[static_cast<std::size_t>(i)] = 0.125f * static_cast<float>(i);
    v.ref_frames = frames;
    v.ref_codes.resize(static_cast<std::size_t>(frames) * 16);
    for (std::size_t i = 0; i < v.ref_codes.size(); ++i) v.ref_codes[i] = static_cast<std::int32_t>(i % 2048);
    return v;
}

const std::vector<std::uint8_t> SAMPLE = {'R', 'I', 'F', 'F', 0, 1, 2, 3, 4, 5, 6, 7};
const std::vector<std::uint8_t> SAMPLE2 = {'R', 'I', 'F', 'F', 9, 9, 9, 9, 9, 9, 9, 9};

std::span<const std::uint8_t> sample(const std::vector<std::uint8_t> & v) {
    return std::span<const std::uint8_t>(v.data(), v.size());
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const fs::path root = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "navi_voice_store_test";
        fs::remove_all(root);
        fs::create_directories(root);
        const std::string dir = root.string();
        const std::string sha = navi::voices::Store::sha256_hex(sample(SAMPLE));

        // --- a round trip, with and without codes --------------------------
        {
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            s.put(make_voice("with", "the transcript", 3), sample(SAMPLE));
            navi::voices::Voice plain = make_voice("plain", "", 0);
            plain.ref_codes.clear();
            s.put(std::move(plain), sample(SAMPLE));
            // a transcript but no codes: what a build with no codec encoder stores
            navi::voices::Voice textonly = make_voice("textonly", "the transcript", 0);
            textonly.ref_codes.clear();
            s.put(std::move(textonly), sample(SAMPLE));
        }
        {
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            const auto a = s.find("with");
            const auto b = s.find("plain");
            check(a && a->icl() && a->ref_frames == 3 && a->ref_codes.size() == 48, "a voice with codes reloads with them");
            check(b && !b->icl() && b->ref_codes.empty(), "a voice without a transcript reloads embedding-only");
            check(s.current("with", sha, "the transcript", true), "the same sample and transcript is a no-op");
            check(!s.current("with", sha, "a different transcript", true), "a changed transcript is not a no-op");
            check(s.current("with", sha, "the transcript", false), "codes already stored survive a model with no encoder");
            check(!s.current("plain", sha, "a transcript", true), "giving a voice a transcript is never a no-op");
            check(!s.current("textonly", sha, "the transcript", true),
                  "a transcript and no codes re-clones when the model can make them");
            check(s.current("textonly", sha, "the transcript", false),
                  "...and is a no-op when it cannot, there being nothing to gain");
            int frames = 0;
            const auto codes = s.codes_for_sample("with", sha, &frames);
            check(codes.size() == 48 && frames == 3, "codes_for_sample returns them for the same bytes");
            check(s.codes_for_sample("with", navi::voices::Store::sha256_hex(sample(SAMPLE2)), &frames).empty(),
                  "...and nothing for different bytes");
        }

        // --- a flipped byte in each payload --------------------------------
        {
            const fs::path codes = root / "with" / "ref_codes.i32";
            const std::string good = read_file(codes);
            std::string bad = good;
            bad[4] = static_cast<char>(bad[4] ^ 0xff);
            write_file(codes, bad);
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            const auto v = s.find("with");
            check(v && !v->icl(), "a flipped byte in ref_codes.i32 demotes the voice to embedding-only");
            check(v && v->embedding.size() == 8, "...and keeps its embedding, which is verified separately");
            write_file(codes, good.substr(0, good.size() - 4));
            navi::voices::Store t = navi::voices::Store::open(dir, 8);
            check(t.find("with") && !t.find("with")->icl(), "a truncated ref_codes.i32 does the same");
            write_file(codes, good);
        }
        {
            const fs::path emb = root / "with" / "embedding.f32";
            const std::string good = read_file(emb);
            std::string bad = good;
            bad[0] = static_cast<char>(bad[0] ^ 0xff);
            write_file(emb, bad);
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            check(!s.find("with"), "a flipped byte in embedding.f32 skips the voice entirely");
            write_file(emb, good);
        }

        // --- the interrupted replacement -----------------------------------
        // Payloads replaced, the manifest still describing the old ones: exactly what a
        // crash between the two writes leaves, and the reason the hashes are there.
        {
            const fs::path codes = root / "with" / "ref_codes.i32";
            const std::string good = read_file(codes);
            std::string other = good;
            for (char & c : other) c = static_cast<char>(c ^ 0x5a);
            write_file(codes, other);
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            check(s.find("with") && !s.find("with")->icl(), "an interrupted replacement is caught, not served");
            write_file(codes, good);
        }

        // --- a manifest from a build that had no hashes ---------------------
        {
            const fs::path meta = root / "with" / "voice.json";
            navi::Json j = navi::Json::parse(read_file(meta));
            navi::Json legacy;
            for (const char * k : {"name", "ref_text", "model", "sample_sha256", "created"}) {
                legacy.set(k, j.str_or(k, ""));
            }
            legacy.set("dim", static_cast<std::int64_t>(8));
            legacy.set("sample_seconds", 1.0);
            write_file(meta, legacy.dump() + "\n");
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            const auto v = s.find("with");
            check(v != nullptr, "a manifest with no hashes still loads (an older build wrote it)");
            check(v && !v->icl(), "...with ICL off, because its codes cannot be verified");

            // and a real replacement upgrades it before touching the payloads
            s.put(make_voice("with", "a new transcript", 4), sample(SAMPLE));
            navi::Json after = navi::Json::parse(read_file(meta));
            check(!after.str_or("embedding_sha256", "").empty() && !after.str_or("ref_codes_sha256", "").empty(),
                  "a replacement leaves both payload hashes in the manifest");
            navi::voices::Store t = navi::voices::Store::open(dir, 8);
            check(t.find("with") && t.find("with")->ref_frames == 4, "...and the new codes load");
        }

        // --- an unreadable embedding stops a replacement before it writes ---
        {
            const fs::path meta = root / "with" / "voice.json";
            navi::Json j = navi::Json::parse(read_file(meta));
            navi::Json legacy;
            for (const char * k : {"name", "ref_text", "model", "sample_sha256", "created"}) {
                legacy.set(k, j.str_or(k, ""));
            }
            legacy.set("dim", static_cast<std::int64_t>(8));
            write_file(meta, legacy.dump() + "\n");
            const fs::path emb = root / "with" / "embedding.f32";
            const std::string good = read_file(emb);
            fs::permissions(emb, fs::perms::none);
            navi::voices::Store s = navi::voices::Store::open(dir, 8);
            bool refused = false;
            std::string before = read_file(root / "with" / "ref_codes.i32");
            try {
                s.put(make_voice("with", "yet another", 5), sample(SAMPLE));
            } catch (const std::exception & e) {
                refused = true;
                std::printf("  refused: %s\n", e.what());
            }
            fs::permissions(emb, fs::perms::owner_read | fs::perms::owner_write);
            check(refused, "an unhashed manifest whose embedding cannot be read refuses the replacement");
            check(read_file(root / "with" / "ref_codes.i32") == before, "...before writing any payload");
            write_file(emb, good);
        }

        fs::remove_all(root);
        std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
