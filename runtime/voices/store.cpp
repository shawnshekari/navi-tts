#include "runtime/voices/store.h"

#include "runtime/common/error.h"
#include "runtime/common/json.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace navi::voices {

namespace {

// SHA-256 (FIPS 180-4), enough to fingerprint a sample. No dependency.
class Sha256 {
public:
    void update(const std::uint8_t * p, std::size_t n) {
        total_ += n;
        while (n > 0) {
            const std::size_t take = std::min(n, 64 - fill_);
            std::memcpy(buf_.data() + fill_, p, take);
            fill_ += take; p += take; n -= take;
            if (fill_ == 64) { block(buf_.data()); fill_ = 0; }
        }
    }
    std::string hex() {
        const std::uint64_t bits = total_ * 8;
        std::uint8_t pad = 0x80;
        update(&pad, 1);
        const std::uint8_t zero = 0;
        while (fill_ != 56) update(&zero, 1);
        std::array<std::uint8_t, 8> len{};
        for (int i = 0; i < 8; ++i) len[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        update(len.data(), 8);
        std::string out;
        static const char * digits = "0123456789abcdef";
        for (const std::uint32_t w : h_) {
            for (int i = 28; i >= 0; i -= 4) out += digits[(w >> i) & 0xF];
        }
        return out;
    }

private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return std::rotr(x, n); }
    void block(const std::uint8_t * p) {
        static constexpr std::uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
                   (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> buf_{};
    std::size_t fill_ = 0;
    std::uint64_t total_ = 0;
};

std::string now_utc() {
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string read_file(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) fail("cannot read " + p.string());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Write to a sibling temp file, then rename over the target.
void write_atomic(const fs::path & p, const void * data, std::size_t n) {
    const fs::path tmp = p.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) fail("cannot write " + tmp.string());
        f.write(static_cast<const char *>(data), static_cast<std::streamsize>(n));
        if (!f) fail("short write to " + tmp.string());
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) fail("cannot rename " + tmp.string() + ": " + ec.message());
}

} // namespace

std::string Store::sha256_hex(std::span<const std::uint8_t> bytes) {
    Sha256 s;
    s.update(bytes.data(), bytes.size());
    return s.hex();
}

std::string Store::id_for(const std::string & name) {
    std::string id;
    for (const unsigned char c : name) {
        const bool ok = std::isalnum(c) || c == '_' || c == '-' || c == '.';
        id += ok ? static_cast<char>(c) : '_';
    }
    while (!id.empty() && id.front() == '.') id.erase(id.begin());   // no hidden / relative entries
    if (id.size() > 128) id.resize(128);
    return id.empty() ? "unnamed" : id;
}

Store Store::open(const std::string & dir, int dim) {
    Store s;
    s.dir_ = dir;
    s.dim_ = dim;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) fail("voices: cannot create " + dir + ": " + ec.message());
    for (const auto & entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory()) continue;
        const fs::path meta = entry.path() / "voice.json";
        if (!fs::exists(meta)) continue;
        const std::string id = entry.path().filename().string();
        try {
            const Json j = Json::parse(read_file(meta));
            auto v = std::make_shared<Voice>();
            v->id = id;
            v->name = j.str_or("name", id);
            v->ref_text = j.str_or("ref_text", "");
            v->model = j.str_or("model", "");
            v->sample_sha256 = j.str_or("sample_sha256", "");
            v->created = j.str_or("created", "");
            v->sample_seconds = j.num_or("sample_seconds", 0.0);
            const int vdim = static_cast<int>(j.int_or("dim", 0));
            const std::string emb = read_file(entry.path() / "embedding.f32");
            const bool sized = emb.size() == static_cast<std::size_t>(vdim) * sizeof(float);
            if (!sized || (dim > 0 && vdim != dim)) {
                std::fprintf(stderr, "voices: skipping %s: embedding is %zu floats, the model wants %d\n", id.c_str(),
                             emb.size() / sizeof(float), dim);
                continue;
            }
            v->embedding.resize(static_cast<std::size_t>(vdim));
            std::memcpy(v->embedding.data(), emb.data(), emb.size());
            s.voices_[id] = std::move(v);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "voices: skipping %s: %s\n", id.c_str(), e.what());
        }
    }
    return s;
}

std::size_t Store::size() const {
    std::lock_guard<std::mutex> l(*m_);
    return voices_.size();
}

std::vector<std::string> Store::ids() const {
    std::lock_guard<std::mutex> l(*m_);
    std::vector<std::string> out;
    out.reserve(voices_.size());
    for (const auto & [id, v] : voices_) out.push_back(id);
    return out;
}

std::vector<VoicePtr> Store::all() const {
    std::lock_guard<std::mutex> l(*m_);
    std::vector<VoicePtr> out;
    out.reserve(voices_.size());
    for (const auto & [id, v] : voices_) out.push_back(v);
    return out;
}

VoicePtr Store::find(const std::string & id) const {
    std::lock_guard<std::mutex> l(*m_);
    const auto it = voices_.find(id);
    return it == voices_.end() ? nullptr : it->second;
}

VoicePtr Store::put(Voice v, std::span<const std::uint8_t> sample_wav) {
    if (v.id != id_for(v.id)) fail("voices: bad id '" + v.id + "'");
    if (dim_ > 0 && static_cast<int>(v.embedding.size()) != dim_) fail("voices: embedding has the wrong size");
    if (v.created.empty()) v.created = now_utc();
    if (v.sample_sha256.empty()) v.sample_sha256 = sha256_hex(sample_wav);
    const fs::path vdir = fs::path(dir_) / v.id;
    std::error_code ec;
    fs::create_directories(vdir, ec);
    if (ec) fail("voices: cannot create " + vdir.string() + ": " + ec.message());
    // Payloads first, the manifest last: a voice exists once voice.json does.
    write_atomic(vdir / "embedding.f32", v.embedding.data(), v.embedding.size() * sizeof(float));
    if (!sample_wav.empty()) write_atomic(vdir / "sample.wav", sample_wav.data(), sample_wav.size());
    Json j;
    j.set("name", v.name);
    j.set("ref_text", v.ref_text);
    j.set("model", v.model);
    j.set("sample_sha256", v.sample_sha256);
    j.set("sample_seconds", v.sample_seconds);
    j.set("dim", static_cast<std::int64_t>(v.embedding.size()));
    j.set("created", v.created);
    const std::string text = j.dump() + "\n";
    write_atomic(vdir / "voice.json", text.data(), text.size());
    auto ptr = std::make_shared<const Voice>(std::move(v));
    std::lock_guard<std::mutex> l(*m_);
    voices_[ptr->id] = ptr;
    return ptr;
}

bool Store::remove(const std::string & id) {
    {
        std::lock_guard<std::mutex> l(*m_);
        if (voices_.erase(id) == 0) return false;
    }
    std::error_code ec;
    fs::remove_all(fs::path(dir_) / id, ec);
    if (ec) std::fprintf(stderr, "voices: could not delete %s: %s\n", id.c_str(), ec.message().c_str());
    return true;
}

} // namespace navi::voices
