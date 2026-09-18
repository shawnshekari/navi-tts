#include "runtime/audio/wav.h"

#include "runtime/common/error.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

namespace navi::audio {

std::vector<std::int16_t> to_s16(std::span<const float> pcm) {
    std::vector<std::int16_t> out(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i];
        if (v > 1.f) v = 1.f;
        if (v < -1.f) v = -1.f;
        out[i] = static_cast<std::int16_t>(std::lrintf(v * 32767.f));
    }
    return out;
}

std::vector<std::uint8_t> wav_bytes(std::span<const std::int16_t> s16, int sample_rate) {
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(s16.size() * 2);
    std::vector<std::uint8_t> out(44 + data_bytes);
    auto put32 = [&](std::size_t at, std::uint32_t v) { std::memcpy(&out[at], &v, 4); };
    auto put16 = [&](std::size_t at, std::uint16_t v) { std::memcpy(&out[at], &v, 2); };
    std::memcpy(&out[0], "RIFF", 4);
    put32(4, 36 + data_bytes);
    std::memcpy(&out[8], "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1);                       // PCM
    put16(22, 1);                       // mono
    put32(24, static_cast<std::uint32_t>(sample_rate));
    put32(28, static_cast<std::uint32_t>(sample_rate) * 2);
    put16(32, 2);
    put16(34, 16);
    std::memcpy(&out[36], "data", 4);
    put32(40, data_bytes);
    std::memcpy(&out[44], s16.data(), data_bytes);
    return out;
}

void write_wav(const std::string & path, std::span<const float> pcm, int sample_rate) {
    const auto bytes = wav_bytes(to_s16(pcm), sample_rate);
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("cannot write " + path);
    f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

} // namespace navi::audio

namespace navi::audio {

namespace {
std::uint32_t rd32(const std::uint8_t * p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
std::uint16_t rd16(const std::uint8_t * p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
}

Wav parse_wav(std::span<const std::uint8_t> b) {
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) fail("wav: not a RIFF/WAVE file");
    Wav w;
    int format = 0, bits = 0;
    std::size_t p = 12;
    const std::uint8_t * data = nullptr;
    std::size_t data_len = 0;
    while (p + 8 <= b.size()) {
        const std::uint8_t * id = b.data() + p;
        std::size_t len = rd32(b.data() + p + 4);
        p += 8;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (len < 16 || p + 16 > b.size()) fail("wav: bad fmt chunk");
            format = rd16(b.data() + p);
            w.channels = rd16(b.data() + p + 2);
            w.sample_rate = static_cast<int>(rd32(b.data() + p + 4));
            bits = rd16(b.data() + p + 14);
            if (format == 0xFFFE && len >= 40) format = rd16(b.data() + p + 24);   // WAVE_FORMAT_EXTENSIBLE: sub-format
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = b.data() + p;
            // streaming headers write 0xFFFFFFFF (or 0): take what is there
            data_len = (len == 0xFFFFFFFFu || len == 0 || p + len > b.size()) ? b.size() - p : len;
            break;
        }
        if (len == 0xFFFFFFFFu) break;
        p += len + (len & 1);
    }
    if (!data) fail("wav: no data chunk");
    if (w.channels < 1 || w.sample_rate < 1) fail("wav: bad format chunk");
    const bool is_float = format == 3;
    if (!(format == 1 || is_float)) fail("wav: unsupported format " + std::to_string(format));
    if (!(bits == 8 || bits == 16 || bits == 24 || bits == 32)) fail("wav: unsupported bit depth " + std::to_string(bits));
    if (is_float && bits != 32) fail("wav: float must be 32-bit");
    const std::size_t bps = static_cast<std::size_t>(bits) / 8, frame = bps * static_cast<std::size_t>(w.channels);
    const std::size_t n = data_len / frame;
    w.pcm.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        float acc = 0.f;
        for (int c = 0; c < w.channels; ++c) {
            const std::uint8_t * s = data + i * frame + static_cast<std::size_t>(c) * bps;
            float v;
            if (is_float) { std::memcpy(&v, s, 4); }
            else if (bits == 8) v = (static_cast<int>(s[0]) - 128) / 128.f;
            else if (bits == 16) v = static_cast<std::int16_t>(rd16(s)) / 32768.f;
            else if (bits == 24) v = static_cast<std::int32_t>((static_cast<std::uint32_t>(s[0]) << 8 | static_cast<std::uint32_t>(s[1]) << 16 | static_cast<std::uint32_t>(s[2]) << 24)) / 2147483648.f;
            else v = static_cast<std::int32_t>(rd32(s)) / 2147483648.f;
            acc += v;
        }
        w.pcm[i] = acc / static_cast<float>(w.channels);
    }
    return w;
}

Wav read_wav(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_wav(bytes);
}

} // namespace navi::audio
