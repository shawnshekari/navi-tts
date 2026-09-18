#include "runtime/audio/wav.h"

#include "runtime/common/error.h"

#include <cmath>
#include <cstring>
#include <fstream>

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
