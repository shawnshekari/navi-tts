#pragma once

// WAV out: 16-bit PCM mono. WAV in: PCM 8/16/24/32-bit and float32, any channel
// count (averaged to mono); streaming headers with 0xFFFFFFFF sizes are accepted
// (DESIGN 3.1). No resampling: the caller checks the rate.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace navi::audio {

std::vector<std::int16_t> to_s16(std::span<const float> pcm);          // clamp, round to nearest
std::vector<std::uint8_t> wav_bytes(std::span<const std::int16_t> s16, int sample_rate);
void write_wav(const std::string & path, std::span<const float> pcm, int sample_rate);

struct Wav {
    int sample_rate = 0;
    int channels = 0;           // as stored; `pcm` is already mono
    std::vector<float> pcm;     // [-1, 1]
};
Wav parse_wav(std::span<const std::uint8_t> bytes);
Wav read_wav(const std::string & path);

} // namespace navi::audio
