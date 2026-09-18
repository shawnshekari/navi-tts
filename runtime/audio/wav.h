#pragma once

// WAV out: 16-bit PCM mono. (WAV in with header repair arrives with the voice store.)

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace navi::audio {

std::vector<std::int16_t> to_s16(std::span<const float> pcm);          // clamp, round to nearest
std::vector<std::uint8_t> wav_bytes(std::span<const std::int16_t> s16, int sample_rate);
void write_wav(const std::string & path, std::span<const float> pcm, int sample_rate);

} // namespace navi::audio
