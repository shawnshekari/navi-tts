#pragma once

// Speaker encoder (docs/model.md "Speaker encoder"): 24 kHz audio -> log-mel
// on the host (STFT, librosa-style slaney filterbank) -> ECAPA-TDNN on the
// device -> 1024-d speaker embedding. Runs once per clone.

#include "model/qwen3tts/params.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"

#include <memory>
#include <span>
#include <vector>

namespace navi::qwen3tts {

class SpeakerEncoder {
public:
    static std::unique_ptr<SpeakerEncoder> create(const Device & dev, const DeviceWeights & w, const Params & p,
                                                  double max_seconds = 30.0);
    ~SpeakerEncoder();

    // Log-mel spectrogram, [n_mels][n_frames] row-major, exactly the model's front end.
    static std::vector<float> mel(std::span<const float> wav_24k, const Params & p, int * n_frames);

    // [hidden] embedding from a mono 24 kHz waveform in [-1, 1].
    std::vector<float> embed(std::span<const float> wav_24k);
    // ... or from a precomputed mel (tests).
    std::vector<float> embed_mel(std::span<const float> mel, int n_frames);

    double last_ms() const;

    struct Impl;

private:
    SpeakerEncoder() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
