#pragma once

// The codec decoder (docs/model.md "Codec decoder"): 16 codebooks per frame
// -> 1920 samples at 24 kHz. Runs on the device per chunk of frames with every
// layer's left context (conv history, attention KV) carried across chunks, so
// a chunked decode equals a one-shot decode (DESIGN 5.2). Scratch is sized for
// max_chunk frames once; longer inputs are looped.

#include "model/qwen3tts/params.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace navi::qwen3tts {

// Tensors the vocoder re-lays out itself (concatenated projections, phase-major
// transposed-conv weights): DeviceWeights::upload should skip them.
bool vocoder_owns_tensor(const std::string & name);

class Vocoder {
public:
    // `file` must still be open (owned tensors are read from the mapping).
    static std::unique_ptr<Vocoder> create(const Device & dev, const NaviFile & file, const DeviceWeights & w,
                                           const Params & p, int max_chunk_frames = 32, int max_frames = 4096);
    ~Vocoder();

    // Start a new utterance: clears histories and the attention caches.
    void reset();

    // Decode `n_frames` x 16 codes (row-major), appending n_frames * 1920
    // samples to `pcm`. Chunks internally; state persists across calls until reset().
    void decode(std::span<const std::int32_t> codes, int n_frames, std::vector<float> & pcm);

    int samples_per_frame() const;
    int max_chunk_frames() const;
    // device time spent in the last decode() call (ms)
    double last_ms() const;

    // Debug: called after every stage of a chunk with (name, device buffer, rows, cols).
    using StageHook = std::function<void(const std::string &, const float *, int, int)>;
    void set_stage_hook(StageHook hook);

    struct Impl;

private:
    Vocoder() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
