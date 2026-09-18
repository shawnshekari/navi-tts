#pragma once

// One request end to end (DESIGN 5): tokenize -> prompt -> prefill -> frames
// until EOS or the cap -> vocoder in batches of frames, PCM out per batch.
// Owns the model modules; the server and the bench harness both call this.

#include "model/qwen3tts/frame.h"
#include "model/qwen3tts/params.h"
#include "model/qwen3tts/speaker.h"
#include "model/qwen3tts/talker.h"
#include "model/qwen3tts/tokenizer.h"
#include "model/qwen3tts/vocoder.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace navi::qwen3tts {

// Streaming batch sizes (frames; 12.5 frames/s). Measured on the XTX: first
// audio 67 ms at 4 frames vs 102 at 8; vocoder 1.17 ms/frame at 16 vs 1.56 at 8.
// Batching never changes the PCM. The server streams with these; the bench
// measures this configuration.
constexpr int STREAM_FIRST_BATCH = 4;
constexpr int STREAM_BATCH = 16;

struct SynthRequest {
    std::string text;
    std::string language = "english";
    std::span<const float> speaker;   // [hidden] speaker embedding
    std::uint64_t seed = 0;
    int max_frames = 600;             // 48 s at 12.5 Hz (DESIGN 2)
    Sampling sampling;                // defaults: the model's generation_config.json
    int vocoder_batch = 0;            // frames per vocoder call; 0 = the vocoder's max. PCM is identical for any batching.
    int first_batch = 0;              // frames in the first call (time to first audio); 0 = vocoder_batch
};

struct SynthStats {
    int n_tokens = 0, n_frames = 0;
    bool hit_cap = false, eos = false;
    double tokenize_ms = 0, prefill_ms = 0, frames_ms = 0, vocoder_ms = 0, total_ms = 0, ttfa_ms = 0;   // prefill_ms includes tokenize_ms
    double audio_s = 0, rtf = 0;
};

class Graph {
public:
    // Loads everything from the .navi: tokenizer, weights to the device, the
    // talker, the frame kernel, the vocoder. The file is closed afterwards.
    static std::unique_ptr<Graph> load(const Device & dev, const std::string & navi_path,
                                       const FrameOptions & fopt = {}, int vocoder_chunk = 32);
    ~Graph();

    const Params & params() const;
    const Tokenizer & tokenizer() const;
    const DeviceWeights & weights() const;

    // Speaker embedding from mono 24 kHz audio (a clone). Runs once per voice.
    std::vector<float> embed_speaker(std::span<const float> wav_24k);

    // Synthesises `req`; every vocoder batch's PCM (24 kHz f32) goes to `on_pcm`
    // as it is produced. Throws navi::Error on a failed frame (barrier timeout).
    SynthStats synth(const SynthRequest & req, const std::function<void(std::span<const float>)> & on_pcm);

    // Convenience: whole utterance into one buffer.
    SynthStats synth(const SynthRequest & req, std::vector<float> & pcm);

    struct Impl;

private:
    Graph() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
