#pragma once

// The codec encoder (docs/model.md "Codec encoder"): mono 24 kHz audio -> 16 RVQ codes per
// 12.5 Hz frame. Those codes are what an in-context clone is conditioned on, beside the
// speaker embedding (DESIGN 9, docs/model.md "Prompt layout (ICL)").
//
// Like the speaker encoder this runs **once per clone**, never per frame, so it is written
// for obvious correctness and bounded memory rather than for ms/frame: the conv stack is
// chunked in time with each conv's causal history threaded across chunks (a chunked encode
// equals a one-shot encode, which is a gate), and everything after it runs once over the
// whole sequence.
//
// Present only in a blob converted with `--with-encoder`; `Params::has_codec_encoder` says
// so. A blob without it still serves voices whose codes were made elsewhere - the codes are
// just numbers, and the codec is the same checkpoint in every model of this family.

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

class CodecEncoder {
public:
    // Scratch is sized once, for `max_seconds` of audio and a conv-stack chunk of
    // `chunk_frames` frames; longer audio is refused, and a bigger chunk needs a bigger
    // encoder. 32 frames (2.56 s) keeps the widest buffer near 16 MB; a chunk covering a
    // whole clip is what the chunked-equals-one-shot gate builds, and costs ~230 MB at 15 s.
    static std::unique_ptr<CodecEncoder> create(const Device & dev, const NaviFile & file,
                                                const DeviceWeights & w, const Params & p,
                                                double max_seconds = 30.0, int chunk_frames = 32);
    ~CodecEncoder();

    // [n_frames][16] codes, row-major - the layout tests/reference/ref_codes.npy and the
    // voice store both use. n_frames = ceil(samples / codec_encoder.downsample_rate), the
    // same count the reference implementation returns. Throws navi::Error on a clip that is
    // empty, longer than max_seconds, or not finite.
    std::vector<std::int32_t> encode(std::span<const float> wav_24k, int * n_frames = nullptr);

    int max_samples() const;
    int frames_for(int n_samples) const;
    int codebooks() const;
    double last_ms() const;          // device time of the last encode()

    // Tests: called after each stage with (name, device buffer, rows, cols). "conv", "xf"
    // and "latents" match the reference dumps; "proj_s" and "proj_a" are the two quantiser
    // groups' inputs, which is what a code mismatch has to be explained against.
    using StageHook = std::function<void(const std::string &, const float *, int, int)>;
    void set_stage_hook(StageHook hook);

    // Tests: the codebook buffer the RVQ step actually reads - the arena's own rows when
    // the converter leaves them adjacent in the right order, and a copy when it does not.
    // A negative control has to write *here*; writing to the arena instead perturbs
    // nothing, which is a mistake this accessor exists because of. [n_valid_q][2048][256].
    const float * d_codebooks() const;

    // Tests: run the conv stack in SMALLER chunks than this encoder was built for, to
    // prove the chunking is exact. 0 restores the built-in size; more than it is refused,
    // because the scratch is sized for it.
    void set_chunk_frames(int frames);
    int chunk_frames() const;

    struct Impl;

private:
    CodecEncoder() = default;
    std::unique_ptr<Impl> impl_;
};

} // namespace navi::qwen3tts
