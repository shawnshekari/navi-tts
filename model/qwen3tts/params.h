#pragma once

// Qwen3-TTS hyperparameters and the tensor table the engine expects, read
// from a .navi file's KV section (docs/model.md). A fine-tune or the 1.7B
// changes these values, not this code.

#include "runtime/weights/navi_file.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace navi::qwen3tts {

struct TransformerParams {
    int    hidden = 0;
    int    n_layer = 0;
    int    n_head = 0;
    int    n_kv_head = 0;
    int    head_dim = 0;
    int    ff = 0;           // intermediate_size
    float  rms_eps = 0.f;
    double rope_theta = 0.0;
    int    vocab = 0;
    int    max_pos = 0;
};

struct Params {
    std::string name;
    std::string dtype;

    // text side
    int im_start_id = 0, im_end_id = 0, assistant_id = 0;
    int tts_bos_id = 0, tts_eos_id = 0, tts_pad_id = 0;
    int text_vocab = 0;
    int text_hidden = 0;

    // talker
    TransformerParams talker;
    int n_code_groups = 0;          // 16 codebooks per frame
    int position_id_per_second = 0;
    std::vector<int> mrope_section;
    bool mrope_interleaved = false;
    int codec_bos_id = 0, codec_eos_id = 0, codec_pad_id = 0;
    int codec_think_id = 0, codec_nothink_id = 0, codec_think_bos_id = 0, codec_think_eos_id = 0;
    std::map<std::string, int> language_ids;

    // code predictor
    TransformerParams cp;

    // speaker encoder + its mel front end
    int spk_dim = 0, spk_sample_rate = 0;
    int mel_n_fft = 0, mel_num_mels = 0, mel_hop = 0, mel_win = 0;
    float mel_fmin = 0.f, mel_fmax = 0.f;

    // codec decoder (vocoder)
    int codec_sample_rate = 0;
    int codec_upsample = 0;          // samples per frame (1920 -> 12.5 Hz at 24 kHz)
    TransformerParams vocoder;       // the pre_transformer
    int vq_latent = 0, vq_codebook_dim = 0, vq_codebook_size = 0, vq_n_q = 0, vq_n_semantic = 0;
    int vocoder_decoder_dim = 0;
    int vocoder_sliding_window = 0;
    std::vector<int> vocoder_upsample_rates;    // [8, 5, 4, 3]
    std::vector<int> vocoder_upsampling_ratios; // [2, 2]
    bool has_codec_encoder = false;

    // generation defaults
    float temperature = 1.f, top_p = 1.f, repetition_penalty = 1.f;
    int   top_k = 0;
    float cp_temperature = 1.f, cp_top_p = 1.f;
    int   cp_top_k = 0;
};

// Throws navi::Error on a file that is not a qwen3-tts .navi.
Params read_params(const NaviFile & f);

struct ExpectedTensor {
    std::string name;
    std::vector<std::uint64_t> dims;
};

// Every tensor the engine will load for these parameters, with its shape.
std::vector<ExpectedTensor> expected_tensors(const Params & p);

struct Validation {
    std::vector<std::string> missing;
    std::vector<std::string> shape_mismatch;   // "name: file [a, b] expected [c, d]"
    std::vector<std::string> unused;           // present but not in the table
    bool ok() const { return missing.empty() && shape_mismatch.empty(); }
};

Validation validate(const NaviFile & f, const Params & p);

} // namespace navi::qwen3tts
