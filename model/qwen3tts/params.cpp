#include "model/qwen3tts/params.h"

#include "runtime/common/error.h"

#include <set>

namespace navi::qwen3tts {

namespace {

TransformerParams read_transformer(const NaviFile & f, const std::string & ns) {
    TransformerParams t;
    t.hidden     = static_cast<int>(f.kv_i64(ns + ".hidden_size"));
    t.n_layer    = static_cast<int>(f.kv_i64(ns + ".num_hidden_layers"));
    t.n_head     = static_cast<int>(f.kv_i64(ns + ".num_attention_heads"));
    t.n_kv_head  = static_cast<int>(f.kv_i64(ns + ".num_key_value_heads"));
    t.head_dim   = static_cast<int>(f.kv_i64(ns + ".head_dim"));
    t.ff         = static_cast<int>(f.kv_i64(ns + ".intermediate_size"));
    t.rms_eps    = static_cast<float>(f.kv_f64(ns + ".rms_norm_eps"));
    t.rope_theta = f.kv_f64(ns + ".rope_theta");
    t.vocab      = static_cast<int>(f.kv_i64_or(ns + ".vocab_size", 0));
    t.max_pos    = static_cast<int>(f.kv_i64_or(ns + ".max_position_embeddings", 0));
    return t;
}

std::vector<int> to_int(const std::vector<std::int64_t> & v) {
    return std::vector<int>(v.begin(), v.end());
}

using Dims = std::vector<std::uint64_t>;

void add_decoder_layers(std::vector<ExpectedTensor> & out, const std::string & prefix, const TransformerParams & t,
                        bool qk_norm) {
    const auto H = static_cast<std::uint64_t>(t.hidden);
    const auto Q = static_cast<std::uint64_t>(t.n_head * t.head_dim);
    const auto K = static_cast<std::uint64_t>(t.n_kv_head * t.head_dim);
    const auto F = static_cast<std::uint64_t>(t.ff);
    const auto D = static_cast<std::uint64_t>(t.head_dim);
    for (int l = 0; l < t.n_layer; ++l) {
        const std::string p = prefix + ".layers." + std::to_string(l) + ".";
        out.push_back({p + "input_layernorm.weight", {H}});
        out.push_back({p + "post_attention_layernorm.weight", {H}});
        out.push_back({p + "self_attn.q_proj.weight", {Q, H}});
        out.push_back({p + "self_attn.k_proj.weight", {K, H}});
        out.push_back({p + "self_attn.v_proj.weight", {K, H}});
        out.push_back({p + "self_attn.o_proj.weight", {H, Q}});
        if (qk_norm) {
            out.push_back({p + "self_attn.q_norm.weight", {D}});
            out.push_back({p + "self_attn.k_norm.weight", {D}});
        }
        out.push_back({p + "mlp.gate_proj.weight", {F, H}});
        out.push_back({p + "mlp.up_proj.weight", {F, H}});
        out.push_back({p + "mlp.down_proj.weight", {H, F}});
    }
}

// The codec encoder (ICL cloning), laid out the way MimiEncoder builds itself so that a
// config with different ratios still lands on the right layer indices: layer 0 is the input
// conv, then per ratio a residual unit, an ELU (no tensors) and a strided downsample, and
// finally an ELU and the output conv. Only what encoding uses is listed - no output_proj,
// and only the codebooks the talker has embeddings for (tools/convert.py::encoder_keep).
void add_codec_encoder(std::vector<ExpectedTensor> & out, const Params & p) {
    const Params::CodecEncoder & e = p.codec_encoder;
    const std::string pre = "codec.encoder.";
    const auto u = [](int v) { return static_cast<std::uint64_t>(v); };

    int dim = e.num_filters, idx = 0, prod = 1;
    for (int r : e.ratios) prod *= r;
    out.push_back({pre + "encoder.layers.0.conv.weight", {u(dim), 1, u(e.kernel_size)}});
    out.push_back({pre + "encoder.layers.0.conv.bias", {u(dim)}});
    ++idx;
    for (std::size_t r = e.ratios.size(); r-- > 0;) {   // the encoder walks the ratios reversed
        const int stride = e.ratios[r], inner = dim / e.compress;
        for (int j = 0; j < e.num_residual_layers; ++j) {
            const std::string b = pre + "encoder.layers." + std::to_string(idx) + ".block.";
            out.push_back({b + "1.conv.weight", {u(inner), u(dim), u(e.residual_kernel_size)}});
            out.push_back({b + "1.conv.bias", {u(inner)}});
            out.push_back({b + "3.conv.weight", {u(dim), u(inner), 1}});
            out.push_back({b + "3.conv.bias", {u(dim)}});
            ++idx;
        }
        ++idx;   // ELU
        const std::string c = pre + "encoder.layers." + std::to_string(idx) + ".conv.";
        out.push_back({c + "weight", {u(dim * 2), u(dim), u(stride * 2)}});   // kernel = 2 x stride
        out.push_back({c + "bias", {u(dim * 2)}});
        ++idx;
        dim *= 2;
    }
    ++idx;   // ELU
    const std::string last = pre + "encoder.layers." + std::to_string(idx) + ".conv.";
    out.push_back({last + "weight", {u(e.latent), u(dim), u(e.last_kernel_size)}});
    out.push_back({last + "bias", {u(e.latent)}});

    // Its transformer is not a Qwen3 decoder: LayerNorm with bias instead of RMSNorm, no GQA,
    // no QK-norm, and an MLP of two bias-free Linears with GELU between them.
    const auto L = u(e.latent), F = u(e.ff);
    for (int l = 0; l < e.n_layer; ++l) {
        const std::string t = pre + "encoder_transformer.layers." + std::to_string(l) + ".";
        for (const char * n : {"input_layernorm", "post_attention_layernorm"}) {
            out.push_back({t + n + ".weight", {L}});
            out.push_back({t + n + ".bias", {L}});
        }
        for (const char * n : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
            out.push_back({t + "self_attn." + n + ".weight", {L, L}});
        }
        out.push_back({t + "mlp.fc1.weight", {F, L}});
        out.push_back({t + "mlp.fc2.weight", {L, F}});
        out.push_back({t + "self_attn_layer_scale.scale", {L}});
        out.push_back({t + "mlp_layer_scale.scale", {L}});
    }

    // 25 Hz -> 12.5 Hz. read_params has already checked that the ratios and the declared
    // samples per frame agree, so this stride is 2 and its kernel 4; it is derived anyway.
    const int ds = prod > 0 ? e.downsample_rate / prod : 0;
    out.push_back({pre + "downsample.conv.weight", {L, L, u(2 * ds)}});

    // Split RVQ. Encoding needs each group's input_proj and the codebooks themselves;
    // the acoustic group quantises the FULL latent, not the semantic residual.
    const auto VQ = u(e.vq_dim), CS = u(e.codebook_size), CD = u(e.codebook_dim);
    const std::string sem = pre + "quantizer.semantic_residual_vector_quantizer.";
    out.push_back({sem + "input_proj.weight", {VQ, L, 1}});
    for (int i = 0; i < e.n_semantic; ++i) {
        out.push_back({sem + "layers." + std::to_string(i) + ".codebook.codebook", {CS, CD}});
    }
    const std::string aco = pre + "quantizer.acoustic_residual_vector_quantizer.";
    out.push_back({aco + "input_proj.weight", {VQ, L, 1}});
    for (int i = 0; i < e.n_valid_q - e.n_semantic; ++i) {
        out.push_back({aco + "layers." + std::to_string(i) + ".codebook.codebook", {CS, CD}});
    }
}

} // namespace

Params read_params(const NaviFile & f) {
    if (f.kv_str("general.arch") != "qwen3-tts") {
        fail(f.path() + ": architecture '" + f.kv_str("general.arch") + "' is not qwen3-tts");
    }
    Params p;
    p.name  = f.kv_str("general.name");
    p.dtype = f.kv_str("general.dtype");

    p.im_start_id  = static_cast<int>(f.kv_i64("text.im_start_id"));
    p.im_end_id    = static_cast<int>(f.kv_i64("text.im_end_id"));
    p.assistant_id = static_cast<int>(f.kv_i64("text.assistant_id"));
    p.tts_bos_id   = static_cast<int>(f.kv_i64("text.tts_bos_id"));
    p.tts_eos_id   = static_cast<int>(f.kv_i64("text.tts_eos_id"));
    p.tts_pad_id   = static_cast<int>(f.kv_i64("text.tts_pad_id"));
    p.text_vocab   = static_cast<int>(f.kv_i64("talker.text_vocab_size"));
    p.text_hidden  = static_cast<int>(f.kv_i64("talker.text_hidden_size"));

    p.talker = read_transformer(f, "talker");
    p.n_code_groups          = static_cast<int>(f.kv_i64("talker.num_code_groups"));
    p.position_id_per_second = static_cast<int>(f.kv_i64_or("talker.position_id_per_seconds", 0));
    p.mrope_section    = to_int(f.kv_i64_array("talker.mrope_section"));
    p.mrope_interleaved = f.kv_i64_or("talker.mrope_interleaved", 0) != 0;
    p.codec_bos_id       = static_cast<int>(f.kv_i64("talker.codec_bos_id"));
    p.codec_eos_id       = static_cast<int>(f.kv_i64("talker.codec_eos_token_id"));
    p.codec_pad_id       = static_cast<int>(f.kv_i64("talker.codec_pad_id"));
    p.codec_think_id     = static_cast<int>(f.kv_i64("talker.codec_think_id"));
    p.codec_nothink_id   = static_cast<int>(f.kv_i64("talker.codec_nothink_id"));
    p.codec_think_bos_id = static_cast<int>(f.kv_i64("talker.codec_think_bos_id"));
    p.codec_think_eos_id = static_cast<int>(f.kv_i64("talker.codec_think_eos_id"));
    for (const auto & [k, v] : f.kv()) {
        const std::string pre = "talker.language.";
        if (k.compare(0, pre.size(), pre) == 0 && v.type == KV::Type::I64) {
            p.language_ids[k.substr(pre.size())] = static_cast<int>(v.i64);
        }
    }

    p.cp = read_transformer(f, "cp");

    p.spk_dim         = static_cast<int>(f.kv_i64("speaker.enc_dim"));
    p.spk_sample_rate = static_cast<int>(f.kv_i64("speaker.sample_rate"));
    p.mel_n_fft    = static_cast<int>(f.kv_i64("speaker.mel.n_fft"));
    p.mel_num_mels = static_cast<int>(f.kv_i64("speaker.mel.num_mels"));
    p.mel_hop      = static_cast<int>(f.kv_i64("speaker.mel.hop_size"));
    p.mel_win      = static_cast<int>(f.kv_i64("speaker.mel.win_size"));
    p.mel_fmin     = static_cast<float>(f.kv_f64("speaker.mel.fmin"));
    p.mel_fmax     = static_cast<float>(f.kv_f64("speaker.mel.fmax"));

    p.codec_sample_rate = static_cast<int>(f.kv_i64("codec.output_sample_rate"));
    p.codec_upsample    = static_cast<int>(f.kv_i64("codec.decode_upsample_rate"));
    p.vocoder = read_transformer(f, "codec.decoder");
    p.vq_latent        = static_cast<int>(f.kv_i64("codec.decoder.latent_dim"));
    p.vq_codebook_dim  = static_cast<int>(f.kv_i64("codec.decoder.codebook_dim"));
    p.vq_codebook_size = static_cast<int>(f.kv_i64("codec.decoder.codebook_size"));
    p.vq_n_q           = static_cast<int>(f.kv_i64("codec.decoder.num_quantizers"));
    p.vq_n_semantic    = static_cast<int>(f.kv_i64("codec.decoder.num_semantic_quantizers"));
    p.vocoder_decoder_dim    = static_cast<int>(f.kv_i64("codec.decoder.decoder_dim"));
    p.vocoder_sliding_window = static_cast<int>(f.kv_i64_or("codec.decoder.sliding_window", 0));
    p.vocoder_upsample_rates     = to_int(f.kv_i64_array("codec.decoder.upsample_rates"));
    p.vocoder_upsampling_ratios  = to_int(f.kv_i64_array("codec.decoder.upsampling_ratios"));
    p.has_codec_encoder = f.kv_i64_or("general.has_codec_encoder", 0) != 0;
    if (p.has_codec_encoder) {
        Params::CodecEncoder & e = p.codec_encoder;
        const std::string ns = "codec.encoder.";
        e.latent      = static_cast<int>(f.kv_i64(ns + "hidden_size"));
        e.n_layer     = static_cast<int>(f.kv_i64(ns + "num_hidden_layers"));
        e.n_head      = static_cast<int>(f.kv_i64(ns + "num_attention_heads"));
        e.head_dim    = static_cast<int>(f.kv_i64(ns + "head_dim"));
        e.ff          = static_cast<int>(f.kv_i64(ns + "intermediate_size"));
        e.norm_eps    = static_cast<float>(f.kv_f64(ns + "norm_eps"));
        e.rope_theta  = f.kv_f64(ns + "rope_theta");
        e.max_pos     = static_cast<int>(f.kv_i64(ns + "max_position_embeddings"));
        e.layer_scale = static_cast<float>(f.kv_f64(ns + "layer_scale_initial_scale"));
        e.num_filters = static_cast<int>(f.kv_i64(ns + "num_filters"));
        e.kernel_size          = static_cast<int>(f.kv_i64(ns + "kernel_size"));
        e.last_kernel_size     = static_cast<int>(f.kv_i64(ns + "last_kernel_size"));
        e.residual_kernel_size = static_cast<int>(f.kv_i64(ns + "residual_kernel_size"));
        e.compress             = static_cast<int>(f.kv_i64(ns + "compress"));
        e.dilation_growth_rate = static_cast<int>(f.kv_i64(ns + "dilation_growth_rate"));
        e.num_residual_layers  = static_cast<int>(f.kv_i64(ns + "num_residual_layers"));
        e.ratios        = to_int(f.kv_i64_array(ns + "upsampling_ratios"));
        e.codebook_dim  = static_cast<int>(f.kv_i64(ns + "codebook_dim"));
        e.codebook_size = static_cast<int>(f.kv_i64(ns + "codebook_size"));
        e.n_q           = static_cast<int>(f.kv_i64(ns + "num_quantizers"));
        e.n_semantic    = static_cast<int>(f.kv_i64(ns + "num_semantic_quantizers"));
        e.vq_dim        = static_cast<int>(f.kv_i64(ns + "vector_quantization_hidden_dimension"));
        e.n_valid_q       = static_cast<int>(f.kv_i64("codec.encoder_valid_num_quantizers"));
        e.downsample_rate = static_cast<int>(f.kv_i64("codec.encode_downsample_rate"));
        e.sample_rate     = static_cast<int>(f.kv_i64("codec.input_sample_rate"));
        e.dtype = f.has_kv("general.codec_encoder_dtype") ? f.kv_str("general.codec_encoder_dtype") : p.dtype;
        // What the engine assumes about this component, checked once here instead of in a kernel.
        int prod = 1;
        for (int r : e.ratios) prod *= r;
        if (e.ratios.empty() || prod * 2 != e.downsample_rate) {
            fail(f.path() + ": the codec encoder's ratios multiply to " + std::to_string(prod)
                 + " and its downsample is stride 2, which is " + std::to_string(prod * 2)
                 + " samples per frame, not the " + std::to_string(e.downsample_rate) + " it declares");
        }
        if (e.n_valid_q != p.n_code_groups) {
            fail(f.path() + ": the codec encoder emits " + std::to_string(e.n_valid_q)
                 + " codebooks but the talker has embeddings for " + std::to_string(p.n_code_groups));
        }
        if (e.n_valid_q < 1 || e.n_valid_q > e.n_q) fail(f.path() + ": bad encoder_valid_num_quantizers");
        if (e.n_semantic != 1) {
            fail(f.path() + ": the engine assumes one semantic quantizer, this blob has "
                 + std::to_string(e.n_semantic));
        }
        if (e.latent != e.n_head * e.head_dim) {
            fail(f.path() + ": the codec encoder's transformer is " + std::to_string(e.latent)
                 + " wide but its heads cover " + std::to_string(e.n_head * e.head_dim));
        }
        if (e.num_filters % e.compress != 0) fail(f.path() + ": num_filters is not a multiple of compress");
        if (e.sample_rate != p.spk_sample_rate) {
            fail(f.path() + ": the codec encoder wants " + std::to_string(e.sample_rate)
                 + " Hz audio and the speaker encoder " + std::to_string(p.spk_sample_rate));
        }
    }

    p.temperature        = static_cast<float>(f.has_kv("gen.temperature") ? f.kv_f64("gen.temperature") : 1.0);
    p.top_p              = static_cast<float>(f.has_kv("gen.top_p") ? f.kv_f64("gen.top_p") : 1.0);
    p.top_k              = static_cast<int>(f.kv_i64_or("gen.top_k", 0));
    p.repetition_penalty = static_cast<float>(f.has_kv("gen.repetition_penalty") ? f.kv_f64("gen.repetition_penalty") : 1.0);
    p.cp_temperature     = static_cast<float>(f.has_kv("gen.subtalker_temperature") ? f.kv_f64("gen.subtalker_temperature") : 1.0);
    p.cp_top_p           = static_cast<float>(f.has_kv("gen.subtalker_top_p") ? f.kv_f64("gen.subtalker_top_p") : 1.0);
    p.cp_top_k           = static_cast<int>(f.kv_i64_or("gen.subtalker_top_k", 0));

    if (p.n_code_groups < 2) fail(f.path() + ": num_code_groups must be >= 2");
    return p;
}

std::vector<ExpectedTensor> expected_tensors(const Params & p) {
    std::vector<ExpectedTensor> out;
    const auto H  = static_cast<std::uint64_t>(p.talker.hidden);
    const auto TH = static_cast<std::uint64_t>(p.text_hidden);
    const auto TV = static_cast<std::uint64_t>(p.text_vocab);
    const auto CV = static_cast<std::uint64_t>(p.talker.vocab);       // 3072 codec vocab (cb0 + control)
    const auto PV = static_cast<std::uint64_t>(p.cp.vocab);           // 2048 per codebook
    const auto G  = static_cast<std::uint64_t>(p.n_code_groups);

    // tokenizer files
    out.push_back({"tokenizer.vocab.json", {}});
    out.push_back({"tokenizer.merges.txt", {}});
    out.push_back({"tokenizer.config.json", {}});

    // talker
    out.push_back({"talker.model.text_embedding.weight", {TV, TH}});
    out.push_back({"talker.text_projection.linear_fc1.weight", {TH, TH}});
    out.push_back({"talker.text_projection.linear_fc1.bias", {TH}});
    out.push_back({"talker.text_projection.linear_fc2.weight", {H, TH}});
    out.push_back({"talker.text_projection.linear_fc2.bias", {H}});
    out.push_back({"talker.model.codec_embedding.weight", {CV, H}});
    add_decoder_layers(out, "talker.model", p.talker, true);
    out.push_back({"talker.model.norm.weight", {H}});
    out.push_back({"talker.codec_head.weight", {CV, H}});

    // Code predictor: codebooks 1..15 each have an input embedding and a head.
    // The embeddings are stored at the TALKER's width, not the predictor's: the
    // same rows also feed the talker's next-frame input (docs/model.md "Code
    // predictor"). When the two widths differ the predictor's input is projected
    // down first; when they are equal that stage is the identity and absent.
    const auto CH = static_cast<std::uint64_t>(p.cp.hidden);
    for (std::uint64_t i = 0; i + 1 < G; ++i) {
        out.push_back({"talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight", {PV, H}});
        out.push_back({"talker.code_predictor.lm_head." + std::to_string(i) + ".weight", {PV, CH}});
    }
    if (p.cp.hidden != p.talker.hidden) {
        out.push_back({"talker.code_predictor.small_to_mtp_projection.weight", {CH, H}});
        out.push_back({"talker.code_predictor.small_to_mtp_projection.bias", {CH}});
    }
    add_decoder_layers(out, "talker.code_predictor.model", p.cp, true);
    out.push_back({"talker.code_predictor.model.norm.weight", {CH}});

    // speaker encoder (ECAPA-TDNN): shapes are fixed by the architecture, listed as shipped
    {
        const std::uint64_t C = 512, R = 64, SE = 128, MFA = 1536, ASP = 128;
        out.push_back({"speaker_encoder.blocks.0.conv.weight", {C, 128, 5}});
        out.push_back({"speaker_encoder.blocks.0.conv.bias", {C}});
        for (int b = 1; b <= 3; ++b) {
            const std::string bp = "speaker_encoder.blocks." + std::to_string(b) + ".";
            out.push_back({bp + "tdnn1.conv.weight", {C, C, 1}});
            out.push_back({bp + "tdnn1.conv.bias", {C}});
            for (int r = 0; r < 7; ++r) {
                const std::string rp = bp + "res2net_block.blocks." + std::to_string(r) + ".conv.";
                out.push_back({rp + "weight", {R, R, 3}});
                out.push_back({rp + "bias", {R}});
            }
            out.push_back({bp + "tdnn2.conv.weight", {C, C, 1}});
            out.push_back({bp + "tdnn2.conv.bias", {C}});
            out.push_back({bp + "se_block.conv1.weight", {SE, C, 1}});
            out.push_back({bp + "se_block.conv1.bias", {SE}});
            out.push_back({bp + "se_block.conv2.weight", {C, SE, 1}});
            out.push_back({bp + "se_block.conv2.bias", {C}});
        }
        out.push_back({"speaker_encoder.mfa.conv.weight", {MFA, MFA, 1}});
        out.push_back({"speaker_encoder.mfa.conv.bias", {MFA}});
        out.push_back({"speaker_encoder.asp.tdnn.conv.weight", {ASP, 3 * MFA, 1}});
        out.push_back({"speaker_encoder.asp.tdnn.conv.bias", {ASP}});
        out.push_back({"speaker_encoder.asp.conv.weight", {MFA, ASP, 1}});
        out.push_back({"speaker_encoder.asp.conv.bias", {MFA}});
        out.push_back({"speaker_encoder.fc.weight", {static_cast<std::uint64_t>(p.spk_dim), 2 * MFA, 1}});
        out.push_back({"speaker_encoder.fc.bias", {static_cast<std::uint64_t>(p.spk_dim)}});
    }

    // codec decoder
    {
        const auto L  = static_cast<std::uint64_t>(p.vq_latent);          // 1024
        const auto CD = static_cast<std::uint64_t>(p.vq_codebook_dim);    // 512
        const auto VQ = CD / 2;                                            // 256: rvq operates on codebook_dim/2
        const auto CS = static_cast<std::uint64_t>(p.vq_codebook_size);   // 2048
        const auto NQ = static_cast<std::uint64_t>(p.vq_n_q);             // 16
        const auto NS = static_cast<std::uint64_t>(p.vq_n_semantic);      // 1
        const auto DD = static_cast<std::uint64_t>(p.vocoder_decoder_dim); // 1536
        const std::string q = "codec.decoder.quantizer.";
        out.push_back({q + "rvq_first.input_proj.weight", {VQ, CD, 1}});
        out.push_back({q + "rvq_first.output_proj.weight", {CD, VQ, 1}});
        for (std::uint64_t i = 0; i < NS; ++i) {
            out.push_back({q + "rvq_first.vq.layers." + std::to_string(i) + "._codebook.codebook", {CS, VQ}});
        }
        out.push_back({q + "rvq_rest.input_proj.weight", {VQ, CD, 1}});
        out.push_back({q + "rvq_rest.output_proj.weight", {CD, VQ, 1}});
        for (std::uint64_t i = 0; i < NQ - NS; ++i) {
            out.push_back({q + "rvq_rest.vq.layers." + std::to_string(i) + "._codebook.codebook", {CS, VQ}});
        }
        out.push_back({"codec.decoder.pre_conv.conv.weight", {L, CD, 3}});
        out.push_back({"codec.decoder.pre_conv.conv.bias", {L}});

        const auto VH = static_cast<std::uint64_t>(p.vocoder.hidden);   // 512
        const std::string pt = "codec.decoder.pre_transformer.";
        out.push_back({pt + "input_proj.weight", {VH, L}});
        out.push_back({pt + "input_proj.bias", {VH}});
        add_decoder_layers(out, "codec.decoder.pre_transformer", p.vocoder, false);
        for (int l = 0; l < p.vocoder.n_layer; ++l) {
            const std::string lp = pt + "layers." + std::to_string(l) + ".";
            out.push_back({lp + "self_attn_layer_scale.scale", {VH}});
            out.push_back({lp + "mlp_layer_scale.scale", {VH}});
        }
        out.push_back({pt + "norm.weight", {VH}});
        out.push_back({pt + "output_proj.weight", {L, VH}});
        out.push_back({pt + "output_proj.bias", {L}});

        for (std::size_t u = 0; u < p.vocoder_upsampling_ratios.size(); ++u) {
            const auto r = static_cast<std::uint64_t>(p.vocoder_upsampling_ratios[u]);
            const std::string up = "codec.decoder.upsample." + std::to_string(u) + ".";
            out.push_back({up + "0.conv.weight", {L, L, r}});
            out.push_back({up + "0.conv.bias", {L}});
            out.push_back({up + "1.dwconv.conv.weight", {L, 1, 7}});
            out.push_back({up + "1.dwconv.conv.bias", {L}});
            out.push_back({up + "1.norm.weight", {L}});
            out.push_back({up + "1.norm.bias", {L}});
            out.push_back({up + "1.pwconv1.weight", {4 * L, L}});
            out.push_back({up + "1.pwconv1.bias", {4 * L}});
            out.push_back({up + "1.pwconv2.weight", {L, 4 * L}});
            out.push_back({up + "1.pwconv2.bias", {L}});
            out.push_back({up + "1.gamma", {L}});
        }

        // decoder.0: conv L -> DD; decoder.1..n: blocks halving the width; then snake + final conv
        out.push_back({"codec.decoder.decoder.0.conv.weight", {DD, L, 7}});
        out.push_back({"codec.decoder.decoder.0.conv.bias", {DD}});
        std::uint64_t width = DD;
        const auto n_rates = p.vocoder_upsample_rates.size();
        for (std::size_t i = 0; i < n_rates; ++i) {
            const auto rate = static_cast<std::uint64_t>(p.vocoder_upsample_rates[i]);
            const std::string bp = "codec.decoder.decoder." + std::to_string(i + 1) + ".block.";
            const std::uint64_t in = width, outw = width / 2;
            out.push_back({bp + "0.alpha", {in}});
            out.push_back({bp + "0.beta", {in}});
            out.push_back({bp + "1.conv.weight", {in, outw, 2 * rate}});
            out.push_back({bp + "1.conv.bias", {outw}});
            for (int r = 0; r < 3; ++r) {
                const std::string rp = bp + std::to_string(2 + r) + ".";
                out.push_back({rp + "act1.alpha", {outw}});
                out.push_back({rp + "act1.beta", {outw}});
                out.push_back({rp + "conv1.conv.weight", {outw, outw, 7}});
                out.push_back({rp + "conv1.conv.bias", {outw}});
                out.push_back({rp + "act2.alpha", {outw}});
                out.push_back({rp + "act2.beta", {outw}});
                out.push_back({rp + "conv2.conv.weight", {outw, outw, 1}});
                out.push_back({rp + "conv2.conv.bias", {outw}});
            }
            width = outw;
        }
        const std::string tail = "codec.decoder.decoder." + std::to_string(n_rates + 1);
        out.push_back({tail + ".alpha", {width}});
        out.push_back({tail + ".beta", {width}});
        const std::string last = "codec.decoder.decoder." + std::to_string(n_rates + 2);
        out.push_back({last + ".conv.weight", {1, width, 7}});
        out.push_back({last + ".conv.bias", {1}});
    }
    if (p.has_codec_encoder) add_codec_encoder(out, p);
    return out;
}

Validation validate(const NaviFile & f, const Params & p) {
    Validation v;
    std::set<std::string> seen;
    for (const ExpectedTensor & e : expected_tensors(p)) {
        seen.insert(e.name);
        const TensorInfo * t = f.find(e.name);
        if (!t) { v.missing.push_back(e.name); continue; }
        if (!e.dims.empty() && t->dims != e.dims) {
            std::string want = "[";
            for (std::size_t i = 0; i < e.dims.size(); ++i) want += (i ? ", " : "") + std::to_string(e.dims[i]);
            v.shape_mismatch.push_back(e.name + ": file " + t->shape_str() + " expected " + want + "]");
        }
    }
    for (const TensorInfo & t : f.tensors()) {
        if (!seen.count(t.name)) v.unused.push_back(t.name);
    }
    return v;
}

} // namespace navi::qwen3tts
