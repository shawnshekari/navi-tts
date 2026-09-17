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

    // code predictor: codebooks 1..15 each have an input embedding and a head
    const auto CH = static_cast<std::uint64_t>(p.cp.hidden);
    for (std::uint64_t i = 0; i + 1 < G; ++i) {
        out.push_back({"talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight", {PV, CH}});
        out.push_back({"talker.code_predictor.lm_head." + std::to_string(i) + ".weight", {PV, CH}});
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
