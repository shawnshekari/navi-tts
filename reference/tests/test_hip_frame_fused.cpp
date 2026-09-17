// Parity test for the single fused talker+cp frame kernel
// (docs/talker_fusion_handoff.md, "THE NEXT TASK").
//
// The fused-frame kernel must be BIT-IDENTICAL to the three-launch
// chained path (HipTalker::run_device + HipCodePredictor::run_device +
// assemble_step_embd) given the same seeds and the same seen-set state.
// This test runs both paths on the same reference step and compares:
//   - cb0 (the talker's sampled codebook-0 token)
//   - the 15 code-predictor codes
//   - the NEXT-frame step embedding (the assembly output)
// byte-for-byte.
//
// usage: test_hip_frame_fused --model <tts.gguf> --ref <talker_step.bin> [--iters N]
//
// The reference talker step (QWEN3_TTS_DUMP_TALKER=<file>
// ./qwen3-tts-cli ...) supplies step_embd, n_past and the KV cache.
// The trailing text row is synthetic (any fixed vector works — both
// paths see the same one).

#include "hip/frame_fused.h"
#include "hip/talker_hip.h"
#include "hip/code_pred_hip.h"

#include "ggml.h"
#include "gguf.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using namespace qwen3_tts;

struct gguf_file {
    gguf_context * g = nullptr;
    ggml_context * ctx = nullptr;
    FILE * fp = nullptr;
    std::vector<void *> device_ptrs;

    bool open(const std::string & path) {
        gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx };
        g = gguf_init_from_file(path.c_str(), ip);
        if (!g) return false;
        fp = fopen(path.c_str(), "rb");
        return fp != nullptr;
    }
    float get_f32(const char * key, float def) const {
        int64_t k = gguf_find_key(g, key);
        return k < 0 ? def : gguf_get_val_f32(g, k);
    }
    uint32_t get_u32(const char * key, uint32_t def) const {
        int64_t k = gguf_find_key(g, key);
        return k < 0 ? def : gguf_get_val_u32(g, k);
    }
    const uint16_t * upload(const std::string & name, ggml_type want, std::string * err) {
        int64_t idx = gguf_find_tensor(g, name.c_str());
        if (idx < 0) { *err = "missing tensor " + name; return nullptr; }
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (t->type != want) { *err = name + ": unexpected type " + ggml_type_name(t->type); return nullptr; }
        const size_t n = ggml_nbytes(t);
        std::vector<uint8_t> buf(n);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, idx);
        if (fseek(fp, (long) off, SEEK_SET) != 0 || fread(buf.data(), 1, n, fp) != n) { *err = "read failed " + name; return nullptr; }
        void * d = nullptr;
        if (hipMalloc(&d, n) != hipSuccess) { *err = "hipMalloc failed for " + name; return nullptr; }
        if (hipMemcpy(d, buf.data(), n, hipMemcpyHostToDevice) != hipSuccess) { *err = "hipMemcpy failed for " + name; return nullptr; }
        device_ptrs.push_back(d);
        return (const uint16_t *) d;
    }
    ~gguf_file() {
        for (void * d : device_ptrs) (void) hipFree(d);
        if (fp) fclose(fp);
        if (ctx) ggml_free(ctx);
        if (g) gguf_free(g);
    }
};

int main(int argc, char ** argv) {
    std::string model, ref;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }
    if (model.empty() || ref.empty()) {
        fprintf(stderr, "usage: %s --model <tts.gguf> --ref <talker_step.bin> [--iters N]\n", argv[0]);
        return 2;
    }

    // --- reference talker step (step_embd, hidden, logits, KV cache) ---
    FILE * rf = fopen(ref.c_str(), "rb");
    if (!rf) { fprintf(stderr, "cannot open %s\n", ref.c_str()); return 1; }
    int32_t hdr[7];
    if (fread(hdr, sizeof(int32_t), 7, rf) != 7) return 1;
    const int hidden = hdr[0], vocab = hdr[1], n_layer = hdr[2];
    const int n_head = hdr[3], n_kv_head = hdr[4], head_dim = hdr[5], n_past = hdr[6];
    const int kv_stride = n_kv_head * head_dim;
    std::vector<float> ref_step_embd(hidden);
    std::vector<uint16_t> ref_kcache((size_t) n_layer * (n_past + 1) * kv_stride);
    std::vector<uint16_t> ref_vcache(ref_kcache.size());
    if (fread(ref_step_embd.data(), sizeof(float), hidden, rf) != (size_t) hidden) return 1;
    fseek(rf, (long) hidden * sizeof(float), SEEK_CUR);  // skip ref hidden
    fseek(rf, (long) vocab * sizeof(float), SEEK_CUR);   // skip ref logits
    const size_t cache_bytes = ref_kcache.size() * sizeof(uint16_t);
    if (fread(ref_kcache.data(), 1, cache_bytes, rf) != cache_bytes) return 1;
    if (fread(ref_vcache.data(), 1, cache_bytes, rf) != cache_bytes) return 1;
    fclose(rf);
    printf("reference: hidden=%d vocab=%d layers=%d heads=%d/%d head_dim=%d n_past=%d\n",
           hidden, vocab, n_layer, n_head, n_kv_head, head_dim, n_past);

    // --- weights ---
    gguf_file gf;
    if (!gf.open(model)) { fprintf(stderr, "cannot open %s\n", model.c_str()); return 1; }
    std::string err;
    auto need = [&](const uint16_t * p) { if (!p) { fprintf(stderr, "%s\n", err.c_str()); exit(1); } return p; };

    tl_params TP;
    TP.hidden    = (int) gf.get_u32("qwen3-tts.talker.embedding_length", 1024);
    TP.n_layer   = (int) gf.get_u32("qwen3-tts.talker.block_count", 28);
    TP.n_head    = (int) gf.get_u32("qwen3-tts.talker.attention.head_count", 16);
    TP.n_kv_head = (int) gf.get_u32("qwen3-tts.talker.attention.head_count_kv", 8);
    TP.ff        = (int) gf.get_u32("qwen3-tts.talker.feed_forward_length", 3072);
    TP.head_dim  = (int) gf.get_u32("qwen3-tts.talker.attention.key_length", 128);
    TP.eps       = gf.get_f32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", 1e-6f);
    TP.rope_theta = gf.get_f32("qwen3-tts.talker.rope.freq_base", 1000000.0f);
    TP.vocab     = (int) gf.get_u32("qwen3-tts.talker.codec_vocab_size", 3072);

    tl_weights TW = {};
    for (int il = 0; il < TP.n_layer; ++il) {
        const std::string b = "talker.blk." + std::to_string(il) + ".";
        TW.attn_norm[il] = need(gf.upload(b + "attn_norm.weight",   GGML_TYPE_F32, &err));
        TW.q_norm[il]    = need(gf.upload(b + "attn_q_norm.weight", GGML_TYPE_F32, &err));
        TW.k_norm[il]    = need(gf.upload(b + "attn_k_norm.weight", GGML_TYPE_F32, &err));
        TW.ffn_norm[il]  = need(gf.upload(b + "ffn_norm.weight",    GGML_TYPE_F32, &err));
        TW.wq[il]        = need(gf.upload(b + "attn_q.weight",      GGML_TYPE_F16, &err));
        TW.wk[il]        = need(gf.upload(b + "attn_k.weight",      GGML_TYPE_F16, &err));
        TW.wv[il]        = need(gf.upload(b + "attn_v.weight",      GGML_TYPE_F16, &err));
        TW.wo[il]        = need(gf.upload(b + "attn_output.weight", GGML_TYPE_F16, &err));
        TW.w_gate[il]    = need(gf.upload(b + "ffn_gate.weight",    GGML_TYPE_F16, &err));
        TW.w_up[il]      = need(gf.upload(b + "ffn_up.weight",      GGML_TYPE_F16, &err));
        TW.w_down[il]    = need(gf.upload(b + "ffn_down.weight",    GGML_TYPE_F16, &err));
    }
    TW.output_norm = need(gf.upload("talker.output_norm.weight", GGML_TYPE_F32, &err));
    TW.codec_head  = need(gf.upload("talker.codec_head.weight",  GGML_TYPE_F16, &err));

    cp_params CPP;
    CPP.hidden    = (int) gf.get_u32("qwen3-tts.code_predictor.embedding_length", 1024);
    CPP.n_head    = (int) gf.get_u32("qwen3-tts.attention.head_count", 16);
    CPP.n_kv_head = (int) gf.get_u32("qwen3-tts.attention.head_count_kv", 8);
    CPP.head_dim  = (int) gf.get_u32("qwen3-tts.attention.key_length", 128);
    CPP.ff        = (int) gf.get_u32("qwen3-tts.code_predictor.feed_forward_length", 3072);
    CPP.vocab     = (int) gf.get_u32("qwen3-tts.code_predictor.vocab_size", 2048);
    CPP.n_layer   = (int) gf.get_u32("qwen3-tts.code_predictor.layer_count", 5);
    CPP.eps       = gf.get_f32("qwen3-tts.attention.layer_norm_rms_epsilon", 1e-6f);
    CPP.rope_theta = gf.get_f32("qwen3-tts.rope.freq_base", 1000000.0f);

    cp_weights CW = {};
    for (int il = 0; il < CPP.n_layer; ++il) {
        const std::string b = "code_pred.blk." + std::to_string(il) + ".";
        CW.attn_norm[il] = need(gf.upload(b + "attn_norm.weight",   GGML_TYPE_F32, &err));
        CW.q_norm[il]    = need(gf.upload(b + "attn_q_norm.weight", GGML_TYPE_F32, &err));
        CW.k_norm[il]    = need(gf.upload(b + "attn_k_norm.weight", GGML_TYPE_F32, &err));
        CW.ffn_norm[il]  = need(gf.upload(b + "ffn_norm.weight",    GGML_TYPE_F32, &err));
        CW.wq[il]        = need(gf.upload(b + "attn_q.weight",      GGML_TYPE_F16, &err));
        CW.wk[il]        = need(gf.upload(b + "attn_k.weight",      GGML_TYPE_F16, &err));
        CW.wv[il]        = need(gf.upload(b + "attn_v.weight",      GGML_TYPE_F16, &err));
        CW.wo[il]        = need(gf.upload(b + "attn_output.weight", GGML_TYPE_F16, &err));
        CW.w_gate[il]    = need(gf.upload(b + "ffn_gate.weight",    GGML_TYPE_F16, &err));
        CW.w_up[il]      = need(gf.upload(b + "ffn_up.weight",      GGML_TYPE_F16, &err));
        CW.w_down[il]    = need(gf.upload(b + "ffn_down.weight",    GGML_TYPE_F16, &err));
    }
    CW.output_norm       = need(gf.upload("code_pred.output_norm.weight", GGML_TYPE_F32, &err));
    CW.talker_codec_embd = need(gf.upload("talker.codec_embd.weight",     GGML_TYPE_F16, &err));
    for (int i = 0; i < CP_N_HEADS_OUT; ++i) {
        CW.codec_embd[i] = need(gf.upload("code_pred.codec_embd." + std::to_string(i) + ".weight", GGML_TYPE_F16, &err));
        CW.lm_head[i]    = need(gf.upload("code_pred.lm_head."    + std::to_string(i) + ".weight", GGML_TYPE_F16, &err));
    }

    // --- external KV cache (two copies: chained and fused, no cross-talk) ---
    void * d_kc_chain = nullptr; void * d_vc_chain = nullptr;
    void * d_kc_fused = nullptr; void * d_vc_fused = nullptr;
    if (hipMalloc(&d_kc_chain, cache_bytes) != hipSuccess ||
        hipMalloc(&d_vc_chain, cache_bytes) != hipSuccess ||
        hipMalloc(&d_kc_fused, cache_bytes) != hipSuccess ||
        hipMalloc(&d_vc_fused, cache_bytes) != hipSuccess) {
        fprintf(stderr, "hipMalloc kv failed\n"); return 1;
    }
    for (void * dst : { d_kc_chain, d_kc_fused })
        if (hipMemcpy(dst, ref_kcache.data(), cache_bytes, hipMemcpyHostToDevice) != hipSuccess) { fprintf(stderr, "memcpy k failed\n"); return 1; }
    for (void * dst : { d_vc_chain, d_vc_fused })
        if (hipMemcpy(dst, ref_vcache.data(), cache_bytes, hipMemcpyHostToDevice) != hipSuccess) { fprintf(stderr, "memcpy v failed\n"); return 1; }
    const size_t layer_elems = (size_t) (n_past + 1) * kv_stride;
    tl_kv kv_chain = {}, kv_fused = {};
    for (int il = 0; il < TP.n_layer; ++il) {
        kv_chain.k[il] = (const uint16_t *) d_kc_chain + (size_t) il * layer_elems;
        kv_chain.v[il] = (const uint16_t *) d_vc_chain + (size_t) il * layer_elems;
        kv_fused.k[il] = (const uint16_t *) d_kc_fused + (size_t) il * layer_elems;
        kv_fused.v[il] = (const uint16_t *) d_vc_fused + (size_t) il * layer_elems;
    }

    // --- step_embd + trailing row on device ---
    float * d_step_embd = nullptr;
    if (hipMalloc(&d_step_embd, hidden * sizeof(float)) != hipSuccess) { fprintf(stderr, "malloc step failed\n"); return 1; }
    if (hipMemcpy(d_step_embd, ref_step_embd.data(), hidden * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) { fprintf(stderr, "memcpy step failed\n"); return 1; }
    std::vector<float> trailing(hidden);
    for (int h = 0; h < hidden; ++h) trailing[h] = 0.5f * (float)((h % 17) - 8);  // deterministic synthetic row

    // greedy by default (bit-identical parity); env override for sampled smoke.
    float temperature = 0.0f;
    int32_t top_k = 0;
    if (const char * e = getenv("QWEN3_TTS_TEST_TEMP")) temperature = (float) atof(e);
    if (const char * e = getenv("QWEN3_TTS_TEST_TOPK")) top_k = atoi(e);
    const uint64_t talker_seed = 12345;
    const uint64_t cp_seed = 67890;
    const int32_t eos_id = 2150;
    const float rep_pen = 1.05f;

    // --- chained path objects ---
    HipTalker tk;
    if (!tk.init(TP, TW, &err)) { fprintf(stderr, "talker init: %s\n", err.c_str()); return 1; }
    HipCodePredictor cp;
    if (!cp.init(CPP, CW, &err)) { fprintf(stderr, "cp init: %s\n", err.c_str()); return 1; }

    // --- fused-frame object (shares the talker's seen set) ---
    HipFrameFusion ff;
    if (!ff.init(TP, TW, CPP, CW, tk.device_seen(), &err)) { fprintf(stderr, "frame fusion init: %s\n", err.c_str()); return 1; }
    printf("chained talker grid: %d blocks, cp grid: %d blocks\n", tk.grid_blocks(), cp.grid_blocks());
    printf("fused-frame grid: %d blocks (%d blocks/CU)\n", ff.grid_blocks(), ff.occupancy_per_cu());

    // --- run chained path ---
    tk.reset_repetition();
    int32_t chain_cb0 = -1;
    std::vector<int32_t> chain_codes(CP_N_HEADS_OUT, -1);
    if (!tk.run_device(d_step_embd, n_past, kv_chain, temperature, top_k, rep_pen, eos_id,
                      talker_seed, &chain_cb0, nullptr, nullptr, /*fused=*/true, &err)) {
        fprintf(stderr, "chained talker: %s\n", err.c_str()); return 1;
    }
    if (!cp.run_device(tk.device_hidden(), chain_cb0, temperature, top_k, cp_seed,
                      chain_codes.data(), nullptr, /*fused=*/true, &err)) {
        fprintf(stderr, "chained cp: %s\n", err.c_str()); return 1;
    }
    if (!cp.assemble_step_embd(trailing.data(), &err)) { fprintf(stderr, "chained assemble: %s\n", err.c_str()); return 1; }
    std::vector<float> chain_step_next(hidden);
    if (hipMemcpy(chain_step_next.data(), cp.device_step_embd(), hidden * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "read chain step_next failed\n"); return 1;
    }

    // --- run fused path (reset seen so both start from the same state) ---
    tk.reset_repetition();
    if (!ff.run_frame(d_step_embd, n_past, kv_fused, trailing.data(), temperature, top_k, rep_pen,
                     eos_id, talker_seed, cp_seed, &err)) {
        fprintf(stderr, "fused run_frame: %s\n", err.c_str()); return 1;
    }
    std::vector<int32_t> fused16(CP_MAX_POS, -1);
    if (!ff.get_codes(fused16.data(), &err)) { fprintf(stderr, "fused get_codes: %s\n", err.c_str()); return 1; }
    std::vector<float> fused_step_next(hidden);
    if (hipMemcpy(fused_step_next.data(), ff.device_step_embd(), hidden * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "read fused step_next failed\n"); return 1;
    }

    // --- compare bit-identical ---
    const int32_t fused_cb0 = fused16[0];
    bool ok = true;
    printf("cb0: chained %d fused %d %s\n", chain_cb0, fused_cb0, chain_cb0 == fused_cb0 ? "OK" : "<-- DIFFERS");
    ok &= (chain_cb0 == fused_cb0);

    int codes_match = 0;
    for (int i = 0; i < CP_N_HEADS_OUT; ++i) codes_match += (chain_codes[i] == fused16[1 + i]);
    printf("codes: %d/%d match\n", codes_match, CP_N_HEADS_OUT);
    ok &= (codes_match == CP_N_HEADS_OUT);

    const bool step_bytes_equal = memcmp(chain_step_next.data(), fused_step_next.data(), hidden * sizeof(float)) == 0;
    float step_maxdiff = 0;
    for (int h = 0; h < hidden; ++h) step_maxdiff = std::max(step_maxdiff, fabsf(chain_step_next[h] - fused_step_next[h]));
    printf("next step_embd: byte-identical=%s max|diff|=%.3e\n", step_bytes_equal ? "YES" : "NO", step_maxdiff);
    ok &= step_bytes_equal;

    // --- timing: fused vs chained (3 launches) ---
    float fused_best = 1e30f, fused_total = 0;
    for (int i = 0; i < iters; ++i) {
        if (!ff.run_frame(d_step_embd, n_past, kv_fused, trailing.data(), temperature, top_k, rep_pen,
                         eos_id, talker_seed, cp_seed, &err)) { fprintf(stderr, "fused: %s\n", err.c_str()); return 1; }
        fused_best = std::min(fused_best, ff.last_run_us());
        fused_total += ff.last_run_us();
    }
    float chain_best = 1e30f, chain_total = 0;
    for (int i = 0; i < iters; ++i) {
        tk.reset_repetition();
        if (!tk.run_device(d_step_embd, n_past, kv_chain, temperature, top_k, rep_pen, eos_id,
                          talker_seed, &chain_cb0, nullptr, nullptr, true, &err)) { fprintf(stderr, "chain tk: %s\n", err.c_str()); return 1; }
        if (!cp.run_device(tk.device_hidden(), chain_cb0, temperature, top_k, cp_seed,
                          chain_codes.data(), nullptr, true, &err)) { fprintf(stderr, "chain cp: %s\n", err.c_str()); return 1; }
        if (!cp.assemble_step_embd(trailing.data(), &err)) { fprintf(stderr, "chain asm: %s\n", err.c_str()); return 1; }
        const float t = tk.last_run_us() + cp.last_run_us();  // assemble is a tiny kernel; fold into overhead
        chain_best = std::min(chain_best, t);
        chain_total += t;
    }
    printf("fused frame: best %.0f us, mean %.0f us over %d runs\n", fused_best, fused_total / iters, iters);
    printf("chained (talker+cp): best %.0f us, mean %.0f us over %d runs\n", chain_best, chain_total / iters, iters);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
