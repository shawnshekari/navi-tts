// Checks the fused HIP code predictor against a reference frame dumped by
// the ggml engine (QWEN3_TTS_DUMP_CODE_PRED=<file> ./qwen3-tts-cli ...).
//
// usage: test_hip_code_pred --model <tts.gguf> --ref <frame.bin> [--iters N]
//
// Loads only the code-predictor tensors from the GGUF, uploads them, runs
// the frame with the per-phase driver and the fused cooperative kernel,
// and reports per-head max abs error, argmax agreement, and timing.

#include "hip/code_pred_hip.h"

#include "ggml.h"
#include "gguf.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
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

    // read one tensor into device memory; returns nullptr if missing
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
        fprintf(stderr, "usage: %s --model <tts.gguf> --ref <frame.bin> [--iters N]\n", argv[0]);
        return 2;
    }

    // --- reference frame ---
    FILE * rf = fopen(ref.c_str(), "rb");
    if (!rf) { fprintf(stderr, "cannot open %s\n", ref.c_str()); return 1; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, rf) != 3) return 1;
    const int hidden = hdr[0], vocab = hdr[1], n_heads = hdr[2];
    std::vector<float> ref_hidden(hidden);
    int32_t cb0 = 0;
    std::vector<float> ref_logits((size_t) n_heads * vocab);
    std::vector<int32_t> ref_codes(n_heads);
    if (fread(ref_hidden.data(), sizeof(float), hidden, rf) != (size_t) hidden) return 1;
    if (fread(&cb0, sizeof(int32_t), 1, rf) != 1) return 1;
    if (fread(ref_logits.data(), sizeof(float), ref_logits.size(), rf) != ref_logits.size()) return 1;
    if (fread(ref_codes.data(), sizeof(int32_t), n_heads, rf) != (size_t) n_heads) return 1;
    fclose(rf);
    printf("reference: hidden=%d vocab=%d heads=%d cb0=%d codes:", hidden, vocab, n_heads, cb0);
    for (int i = 0; i < n_heads; ++i) printf(" %d", ref_codes[i]);
    printf("\n");

    // --- weights ---
    gguf_file gf;
    if (!gf.open(model)) { fprintf(stderr, "cannot open %s\n", model.c_str()); return 1; }
    cp_params P;
    P.hidden    = (int) gf.get_u32("qwen3-tts.code_predictor.embedding_length", 1024);
    P.n_head    = (int) gf.get_u32("qwen3-tts.attention.head_count", 16);
    P.n_kv_head = (int) gf.get_u32("qwen3-tts.attention.head_count_kv", 8);
    P.head_dim  = (int) gf.get_u32("qwen3-tts.attention.key_length", 128);
    P.ff        = (int) gf.get_u32("qwen3-tts.code_predictor.feed_forward_length", 3072);
    P.vocab     = (int) gf.get_u32("qwen3-tts.code_predictor.vocab_size", 2048);
    P.n_layer   = (int) gf.get_u32("qwen3-tts.code_predictor.layer_count", 5);
    P.eps       = gf.get_f32("qwen3-tts.attention.layer_norm_rms_epsilon", 1e-6f);
    P.rope_theta = gf.get_f32("qwen3-tts.rope.freq_base", 1000000.0f);
    if (P.hidden != hidden || P.vocab != vocab) { fprintf(stderr, "model/reference shape mismatch\n"); return 1; }
    printf("model: hidden=%d heads=%d/%d head_dim=%d ff=%d layers=%d eps=%g theta=%g\n",
           P.hidden, P.n_head, P.n_kv_head, P.head_dim, P.ff, P.n_layer, P.eps, P.rope_theta);

    cp_weights W = {};
    std::string err;
    auto need = [&](const uint16_t * p) { if (!p) { fprintf(stderr, "%s\n", err.c_str()); exit(1); } return p; };
    for (int il = 0; il < P.n_layer; ++il) {
        const std::string b = "code_pred.blk." + std::to_string(il) + ".";
        W.attn_norm[il] = need(gf.upload(b + "attn_norm.weight",   GGML_TYPE_F32, &err));
        W.q_norm[il]    = need(gf.upload(b + "attn_q_norm.weight", GGML_TYPE_F32, &err));
        W.k_norm[il]    = need(gf.upload(b + "attn_k_norm.weight", GGML_TYPE_F32, &err));
        W.ffn_norm[il]  = need(gf.upload(b + "ffn_norm.weight",    GGML_TYPE_F32, &err));
        W.wq[il]        = need(gf.upload(b + "attn_q.weight",      GGML_TYPE_F16, &err));
        W.wk[il]        = need(gf.upload(b + "attn_k.weight",      GGML_TYPE_F16, &err));
        W.wv[il]        = need(gf.upload(b + "attn_v.weight",      GGML_TYPE_F16, &err));
        W.wo[il]        = need(gf.upload(b + "attn_output.weight", GGML_TYPE_F16, &err));
        W.w_gate[il]    = need(gf.upload(b + "ffn_gate.weight",    GGML_TYPE_F16, &err));
        W.w_up[il]      = need(gf.upload(b + "ffn_up.weight",      GGML_TYPE_F16, &err));
        W.w_down[il]    = need(gf.upload(b + "ffn_down.weight",    GGML_TYPE_F16, &err));
    }
    W.output_norm       = need(gf.upload("code_pred.output_norm.weight", GGML_TYPE_F32, &err));
    W.talker_codec_embd = need(gf.upload("talker.codec_embd.weight",     GGML_TYPE_F16, &err));
    for (int i = 0; i < CP_N_HEADS_OUT; ++i) {
        W.codec_embd[i] = need(gf.upload("code_pred.codec_embd." + std::to_string(i) + ".weight", GGML_TYPE_F16, &err));
        W.lm_head[i]    = need(gf.upload("code_pred.lm_head."    + std::to_string(i) + ".weight", GGML_TYPE_F16, &err));
    }

    HipCodePredictor cp;
    if (!cp.init(P, W, &err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    printf("grid: %d blocks\n", cp.grid_blocks());

    // greedy by default: sampled codes must match ref argmax. Override via
    // env to smoke-test the temperature/top-k path (codes then legitimately
    // differ from the reference; only agreement of logits is checked).
    float temperature = 0.0f;
    int32_t top_k = 0;
    if (const char * e = getenv("QWEN3_TTS_TEST_TEMP")) temperature = (float) atof(e);
    if (const char * e = getenv("QWEN3_TTS_TEST_TOPK")) top_k = atoi(e);
    const uint64_t seed = 42;
    std::vector<int32_t> got_codes(n_heads, -1);

    auto compare = [&](const char * label, const std::vector<float> & out) {
        int agree = 0;
        float worst = 0;
        printf("%s:\n", label);
        for (int h = 0; h < n_heads; ++h) {
            const float * a = &ref_logits[(size_t) h * vocab];
            const float * b = &out[(size_t) h * vocab];
            float max_abs = 0, max_ref = 0;
            int am_a = 0, am_b = 0;
            for (int i = 0; i < vocab; ++i) {
                max_abs = std::max(max_abs, fabsf(a[i] - b[i]));
                max_ref = std::max(max_ref, fabsf(a[i]));
                if (a[i] > a[am_a]) am_a = i;
                if (b[i] > b[am_b]) am_b = i;
            }
            worst = std::max(worst, max_abs / max_ref);
            agree += (am_a == am_b);
            printf("  head %2d: max|diff| %.4f (%.2f%% of max|ref| %.2f)  argmax ref %4d hip %4d%s\n",
                   h, max_abs, 100.0 * max_abs / max_ref, max_ref, am_a, am_b, am_a == am_b ? "" : "  <-- differs");
        }
        printf("  argmax agreement %d/%d, worst relative error %.3f%%\n", agree, n_heads, 100.0 * worst);
        return agree == n_heads;
    };

    auto check_codes = [&](const char * label, const std::vector<int32_t> & c) {
        if (temperature > 0.0f) {
            int in_range = 0;
            for (int i = 0; i < n_heads; ++i) in_range += (c[i] >= 0 && c[i] < vocab);
            printf("%s sampled codes (temp %.2f top_k %d): %d/%d in vocab range\n",
                   label, temperature, top_k, in_range, n_heads);
            return in_range == n_heads;
        }
        int agree = 0;
        for (int i = 0; i < n_heads; ++i) agree += (c[i] == ref_codes[i]);
        printf("%s sampled codes: %d/%d match reference\n", label, agree, n_heads);
        return agree == n_heads;
    };

    // Logit comparison against the reference is only valid under greedy:
    // with temperature>0 the sampled codes diverge, so every downstream
    // head sees a different input. There we only check code validity and
    // that fused and phased agree with each other.
    std::vector<float> out((size_t) n_heads * vocab);
    if (!cp.run(ref_hidden.data(), cb0, temperature, top_k, seed, got_codes.data(), out.data(), /*fused=*/false, &err)) { fprintf(stderr, "run(phased): %s\n", err.c_str()); return 1; }
    bool ok_phased = check_codes("per-phase", got_codes);
    if (temperature <= 0.0f) ok_phased = compare("per-phase launches", out) && ok_phased;
    printf("  time %.0f us\n", cp.last_run_us());
    std::vector<float> phased_out = out;
    std::vector<int32_t> phased_codes = got_codes;

    std::fill(got_codes.begin(), got_codes.end(), -1);
    std::fill(out.begin(), out.end(), 0.0f);
    if (!cp.run(ref_hidden.data(), cb0, temperature, top_k, seed, got_codes.data(), out.data(), /*fused=*/true, &err)) { fprintf(stderr, "run(fused): %s\n", err.c_str()); return 1; }
    bool ok_fused = check_codes("fused", got_codes);
    if (temperature <= 0.0f) {
        ok_fused = compare("fused cooperative kernel", out) && ok_fused;
    } else {
        int agree = 0;
        for (int i = 0; i < n_heads; ++i) agree += (got_codes[i] == phased_codes[i]);
        printf("fused vs per-phase sampled code agreement: %d/%d\n", agree, n_heads);
        ok_fused = ok_fused && agree == n_heads;
    }

    // timing: repeat the fused frame
    float best = 1e30f, total = 0;
    for (int i = 0; i < iters; ++i) {
        if (!cp.run(ref_hidden.data(), cb0, temperature, top_k, seed, got_codes.data(), out.data(), true, &err)) { fprintf(stderr, "run: %s\n", err.c_str()); return 1; }
        best = std::min(best, cp.last_run_us());
        total += cp.last_run_us();
    }
    printf("fused frame (16 positions, 15 heads): best %.0f us, mean %.0f us over %d runs\n", best, total / iters, iters);

    return (ok_phased && ok_fused) ? 0 : 1;
}
