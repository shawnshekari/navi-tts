// Checks the fused HIP talker against a reference decode step dumped by
// the ggml engine (QWEN3_TTS_DUMP_TALKER=<file> ./qwen3-tts-cli ...).
//
// usage: test_hip_talker --model <tts.gguf> --ref <step.bin> [--iters N]
//
// Loads only the talker tensors from the GGUF, uploads them plus the
// dumped KV cache, runs the step with the per-phase driver and the
// fused cooperative kernel, and reports hidden/logit max abs error,
// argmax agreement, and timing.

#include "hip/talker_hip.h"

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
        fprintf(stderr, "usage: %s --model <tts.gguf> --ref <step.bin> [--iters N]\n", argv[0]);
        return 2;
    }

    // --- reference step ---
    FILE * rf = fopen(ref.c_str(), "rb");
    if (!rf) { fprintf(stderr, "cannot open %s\n", ref.c_str()); return 1; }
    int32_t hdr[7];
    if (fread(hdr, sizeof(int32_t), 7, rf) != 7) return 1;
    const int hidden = hdr[0], vocab = hdr[1], n_layer = hdr[2];
    const int n_head = hdr[3], n_kv_head = hdr[4], head_dim = hdr[5], n_past = hdr[6];
    const int kv_stride = n_kv_head * head_dim;
    std::vector<float> ref_step_embd(hidden), ref_hidden(hidden), ref_logits(vocab);
    std::vector<uint16_t> ref_kcache((size_t) n_layer * (n_past + 1) * kv_stride);
    std::vector<uint16_t> ref_vcache(ref_kcache.size());
    if (fread(ref_step_embd.data(), sizeof(float), hidden, rf) != (size_t) hidden) return 1;
    if (fread(ref_hidden.data(), sizeof(float), hidden, rf) != (size_t) hidden) return 1;
    if (fread(ref_logits.data(), sizeof(float), vocab, rf) != (size_t) vocab) return 1;
    const size_t cache_bytes = ref_kcache.size() * sizeof(uint16_t);
    if (fread(ref_kcache.data(), 1, cache_bytes, rf) != cache_bytes) return 1;
    if (fread(ref_vcache.data(), 1, cache_bytes, rf) != cache_bytes) return 1;
    fclose(rf);
    int ref_cb0 = 0;
    for (int i = 1; i < vocab; ++i) if (ref_logits[i] > ref_logits[ref_cb0]) ref_cb0 = i;
    printf("reference: hidden=%d vocab=%d layers=%d heads=%d/%d head_dim=%d n_past=%d ref_cb0=%d\n",
           hidden, vocab, n_layer, n_head, n_kv_head, head_dim, n_past, ref_cb0);

    // --- weights ---
    gguf_file gf;
    if (!gf.open(model)) { fprintf(stderr, "cannot open %s\n", model.c_str()); return 1; }
    tl_params P;
    P.hidden    = (int) gf.get_u32("qwen3-tts.talker.embedding_length", 1024);
    P.n_layer   = (int) gf.get_u32("qwen3-tts.talker.block_count", 28);
    P.n_head    = (int) gf.get_u32("qwen3-tts.talker.attention.head_count", 16);
    P.n_kv_head = (int) gf.get_u32("qwen3-tts.talker.attention.head_count_kv", 8);
    P.ff        = (int) gf.get_u32("qwen3-tts.talker.feed_forward_length", 3072);
    P.head_dim  = (int) gf.get_u32("qwen3-tts.talker.attention.key_length", 128);
    P.eps       = gf.get_f32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", 1e-6f);
    P.rope_theta = gf.get_f32("qwen3-tts.talker.rope.freq_base", 1000000.0f);
    P.vocab     = (int) gf.get_u32("qwen3-tts.talker.codec_vocab_size", 3072);
    if (P.hidden != hidden || P.vocab != vocab || P.n_layer != n_layer ||
        P.n_head != n_head || P.n_kv_head != n_kv_head || P.head_dim != head_dim) {
        fprintf(stderr, "model/reference shape mismatch\n");
        return 1;
    }
    printf("model: hidden=%d heads=%d/%d head_dim=%d ff=%d layers=%d eps=%g theta=%g vocab=%d\n",
           P.hidden, P.n_head, P.n_kv_head, P.head_dim, P.ff, P.n_layer, P.eps, P.rope_theta, P.vocab);

    tl_weights W = {};
    std::string err;
    auto need = [&](const uint16_t * p) { if (!p) { fprintf(stderr, "%s\n", err.c_str()); exit(1); } return p; };
    for (int il = 0; il < P.n_layer; ++il) {
        const std::string b = "talker.blk." + std::to_string(il) + ".";
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
    W.output_norm = need(gf.upload("talker.output_norm.weight", GGML_TYPE_F32, &err));
    W.codec_head  = need(gf.upload("talker.codec_head.weight",  GGML_TYPE_F16, &err));

    // --- dumped KV cache as the external cache ---
    tl_kv kv = {};
    void * d_kcache = nullptr;
    void * d_vcache = nullptr;
    if (hipMalloc(&d_kcache, cache_bytes) != hipSuccess ||
        hipMalloc(&d_vcache, cache_bytes) != hipSuccess) {
        fprintf(stderr, "hipMalloc kv failed\n");
        return 1;
    }
    if (hipMemcpy(d_kcache, ref_kcache.data(), cache_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_vcache, ref_vcache.data(), cache_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "hipMemcpy kv failed\n");
        return 1;
    }
    const size_t layer_elems = (size_t) (n_past + 1) * kv_stride;
    for (int il = 0; il < P.n_layer; ++il) {
        kv.k[il] = (const uint16_t *) d_kcache + (size_t) il * layer_elems;
        kv.v[il] = (const uint16_t *) d_vcache + (size_t) il * layer_elems;
    }

    HipTalker tk;
    if (!tk.init(P, W, &err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    printf("grid: %d blocks\n", tk.grid_blocks());

    // greedy by default: sampled cb0 must match the reference argmax.
    // Override via env to smoke-test the temperature/top-k path (cb0
    // then legitimately differs; only validity is checked).
    float temperature = 0.0f;
    int32_t top_k = 0;
    if (const char * e = getenv("QWEN3_TTS_TEST_TEMP")) temperature = (float) atof(e);
    if (const char * e = getenv("QWEN3_TTS_TEST_TOPK")) top_k = atoi(e);
    const uint64_t seed = 42;
    const int32_t eos_id = 2150;

    std::vector<float> got_hidden(hidden), got_logits(vocab);
    int32_t got_cb0 = -1;

    auto compare = [&](const char * label) {
        bool ok = true;
        float h_abs = 0, h_ref = 0;
        for (int i = 0; i < hidden; ++i) {
            h_abs = std::max(h_abs, fabsf(ref_hidden[i] - got_hidden[i]));
            h_ref = std::max(h_ref, fabsf(ref_hidden[i]));
        }
        const float h_rel = h_abs / h_ref;
        float l_abs = 0, l_ref = 0;
        int am_a = 0, am_b = 0;
        for (int i = 0; i < vocab; ++i) {
            l_abs = std::max(l_abs, fabsf(ref_logits[i] - got_logits[i]));
            l_ref = std::max(l_ref, fabsf(ref_logits[i]));
            if (ref_logits[i] > ref_logits[am_a]) am_a = i;
            if (got_logits[i] > got_logits[am_b]) am_b = i;
        }
        const float l_rel = l_abs / l_ref;
        printf("%s:\n", label);
        printf("  hidden: max|diff| %.4f (%.3f%% of max|ref| %.2f)\n", h_abs, 100.0 * h_rel, h_ref);
        printf("  logits: max|diff| %.4f (%.2f%% of max|ref| %.2f)  argmax ref %4d hip %4d%s\n",
               l_abs, 100.0 * l_rel, l_ref, am_a, am_b, am_a == am_b ? "" : "  <-- differs");
        if (temperature <= 0.0f) {
            printf("  cb0: ref %d hip %d %s\n", ref_cb0, got_cb0, got_cb0 == ref_cb0 ? "OK" : "<-- differs");
            ok = (got_cb0 == ref_cb0);
        } else {
            printf("  cb0 sampled: %d (range %s)\n", got_cb0,
                   (got_cb0 >= 0 && got_cb0 < vocab) ? "ok" : "BAD");
            ok = (got_cb0 >= 0 && got_cb0 < vocab);
        }
        // F16 matvec tolerance: hidden and logits within a few percent,
        // argmax agreement under greedy.
        ok = ok && h_rel < 0.05f && l_rel < 0.10f;
        printf("  %s\n", ok ? "PASS" : "FAIL");
        return ok;
    };

    if (!tk.run(ref_step_embd.data(), n_past, kv, temperature, top_k, 0.0f, eos_id, seed,
               &got_cb0, got_hidden.data(), got_logits.data(), /*fused=*/false, &err)) {
        fprintf(stderr, "run(phased): %s\n", err.c_str());
        return 1;
    }
    bool ok_phased = compare("per-phase launches");

    std::fill(got_hidden.begin(), got_hidden.end(), 0.0f);
    std::fill(got_logits.begin(), got_logits.end(), 0.0f);
    got_cb0 = -1;
    if (!tk.run(ref_step_embd.data(), n_past, kv, temperature, top_k, 0.0f, eos_id, seed,
               &got_cb0, got_hidden.data(), got_logits.data(), /*fused=*/true, &err)) {
        fprintf(stderr, "run(fused): %s\n", err.c_str());
        return 1;
    }
    bool ok_fused = compare("fused cooperative kernel");

    // timing: repeat the fused step
    float best = 1e30f, total = 0;
    for (int i = 0; i < iters; ++i) {
        if (!tk.run(ref_step_embd.data(), n_past, kv, temperature, top_k, 0.0f, eos_id, seed,
                   &got_cb0, nullptr, nullptr, true, &err)) {
            fprintf(stderr, "run: %s\n", err.c_str());
            return 1;
        }
        best = std::min(best, tk.last_run_us());
        total += tk.last_run_us();
    }
    printf("fused talker step (n_past=%d): best %.0f us, mean %.0f us over %d runs\n",
           n_past, best, total / iters, iters);

    return (ok_phased && ok_fused) ? 0 : 1;
}
