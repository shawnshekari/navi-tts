#include "model/qwen3tts/talker.h"

#include "model/qwen3tts/kernels.h"
#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <cstdint>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>

namespace navi::qwen3tts {

namespace {

using namespace navi::qwen3tts::kern;

// rows[t] = table[ids[t]] (f16 -> f32)
__global__ void k_gather_rows(const unsigned short * table, const int * ids, int width, float * out) {
    const unsigned short * src = table + static_cast<std::size_t>(ids[blockIdx.x]) * width;
    float * dst = out + static_cast<std::size_t>(blockIdx.x) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = dev::half_bits_to_float(src[i]);
}

// out[t] = (ia[t] >= 0 ? A[ia[t]] : 0) + (ib[t] >= 0 ? B[ib[t]] : 0) + (t == spk_row ? spk : 0)
__global__ void k_assemble_rows(const float * A, const float * B, const float * spk, const int * ia, const int * ib,
                                int spk_row, int width, float * out) {
    const int t = blockIdx.x;
    float * dst = out + static_cast<std::size_t>(t) * width;
    const float * a = ia[t] >= 0 ? A + static_cast<std::size_t>(ia[t]) * width : nullptr;
    const float * b = ib[t] >= 0 ? B + static_cast<std::size_t>(ib[t]) * width : nullptr;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = 0.f;
        if (a) v += a[i];
        if (b) v += b[i];
        if (t == spk_row) v += spk[i];
        dst[i] = v;
    }
}

// Per (position, head): RMSNorm over head_dim, RoPE at `pos0 + t`, and for
// k/v the f16 store into the cache row. q stays f32 in place. One block per
// position; threads sweep heads x pairs.
__global__ void k_qk_norm_rope(float * q, float * k, const float * v, int row_stride, const float * q_norm, const float * k_norm,
                               int n_head, int n_kv_head, int head_dim, float eps, double theta, int pos0,
                               unsigned short * kcache, unsigned short * vcache, int kv_dim) {
    __shared__ float red[32];
    const int t = blockIdx.x, pos = pos0 + t, half = head_dim / 2;
    float * qr = q + static_cast<std::size_t>(t) * row_stride;
    float * kr = k + static_cast<std::size_t>(t) * row_stride;
    const float * vr = v + static_cast<std::size_t>(t) * row_stride;
    // norms: one wave per head would be neater; heads are few, do them serially with the block
    for (int h = 0; h < n_head + n_kv_head; ++h) {
        float * row = h < n_head ? qr + h * head_dim : kr + (h - n_head) * head_dim;
        const float * w = h < n_head ? q_norm : k_norm;
        float ss = 0.f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += row[i] * row[i];
        const float scale = rsqrtf(dev::block_sum(ss, red) / head_dim + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] = row[i] * scale * w[i];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < (n_head + n_kv_head) * half; i += blockDim.x) {
        const int h = i / half, p = i % half;
        float * row = h < n_head ? qr + h * head_dim : kr + (h - n_head) * head_dim;
        const float inv_freq = static_cast<float>(1.0 / pow(theta, (2.0 * p) / head_dim));
        const float ang = static_cast<float>(pos) * inv_freq;
        const float cs = cosf(ang), sn = sinf(ang);
        const float x0 = row[p], x1 = row[p + half];
        row[p] = x0 * cs - x1 * sn;
        row[p + half] = x1 * cs + x0 * sn;
    }
    __syncthreads();
    unsigned short * kc = kcache + static_cast<std::size_t>(pos) * kv_dim;
    unsigned short * vc = vcache + static_cast<std::size_t>(pos) * kv_dim;
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        _Float16 kh = static_cast<_Float16>(kr[i]), vh = static_cast<_Float16>(vr[i]);
        __builtin_memcpy(&kc[i], &kh, 2);
        __builtin_memcpy(&vc[i], &vh, 2);
    }
}

// Causal GQA attention for the prefill: block per (t, q-head), keys 0..pos0+t
// from the f16 cache. Scores in LDS (MAX_KEYS), softmax in f32.
constexpr int MAX_KEYS = 1024;

__global__ void k_prefill_attention(const float * q, int q_stride, const unsigned short * kcache, const unsigned short * vcache,
                                    int n_head, int n_kv_head, int head_dim, int kv_dim, int pos0, float * out) {
    __shared__ float sc[MAX_KEYS];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, pos = pos0 + t;
    const int g = h / (n_head / n_kv_head);
    const int n = pos + 1;
    const float * qh = q + static_cast<std::size_t>(t) * q_stride + h * head_dim;
    const float scaling = rsqrtf(static_cast<float>(head_dim));
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const unsigned short * kh = kcache + static_cast<std::size_t>(j) * kv_dim + g * head_dim;
        float d = 0.f;
        for (int i = 0; i < head_dim; ++i) d += qh[i] * dev::half_bits_to_float(kh[i]);
        sc[j] = d * scaling;
        m = fmaxf(m, sc[j]);
    }
    m = dev::block_max(m, red);
    float s = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) { sc[j] = __expf(sc[j] - m); s += sc[j]; }
    s = dev::block_sum(s, red);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            acc += sc[j] * dev::half_bits_to_float(vcache[static_cast<std::size_t>(j) * kv_dim + g * head_dim + i]);
        }
        out[(static_cast<std::size_t>(t) * n_head + h) * head_dim + i] = acc / s;
    }
}

struct Layer {
    const float * ln1, * ln2, * q_norm, * k_norm;
    const unsigned short * wq, * wk, * wv, * wo, * wg, * wu, * wd;
    const unsigned short * wqkv, * wgu;   // [q; k; v] and [gate; up] when adjacent in the arena, else null
};

} // namespace

struct Talker::Impl {
    hipStream_t stream = nullptr;
    Params p;
    int max_prompt = 0;
    std::vector<void *> owned;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    double last_ms = 0;

    // weights
    const unsigned short * text_emb = nullptr, * codec_emb = nullptr, * fc1 = nullptr, * fc2 = nullptr, * head = nullptr;
    const float * fc1_b = nullptr, * fc2_b = nullptr, * final_norm = nullptr;
    std::vector<Layer> layers;
    KvCache kv;

    // prompt scratch
    int * d_ids = nullptr, * d_cids = nullptr, * d_ia = nullptr, * d_ib = nullptr;
    float * text_rows = nullptr, * fc1_out = nullptr, * proj = nullptr, * codec_rows = nullptr, * d_spk = nullptr;
    Prompt prompt;

    // prefill scratch
    float * x = nullptr, * n = nullptr, * qkv = nullptr, * att = nullptr, * gu = nullptr,
          * logits = nullptr, * gate_tmp = nullptr;   // qkv: [M][Q + 2 KV] as [q | k | v]; gate_tmp: [32][ff], the skinny GEMM's gated input

    template <class T> T * alloc(std::size_t n_elems) {
        void * ptr = nullptr;
        NAVI_HIP(hipMalloc(&ptr, n_elems * sizeof(T)));
        NAVI_HIP(hipMemset(ptr, 0, n_elems * sizeof(T)));
        owned.push_back(ptr);
        return static_cast<T *>(ptr);
    }

    void gemm(const unsigned short * W, const float * bias, const float * in, int T, int Cin, int Cout, float * out,
              int out_stride, int pro, int epi, const float * residual = nullptr, int in_stride = 0) {
        ConvArgs a{};
        a.W = W; a.bias = bias; a.x = in; a.T_in = T; a.Cin = Cin; a.Cout = Cout; a.dil = 1;
        a.x_stride = in_stride ? in_stride : Cin; a.phases = 1; a.y = out; a.y_stride = out_stride; a.residual = residual;
        // T <= 32 (every prefill GEMM): the skinny kernel, with the gate as
        // its own pass. Same summation order as k_conv - bit-identical.
        const bool aligned = (Cin % 8 == 0) && (a.x_stride % 4 == 0)
            && ((reinterpret_cast<std::uintptr_t>(in) | reinterpret_cast<std::uintptr_t>(W)) & 15u) == 0;
        if (T <= 32 && aligned && (pro == PRO_NONE || pro == PRO_SILU_GATE)
            && (epi == EPI_NONE || epi == EPI_SILU || epi == EPI_RESIDUAL)) {
            if (pro == PRO_SILU_GATE) {
                k_silu_gate<<<T, 256, 0, stream>>>(in, Cin, a.x_stride, gate_tmp);
                a.x = gate_tmp; a.x_stride = Cin;
            }
            const dim3 grid((Cout + SK_ROWS - 1) / SK_ROWS), block(SK_NT);
            if (epi == EPI_NONE) k_gemm_skinny<EPI_NONE><<<grid, block, 0, stream>>>(a);
            else if (epi == EPI_SILU) k_gemm_skinny<EPI_SILU><<<grid, block, 0, stream>>>(a);
            else k_gemm_skinny<EPI_RESIDUAL><<<grid, block, 0, stream>>>(a);
            NAVI_HIP(hipGetLastError());
            return;
        }
        const dim3 grid((T + TT - 1) / TT, (Cout + CT - 1) / CT, 1), block(NTHREADS);
        if (pro == PRO_NONE && epi == EPI_NONE) k_conv<unsigned short, 1, PRO_NONE, EPI_NONE><<<grid, block, 0, stream>>>(a);
        else if (pro == PRO_NONE && epi == EPI_SILU) k_conv<unsigned short, 1, PRO_NONE, EPI_SILU><<<grid, block, 0, stream>>>(a);
        else if (pro == PRO_NONE && epi == EPI_RESIDUAL) k_conv<unsigned short, 1, PRO_NONE, EPI_RESIDUAL><<<grid, block, 0, stream>>>(a);
        else if (pro == PRO_SILU_GATE && epi == EPI_RESIDUAL) k_conv<unsigned short, 1, PRO_SILU_GATE, EPI_RESIDUAL><<<grid, block, 0, stream>>>(a);
        else fail("talker: no gemm instantiation");
        NAVI_HIP(hipGetLastError());
    }
};

DeviceWeights::Groups Talker::fused_groups(const Params & p) {
    DeviceWeights::Groups g;
    for (int l = 0; l < p.talker.n_layer; ++l) {
        const std::string lp = "talker.model.layers." + std::to_string(l) + ".";
        g.push_back({lp + "self_attn.q_proj.weight", lp + "self_attn.k_proj.weight", lp + "self_attn.v_proj.weight"});
        g.push_back({lp + "mlp.gate_proj.weight", lp + "mlp.up_proj.weight"});
    }
    return g;
}

std::unique_ptr<Talker> Talker::create(const Device & dev, const DeviceWeights & w, const Params & p,
                                       int max_prompt_tokens, int max_positions) {
    std::unique_ptr<Talker> t(new Talker());
    t->impl_ = std::make_unique<Impl>();
    Impl & I = *t->impl_;
    I.stream = dev.stream();
    I.p = p;
    I.max_prompt = max_prompt_tokens;
    if (max_prompt_tokens > MAX_KEYS) fail("talker: max_prompt_tokens exceeds the prefill attention limit");
    const auto & tp = p.talker;
    const int H = tp.hidden, Q = tp.n_head * tp.head_dim, KV = tp.n_kv_head * tp.head_dim, F = tp.ff;
    auto f16p = [&](const std::string & n) { return w.get(n).as<unsigned short>(); };
    auto f32p = [&](const std::string & n) { return w.get(n).as<float>(); };

    I.text_emb = f16p("talker.model.text_embedding.weight");
    I.codec_emb = f16p("talker.model.codec_embedding.weight");
    I.fc1 = f16p("talker.text_projection.linear_fc1.weight");
    I.fc1_b = f32p("talker.text_projection.linear_fc1.bias");
    I.fc2 = f16p("talker.text_projection.linear_fc2.weight");
    I.fc2_b = f32p("talker.text_projection.linear_fc2.bias");
    I.head = f16p("talker.codec_head.weight");
    I.final_norm = f32p("talker.model.norm.weight");
    for (int l = 0; l < tp.n_layer; ++l) {
        const std::string lp = "talker.model.layers." + std::to_string(l) + ".";
        Layer ly{};
        ly.ln1 = f32p(lp + "input_layernorm.weight");
        ly.ln2 = f32p(lp + "post_attention_layernorm.weight");
        ly.q_norm = f32p(lp + "self_attn.q_norm.weight");
        ly.k_norm = f32p(lp + "self_attn.k_norm.weight");
        ly.wq = f16p(lp + "self_attn.q_proj.weight");
        ly.wk = f16p(lp + "self_attn.k_proj.weight");
        ly.wv = f16p(lp + "self_attn.v_proj.weight");
        ly.wo = f16p(lp + "self_attn.o_proj.weight");
        ly.wg = f16p(lp + "mlp.gate_proj.weight");
        ly.wu = f16p(lp + "mlp.up_proj.weight");
        ly.wd = f16p(lp + "mlp.down_proj.weight");
        ly.wqkv = (ly.wk == ly.wq + static_cast<std::size_t>(Q) * H && ly.wv == ly.wk + static_cast<std::size_t>(KV) * H) ? ly.wq : nullptr;
        ly.wgu = (ly.wu == ly.wg + static_cast<std::size_t>(F) * H) ? ly.wg : nullptr;
        I.layers.push_back(ly);
    }
    {
        int unfused = 0;
        for (const Layer & ly : I.layers) unfused += (!ly.wqkv) + (!ly.wgu);
        if (unfused) std::fprintf(stderr, "talker: %d of %d q/k/v and gate/up groups are not adjacent in the arena; separate GEMMs\n",
                                  unfused, 2 * tp.n_layer);
    }

    I.kv.n_layer = tp.n_layer;
    I.kv.max_pos = max_positions;
    I.kv.kv_dim = KV;
    for (int l = 0; l < tp.n_layer; ++l) {
        I.kv.k.push_back(I.alloc<unsigned short>(static_cast<std::size_t>(max_positions) * KV));
        I.kv.v.push_back(I.alloc<unsigned short>(static_cast<std::size_t>(max_positions) * KV));
    }

    const std::size_t M = static_cast<std::size_t>(max_prompt_tokens) + 8;
    I.d_ids = I.alloc<int>(M); I.d_cids = I.alloc<int>(16); I.d_ia = I.alloc<int>(M); I.d_ib = I.alloc<int>(M);
    I.text_rows = I.alloc<float>(M * p.text_hidden);
    I.fc1_out = I.alloc<float>(M * p.text_hidden);
    I.proj = I.alloc<float>(M * H);
    I.codec_rows = I.alloc<float>(16 * H);
    I.d_spk = I.alloc<float>(H);
    I.prompt.prefill = I.alloc<float>(M * H);
    I.prompt.trailing = I.alloc<float>(M * H);
    I.prompt.tts_pad = I.alloc<float>(H);

    I.x = I.alloc<float>(M * H); I.n = I.alloc<float>(M * H);
    I.qkv = I.alloc<float>(M * (Q + 2 * KV));
    I.att = I.alloc<float>(M * Q); I.gu = I.alloc<float>(M * 2 * F);
    I.gate_tmp = I.alloc<float>(32 * F);
    I.logits = I.alloc<float>(M * tp.vocab);
    NAVI_HIP(hipEventCreate(&I.ev0));
    NAVI_HIP(hipEventCreate(&I.ev1));
    return t;
}

Talker::~Talker() {
    if (!impl_) return;
    if (impl_->ev0) (void) hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void) hipEventDestroy(impl_->ev1);
    for (void * ptr : impl_->owned) (void) hipFree(ptr);
}

const Prompt & Talker::build_prompt(std::span<const std::int32_t> ids, const std::string & language,
                                    std::span<const float> speaker) {
    Impl & I = *impl_;
    const Params & p = I.p;
    const int H = p.talker.hidden;
    if (ids.size() < 9) fail("talker: prompt ids too short (need role header, text, and the trailer)");
    if (static_cast<int>(ids.size()) > I.max_prompt) fail("talker: prompt longer than max_prompt_tokens");
    if (static_cast<int>(speaker.size()) != H) fail("talker: speaker embedding must be [hidden]");

    // text rows to project: role header (3), text tokens ids[3:-5], tts_bos, tts_eos, tts_pad
    const int n_text = static_cast<int>(ids.size()) - 8;
    if (n_text < 1) fail("talker: empty text");
    std::vector<int> tids(ids.begin(), ids.begin() + 3);
    tids.insert(tids.end(), ids.begin() + 3, ids.begin() + 3 + n_text);
    const int i_bos = static_cast<int>(tids.size()); tids.push_back(p.tts_bos_id);
    const int i_eos = static_cast<int>(tids.size()); tids.push_back(p.tts_eos_id);
    const int i_pad = static_cast<int>(tids.size()); tids.push_back(p.tts_pad_id);
    const int n_rows = static_cast<int>(tids.size());

    // codec rows: think block (+ language), speaker slot handled separately, pad, bos
    std::vector<int> cids;
    if (language.empty() || language == "auto") {
        cids = {p.codec_nothink_id, p.codec_think_bos_id, p.codec_think_eos_id};
    } else {
        const auto it = p.language_ids.find(language);
        if (it == p.language_ids.end()) fail("talker: unknown language '" + language + "'");
        cids = {p.codec_think_id, p.codec_think_bos_id, it->second, p.codec_think_eos_id};
    }
    const int n_think = static_cast<int>(cids.size());
    const int i_cpad = n_think; cids.push_back(p.codec_pad_id);
    const int i_cbos = n_think + 1; cids.push_back(p.codec_bos_id);

    // prefill layout (docs/model.md): role rows, think rows (+tts_pad), speaker (+tts_pad),
    // codec_pad (+tts_bos), codec_bos (+ first text token)
    std::vector<int> ia, ib;
    for (int r = 0; r < 3; ++r) { ia.push_back(r); ib.push_back(-1); }
    for (int r = 0; r < n_think; ++r) { ia.push_back(i_pad); ib.push_back(r); }
    const int spk_row = static_cast<int>(ia.size());
    ia.push_back(i_pad); ib.push_back(-1);
    ia.push_back(i_bos); ib.push_back(i_cpad);
    ia.push_back(3); ib.push_back(i_cbos);
    const int n_prefill = static_cast<int>(ia.size());
    // trailing: text tokens[1:] then tts_eos
    std::vector<int> ta, tb;
    for (int r = 1; r < n_text; ++r) { ta.push_back(3 + r); tb.push_back(-1); }
    ta.push_back(i_eos); tb.push_back(-1);
    const int n_trailing = static_cast<int>(ta.size());

    NAVI_HIP(hipMemcpyAsync(I.d_ids, tids.data(), tids.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.d_cids, cids.data(), cids.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.d_spk, speaker.data(), H * sizeof(float), hipMemcpyHostToDevice, I.stream));
    k_gather_rows<<<n_rows, 256, 0, I.stream>>>(I.text_emb, I.d_ids, p.text_hidden, I.text_rows);
    I.gemm(I.fc1, I.fc1_b, I.text_rows, n_rows, p.text_hidden, p.text_hidden, I.fc1_out, p.text_hidden, PRO_NONE, EPI_SILU);
    I.gemm(I.fc2, I.fc2_b, I.fc1_out, n_rows, p.text_hidden, H, I.proj, H, PRO_NONE, EPI_NONE);
    k_gather_rows<<<static_cast<unsigned>(cids.size()), 256, 0, I.stream>>>(I.codec_emb, I.d_cids, H, I.codec_rows);

    NAVI_HIP(hipMemcpyAsync(I.d_ia, ia.data(), ia.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.d_ib, ib.data(), ib.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    k_assemble_rows<<<n_prefill, 256, 0, I.stream>>>(I.proj, I.codec_rows, I.d_spk, I.d_ia, I.d_ib, spk_row, H, I.prompt.prefill);
    NAVI_HIP(hipStreamSynchronize(I.stream));   // ia/ib are reused for the trailing rows
    NAVI_HIP(hipMemcpyAsync(I.d_ia, ta.data(), ta.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.d_ib, tb.data(), tb.size() * sizeof(int), hipMemcpyHostToDevice, I.stream));
    k_assemble_rows<<<n_trailing, 256, 0, I.stream>>>(I.proj, I.codec_rows, I.d_spk, I.d_ia, I.d_ib, -1, H, I.prompt.trailing);
    NAVI_HIP(hipMemcpyAsync(I.prompt.tts_pad, I.proj + static_cast<std::size_t>(i_pad) * H, H * sizeof(float),
                            hipMemcpyDeviceToDevice, I.stream));
    NAVI_HIP(hipGetLastError());
    I.prompt.n_prefill = n_prefill;
    I.prompt.n_trailing = n_trailing;
    return I.prompt;
}

void Talker::prefill(const Prompt & pr) {
    Impl & I = *impl_;
    const auto & tp = I.p.talker;
    const int T = pr.n_prefill, H = tp.hidden, Q = tp.n_head * tp.head_dim, KV = tp.n_kv_head * tp.head_dim, F = tp.ff;
    const int QKV = Q + 2 * KV;
    if (T > I.kv.max_pos) fail("talker: prompt exceeds the KV cache");
    NAVI_HIP(hipEventRecord(I.ev0, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.x, pr.prefill, static_cast<std::size_t>(T) * H * sizeof(float), hipMemcpyDeviceToDevice, I.stream));
    for (int l = 0; l < tp.n_layer; ++l) {
        const Layer & ly = I.layers[static_cast<std::size_t>(l)];
        k_rmsnorm<<<T, 256, 0, I.stream>>>(I.x, ly.ln1, H, tp.rms_eps, I.n);
        // q, k, v as one GEMM into [q | k | v] rows when the weights are adjacent
        float * q = I.qkv, * k = I.qkv + Q, * v = I.qkv + Q + KV;
        if (ly.wqkv) {
            I.gemm(ly.wqkv, nullptr, I.n, T, H, QKV, I.qkv, QKV, PRO_NONE, EPI_NONE);
        } else {
            I.gemm(ly.wq, nullptr, I.n, T, H, Q, q, QKV, PRO_NONE, EPI_NONE);
            I.gemm(ly.wk, nullptr, I.n, T, H, KV, k, QKV, PRO_NONE, EPI_NONE);
            I.gemm(ly.wv, nullptr, I.n, T, H, KV, v, QKV, PRO_NONE, EPI_NONE);
        }
        k_qk_norm_rope<<<T, 128, 0, I.stream>>>(q, k, v, QKV, ly.q_norm, ly.k_norm, tp.n_head, tp.n_kv_head,
                                                  tp.head_dim, tp.rms_eps, tp.rope_theta, 0,
                                                  I.kv.k[static_cast<std::size_t>(l)], I.kv.v[static_cast<std::size_t>(l)], KV);
        k_prefill_attention<<<dim3(T, tp.n_head), 128, 0, I.stream>>>(q, QKV, I.kv.k[static_cast<std::size_t>(l)],
                                                                         I.kv.v[static_cast<std::size_t>(l)], tp.n_head,
                                                                         tp.n_kv_head, tp.head_dim, KV, 0, I.att);
        I.gemm(ly.wo, nullptr, I.att, T, Q, H, I.x, H, PRO_NONE, EPI_RESIDUAL, I.x);
        k_rmsnorm<<<T, 256, 0, I.stream>>>(I.x, ly.ln2, H, tp.rms_eps, I.n);
        if (ly.wgu) {
            I.gemm(ly.wgu, nullptr, I.n, T, H, 2 * F, I.gu, 2 * F, PRO_NONE, EPI_NONE);
        } else {
            I.gemm(ly.wg, nullptr, I.n, T, H, F, I.gu, 2 * F, PRO_NONE, EPI_NONE);
            I.gemm(ly.wu, nullptr, I.n, T, H, F, I.gu + F, 2 * F, PRO_NONE, EPI_NONE);
        }
        I.gemm(ly.wd, nullptr, I.gu, T, F, H, I.x, H, PRO_SILU_GATE, EPI_RESIDUAL, I.x, 2 * F);
    }
    k_rmsnorm<<<T, 256, 0, I.stream>>>(I.x, I.final_norm, H, tp.rms_eps, I.n);
    I.gemm(I.head, nullptr, I.n, T, H, tp.vocab, I.logits, tp.vocab, PRO_NONE, EPI_NONE);
    NAVI_HIP(hipGetLastError());
    NAVI_HIP(hipEventRecord(I.ev1, I.stream));
    NAVI_HIP(hipEventSynchronize(I.ev1));
    float ms = 0.f;
    NAVI_HIP(hipEventElapsedTime(&ms, I.ev0, I.ev1));
    I.last_ms = ms;
    I.kv.n_used = T;
}

KvCache & Talker::kv() { return impl_->kv; }
const float * Talker::d_hidden() const { return impl_->n + static_cast<std::size_t>(impl_->kv.n_used - 1) * impl_->p.talker.hidden; }
const float * Talker::d_logits() const { return impl_->logits + static_cast<std::size_t>(impl_->kv.n_used - 1) * impl_->p.talker.vocab; }
double Talker::last_ms() const { return impl_->last_ms; }

} // namespace navi::qwen3tts
