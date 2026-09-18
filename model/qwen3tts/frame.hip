#include "model/qwen3tts/frame.h"

#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstring>

// Device code grown from the fork-era fused kernel (reference/hip/frame_fused.hip
// and hip_dev.h, the author's own): same phases, same barrier schedule, same
// summation order. What is new is the host side, the frame-0 path, and the
// min-frames EOS rule.

namespace navi::qwen3tts {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_LAYERS = 32;
constexpr int CP_MAX_LAYERS = 8;
constexpr int CP_POS = 16;          // hidden, cb0, then 14 fed-back codes
constexpr int CP_HEADS = 15;        // codebooks 1..15
constexpr int VEC_MAX = 4096;       // largest LDS vector: ff 3072, vocab 3072 padded to 4096 for the sort
constexpr int MV_ROWS = 4;

struct TlParams { int hidden, n_head, n_kv_head, head_dim, ff, vocab, n_layer; float eps, rope_theta; };
struct CpParams { int hidden, n_head, n_kv_head, head_dim, ff, vocab, n_layer; float eps, rope_theta; };

struct TlWeights {
    const float * attn_norm[MAX_LAYERS], * q_norm[MAX_LAYERS], * k_norm[MAX_LAYERS], * ffn_norm[MAX_LAYERS];
    const uint16_t * wq[MAX_LAYERS], * wk[MAX_LAYERS], * wv[MAX_LAYERS], * wo[MAX_LAYERS];
    const uint16_t * w_gate[MAX_LAYERS], * w_up[MAX_LAYERS], * w_down[MAX_LAYERS];
    const float * output_norm;
    const uint16_t * codec_head;
};

struct CpWeights {
    const float * attn_norm[CP_MAX_LAYERS], * q_norm[CP_MAX_LAYERS], * k_norm[CP_MAX_LAYERS], * ffn_norm[CP_MAX_LAYERS];
    const uint16_t * wq[CP_MAX_LAYERS], * wk[CP_MAX_LAYERS], * wv[CP_MAX_LAYERS], * wo[CP_MAX_LAYERS];
    const uint16_t * w_gate[CP_MAX_LAYERS], * w_up[CP_MAX_LAYERS], * w_down[CP_MAX_LAYERS];
    const float * output_norm;
    const uint16_t * talker_codec_embd;
    const uint16_t * codec_embd[CP_HEADS];
    const uint16_t * lm_head[CP_HEADS];
};

struct Work {
    // talker
    float * x, * q, * k, * v, * attn, * h, * hidden_out, * logits;
    uint8_t * seen;
    uint16_t * kcache[MAX_LAYERS], * vcache[MAX_LAYERS];   // [pos][n_kv_head*head_dim] f16
    int n_past;
    float temperature; int top_k; float repetition_penalty; int eos_id; int suppress_from; int min_frames; int frame_index;
    uint64_t talker_seed;
    // cp
    float * cx, * cq, * ck, * cv, * cattn, * ch, * clogits;
    uint16_t * ckcache, * cvcache;     // [n_layer][CP_POS][kv]
    int32_t * codes;                   // [CP_POS]
    float cp_temperature; int cp_top_k;
    uint64_t cp_seed;
    // frame plumbing
    int skip_talker;                   // frame 0: hidden/logits come from the prefill
    const float * hidden_in, * logits_in;
    const float * step_in, * trailing;
    float * step_out;
    volatile unsigned * bar;           // [0] count, [1] sense, [2] abort
    unsigned long long spin_cap;
};

// --- device helpers ---------------------------------------------------------

// Deadline-aware sense-reversing grid barrier. Returns false once any block
// has waited past spin_cap; every later barrier then short-circuits so the
// whole grid unwinds instead of deadlocking.
__device__ __forceinline__ bool grid_barrier(volatile unsigned * bar, unsigned nblocks, unsigned long long spin_cap) {
    __shared__ unsigned lsense;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (bar[2]) {
            lsense = bar[1];
        } else {
            const unsigned s = bar[1];
            __threadfence();
            const unsigned prev = atomicAdd((unsigned *) &bar[0], 1u);
            if (prev + 1u == nblocks) {
                bar[0] = 0u;
                __threadfence();
                bar[1] = s ^ 1u;
                lsense = s ^ 1u;
            } else {
                unsigned long long spins = 0;
                while (bar[1] == s && !bar[2]) {
                    if (++spins >= spin_cap) { bar[2] = 1u; break; }
                }
                lsense = bar[1];
            }
        }
    }
    __syncthreads();
    __threadfence();
    return bar[2] == 0u;
}

__device__ __forceinline__ float h2f(const uint16_t * p, int i) { return dev::half_bits_to_float(p[i]); }
__device__ __forceinline__ void f2h(uint16_t * p, int i, float v) {
    _Float16 h = static_cast<_Float16>(v);
    __builtin_memcpy(&p[i], &h, 2);
}

__device__ void block_rmsnorm(const float * x, const float * w, int n, float eps, float * out, float * red) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = dev::block_sum(ss, red);
    const float scale = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * w[i];
    __syncthreads();
}

__device__ __forceinline__ void block_stage(const float * src, int n, float * dst) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}

__device__ __forceinline__ float dot8(const uint4 & u, const float * xs) {
    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
    const float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]), f2 = __half22float2(h2[2]), f3 = __half22float2(h2[3]);
    return f0.x * xs[0] + f0.y * xs[1] + f1.x * xs[2] + f1.y * xs[3] + f2.x * xs[4] + f2.y * xs[5] + f3.x * xs[6] + f3.y * xs[7];
}

// R rows of W (row-major f16) dotted with xs (f32 in LDS) by one wave; R*(n_cols/256)
// 16-byte loads in flight per lane. Rows past n_rows contribute zero. n_cols % 8 == 0.
template <int R>
__device__ __forceinline__ void wave_dot_rows(const uint16_t * W, int row0, int n_rows, int n_cols, const float * xs, float * acc) {
    const int lane = threadIdx.x % warpSize;
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll 2
    for (int c = lane * 8; c < n_cols; c += warpSize * 8) {
        uint4 u[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            u[r] = (row0 + r < n_rows) ? *reinterpret_cast<const uint4 *>(W + static_cast<size_t>(row0 + r) * n_cols + c)
                                       : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += dot8(u[r], xs + c);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = dev::wave_sum(acc[r]);
}

// y[row] (=, or += when ACCUMULATE) W*xs over all rows, partitioned across the grid's waves
template <bool ACCUMULATE>
__device__ void grid_matvec(const uint16_t * W, int n_rows, int n_cols, const float * xs, float * y) {
    const int lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize, nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < n_rows; row0 += nwave * MV_ROWS) {
        float acc[MV_ROWS];
        wave_dot_rows<MV_ROWS>(W, row0, n_rows, n_cols, xs, acc);
        if (lane < MV_ROWS && row0 + lane < n_rows) {
            float d = acc[0];
#pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) d = acc[r];
            if (ACCUMULATE) y[row0 + lane] += d; else y[row0 + lane] = d;
        }
    }
}

__device__ __forceinline__ uint32_t pcg32_next(uint64_t & s) {
    const uint64_t old = s;
    s = old * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t xs = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    const uint32_t rot = static_cast<uint32_t>(old >> 59u);
    return (xs >> rot) | (xs << ((0u - rot) & 31u));
}

// Block-wide argmax over n floats; lowest index wins ties. red: >= 32 floats.
__device__ int block_argmax(const float * v, int n, float * red) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    float bv = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > bv || (x == bv && i < bi)) { bv = x; bi = i; }
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o, warpSize);
        const int oi = __shfl_xor(bi, o, warpSize);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    float * rv = red;
    int * ri = reinterpret_cast<int *>(red + 16);
    __syncthreads();
    if (lane == 0) { rv[wid] = bv; ri[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < nw; ++w)
            if (rv[w] > rv[0] || (rv[w] == rv[0] && ri[w] < ri[0])) { rv[0] = rv[w]; ri[0] = ri[w]; }
    }
    __syncthreads();
    const int r = ri[0];
    __syncthreads();
    return r;
}

// Multinomial draw over probabilities p[0..V) (unnormalised, sum given) with
// PCG32 seeded from `seed`; thread 0 only.
__device__ __forceinline__ int draw(const float * p, int V, float inv_sum, uint64_t seed) {
    uint64_t st = seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
    (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
    const float u = static_cast<float>(pcg32_next(st) >> 8) * (1.0f / 16777216.0f);
    float cum = 0.f;
    int chosen = V - 1;
    for (int i = 0; i < V; ++i) {
        cum += p[i] * inv_sum;
        if (cum >= u) { chosen = i; break; }
    }
    return chosen;
}

// Bitonic sort (descending) of xs[0..N), N a power of two, whole block.
__device__ void block_sort_desc(float * xs, int N) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < (N >> 1); i += blockDim.x) {
                const int l = ((i & ~(j - 1)) << 1) | (i & (j - 1));
                const int m = l + j;
                const bool desc = (l & k) == 0;
                const float x = xs[l], y = xs[m];
                if (desc ? (x < y) : (x > y)) { xs[l] = y; xs[m] = x; }
            }
        }
    }
    __syncthreads();
}

// --- talker phases ----------------------------------------------------------

__device__ void tl_qkv(const TlParams & P, const TlWeights & W, const Work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim, P.hidden, xs, B.q);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.k);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.v);
}

// blocks [0, n_head): q heads; [n_head, +n_kv_head): k heads (+cache); [.., +n_kv_head): v -> cache
template <class WP>
__device__ void norm_rope_store(const WP & P, const float * q_norm, const float * k_norm, float * q, float * k,
                                const float * v, int pos, uint16_t * kcache, uint16_t * vcache, float * xs, float * red) {
    const int hd = P.head_dim, b = blockIdx.x, t = threadIdx.x, kv_stride = P.n_kv_head * hd;
    if (b < P.n_head + P.n_kv_head) {
        const bool is_q = b < P.n_head;
        const int head = is_q ? b : b - P.n_head;
        float * vec = (is_q ? q : k) + head * hd;
        const float * nw = is_q ? q_norm : k_norm;
        const float e = (t < hd) ? vec[t] : 0.f;
        const float ss = dev::block_sum(e * e, red);
        const float scale = rsqrtf(ss / hd + P.eps);
        if (t < hd) xs[t] = e * scale * nw[t];
        __syncthreads();
        if (t < hd / 2) {
            const float theta = static_cast<float>(pos) * powf(P.rope_theta, -2.0f * static_cast<float>(t) / static_cast<float>(hd));
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = xs[t], x1 = xs[t + hd / 2];
            const float r0 = x0 * c - x1 * s, r1 = x0 * s + x1 * c;
            vec[t] = r0;
            vec[t + hd / 2] = r1;
            if (!is_q) {
                uint16_t * kc = kcache + static_cast<size_t>(pos) * kv_stride + head * hd;
                f2h(kc, t, r0);
                f2h(kc, t + hd / 2, r1);
            }
        }
    } else if (b < P.n_head + 2 * P.n_kv_head) {
        const int head = b - P.n_head - P.n_kv_head;
        if (t < hd) f2h(vcache + static_cast<size_t>(pos) * kv_stride + head * hd, t, v[head * hd + t]);
    }
}

// one block per q head: scores over positions 0..pos, softmax, weighted V sum
template <class WP>
__device__ void attend(const WP & P, const float * q, const uint16_t * kcache, const uint16_t * vcache, int pos,
                       float * out, float * xs) {
    const int h = blockIdx.x;
    if (h >= P.n_head) return;
    const int hd = P.head_dim, kvh = h / (P.n_head / P.n_kv_head), kv_stride = P.n_kv_head * hd;
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    const float scale = rsqrtf(static_cast<float>(hd));
    float * scores = xs;
    const float * qh = q + h * hd;
    for (int j = wid; j <= pos; j += nw) {
        const uint16_t * kj = kcache + static_cast<size_t>(j) * kv_stride + kvh * hd;
        float d = 0.f;
        for (int i = lane; i < hd; i += warpSize) d += qh[i] * h2f(kj, i);
        d = dev::wave_sum(d);
        if (lane == 0) scores[j] = d * scale;
    }
    __syncthreads();
    if (wid == 0) {
        float m = -INFINITY;
        for (int j = lane; j <= pos; j += warpSize) m = fmaxf(m, scores[j]);
        m = dev::wave_max(m);
        float s = 0.f;
        for (int j = lane; j <= pos; j += warpSize) { const float e = expf(scores[j] - m); scores[j] = e; s += e; }
        s = dev::wave_sum(s);
        const float inv = 1.f / s;
        for (int j = lane; j <= pos; j += warpSize) scores[j] *= inv;
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        const int i = threadIdx.x;
        float acc = 0.f;
        for (int j = 0; j <= pos; ++j) acc += scores[j] * h2f(vcache + static_cast<size_t>(j) * kv_stride + kvh * hd, i);
        out[h * hd + i] = acc;
    }
}

template <class WP>
__device__ void gate_up(const WP & P, const uint16_t * wg, const uint16_t * wu, const float * xs, float * hbuf) {
    const int lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize, nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < P.ff; row0 += nwave * MV_ROWS) {
        float g[MV_ROWS], u[MV_ROWS];
        wave_dot_rows<MV_ROWS>(wg, row0, P.ff, P.hidden, xs, g);
        wave_dot_rows<MV_ROWS>(wu, row0, P.ff, P.hidden, xs, u);
        if (lane < MV_ROWS && row0 + lane < P.ff) {
            float gg = g[0], uu = u[0];
#pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) { gg = g[r]; uu = u[r]; }
            hbuf[row0 + lane] = gg / (1.f + expf(-gg)) * uu;
        }
    }
}

__device__ void tl_head(const TlParams & P, const TlWeights & W, const Work & B, float * xs, float * red) {
    block_rmsnorm(B.x, W.output_norm, P.hidden, P.eps, xs, red);
    for (int i = threadIdx.x; i < P.hidden; i += blockDim.x) B.hidden_out[i] = xs[i];
    __syncthreads();
    grid_matvec<false>(W.codec_head, P.vocab, P.hidden, xs, B.logits);
}

// The logit the sampler sees for token i: control ids suppressed except EOS,
// EOS suppressed before min_frames, then the HF repetition penalty.
__device__ __forceinline__ float tl_modified_logit(const Work & B, int i) {
    float v = B.logits[i];
    if (i >= B.suppress_from && i != B.eos_id) return -INFINITY;
    if (i == B.eos_id && B.frame_index < B.min_frames) return -INFINITY;
    if (B.repetition_penalty > 0.f && B.seen[i]) v = (v > 0.f) ? v / B.repetition_penalty : v * B.repetition_penalty;
    return v;
}

// Block 0 samples cb0 into codes[0]: greedy if temperature <= 0, else
// temperature, top-k threshold (bitonic sort), softmax, multinomial.
__device__ void tl_sample(const TlParams & P, const Work & B, float * xs, float * red) {
    if (blockIdx.x != 0) return;
    const int V = P.vocab;
    if (B.temperature <= 0.f) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = tl_modified_logit(B, i);
        __syncthreads();
        const int best = block_argmax(xs, V, red);
        if (threadIdx.x == 0) { B.codes[0] = best; if (best != B.eos_id) B.seen[best] = 1u; }
        return;
    }
    const int N = VEC_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) xs[i] = (i < V) ? tl_modified_logit(B, i) / B.temperature : -INFINITY;
    __syncthreads();
    float threshold = -INFINITY, mx;
    if (B.top_k > 0 && B.top_k < V) {
        block_sort_desc(xs, N);
        threshold = xs[B.top_k - 1];
        mx = xs[0];
    } else {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, xs[i]);
        mx = dev::block_max(m, red);
    }
    __syncthreads();
    float ssum = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = tl_modified_logit(B, i) / B.temperature;
        const float e = (v >= threshold) ? expf(v - mx) : 0.f;
        xs[i] = e;
        ssum += e;
    }
    ssum = dev::block_sum(ssum, red);
    __syncthreads();
    if (threadIdx.x == 0) {
        const int chosen = draw(xs, V, 1.f / ssum, B.talker_seed);
        B.codes[0] = chosen;
        if (chosen != B.eos_id) B.seen[chosen] = 1u;
    }
    __syncthreads();
}

// --- code-predictor phases ---------------------------------------------------

__device__ void cp_embed(const CpParams & P, const CpWeights & W, const Work & B, int pos) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
    const uint16_t * row = nullptr;
    if (pos == 1) row = W.talker_codec_embd + static_cast<size_t>(B.codes[0]) * P.hidden;
    else if (pos >= 2) row = W.codec_embd[pos - 2] + static_cast<size_t>(B.codes[pos - 1]) * P.hidden;
    for (int i = tid; i < P.hidden; i += nth) B.cx[i] = (pos == 0) ? B.hidden_out[i] : h2f(row, i);
}

__device__ void cp_head(const CpParams & P, const CpWeights & W, const Work & B, int pos, float * xs, float * red) {
    if (pos < 1) return;
    block_rmsnorm(B.cx, W.output_norm, P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.lm_head[pos - 1], P.vocab, P.hidden, xs, B.clogits + static_cast<size_t>(pos - 1) * P.vocab);
}

// Block 0 samples codebook `pos` from head pos-1 into codes[pos].
__device__ void cp_sample(const CpParams & P, const Work & B, int pos, float * xs, float * red) {
    if (pos < 1 || blockIdx.x != 0) return;
    const int V = P.vocab;
    const float * lg = B.clogits + static_cast<size_t>(pos - 1) * V;
    if (B.cp_temperature <= 0.f) {
        const int best = block_argmax(lg, V, red);
        if (threadIdx.x == 0) B.codes[pos] = best;
        return;
    }
    for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = lg[i] / B.cp_temperature;
    __syncthreads();
    float threshold = -INFINITY, mx;
    if (B.cp_top_k > 0 && B.cp_top_k < V) {
        block_sort_desc(xs, V);   // 2048 is a power of two
        threshold = xs[B.cp_top_k - 1];
        mx = xs[0];
    } else {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < V; i += blockDim.x) m = fmaxf(m, xs[i]);
        mx = dev::block_max(m, red);
    }
    __syncthreads();
    float ssum = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = lg[i] / B.cp_temperature;
        const float e = (v >= threshold) ? expf(v - mx) : 0.f;
        xs[i] = e;
        ssum += e;
    }
    ssum = dev::block_sum(ssum, red);
    __syncthreads();
    if (threadIdx.x == 0) B.codes[pos] = draw(xs, V, 1.f / ssum, B.cp_seed + static_cast<uint64_t>(pos) * 0xD1B54A32D192ED03ull);
    __syncthreads();
}

// next frame's step embedding: codec_embd[cb0] + sum_cb cp_embd[cb][code] + trailing
__device__ void assemble_step(const CpParams & P, const CpWeights & W, const Work & B) {
    const int hidden = P.hidden;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden; h += gridDim.x * blockDim.x) {
        float v = h2f(W.talker_codec_embd + static_cast<size_t>(B.codes[0]) * hidden, h);
#pragma unroll
        for (int cb = 0; cb < CP_HEADS; ++cb) v += h2f(W.codec_embd[cb] + static_cast<size_t>(B.codes[1 + cb]) * hidden, h);
        v += B.trailing[h];
        B.step_out[h] = v;
    }
}

// --- the frame ----------------------------------------------------------------

__global__ void __launch_bounds__(BLOCK) k_frame(TlParams TP, TlWeights TW, CpParams CP, CpWeights CW, Work B) {
    __shared__ float xs[VEC_MAX];
    __shared__ float red[32];
    const unsigned nb = gridDim.x;
#define GBSYNC() do { if (!grid_barrier(B.bar, nb, B.spin_cap)) return; } while (0)

    if (B.skip_talker) {
        // frame 0: the prefill's last position already gave hidden and logits
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < TP.hidden; i += gridDim.x * blockDim.x) B.hidden_out[i] = B.hidden_in[i];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < TP.vocab; i += gridDim.x * blockDim.x) B.logits[i] = B.logits_in[i];
        GBSYNC();
    } else {
        for (int i = threadIdx.x; i < TP.hidden; i += blockDim.x) B.x[i] = B.step_in[i];
        GBSYNC();
        for (int il = 0; il < TP.n_layer; ++il) {
            tl_qkv(TP, TW, B, il, xs, red);                                                                  GBSYNC();
            norm_rope_store(TP, TW.q_norm[il], TW.k_norm[il], B.q, B.k, B.v, B.n_past, B.kcache[il], B.vcache[il], xs, red); GBSYNC();
            attend(TP, B.q, B.kcache[il], B.vcache[il], B.n_past, B.attn, xs);                                 GBSYNC();
            block_stage(B.attn, TP.n_head * TP.head_dim, xs);
            grid_matvec<true>(TW.wo[il], TP.hidden, TP.n_head * TP.head_dim, xs, B.x);                        GBSYNC();
            block_rmsnorm(B.x, TW.ffn_norm[il], TP.hidden, TP.eps, xs, red);
            gate_up(TP, TW.w_gate[il], TW.w_up[il], xs, B.h);                                                 GBSYNC();
            block_stage(B.h, TP.ff, xs);
            grid_matvec<true>(TW.w_down[il], TP.hidden, TP.ff, xs, B.x);                                      GBSYNC();
        }
        tl_head(TP, TW, B, xs, red);                                                                          GBSYNC();
    }
    tl_sample(TP, B, xs, red);                                                                                GBSYNC();

    const int ckv = CP.n_kv_head * CP.head_dim;
    for (int pos = 0; pos < CP_POS; ++pos) {
        cp_embed(CP, CW, B, pos);                                                                             GBSYNC();
        for (int il = 0; il < CP.n_layer; ++il) {
            uint16_t * kc = B.ckcache + static_cast<size_t>(il) * CP_POS * ckv;
            uint16_t * vc = B.cvcache + static_cast<size_t>(il) * CP_POS * ckv;
            block_rmsnorm(B.cx, CW.attn_norm[il], CP.hidden, CP.eps, xs, red);
            grid_matvec<false>(CW.wq[il], CP.n_head * CP.head_dim, CP.hidden, xs, B.cq);
            grid_matvec<false>(CW.wk[il], ckv, CP.hidden, xs, B.ck);
            grid_matvec<false>(CW.wv[il], ckv, CP.hidden, xs, B.cv);                                          GBSYNC();
            norm_rope_store(CP, CW.q_norm[il], CW.k_norm[il], B.cq, B.ck, B.cv, pos, kc, vc, xs, red);       GBSYNC();
            attend(CP, B.cq, kc, vc, pos, B.cattn, xs);                                                       GBSYNC();
            block_stage(B.cattn, CP.n_head * CP.head_dim, xs);
            grid_matvec<true>(CW.wo[il], CP.hidden, CP.n_head * CP.head_dim, xs, B.cx);                       GBSYNC();
            block_rmsnorm(B.cx, CW.ffn_norm[il], CP.hidden, CP.eps, xs, red);
            gate_up(CP, CW.w_gate[il], CW.w_up[il], xs, B.ch);                                                GBSYNC();
            block_stage(B.ch, CP.ff, xs);
            grid_matvec<true>(CW.w_down[il], CP.hidden, CP.ff, xs, B.cx);                                     GBSYNC();
        }
        cp_head(CP, CW, B, pos, xs, red);                                                                     GBSYNC();
        cp_sample(CP, B, pos, xs, red);                                                                       GBSYNC();
    }
    assemble_step(CP, CW, B);
    GBSYNC();
#undef GBSYNC
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

} // namespace

struct Frame::Impl {
    hipStream_t stream = nullptr;
    Params p;
    TlParams TP{};
    TlWeights TW{};
    CpParams CP{};
    CpWeights CW{};
    Work B{};
    FrameOptions opt;
    int grid = 0, occ = 0;
    std::vector<void *> owned;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    int32_t h_codes[CP_POS] = {};

    template <class T> T * alloc(std::size_t n) {
        void * ptr = nullptr;
        NAVI_HIP(hipMalloc(&ptr, n * sizeof(T)));
        NAVI_HIP(hipMemset(ptr, 0, n * sizeof(T)));
        owned.push_back(ptr);
        return static_cast<T *>(ptr);
    }

    FrameResult launch(const Sampling & s, uint64_t seed, int frame_index) {
        FrameResult r;
        B.temperature = s.temperature; B.top_k = s.top_k; B.repetition_penalty = s.repetition_penalty;
        B.cp_temperature = s.cp_temperature; B.cp_top_k = s.cp_top_k; B.min_frames = s.min_frames;
        B.frame_index = frame_index;
        B.talker_seed = splitmix64(seed ^ (static_cast<uint64_t>(frame_index) * 0xA24BAED4963EE407ull));
        B.cp_seed = splitmix64(B.talker_seed ^ 0x9FB21C651E98DF25ull);
        NAVI_HIP(hipMemsetAsync((void *) B.bar, 0, 3 * sizeof(unsigned), stream));
        NAVI_HIP(hipEventRecord(ev0, stream));
        void * args[] = {&TP, &TW, &CP, &CW, &B};
        const hipError_t e = hipLaunchCooperativeKernel((const void *) k_frame, dim3(grid), dim3(BLOCK), args, 0, stream);
        if (e != hipSuccess) { r.error = std::string("cooperative launch failed: ") + hipGetErrorString(e); return r; }
        NAVI_HIP(hipEventRecord(ev1, stream));
        NAVI_HIP(hipMemcpyAsync(h_codes, B.codes, sizeof(h_codes), hipMemcpyDeviceToHost, stream));
        unsigned hbar[3] = {0, 0, 0};
        NAVI_HIP(hipMemcpyAsync(hbar, (const void *) B.bar, sizeof(hbar), hipMemcpyDeviceToHost, stream));
        NAVI_HIP(hipStreamSynchronize(stream));
        float ms = 0.f;
        NAVI_HIP(hipEventElapsedTime(&ms, ev0, ev1));
        r.us = ms * 1000.0;
        if (hbar[2]) { r.error = "grid barrier timed out: the cooperative grid could not run to completion"; return r; }
        std::memcpy(r.codes, h_codes, sizeof(r.codes));
        r.eos = r.codes[0] == B.eos_id;
        r.ok = true;
        return r;
    }
};

std::unique_ptr<Frame> Frame::create(const Device & dev, const DeviceWeights & w, const Params & p, const FrameOptions & opt) {
    std::unique_ptr<Frame> f(new Frame());
    f->impl_ = std::make_unique<Impl>();
    Impl & I = *f->impl_;
    I.stream = dev.stream();
    I.p = p;
    I.opt = opt;
    const auto & tp = p.talker;
    const auto & cp = p.cp;
    if (tp.n_layer > MAX_LAYERS || cp.n_layer > CP_MAX_LAYERS) fail("frame: too many layers for the kernel");
    if (tp.ff > VEC_MAX || tp.vocab > VEC_MAX || cp.ff > VEC_MAX || cp.vocab > VEC_MAX || tp.head_dim > 128 || cp.head_dim > 128) {
        fail("frame: model shape exceeds the kernel's LDS vector");
    }
    if (p.n_code_groups != CP_POS) fail("frame: kernel is built for 16 codebooks");
    if ((cp.vocab & (cp.vocab - 1)) != 0) fail("frame: cp vocab must be a power of two for the top-k sort");
    if (tp.hidden != cp.hidden) fail("frame: talker/cp hidden sizes differ (small_to_mtp_projection not supported)");

    I.TP = {tp.hidden, tp.n_head, tp.n_kv_head, tp.head_dim, tp.ff, tp.vocab, tp.n_layer, tp.rms_eps, static_cast<float>(tp.rope_theta)};
    I.CP = {cp.hidden, cp.n_head, cp.n_kv_head, cp.head_dim, cp.ff, cp.vocab, cp.n_layer, cp.rms_eps, static_cast<float>(cp.rope_theta)};
    auto f16p = [&](const std::string & n) { return w.get(n).as<uint16_t>(); };
    auto f32p = [&](const std::string & n) { return w.get(n).as<float>(); };
    for (int l = 0; l < tp.n_layer; ++l) {
        const std::string lp = "talker.model.layers." + std::to_string(l) + ".";
        I.TW.attn_norm[l] = f32p(lp + "input_layernorm.weight");
        I.TW.ffn_norm[l] = f32p(lp + "post_attention_layernorm.weight");
        I.TW.q_norm[l] = f32p(lp + "self_attn.q_norm.weight");
        I.TW.k_norm[l] = f32p(lp + "self_attn.k_norm.weight");
        I.TW.wq[l] = f16p(lp + "self_attn.q_proj.weight");
        I.TW.wk[l] = f16p(lp + "self_attn.k_proj.weight");
        I.TW.wv[l] = f16p(lp + "self_attn.v_proj.weight");
        I.TW.wo[l] = f16p(lp + "self_attn.o_proj.weight");
        I.TW.w_gate[l] = f16p(lp + "mlp.gate_proj.weight");
        I.TW.w_up[l] = f16p(lp + "mlp.up_proj.weight");
        I.TW.w_down[l] = f16p(lp + "mlp.down_proj.weight");
    }
    I.TW.output_norm = f32p("talker.model.norm.weight");
    I.TW.codec_head = f16p("talker.codec_head.weight");
    for (int l = 0; l < cp.n_layer; ++l) {
        const std::string lp = "talker.code_predictor.model.layers." + std::to_string(l) + ".";
        I.CW.attn_norm[l] = f32p(lp + "input_layernorm.weight");
        I.CW.ffn_norm[l] = f32p(lp + "post_attention_layernorm.weight");
        I.CW.q_norm[l] = f32p(lp + "self_attn.q_norm.weight");
        I.CW.k_norm[l] = f32p(lp + "self_attn.k_norm.weight");
        I.CW.wq[l] = f16p(lp + "self_attn.q_proj.weight");
        I.CW.wk[l] = f16p(lp + "self_attn.k_proj.weight");
        I.CW.wv[l] = f16p(lp + "self_attn.v_proj.weight");
        I.CW.wo[l] = f16p(lp + "self_attn.o_proj.weight");
        I.CW.w_gate[l] = f16p(lp + "mlp.gate_proj.weight");
        I.CW.w_up[l] = f16p(lp + "mlp.up_proj.weight");
        I.CW.w_down[l] = f16p(lp + "mlp.down_proj.weight");
    }
    I.CW.output_norm = f32p("talker.code_predictor.model.norm.weight");
    I.CW.talker_codec_embd = f16p("talker.model.codec_embedding.weight");
    for (int i = 0; i < CP_HEADS; ++i) {
        I.CW.codec_embd[i] = f16p("talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight");
        I.CW.lm_head[i] = f16p("talker.code_predictor.lm_head." + std::to_string(i) + ".weight");
    }

    Work & B = I.B;
    const int Q = tp.n_head * tp.head_dim, KV = tp.n_kv_head * tp.head_dim;
    B.x = I.alloc<float>(tp.hidden); B.q = I.alloc<float>(Q); B.k = I.alloc<float>(KV); B.v = I.alloc<float>(KV);
    B.attn = I.alloc<float>(Q); B.h = I.alloc<float>(tp.ff); B.hidden_out = I.alloc<float>(tp.hidden);
    B.logits = I.alloc<float>(tp.vocab);
    B.seen = I.alloc<uint8_t>(tp.vocab);
    const int CQ = cp.n_head * cp.head_dim, CKV = cp.n_kv_head * cp.head_dim;
    B.cx = I.alloc<float>(cp.hidden); B.cq = I.alloc<float>(CQ); B.ck = I.alloc<float>(CKV); B.cv = I.alloc<float>(CKV);
    B.cattn = I.alloc<float>(CQ); B.ch = I.alloc<float>(cp.ff); B.clogits = I.alloc<float>(static_cast<std::size_t>(CP_HEADS) * cp.vocab);
    B.ckcache = I.alloc<uint16_t>(static_cast<std::size_t>(cp.n_layer) * CP_POS * CKV);
    B.cvcache = I.alloc<uint16_t>(static_cast<std::size_t>(cp.n_layer) * CP_POS * CKV);
    B.codes = I.alloc<int32_t>(CP_POS);
    B.step_out = I.alloc<float>(tp.hidden);
    B.bar = I.alloc<unsigned>(3);
    B.eos_id = p.codec_eos_id;
    B.suppress_from = tp.vocab - 1024;   // HF: suppress_tokens = range(vocab - 1024, vocab) minus EOS
    B.spin_cap = opt.spin_cap;

    // Grid: blocks_per_mp co-resident blocks per multiprocessor (WGP), checked
    // against what the kernel's registers/LDS actually allow.
    NAVI_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&I.occ, k_frame, BLOCK, 0));
    if (I.occ < 1) fail("frame: k_frame cannot be resident at all");
    if (opt.blocks_per_mp > I.occ) {
        fail("frame: blocks_per_mp " + std::to_string(opt.blocks_per_mp) + " exceeds the kernel's occupancy " + std::to_string(I.occ));
    }
    I.grid = dev.info().multiprocessors * opt.blocks_per_mp;
    NAVI_HIP(hipEventCreate(&I.ev0));
    NAVI_HIP(hipEventCreate(&I.ev1));
    return f;
}

Frame::~Frame() {
    if (!impl_) return;
    if (impl_->ev0) (void) hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void) hipEventDestroy(impl_->ev1);
    for (void * p : impl_->owned) (void) hipFree(p);
}

void Frame::reset() {
    NAVI_HIP(hipMemsetAsync(impl_->B.seen, 0, static_cast<std::size_t>(impl_->TP.vocab), impl_->stream));
}

FrameResult Frame::run_first(const float * d_hidden, const float * d_logits, const float * d_trailing, const Sampling & s,
                             std::uint64_t seed, int frame_index) {
    Impl & I = *impl_;
    I.B.skip_talker = 1;
    I.B.hidden_in = d_hidden;
    I.B.logits_in = d_logits;
    I.B.trailing = d_trailing;
    I.B.step_in = nullptr;
    return I.launch(s, seed, frame_index);
}

FrameResult Frame::run(KvCache & kv, int n_past, const float * d_trailing, const Sampling & s, std::uint64_t seed, int frame_index) {
    Impl & I = *impl_;
    if (n_past >= kv.max_pos) { FrameResult r; r.error = "KV cache full"; return r; }
    if (kv.n_layer != I.TP.n_layer) { FrameResult r; r.error = "KV cache layer count mismatch"; return r; }
    for (int l = 0; l < kv.n_layer; ++l) { I.B.kcache[l] = kv.k[static_cast<std::size_t>(l)]; I.B.vcache[l] = kv.v[static_cast<std::size_t>(l)]; }
    I.B.skip_talker = 0;
    I.B.n_past = n_past;
    I.B.step_in = I.B.step_out;
    I.B.trailing = d_trailing;
    FrameResult r = I.launch(s, seed, frame_index);
    if (r.ok) kv.n_used = n_past + 1;
    return r;
}

const float * Frame::d_step_out() const { return impl_->B.step_out; }
int Frame::grid_blocks() const { return impl_->grid; }
int Frame::occupancy_per_mp() const { return impl_->occ; }

} // namespace navi::qwen3tts
