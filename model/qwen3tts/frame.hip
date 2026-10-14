#include "model/qwen3tts/frame.h"

#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <hip/hip_fp16.h>

#include <algorithm>
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
// The LDS vector is sized per model at launch (Frame::create), not by a constant:
// the 1.7B's talker ff is 6144 where the 0.6B's is 3072, and paying the larger of
// the two on both would cost the smaller model occupancy it currently has.
// VEC_FLOOR is the historical size and stays a floor for two reasons: attend()
// keeps one score per key position in the vector, and Frame::create is not told
// the Talker's max_positions (2048 by default), so the floor is what guarantees
// the shipped configuration fits; and it keeps the 0.6B's LDS layout unchanged.
constexpr int VEC_FLOOR = 4096;
// Floats in the LDS block for a given vector size: [xs][red: 32][hist: 256][idx].
constexpr int lds_floats(int vec) { return 2 * vec + 32 + 256; }
constexpr int MV_ROWS = 2;         // rows per wave in a matvec: 2 rows x 4 loads in flight streams ~20% faster than 4 x 2 (bench/README)

struct TlParams { int hidden, n_head, n_kv_head, head_dim, ff, vocab, n_layer; float eps, rope_theta; };
// in_dim: the width the code predictor's input embeddings are stored at, which
// is the talker's hidden size (the same rows feed the talker's next-frame input).
// It equals `hidden` on a model whose two widths match, and then there is no
// projection stage and CpWeights::proj_w is null.
struct CpParams { int hidden, in_dim, n_head, n_kv_head, head_dim, ff, vocab, n_layer; float eps, rope_theta; };

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
    const uint16_t * proj_w;   // small_to_mtp_projection [hidden][in_dim], null when the widths match
    const float    * proj_b;   // its bias [hidden]
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
    int vec_max;                       // floats in the dynamic LDS vector (Frame::create)
    float * cp_input_log;              // [CP_POS][cp hidden] capture of the predictor's projected
                                       // inputs for the parity gate; null in production
    volatile unsigned * bar;           // [0] count, [1] sense, [2] abort
    unsigned long long spin_cap;
#ifdef NAVI_FRAME_PROFILE
    unsigned long long * prof;   // [2*i]: block 0 arrives at barrier i, [2*i+1]: it is released
#endif
};

// --- device helpers ---------------------------------------------------------

// Deadline-aware grid barrier on one monotonic counter: a block's target is
// the next multiple of nblocks past the value its arrival returned, so the
// last arriver writes nothing extra and waiters spin on the same word (the
// sense-reversing form cost two more writes and a fence on the critical
// path: 0.81 us, this 0.6). Returns false once any block has waited past
// spin_cap; every later barrier then short-circuits so the whole grid
// unwinds instead of deadlocking. bar[0]: arrivals, bar[2]: abort.
__device__ __forceinline__ bool grid_barrier(volatile unsigned * bar, unsigned nblocks, unsigned long long spin_cap) {
    __syncthreads();
    if (threadIdx.x == 0 && !bar[2]) {
        __threadfence();
        const unsigned prev = atomicAdd((unsigned *) &bar[0], 1u);
        const unsigned target = (prev / nblocks + 1u) * nblocks;
        unsigned long long spins = 0;
        while (bar[0] < target && !bar[2]) {
            if (++spins >= spin_cap) { bar[2] = 1u; break; }
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
#pragma unroll 4
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
// in cycling groups of MV_ROWS. (A balanced contiguous partition, and fusing
// q/k/v into one virtual matvec, were measured 2026-09-17: no change - the
// phase ends when the memory system does, not when the last wave does.)
// BIAS (non-accumulating only) adds `bias` on the wave that owns the row, so no
// second pass and no barrier is needed to make it visible. It is a template
// parameter, not a nullable pointer, so the callers that have no bias - every
// matvec on the per-frame path - compile exactly as they did before.
template <bool ACCUMULATE, bool BIAS = false>
__device__ void grid_matvec(const uint16_t * W, int n_rows, int n_cols, const float * xs, float * y,
                            const float * bias = nullptr) {
    const int lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize, nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < n_rows; row0 += nwave * MV_ROWS) {
        float acc[MV_ROWS];
        wave_dot_rows<MV_ROWS>(W, row0, n_rows, n_cols, xs, acc);
        if (lane < MV_ROWS && row0 + lane < n_rows) {
            float d = acc[0];
#pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) d = acc[r];
            if (ACCUMULATE) y[row0 + lane] += d;
            else if (BIAS) y[row0 + lane] = d + bias[row0 + lane];
            else y[row0 + lane] = d;
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

// Sampling scratch lives in the same dynamic LDS block as the vector: `hist` is
// one 8-bit digit's histogram for the radix select, `idx` the candidate indices
// for the multinomial draw. One block (block 0), BLOCK threads.

// float -> monotone unsigned key: larger float, larger key
__device__ __forceinline__ unsigned float_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_float(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

// The k-th largest of xs[0..V) (1 <= k <= V), exactly: four 8-bit radix
// passes, each a 256-bin histogram walked in parallel (one thread per bin,
// wave scans from the top digit down). Ties at the threshold are the
// caller's business, as they were with a sort. red: >= 12 ints.
__device__ float block_kth_largest(const float * xs, int V, int k, unsigned * hist, float * redf) {
    int * red = reinterpret_cast<int *>(redf);
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    unsigned prefix = 0, mask = 0;
    int need = k;
    for (int pass = 3; pass >= 0; --pass) {
        const int shift = pass * 8;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const unsigned key = float_key(xs[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        // thread t owns digit 255 - t; `above` = keys with a larger digit
        const int d = 255 - static_cast<int>(threadIdx.x);
        const int c = static_cast<int>(hist[d]);
        int incl = c;
        for (int o = 1; o < warpSize; o <<= 1) { const int v = __shfl_up(incl, o, warpSize); if (lane >= o) incl += v; }
        if (lane == warpSize - 1) red[wid] = incl;
        __syncthreads();
        int above = incl - c;
        for (int w = 0; w < wid; ++w) above += red[w];
        if (above < need && above + c >= need) { red[8] = d; red[9] = need - above; }
        __syncthreads();
        need = red[9];
        prefix |= static_cast<unsigned>(red[8]) << shift;
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    return key_float(prefix);
}

// Multinomial draw over probabilities p[0..V) (unnormalised, sum given),
// PCG32 seeded from `seed`. Only the nonzero entries are visited, in index
// order: adding an exact zero leaves the running sum unchanged, so this is
// bit-identical to walking all of p, but thread 0 walks <= top_k entries.
// Returns the index on thread 0 (garbage elsewhere). red: >= 12 ints.
__device__ int block_draw(const float * p, int V, float inv_sum, uint64_t seed, int * idx, float * redf) {
    int * red = reinterpret_cast<int *>(redf);
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    if (threadIdx.x == 0) red[0] = 0;
    __syncthreads();
    for (int base = 0; base < V; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool nz = i < V && p[i] != 0.f;
        const unsigned long long m = __ballot(nz);
        const int in_wave = __popcll(m & ((1ull << lane) - 1ull));
        if (lane == 0) red[1 + wid] = __popcll(m);
        __syncthreads();
        int before = red[0];
        for (int w = 0; w < wid; ++w) before += red[1 + w];
        if (nz) idx[before + in_wave] = i;
        __syncthreads();
        if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < nw; ++w) t += red[1 + w]; red[0] += t; }
        __syncthreads();
    }
    int chosen = V - 1;
    if (threadIdx.x == 0) {
        uint64_t st = seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
        (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
        const float u = static_cast<float>(pcg32_next(st) >> 8) * (1.0f / 16777216.0f);
        const int n = red[0];
        float cum = 0.f;
        for (int j = 0; j < n; ++j) {
            const int i = idx[j];
            cum += p[i] * inv_sum;
            if (cum >= u) { chosen = i; break; }
        }
    }
    return chosen;
}

// --- talker phases ----------------------------------------------------------

__device__ void tl_qkv(const TlParams & P, const TlWeights & W, const Work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim, P.hidden, xs, B.q);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.k);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.v);
}

// head slots [0, n_head): q heads; [n_head, +n_kv_head): k heads (+cache); [.., +n_kv_head):
// v -> cache. A block takes slots blockIdx.x, +gridDim.x, ...: one each on the XTX's
// 48 blocks, two on Strix Halo's 20 (32 slots; with one slot per block the last 12
// were never done - found at M3). Each slot's arithmetic is the same either way.
template <class WP>
__device__ void norm_rope_store(const WP & P, const float * q_norm, const float * k_norm, float * q, float * k,
                                const float * v, int pos, uint16_t * kcache, uint16_t * vcache, float * xs, float * red) {
    const int hd = P.head_dim, t = threadIdx.x, kv_stride = P.n_kv_head * hd;
    for (int b = blockIdx.x; b < P.n_head + 2 * P.n_kv_head; b += gridDim.x) {
    __syncthreads();   // xs / red free from the previous slot
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
    } else {
        const int head = b - P.n_head - P.n_kv_head;
        if (t < hd) f2h(vcache + static_cast<size_t>(pos) * kv_stride + head * hd, t, v[head * hd + t]);
    }
    }
}

// one block per q head: scores over positions 0..pos, softmax, weighted V sum
template <class WP>
__device__ void attend(const WP & P, const float * q, const uint16_t * kcache, const uint16_t * vcache, int pos,
                       float * out, float * xs) {
    for (int h = blockIdx.x; h < P.n_head; h += gridDim.x) {   // one head per block when the grid allows
    __syncthreads();
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
__device__ void tl_sample(const TlParams & P, const Work & B, float * xs, float * red, unsigned * hist, int * idx) {
    if (blockIdx.x != 0) return;
    const int V = P.vocab;
    if (B.temperature <= 0.f) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = tl_modified_logit(B, i);
        __syncthreads();
        const int best = block_argmax(xs, V, red);
        if (threadIdx.x == 0) { B.codes[0] = best; if (best != B.eos_id) B.seen[best] = 1u; }
        return;
    }
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) { xs[i] = tl_modified_logit(B, i) / B.temperature; m = fmaxf(m, xs[i]); }
    const float mx = dev::block_max(m, red);
    __syncthreads();
    const float threshold = (B.top_k > 0 && B.top_k < V) ? block_kth_largest(xs, V, B.top_k, hist, red) : -INFINITY;
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
    const int chosen = block_draw(xs, V, 1.f / ssum, B.talker_seed, idx, red);
    if (threadIdx.x == 0) {
        B.codes[0] = chosen;
        if (chosen != B.eos_id) B.seen[chosen] = 1u;
    }
    __syncthreads();
}

// --- code-predictor phases ---------------------------------------------------

// The predictor's input at step `pos`: the talker's hidden at pos 0, the talker's
// codebook-0 embedding at pos 1, then the fed-back code of step pos-1 through that
// step's own table - all at the talker's width - projected to the predictor's width
// when the two differ (HF small_to_mtp_projection; the identity when they match).
//
// Every block can name the source row on its own: codes[] and hidden_out were
// written before the barrier that precedes this phase. So the projecting path
// stages the row into LDS per block and needs no extra buffer and no extra barrier.
__device__ void cp_embed(const CpParams & P, const CpWeights & W, const Work & B, int pos, float * xs) {
    const uint16_t * row = nullptr;
    if (pos == 1) row = W.talker_codec_embd + static_cast<size_t>(B.codes[0]) * P.in_dim;
    else if (pos >= 2) row = W.codec_embd[pos - 2] + static_cast<size_t>(B.codes[pos - 1]) * P.in_dim;
    if (!W.proj_w) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
        for (int i = tid; i < P.hidden; i += nth) B.cx[i] = (pos == 0) ? B.hidden_out[i] : h2f(row, i);
        return;
    }
    for (int i = threadIdx.x; i < P.in_dim; i += blockDim.x) xs[i] = (pos == 0) ? B.hidden_out[i] : h2f(row, i);
    __syncthreads();
    grid_matvec<false, true>(W.proj_w, P.hidden, P.in_dim, xs, B.cx, W.proj_b);
}

__device__ void cp_head(const CpParams & P, const CpWeights & W, const Work & B, int pos, float * xs, float * red) {
    if (pos < 1) return;
    block_rmsnorm(B.cx, W.output_norm, P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.lm_head[pos - 1], P.vocab, P.hidden, xs, B.clogits + static_cast<size_t>(pos - 1) * P.vocab);
}

// Block 0 samples codebook `pos` from head pos-1 into codes[pos].
__device__ void cp_sample(const CpParams & P, const Work & B, int pos, float * xs, float * red, unsigned * hist, int * idx) {
    if (pos < 1 || blockIdx.x != 0) return;
    const int V = P.vocab;
    const float * lg = B.clogits + static_cast<size_t>(pos - 1) * V;
    if (B.cp_temperature <= 0.f) {
        const int best = block_argmax(lg, V, red);
        if (threadIdx.x == 0) B.codes[pos] = best;
        return;
    }
    float m = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) { xs[i] = lg[i] / B.cp_temperature; m = fmaxf(m, xs[i]); }
    const float mx = dev::block_max(m, red);
    __syncthreads();
    const float threshold = (B.cp_top_k > 0 && B.cp_top_k < V) ? block_kth_largest(xs, V, B.cp_top_k, hist, red) : -INFINITY;
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
    const int chosen = block_draw(xs, V, 1.f / ssum, B.cp_seed + static_cast<uint64_t>(pos) * 0xD1B54A32D192ED03ull, idx, red);
    if (threadIdx.x == 0) B.codes[pos] = chosen;
    __syncthreads();
}


// next frame's step embedding: codec_embd[cb0] + sum_cb cp_embd[cb][code] + trailing.
// This is the talker's input, so it is assembled at the talker's width (in_dim) -
// the width the embedding tables are stored at - not the predictor's.
__device__ void assemble_step(const CpParams & P, const CpWeights & W, const Work & B) {
    const int hidden = P.in_dim;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden; h += gridDim.x * blockDim.x) {
        float v = h2f(W.talker_codec_embd + static_cast<size_t>(B.codes[0]) * hidden, h);
#pragma unroll
        for (int cb = 0; cb < CP_HEADS; ++cb) v += h2f(W.codec_embd[cb] + static_cast<size_t>(B.codes[1 + cb]) * hidden, h);
        v += B.trailing[h];
        B.step_out[h] = v;
    }
}

// --- the frame ----------------------------------------------------------------

// FLOOR_LDS: the vector fits VEC_FLOOR, so the block is a compile-time __shared__
// array and the carve is a compile-time offset - which is how this kernel was
// written before the 1.7B needed a bigger vector. Measured on gfx1151, going to a
// runtime LDS base costs the 0.6B about 1.5% of its frame time (LDS addressing
// loses its immediate offsets), so the model that fits keeps the static form and
// only the model that does not pays for the flexibility.
template <bool FLOOR_LDS>
__global__ void __launch_bounds__(BLOCK) k_frame_t(TlParams TP, TlWeights TW, CpParams CP, CpWeights CW, Work B) {
    __shared__ float floor_lds[FLOOR_LDS ? lds_floats(VEC_FLOOR) : 1];
    extern __shared__ float dyn_lds[];
    float * const base = FLOOR_LDS ? floor_lds : dyn_lds;
    const int vec = FLOOR_LDS ? VEC_FLOOR : B.vec_max;
    float * xs = base;
    float * red = xs + vec;
    unsigned * hist = reinterpret_cast<unsigned *>(red + 32);
    int * idx = reinterpret_cast<int *>(hist + 256);
    const unsigned nb = gridDim.x;
#ifdef NAVI_FRAME_PROFILE
    int prof_i = 0;
#define GBSYNC() do { if (blockIdx.x == 0 && threadIdx.x == 0) B.prof[2 * prof_i] = wall_clock64(); \
                      if (!grid_barrier(B.bar, nb, B.spin_cap)) return; \
                      if (blockIdx.x == 0 && threadIdx.x == 0) B.prof[2 * prof_i + 1] = wall_clock64(); ++prof_i; } while (0)
#else
#define GBSYNC() do { if (!grid_barrier(B.bar, nb, B.spin_cap)) return; } while (0)
#endif

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
    tl_sample(TP, B, xs, red, hist, idx);                                                                          GBSYNC();

    const int ckv = CP.n_kv_head * CP.head_dim;
    for (int pos = 0; pos < CP_POS; ++pos) {
        cp_embed(CP, CW, B, pos, xs);                                                                         GBSYNC();
        // Parity capture (null in production), after the barrier: cp_embed's projection
        // writes B.cx partitioned across the grid, so reading it grid-strided is only
        // safe once every block has arrived. The next phase only reads B.cx too, so
        // this needs no barrier of its own.
        if (B.cp_input_log) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < CP.hidden; i += gridDim.x * blockDim.x)
                B.cp_input_log[static_cast<size_t>(pos) * CP.hidden + i] = B.cx[i];
        }
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
        cp_sample(CP, B, pos, xs, red, hist, idx);                                                                 GBSYNC();
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
    std::size_t lds_bytes = 0;    // dynamic LDS for the launch; 0 when the static form is used
    bool floor_lds = false;
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

#ifdef NAVI_FRAME_PROFILE
    // Phase names in barrier order: the talker (6 per layer) then the code predictor.
    void dump_profile(int frame_index) {
        static std::vector<unsigned long long> h(2 * 4096);
        NAVI_HIP(hipMemcpy(h.data(), B.prof, h.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
        int khz = 0;
        NAVI_HIP(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
        const double us_per_tick = 1e3 / khz;
        std::vector<std::string> names;
        if (B.skip_talker) names.push_back("tl:copy");
        else {
            names.push_back("tl:step_in");
            for (int il = 0; il < TP.n_layer; ++il) for (const char * n : {"tl:qkv", "tl:norm_rope", "tl:attend", "tl:wo", "tl:gate_up", "tl:down"}) names.push_back(n);
            names.push_back("tl:head");
        }
        names.push_back("tl:sample");
        for (int pos = 0; pos < CP_POS; ++pos) {
            names.push_back("cp:embed");
            for (int il = 0; il < CP.n_layer; ++il) for (const char * n : {"cp:qkv", "cp:norm_rope", "cp:attend", "cp:wo", "cp:gate_up", "cp:down"}) names.push_back(n);
            names.push_back("cp:head");
            names.push_back("cp:sample");
        }
        names.push_back("assemble");
        std::map<std::string, std::pair<double, double>> acc;   // name -> (phase us, barrier wait us)
        std::map<std::string, int> cnt;
        double total_phase = 0, total_wait = 0;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const double release_prev = i == 0 ? 0 : static_cast<double>(h[2 * (i - 1) + 1]);
            const double arrive = static_cast<double>(h[2 * i]), release = static_cast<double>(h[2 * i + 1]);
            if (i == 0) continue;
            const double phase = (arrive - release_prev) * us_per_tick, wait = (release - arrive) * us_per_tick;
            acc[names[i]].first += phase; acc[names[i]].second += wait; cnt[names[i]]++;
            total_phase += phase; total_wait += wait;
        }
        if (frame_index != 8) return;   // one representative frame
        std::fprintf(stderr, "frame %d profile (block 0): %.0f us in phases + %.0f us waiting at %zu barriers\n", frame_index, total_phase, total_wait, names.size());
        std::fprintf(stderr, "  %-14s %5s %9s %9s %9s\n", "phase", "n", "phase us", "wait us", "us each");
        for (const auto & [n, v] : acc) std::fprintf(stderr, "  %-14s %5d %9.0f %9.0f %9.2f\n", n.c_str(), cnt[n], v.first, v.second, v.first / cnt[n]);
    }
#endif

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
        const void * k = floor_lds ? (const void *) k_frame_t<true> : (const void *) k_frame_t<false>;
        const hipError_t e = hipLaunchCooperativeKernel(k, dim3(grid), dim3(BLOCK), args, lds_bytes, stream);
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
#ifdef NAVI_FRAME_PROFILE
        dump_profile(frame_index);
#endif
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
    if (tp.head_dim > 128 || cp.head_dim > 128) fail("frame: head_dim exceeds the kernel's block layout");
    if (p.n_code_groups != CP_POS) fail("frame: kernel is built for 16 codebooks");
    if ((cp.vocab & (cp.vocab - 1)) != 0) fail("frame: cp vocab must be a power of two for the top-k sort");

    // Every vector the kernel stages in LDS, plus the floor (see VEC_FLOOR).
    const int vec_max = std::max({VEC_FLOOR, tp.hidden, tp.ff, tp.vocab, tp.n_head * tp.head_dim,
                                  cp.hidden, cp.ff, cp.vocab, cp.n_head * cp.head_dim});
    I.floor_lds = vec_max == VEC_FLOOR;
    const std::size_t need = static_cast<std::size_t>(lds_floats(vec_max)) * sizeof(float);
    if (need > dev.info().lds_per_block) {
        fail("frame: the model needs " + std::to_string(need) + " B of LDS per block, the device has "
             + std::to_string(dev.info().lds_per_block));
    }
    // The static form carries its own LDS; only the larger form asks the launch for it.
    I.lds_bytes = I.floor_lds ? 0 : need;

    I.TP = {tp.hidden, tp.n_head, tp.n_kv_head, tp.head_dim, tp.ff, tp.vocab, tp.n_layer, tp.rms_eps, static_cast<float>(tp.rope_theta)};
    I.CP = {cp.hidden, tp.hidden, cp.n_head, cp.n_kv_head, cp.head_dim, cp.ff, cp.vocab, cp.n_layer, cp.rms_eps, static_cast<float>(cp.rope_theta)};
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
    // The predictor's inputs arrive at the talker's width (docs/model.md). When the
    // two widths differ they are projected down first; when they match, HF's stage is
    // nn.Identity() and the tensors are absent, so the kernel takes the direct path.
    if (cp.hidden != tp.hidden) {
        I.CW.proj_w = f16p("talker.code_predictor.small_to_mtp_projection.weight");
        I.CW.proj_b = f32p("talker.code_predictor.small_to_mtp_projection.bias");
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
    B.vec_max = vec_max;
    B.bar = I.alloc<unsigned>(3);
#ifdef NAVI_FRAME_PROFILE
    B.prof = I.alloc<unsigned long long>(2 * 4096);
#endif
    B.eos_id = p.codec_eos_id;
    B.suppress_from = tp.vocab - 1024;   // HF: suppress_tokens = range(vocab - 1024, vocab) minus EOS
    B.spin_cap = opt.spin_cap;

    // Grid: blocks_per_mp co-resident blocks per multiprocessor (WGP), checked
    // against what the kernel's registers/LDS actually allow.
    NAVI_HIP(I.floor_lds ? hipOccupancyMaxActiveBlocksPerMultiprocessor(&I.occ, k_frame_t<true>, BLOCK, 0)
                         : hipOccupancyMaxActiveBlocksPerMultiprocessor(&I.occ, k_frame_t<false>, BLOCK, I.lds_bytes));
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
    // attend() keeps one score per key position in the LDS vector, so a position
    // past it would write off the end of xs. The KV cache's max_pos is the
    // Talker's business and says nothing about the kernel's LDS, so check both.
    if (n_past < 0 || n_past >= I.B.vec_max) {
        FrameResult r;
        r.error = "position " + std::to_string(n_past) + " is outside the frame kernel's score vector ("
                  + std::to_string(I.B.vec_max) + ")";
        return r;
    }
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

void Frame::enable_cp_input_capture() {
    if (!impl_->B.cp_input_log) impl_->B.cp_input_log = impl_->alloc<float>(static_cast<std::size_t>(CP_POS) * impl_->CP.hidden);
}
const float * Frame::d_cp_inputs() const { return impl_->B.cp_input_log; }
int Frame::cp_hidden() const { return impl_->CP.hidden; }
int Frame::max_position() const { return impl_->B.vec_max; }
std::size_t Frame::lds_bytes() const {
    return impl_->floor_lds ? static_cast<std::size_t>(lds_floats(VEC_FLOOR)) * sizeof(float) : impl_->lds_bytes;
}
int Frame::grid_blocks() const { return impl_->grid; }
int Frame::occupancy_per_mp() const { return impl_->occ; }

} // namespace navi::qwen3tts
