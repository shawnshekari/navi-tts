// Single fused talker+code-predictor frame kernel — see frame_fused.h.
//
// One cooperative launch per frame:
//   step 0  copy the incoming step embedding into the residual stream
//   step 1  talker 28-layer step (tl_* phases, verbatim from
//           talker_hip.hip) -> hidden_out + cb0 (into codes[0]) + the
//           F16 KV row at n_past
//   step 2  cp 16-position loop (cp_* phases, verbatim from
//           code_pred_hip.hip) -> codes[1..15]
//   step 3  next-frame step_embd assembly (k_assemble_step_embd math,
//           same summation order) -> step_out
//
// All phases share ONE deadline-aware grid_barrier (hip/hip_dev.h) with
// ONE bar buffer and ONE spin cap, so the talker's ~170 barriers/step
// and the cp's 16 x ~8-phase barriers interleave in a single grid.
// A barrier timeout sets bar[2]; every later barrier short-circuits and
// the grid unwinds — run_frame() then returns false and the caller falls
// back to the three-launch chained path.
//
// Parity: the phase bodies are the shipped kernels' device code with the
// same summation order and the same per-position sampling RNG seeds, so
// cb0, the 15 codes and the next step_embd are bit-identical to the
// chained path given the same seeds.

#include "frame_fused.h"
#include "hip/hip_dev.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace qwen3_tts::hipdev;

namespace qwen3_tts {

namespace {

constexpr int BLOCK = 256;                 // threads per block
// Largest vector staged in LDS: talker ff (3072) and the padded vocab
// for the bitonic top-k (codec vocab 3072 -> next pow2 4096).
constexpr int FF_MAX_VEC = 4096;

struct ff_work {
    // --- talker side (field names match talker_hip.hip's tl_work) ---
    float * x;          // [hidden]        residual stream
    float * q;          // [n_head*head_dim]
    float * k;          // [n_kv_head*head_dim]
    float * v;          // [n_kv_head*head_dim]
    float * attn;       // [n_head*head_dim]
    float * h;          // [ff]
    float * hidden_out; // [hidden]
    float * logits;     // [talker vocab]
    uint8_t * seen;     // [talker vocab] repetition set, shared with HipTalker
    const uint16_t * kcache[TL_MAX_LAYERS];  // external, row stride n_kv_head*head_dim
    const uint16_t * vcache[TL_MAX_LAYERS];
    int32_t n_past;
    float temperature;
    int32_t top_k;
    float repetition_penalty;   // <= 0 disables
    int32_t eos_id;
    uint64_t talker_seed;
    // --- code-predictor side (cp_* phases read/write these) ---
    float * cx;         // [hidden]
    float * cq;         // [n_head*head_dim]
    float * ck;         // [n_kv_head*head_dim]
    float * cv;         // [n_kv_head*head_dim]
    __half * ckcache;   // [n_layer][CP_MAX_POS][n_kv_head*head_dim]
    __half * csvcache;
    float * cattn;      // [n_head*head_dim]
    float * ch;         // [ff]
    float * clogits;    // [15][vocab]
    int32_t * codes;    // [CP_MAX_POS]: [0] = cb0 (written by the talker
                        //                  sample here), [1..15] sampled by the cp
    uint64_t cp_seed;
    // --- frame plumbing ---
    const float * step_in;   // [hidden] this frame's step embedding (device)
    const float * trailing;  // [hidden] next frame's trailing row (device)
    float * step_out;        // [hidden] next frame's step embedding (device)
    // Cooperative barrier state: [0]=count, [1]=sense, [2]=abort. ONE
    // buffer shared by the talker and cp barrier schedules (the fused
    // kernel's whole grid). Zeroed before each cooperative launch.
    volatile unsigned * bar;
    unsigned long long spin_cap;
};

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { if (err) *err = std::string(#x) + ": " + hipGetErrorString(e_); return false; } } while (0)

// ---------------------------------------------------------------------------
// Talker phases — copied from talker_hip.hip (tl_work -> ff_work; the
// sampled cb0 lands in codes[0] instead of a separate cb0 buffer, and
// the seed field is talker_seed).
// ---------------------------------------------------------------------------

__device__ void tl_qkv(const tl_params & P, const tl_weights & W, const ff_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim,    P.hidden, xs, B.q);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.k);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.v);
}

// blocks [0, n_head): q heads; [n_head, n_head+n_kv_head): k heads (+cache);
// [.., +n_kv_head): v -> cache. Threads [0, head_dim) active.
__device__ void tl_norm_rope(const tl_params & P, const tl_weights & W, const ff_work & B, int il, float * xs, float * red) {
    const int hd = P.head_dim;
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int kv_stride = P.n_kv_head * hd;
    if (b < P.n_head + P.n_kv_head) {
        const bool is_q = b < P.n_head;
        const int head = is_q ? b : b - P.n_head;
        float * vec = (is_q ? B.q : B.k) + head * hd;
        const uint16_t * nw = is_q ? W.q_norm[il] : W.k_norm[il];

        float e = (t < hd) ? vec[t] : 0.0f;
        const float ss = block_sum(e * e, red);
        const float scale = rsqrtf(ss / hd + P.eps);
        if (t < hd) xs[t] = e * scale * f32_from_bits(nw, t);
        __syncthreads();

        if (t < hd / 2) {
            // NEOX: pair (t, t + hd/2), theta = pos * base^(-2t/hd)
            const float theta = (float) B.n_past * powf(P.rope_theta, -2.0f * (float) t / (float) hd);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = xs[t], x1 = xs[t + hd / 2];
            const float r0 = x0 * c - x1 * s;
            const float r1 = x0 * s + x1 * c;
            vec[t] = r0;
            vec[t + hd / 2] = r1;
            if (!is_q) {
                __half * kc = (__half *) B.kcache[il] + (size_t) B.n_past * kv_stride + head * hd;
                kc[t] = __float2half(r0);
                kc[t + hd / 2] = __float2half(r1);
            }
        }
    } else if (b < P.n_head + 2 * P.n_kv_head) {
        const int head = b - P.n_head - P.n_kv_head;
        if (t < hd) {
            __half * vc = (__half *) B.vcache[il] + (size_t) B.n_past * kv_stride + head * hd;
            vc[t] = __float2half(B.v[head * hd + t]);
        }
    }
}

// one block per q head: scores over positions 0..n_past, softmax, weighted V sum
__device__ void tl_attn(const tl_params & P, const ff_work & B, int il, float * xs, float * red) {
    const int h = blockIdx.x;
    if (h >= P.n_head) return;
    const int hd = P.head_dim;
    const int kvh = h / (P.n_head / P.n_kv_head);
    const int kv_stride = P.n_kv_head * hd;
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nw   = blockDim.x / warpSize;
    const float scale = rsqrtf((float) hd);
    float * scores = xs;             // [n_past + 1]
    const float * q = B.q + h * hd;
    const __half * kc = (const __half *) B.kcache[il];
    const __half * vc = (const __half *) B.vcache[il];

    for (int j = wid; j <= B.n_past; j += nw) {
        const __half * kj = kc + (size_t) j * kv_stride + kvh * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += warpSize) d += q[i] * __half2float(kj[i]);
        d = warp_sum(d);
        if (lane == 0) scores[j] = d * scale;
    }
    __syncthreads();
    if (wid == 0) {
        float m = -INFINITY;
        for (int j = lane; j <= B.n_past; j += warpSize) m = fmaxf(m, scores[j]);
        for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, warpSize));
        float s = 0.0f;
        for (int j = lane; j <= B.n_past; j += warpSize) { const float e = expf(scores[j] - m); scores[j] = e; s += e; }
        s = warp_sum(s);
        const float inv = 1.0f / s;
        for (int j = lane; j <= B.n_past; j += warpSize) scores[j] *= inv;
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        const int i = threadIdx.x;
        float acc = 0.0f;
        for (int j = 0; j <= B.n_past; ++j) {
            acc += scores[j] * __half2float(vc[(size_t) j * kv_stride + kvh * hd + i]);
        }
        B.attn[h * hd + i] = acc;
    }
    (void) red;
}

__device__ void tl_oproj(const tl_params & P, const tl_weights & W, const ff_work & B, int il, float * xs) {
    block_stage(B.attn, P.n_head * P.head_dim, xs);
    grid_matvec<true>(W.wo[il], P.hidden, P.n_head * P.head_dim, xs, B.x);
}

__device__ void tl_gate_up(const tl_params & P, const tl_weights & W, const ff_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.ffn_norm[il], P.hidden, P.eps, xs, red);
    const int lane  = threadIdx.x % warpSize;
    const int wpb   = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize;
    const int nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < P.ff; row0 += nwave * MV_ROWS) {
        float g[MV_ROWS], u[MV_ROWS];
        wave_dot_rows<MV_ROWS>(W.w_gate[il], row0, P.ff, P.hidden, xs, g);
        wave_dot_rows<MV_ROWS>(W.w_up[il],   row0, P.ff, P.hidden, xs, u);
        if (lane < MV_ROWS && row0 + lane < P.ff) {
            float gg = g[0], uu = u[0];
            #pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) { gg = g[r]; uu = u[r]; }
            B.h[row0 + lane] = gg / (1.0f + expf(-gg)) * uu;
        }
    }
}

__device__ void tl_down(const tl_params & P, const tl_weights & W, const ff_work & B, int il, float * xs) {
    block_stage(B.h, P.ff, xs);
    grid_matvec<true>(W.w_down[il], P.hidden, P.ff, xs, B.x);
}

__device__ void tl_head(const tl_params & P, const tl_weights & W, const ff_work & B, float * xs, float * red) {
    block_rmsnorm(B.x, W.output_norm, P.hidden, P.eps, xs, red);
    // The normalized hidden is the code predictor's position-0 input;
    // stash it before the head matvec (xs stays intact for it).
    for (int i = threadIdx.x; i < P.hidden; i += blockDim.x) B.hidden_out[i] = xs[i];
    __syncthreads();
    grid_matvec<false>(W.codec_head, P.vocab, P.hidden, xs, B.logits);
}

// The logit the sampler sees for token i: suppression of
// [vocab - 1024, vocab) except eos, then the HuggingFace repetition
// penalty on tokens seen in earlier frames. Mirrors generate() exactly.
__device__ __forceinline__ float tl_modified_logit(const tl_params & P, const ff_work & B, int i) {
    float v = B.logits[i];
    if (i >= P.vocab - 1024 && i != B.eos_id) return -INFINITY;
    if (B.repetition_penalty > 0.0f && B.seen[i]) {
        v = (v > 0.0f) ? v / B.repetition_penalty : v * B.repetition_penalty;
    }
    return v;
}

// One block (blockIdx.x == 0) samples cb0 and writes B.codes[0]. Greedy if
// temperature <= 0, else temperature scaling, top-k threshold (bitonic
// sort), softmax, PCG32 multinomial seeded per call. The chosen token is
// added to the seen set unless it is EOS (the host stops before emitting
// it, matching generate()).
__device__ void tl_sample(const tl_params & P, const ff_work & B, float * xs, float * red) {
    if (blockIdx.x != 0) return;
    const int V = P.vocab;

    if (B.temperature <= 0.0f) {
        // Greedy over the modified logits: copy them into xs, then argmax.
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = tl_modified_logit(P, B, i);
        __syncthreads();
        const int best = block_argmax(xs, V, red);
        if (threadIdx.x == 0) {
            B.codes[0] = best;
            if (best != B.eos_id) B.seen[best] = 1u;
        }
        return;
    }

    // Sampled path. Bitonic sort of the scaled modified logits in xs,
    // padded to the next power of two (codec vocab 3072 -> 4096) with
    // -INF so the network is well-formed; the threshold and the CDF scan
    // only look at the first V entries.
    const int N = 4096;   // next pow2 >= vocab, == FF_MAX_VEC
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        xs[i] = (i < V) ? tl_modified_logit(P, B, i) / B.temperature : -INFINITY;
    }
    __syncthreads();

    if (B.top_k > 0 && B.top_k < V) {
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
        const float threshold = xs[B.top_k - 1];  // k-th largest scaled logit
        const float mx = xs[0];                  // max scaled logit
        // Masked softmax over the ORIGINAL order so the CDF index is the
        // vocab token id (the sort only gave us the threshold).
        float ssum = 0.0f;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = tl_modified_logit(P, B, i) / B.temperature;
            const float e = (v >= threshold) ? expf(v - mx) : 0.0f;
            xs[i] = e;
            ssum += e;
        }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.talker_seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.codes[0] = chosen;
            if (chosen != B.eos_id) B.seen[chosen] = 1u;
        }
    } else {
        // temperature only, no top-k
        float maxl = -INFINITY;
        for (int i = threadIdx.x; i < V; i += blockDim.x) maxl = fmaxf(maxl, tl_modified_logit(P, B, i) / B.temperature);
        maxl = block_sum(maxl, red);
        float ssum = 0.0f;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const float e = expf(tl_modified_logit(P, B, i) / B.temperature - maxl);
            xs[i] = e;
            ssum += e;
        }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.talker_seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.codes[0] = chosen;
            if (chosen != B.eos_id) B.seen[chosen] = 1u;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Code-predictor phases — copied from code_pred_hip.hip (cp_work ->
// ff_work with the c-prefixed fields; position-0 input reads the
// talker's hidden_out; the seed field is cp_seed).
// ---------------------------------------------------------------------------

__device__ void cp_embed(const cp_params & P, const cp_weights & W, const ff_work & B, int pos) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;
    const uint16_t * row = nullptr;
    if (pos == 1) {
        row = W.talker_codec_embd + (size_t) B.codes[0] * P.hidden;
    } else if (pos >= 2) {
        row = W.codec_embd[pos - 2] + (size_t) B.codes[pos - 1] * P.hidden;
    }
    for (int i = tid; i < P.hidden; i += nth) {
        B.cx[i] = (pos == 0) ? B.hidden_out[i] : __half2float(reinterpret_cast<const __half *>(row)[i]);
    }
}

__device__ void cp_qkv(const cp_params & P, const cp_weights & W, const ff_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.cx, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim,    P.hidden, xs, B.cq);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.ck);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.cv);
}

// blocks [0, n_head): q heads; [n_head, n_head+n_kv_head): k heads (+cache);
// [.., +n_kv_head): v -> cache. Threads [0, head_dim) active.
__device__ void cp_norm_rope(const cp_params & P, const cp_weights & W, const ff_work & B, int il, int pos, float * xs, float * red) {
    const int hd = P.head_dim;
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int kv_stride = P.n_kv_head * hd;
    if (b < P.n_head + P.n_kv_head) {
        const bool is_q = b < P.n_head;
        const int head = is_q ? b : b - P.n_head;
        float * vec = (is_q ? B.cq : B.ck) + head * hd;
        const uint16_t * nw = is_q ? W.q_norm[il] : W.k_norm[il];

        float e = (t < hd) ? vec[t] : 0.0f;
        const float ss = block_sum(e * e, red);
        const float scale = rsqrtf(ss / hd + P.eps);
        if (t < hd) xs[t] = e * scale * f32_from_bits(nw, t);
        __syncthreads();

        if (t < hd / 2) {
            // NEOX: pair (t, t + hd/2), theta = pos * base^(-2t/hd)
            const float theta = (float) pos * powf(P.rope_theta, -2.0f * (float) t / (float) hd);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = xs[t], x1 = xs[t + hd / 2];
            const float r0 = x0 * c - x1 * s;
            const float r1 = x0 * s + x1 * c;
            vec[t] = r0;
            vec[t + hd / 2] = r1;
            if (!is_q) {
                __half * kc = B.ckcache + ((size_t) il * CP_MAX_POS + pos) * kv_stride + head * hd;
                kc[t] = __float2half(r0);
                kc[t + hd / 2] = __float2half(r1);
            }
        }
    } else if (b < P.n_head + 2 * P.n_kv_head) {
        const int head = b - P.n_head - P.n_kv_head;
        if (t < hd) {
            __half * vc = B.csvcache + ((size_t) il * CP_MAX_POS + pos) * kv_stride + head * hd;
            vc[t] = __float2half(B.cv[head * hd + t]);
        }
    }
}

// one block per q head: scores over positions 0..pos, softmax, weighted V sum
__device__ void cp_attn(const cp_params & P, const ff_work & B, int il, int pos, float * xs, float * red) {
    const int h = blockIdx.x;
    if (h >= P.n_head) return;
    const int hd = P.head_dim;
    const int kvh = h / (P.n_head / P.n_kv_head);
    const int kv_stride = P.n_kv_head * hd;
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nw   = blockDim.x / warpSize;
    const float scale = rsqrtf((float) hd);
    float * scores = xs;             // [CP_MAX_POS]
    const float * q = B.cq + h * hd;

    for (int j = wid; j <= pos; j += nw) {
        const __half * kc = B.ckcache + ((size_t) il * CP_MAX_POS + j) * kv_stride + kvh * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += warpSize) d += q[i] * __half2float(kc[i]);
        d = warp_sum(d);
        if (lane == 0) scores[j] = d * scale;
    }
    __syncthreads();
    if (wid == 0) {
        float m = -INFINITY;
        for (int j = lane; j <= pos; j += warpSize) m = fmaxf(m, scores[j]);
        for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, warpSize));
        float s = 0.0f;
        for (int j = lane; j <= pos; j += warpSize) { const float e = expf(scores[j] - m); scores[j] = e; s += e; }
        s = warp_sum(s);
        const float inv = 1.0f / s;
        for (int j = lane; j <= pos; j += warpSize) scores[j] *= inv;
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        const int i = threadIdx.x;
        float acc = 0.0f;
        for (int j = 0; j <= pos; ++j) {
            const __half * vc = B.csvcache + ((size_t) il * CP_MAX_POS + j) * kv_stride + kvh * hd;
            acc += scores[j] * __half2float(vc[i]);
        }
        B.cattn[h * hd + i] = acc;
    }
    (void) red;
}

__device__ void cp_oproj(const cp_params & P, const cp_weights & W, const ff_work & B, int il, float * xs) {
    block_stage(B.cattn, P.n_head * P.head_dim, xs);
    grid_matvec<true>(W.wo[il], P.hidden, P.n_head * P.head_dim, xs, B.cx);
}

__device__ void cp_gate_up(const cp_params & P, const cp_weights & W, const ff_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.cx, W.ffn_norm[il], P.hidden, P.eps, xs, red);
    const int lane  = threadIdx.x % warpSize;
    const int wpb   = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize;
    const int nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < P.ff; row0 += nwave * MV_ROWS) {
        float g[MV_ROWS], u[MV_ROWS];
        wave_dot_rows<MV_ROWS>(W.w_gate[il], row0, P.ff, P.hidden, xs, g);
        wave_dot_rows<MV_ROWS>(W.w_up[il],   row0, P.ff, P.hidden, xs, u);
        if (lane < MV_ROWS && row0 + lane < P.ff) {
            float gg = g[0], uu = u[0];
            #pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) { gg = g[r]; uu = u[r]; }
            B.ch[row0 + lane] = gg / (1.0f + expf(-gg)) * uu;
        }
    }
}

__device__ void cp_down(const cp_params & P, const cp_weights & W, const ff_work & B, int il, float * xs) {
    block_stage(B.ch, P.ff, xs);
    grid_matvec<true>(W.w_down[il], P.hidden, P.ff, xs, B.cx);
}

__device__ void cp_head(const cp_params & P, const cp_weights & W, const ff_work & B, int pos, float * xs, float * red) {
    if (pos < 1) return;
    block_rmsnorm(B.cx, W.output_norm, P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.lm_head[pos - 1], P.vocab, P.hidden, xs, B.clogits + (size_t) (pos - 1) * P.vocab);
}

// One block (blockIdx.x == 0) samples head h = pos-1 and writes B.codes[pos].
// Mirrors the host sample_or_argmax: greedy if temperature <= 0, else
// temperature scaling, top-k threshold (bitonic sort), softmax,
// multinomial draw. RNG: PCG32 seeded per (seed, pos) so a frame is
// reproducible and positions are independent.
__device__ void cp_sample(const cp_params & P, const ff_work & B, int pos, float * xs, float * red) {
    if (pos < 1 || blockIdx.x != 0) return;
    const int h = pos - 1;
    const int V = P.vocab;
    const float * lg = B.clogits + (size_t) h * V;

    if (B.temperature <= 0.0f) {
        const int best = block_argmax(lg, V, red);  // all threads must participate
        if (threadIdx.x == 0) B.codes[pos] = best;
        return;
    }

    // Sampled path: bitonic sort of the whole vocab in LDS for the top-k
    // threshold; the CDF scan runs over the original order so the index
    // is the vocab token id.
    for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = lg[i] / B.temperature;
    __syncthreads();

    if (B.top_k > 0 && B.top_k < V) {
        for (int k = 2; k <= V; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                __syncthreads();
                for (int i = threadIdx.x; i < (V >> 1); i += blockDim.x) {
                    const int l = ((i & ~(j - 1)) << 1) | (i & (j - 1));
                    const int m = l + j;
                    const bool desc = (l & k) == 0;
                    const float x = xs[l], y = xs[m];
                    if (desc ? (x < y) : (x > y)) { xs[l] = y; xs[m] = x; }
                }
            }
        }
        __syncthreads();
        const float threshold = xs[B.top_k - 1];  // k-th largest scaled logit
        const float mx = xs[0];                  // max scaled logit
        // Masked softmax over the ORIGINAL order so the CDF index is the
        // vocab token id (the sort only gave us the threshold).
        float ssum = 0.0f;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const float v = lg[i] / B.temperature;
            const float e = (v >= threshold) ? expf(v - mx) : 0.0f;
            xs[i] = e;
            ssum += e;
        }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.cp_seed * 0x9E3779B97F4A7C15ull + (uint64_t) pos * 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.codes[pos] = chosen;
        }
    } else {
        // temperature only, no top-k; xs already holds the scaled logits
        float maxl = -INFINITY;
        for (int i = threadIdx.x; i < V; i += blockDim.x) maxl = fmaxf(maxl, xs[i]);
        maxl = block_sum(maxl, red);
        float ssum = 0.0f;
        for (int i = threadIdx.x; i < V; i += blockDim.x) { const float e = expf(xs[i] - maxl); xs[i] = e; ssum += e; }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.cp_seed * 0x9E3779B97F4A7C15ull + (uint64_t) pos * 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.codes[pos] = chosen;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Next-frame step embedding assembly — same math and summation order as
// k_assemble_step_embd in code_pred_hip.hip (bit-identical to the host
// path): out[h] = codec_embd[cb0][h] + sum_cb cp_embd[cb][code][h] +
// trailing[h].
// ---------------------------------------------------------------------------
__device__ void ff_assemble(const cp_params & P, const cp_weights & W, const ff_work & B) {
    const int hidden = P.hidden;
    const int32_t * codes = B.codes;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden;
         h += gridDim.x * blockDim.x) {
        float v = __half2float(reinterpret_cast<const __half *>(W.talker_codec_embd)[(size_t)codes[0] * hidden + h]);
        #pragma unroll
        for (int cb = 0; cb < CP_N_HEADS_OUT; ++cb) {
            v += __half2float(reinterpret_cast<const __half *>(W.codec_embd[cb])[(size_t)codes[1 + cb] * hidden + h]);
        }
        v += B.trailing[h];
        B.step_out[h] = v;
    }
}

// ---------------------------------------------------------------------------
// The fused frame: one cooperative launch, one barrier schedule.
// ---------------------------------------------------------------------------
__global__ void k_frame_fused(tl_params TP, tl_weights TW, cp_params CP, cp_weights CW, ff_work B) {
    __shared__ float xs[FF_MAX_VEC];
    __shared__ float red[32];
    const unsigned nb = gridDim.x;
#define GBSYNC() do { if (!grid_barrier(B.bar, nb, B.spin_cap)) return; } while (0)

    // 0. incoming step embedding -> residual stream
    for (int i = threadIdx.x; i < TP.hidden; i += blockDim.x) B.x[i] = B.step_in[i];
    GBSYNC();

    // 1. talker 28-layer step -> hidden_out + codes[0] (cb0) + KV row at n_past
    for (int il = 0; il < TP.n_layer; ++il) {
        tl_qkv(TP, TW, B, il, xs, red);        GBSYNC();
        tl_norm_rope(TP, TW, B, il, xs, red);  GBSYNC();
        tl_attn(TP, B, il, xs, red);           GBSYNC();
        tl_oproj(TP, TW, B, il, xs);           GBSYNC();
        tl_gate_up(TP, TW, B, il, xs, red);    GBSYNC();
        tl_down(TP, TW, B, il, xs);            GBSYNC();
    }
    tl_head(TP, TW, B, xs, red);               GBSYNC();
    tl_sample(TP, B, xs, red);                 GBSYNC();

    // 2. cp 16-position loop -> codes[1..15]
    for (int pos = 0; pos < CP_MAX_POS; ++pos) {
        cp_embed(CP, CW, B, pos);              GBSYNC();
        for (int il = 0; il < CP.n_layer; ++il) {
            cp_qkv(CP, CW, B, il, xs, red);              GBSYNC();
            cp_norm_rope(CP, CW, B, il, pos, xs, red);   GBSYNC();
            cp_attn(CP, B, il, pos, xs, red);             GBSYNC();
            cp_oproj(CP, CW, B, il, xs);                  GBSYNC();
            cp_gate_up(CP, CW, B, il, xs, red);           GBSYNC();
            cp_down(CP, CW, B, il, xs);                   GBSYNC();
        }
        cp_head(CP, CW, B, pos, xs, red);      GBSYNC();
        cp_sample(CP, B, pos, xs, red);        GBSYNC();
    }

    // 3. next-frame step embedding assembly
    ff_assemble(CP, CW, B);
    GBSYNC();
#undef GBSYNC
}

} // namespace

struct HipFrameFusion::impl {
    tl_params TP;
    tl_weights TW;
    cp_params CP;
    cp_weights CW;
    ff_work B = {};
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    hipStream_t stream = nullptr;
    unsigned long long normal_spin_cap = 0;
    unsigned long long probe_spin_cap = 0;
};

HipFrameFusion::HipFrameFusion() = default;
HipFrameFusion::~HipFrameFusion() { release(); }

bool HipFrameFusion::init(const tl_params & tp, const tl_weights & tw,
                         const cp_params & cpp, const cp_weights & cw,
                         uint8_t * d_seen, std::string * err) {
    release();
    if (!d_seen) { if (err) *err = "frame fusion: null seen set"; return false; }
    if (tp.n_layer > TL_MAX_LAYERS) { if (err) *err = "too many talker layers"; return false; }
    if (cpp.n_layer > CP_MAX_LAYERS) { if (err) *err = "too many cp layers"; return false; }
    if (tp.ff > FF_MAX_VEC || tp.n_head * tp.head_dim > FF_MAX_VEC || tp.head_dim > 128) {
        if (err) *err = "talker shape exceeds kernel limits"; return false;
    }
    if (cpp.ff > FF_MAX_VEC || cpp.n_head * cpp.head_dim > FF_MAX_VEC || cpp.head_dim > 128) {
        if (err) *err = "cp shape exceeds kernel limits"; return false;
    }
    if (tp.vocab > FF_MAX_VEC) { if (err) *err = "talker vocab exceeds LDS vector"; return false; }
    if (cpp.vocab > FF_MAX_VEC) { if (err) *err = "cp vocab exceeds LDS vector"; return false; }
    p_ = new impl();
    p_->TP = tp; p_->TW = tw; p_->CP = cpp; p_->CW = cw;
    ff_work & B = p_->B;
    const size_t kv = (size_t) cpp.n_kv_head * cpp.head_dim;
    CK(hipMalloc(&B.x,          tp.hidden * sizeof(float)));
    CK(hipMalloc(&B.q,          tp.n_head * tp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.k,          (size_t) tp.n_kv_head * tp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.v,          (size_t) tp.n_kv_head * tp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.attn,       (size_t) tp.n_head * tp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.h,          tp.ff * sizeof(float)));
    CK(hipMalloc(&B.hidden_out, tp.hidden * sizeof(float)));
    CK(hipMalloc(&B.logits,     tp.vocab * sizeof(float)));
    B.seen = d_seen;
    CK(hipMalloc(&B.cx,         cpp.hidden * sizeof(float)));
    CK(hipMalloc(&B.cq,         (size_t) cpp.n_head * cpp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.ck,         kv * sizeof(float)));
    CK(hipMalloc(&B.cv,         kv * sizeof(float)));
    CK(hipMalloc(&B.ckcache,    (size_t) cpp.n_layer * CP_MAX_POS * kv * sizeof(__half)));
    CK(hipMalloc(&B.csvcache,   (size_t) cpp.n_layer * CP_MAX_POS * kv * sizeof(__half)));
    CK(hipMalloc(&B.cattn,      (size_t) cpp.n_head * cpp.head_dim * sizeof(float)));
    CK(hipMalloc(&B.ch,         cpp.ff * sizeof(float)));
    CK(hipMalloc(&B.clogits,    (size_t) CP_N_HEADS_OUT * cpp.vocab * sizeof(float)));
    CK(hipMalloc(&B.codes,      CP_MAX_POS * sizeof(int32_t)));
    CK(hipMalloc((void **) &B.trailing, cpp.hidden * sizeof(float)));
    CK(hipMalloc(&B.step_out,   cpp.hidden * sizeof(float)));
    d_step_embd_ = B.step_out;
    CK(hipStreamCreateWithFlags(&p_->stream, hipStreamNonBlocking));
    CK(hipEventCreate(&p_->ev0));
    CK(hipEventCreate(&p_->ev1));

    // Grid: as many co-resident blocks as the cooperative launch allows,
    // up to 4/CU. RESIDENCY IS THE RISK: the fused kernel carries the
    // union of the talker's and cp's register/LDS pressure, so measure
    // occupancy for k_frame_fused itself (not the standalone kernels)
    // before sizing the grid.
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    if (!prop.cooperativeLaunch) { if (err) *err = "device does not support cooperative launch"; return false; }
    int per_cu = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_frame_fused, BLOCK, 0));
    occ_per_cu_ = per_cu;
    if (per_cu < 1) { if (err) *err = "k_frame_fused cannot be resident"; return false; }
    int cap = 4;
    if (const char * e = getenv("QWEN3_TTS_HIP_BLOCKS_PER_CU")) cap = atoi(e);
    grid_blocks_ = prop.multiProcessorCount * (per_cu < cap ? per_cu : cap);

    CK(hipMalloc((void **) &B.bar, 3 * sizeof(unsigned)));
    CK(hipMemset((void *) B.bar, 0, 3 * sizeof(unsigned)));
    // ~0.5 s at a few ns per spin; legitimate stragglers are <1 ms.
    p_->normal_spin_cap = 50000000ull;
    if (const char * e = getenv("QWEN3_TTS_HIP_BARRIER_SPINS")) p_->normal_spin_cap = strtoull(e, nullptr, 10);
    p_->probe_spin_cap = p_->normal_spin_cap / 10;
    if (p_->probe_spin_cap < 1) p_->probe_spin_cap = 1;
    p_->B.spin_cap = p_->normal_spin_cap;
    return true;
}

void HipFrameFusion::set_probe_mode(bool tight) {
    if (!p_) return;
    p_->B.spin_cap = tight ? p_->probe_spin_cap : p_->normal_spin_cap;
}

void HipFrameFusion::release() {
    if (!p_) return;
    void * bufs[] = { p_->B.x, p_->B.q, p_->B.k, p_->B.v, p_->B.attn, p_->B.h,
                      p_->B.hidden_out, p_->B.logits,
                      p_->B.cx, p_->B.cq, p_->B.ck, p_->B.cv,
                      p_->B.ckcache, p_->B.csvcache, p_->B.cattn, p_->B.ch,
                      p_->B.clogits, (void *) p_->B.codes, (void *) p_->B.trailing,
                      (void *) p_->B.step_out, (void *) p_->B.bar };
    for (void * b : bufs) (void) hipFree(b);
    if (p_->ev0) (void) hipEventDestroy(p_->ev0);
    if (p_->ev1) (void) hipEventDestroy(p_->ev1);
    if (p_->stream) (void) hipStreamDestroy(p_->stream);
    d_step_embd_ = nullptr;
    delete p_;
    p_ = nullptr;
}

bool HipFrameFusion::run_frame(const float * d_step_embd, int32_t n_past, const tl_kv & kv,
                              const float * trailing_row,
                              float temperature, int32_t top_k, float repetition_penalty,
                              int32_t eos_id, uint64_t talker_seed, uint64_t cp_seed,
                              std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    if (!d_step_embd) { if (err) *err = "run_frame: null device step_embd"; return false; }
    if (!trailing_row) { if (err) *err = "run_frame: null trailing row"; return false; }
    ff_work & B = p_->B;
    // Test hook: simulate a mid-generation barrier timeout by failing the
    // Nth call (0-based), so the fused-frame -> chained-path fallback in
    // generate() can be exercised deterministically without a GPU hog.
    // QWEN3_TTS_TEST_FRAME_FAIL_AFTER=<n>; -1 (default) never fails.
    static long test_fail_after = -1;
    static long test_calls = 0;
    static bool test_init = false;
    if (!test_init) {
        test_init = true;
        if (const char * e = getenv("QWEN3_TTS_TEST_FRAME_FAIL_AFTER")) test_fail_after = strtol(e, nullptr, 10);
    }
    if (test_fail_after >= 0 && test_calls == test_fail_after) {
        test_calls++;
        if (err) *err = "injected barrier timeout (test hook)";
        return false;
    }
    test_calls++;
    B.step_in = d_step_embd;
    B.n_past = n_past;
    B.temperature = temperature;
    B.top_k = top_k;
    B.repetition_penalty = repetition_penalty;
    B.eos_id = eos_id;
    B.talker_seed = talker_seed;
    B.cp_seed = cp_seed;
    for (int il = 0; il < p_->TP.n_layer; ++il) {
        B.kcache[il] = kv.k[il];
        B.vcache[il] = kv.v[il];
        if (!kv.k[il] || !kv.v[il]) { if (err) *err = "kv cache pointer missing"; return false; }
    }
    // Stage the next frame's trailing text row (host -> device).
    CK(hipMemcpyAsync((void *) B.trailing, trailing_row, p_->CP.hidden * sizeof(float),
                      hipMemcpyHostToDevice, p_->stream));
    CK(hipMemsetAsync((void *) B.bar, 0, 3 * sizeof(unsigned), p_->stream));
    CK(hipEventRecord(p_->ev0, p_->stream));
    void * args[] = { (void *) &p_->TP, (void *) &p_->TW, (void *) &p_->CP, (void *) &p_->CW, (void *) &B };
    CK(hipLaunchCooperativeKernel((const void *) k_frame_fused, dim3(grid_blocks_), dim3(BLOCK), args, 0, p_->stream));
    CK(hipEventRecord(p_->ev1, p_->stream));
    CK(hipStreamSynchronize(p_->stream));
    CK(hipGetLastError());
    unsigned hbar[3] = {0, 0, 0};
    CK(hipMemcpy(hbar, (const void *) B.bar, sizeof(hbar), hipMemcpyDeviceToHost));
    if (hbar[2]) {
        if (err) *err = "grid barrier timed out (aborted): the fused-frame cooperative grid could not run to completion";
        return false;
    }
    float ms = 0;
    CK(hipEventElapsedTime(&ms, p_->ev0, p_->ev1));
    last_run_us_ = ms * 1000.0f;
    return true;
}

bool HipFrameFusion::get_codes(int32_t * out16, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    CK(hipMemcpy(out16, (const void *) p_->B.codes, CP_MAX_POS * sizeof(int32_t), hipMemcpyDeviceToHost));
    return true;
}

} // namespace qwen3_tts
