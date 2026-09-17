// Fused HIP code predictor — see code_pred_hip.h.
//
// Phase structure per position p (n_past = p), per layer:
//   A  rmsnorm(x)*attn_norm (every block, into LDS) -> q,k,v = W*xn
//   B  per-head rmsnorm(q,k) + NEOX rope, K/V -> cache (f16)
//   C  attention over positions 0..p, one block per q head
//   D  x += Wo * attn
//   E  rmsnorm(x)*ffn_norm -> h = silu(Wg*xn) * (Wu*xn)
//   F  x += Wd * h
// then rmsnorm(x)*output_norm -> logits[p-1] = head[p-1] * xn, and the next
// position's embedding is loaded into x. Every phase writes to global memory
// only what the next phase reads, and a grid-wide barrier separates them.
//
// Matvecs partition output rows across all waves in the grid (one wave per
// row, 8 f16 per lane per iteration); the input vector is staged in LDS.

#include "code_pred_hip.h"
#include "hip/hip_dev.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qwen3_tts::hipdev;

namespace cg = cooperative_groups;

namespace qwen3_tts {

namespace {

constexpr int BLOCK = 256;                 // threads per block
constexpr int MAX_VEC = 3072;              // largest matvec input (ff)
constexpr int MAX_HEAD_DIM = 128;

struct cp_work {
    float * x;          // [hidden]        residual stream
    float * q;          // [n_head*head_dim]
    float * k;          // [n_kv_head*head_dim]
    float * v;          // [n_kv_head*head_dim]
    __half * kcache;    // [n_layer][CP_MAX_POS][n_kv_head*head_dim]
    __half * vcache;
    float * attn;       // [n_head*head_dim]
    float * h;          // [ff]
    float * logits;     // [15][vocab]
    const float * hidden_in;   // [hidden]
    int32_t * codes;    // [CP_MAX_POS]: [0] = cb0 token (host), [1..15] sampled on device
    float temperature;  // sampling config, per frame
    int32_t top_k;
    uint64_t seed;
    // Cooperative barrier state: [0]=count, [1]=sense, [2]=abort. The
    // occupancy API only knows about this process; if other processes hold
    // CUs a cooperative grid can never become fully resident and cg::grid
    // .sync() spins forever with the display blocked. The spin cap turns
    // that into a clean abort instead of a hung workstation.
    volatile unsigned * bar;
    unsigned long long spin_cap;
};

enum cp_phase : int {
    PH_EMBED = 0,
    PH_QKV,
    PH_NORM_ROPE,
    PH_ATTN,
    PH_OPROJ,
    PH_GATE_UP,
    PH_DOWN,
    PH_HEAD,
    PH_SAMPLE,
    PH_COUNT
};

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { if (err) *err = std::string(#x) + ": " + hipGetErrorString(e_); return false; } } while (0)

// warp_sum / block_sum / grid_barrier / block_rmsnorm / block_stage /
// dot8 / wave_dot_rows / MV_ROWS / grid_matvec / pcg32_next /
// block_argmax come from hip/hip_dev.h (shared with the fused talker and
// the fused-frame kernel).

__device__ void phase_embed(const cp_params & P, const cp_weights & W, const cp_work & B, int pos) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;
    const uint16_t * row = nullptr;
    if (pos == 1) {
        row = W.talker_codec_embd + (size_t) B.codes[0] * P.hidden;
    } else if (pos >= 2) {
        row = W.codec_embd[pos - 2] + (size_t) B.codes[pos - 1] * P.hidden;
    }
    for (int i = tid; i < P.hidden; i += nth) {
        B.x[i] = (pos == 0) ? B.hidden_in[i] : __half2float(reinterpret_cast<const __half *>(row)[i]);
    }
}

__device__ void phase_qkv(const cp_params & P, const cp_weights & W, const cp_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim,    P.hidden, xs, B.q);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.k);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.v);
}

// blocks [0, n_head): q heads; [n_head, n_head+n_kv_head): k heads (+cache);
// [.., +n_kv_head): v -> cache. Threads [0, head_dim) active.
__device__ void phase_norm_rope(const cp_params & P, const cp_weights & W, const cp_work & B, int il, int pos, float * xs, float * red) {
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
            const float theta = (float) pos * powf(P.rope_theta, -2.0f * (float) t / (float) hd);
            float s, c;
            sincosf(theta, &s, &c);
            const float x0 = xs[t], x1 = xs[t + hd / 2];
            const float r0 = x0 * c - x1 * s;
            const float r1 = x0 * s + x1 * c;
            vec[t] = r0;
            vec[t + hd / 2] = r1;
            if (!is_q) {
                __half * kc = B.kcache + ((size_t) il * CP_MAX_POS + pos) * kv_stride + head * hd;
                kc[t] = __float2half(r0);
                kc[t + hd / 2] = __float2half(r1);
            }
        }
    } else if (b < P.n_head + 2 * P.n_kv_head) {
        const int head = b - P.n_head - P.n_kv_head;
        if (t < hd) {
            __half * vc = B.vcache + ((size_t) il * CP_MAX_POS + pos) * kv_stride + head * hd;
            vc[t] = __float2half(B.v[head * hd + t]);
        }
    }
}

// one block per q head: scores over positions 0..pos, softmax, weighted V sum
__device__ void phase_attn(const cp_params & P, const cp_work & B, int il, int pos, float * xs, float * red) {
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
    const float * q = B.q + h * hd;

    for (int j = wid; j <= pos; j += nw) {
        const __half * kc = B.kcache + ((size_t) il * CP_MAX_POS + j) * kv_stride + kvh * hd;
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
            const __half * vc = B.vcache + ((size_t) il * CP_MAX_POS + j) * kv_stride + kvh * hd;
            acc += scores[j] * __half2float(vc[i]);
        }
        B.attn[h * hd + i] = acc;
    }
    (void) red;
}

__device__ void phase_oproj(const cp_params & P, const cp_weights & W, const cp_work & B, int il, float * xs) {
    block_stage(B.attn, P.n_head * P.head_dim, xs);
    grid_matvec<true>(W.wo[il], P.hidden, P.n_head * P.head_dim, xs, B.x);
}

__device__ void phase_gate_up(const cp_params & P, const cp_weights & W, const cp_work & B, int il, float * xs, float * red) {
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

__device__ void phase_down(const cp_params & P, const cp_weights & W, const cp_work & B, int il, float * xs) {
    block_stage(B.h, P.ff, xs);
    grid_matvec<true>(W.w_down[il], P.hidden, P.ff, xs, B.x);
}

__device__ void phase_head(const cp_params & P, const cp_weights & W, const cp_work & B, int pos, float * xs, float * red) {
    if (pos < 1) return;
    block_rmsnorm(B.x, W.output_norm, P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.lm_head[pos - 1], P.vocab, P.hidden, xs, B.logits + (size_t) (pos - 1) * P.vocab);
}


// One block (blockIdx.x == 0) samples head h = pos-1 and writes B.codes[pos].
// Mirrors the host sample_or_argmax: greedy if temperature <= 0, else
// temperature scaling, top-k threshold (iterative max removal), softmax,
// multinomial draw. RNG: PCG32 seeded per (seed, pos) so a frame is
// reproducible and positions are independent.
__device__ void phase_sample(const cp_params & P, const cp_work & B, int pos, float * xs, float * red) {
    if (pos < 1 || blockIdx.x != 0) return;
    const int h = pos - 1;
    const int V = P.vocab;
    const float * lg = B.logits + (size_t) h * V;

    if (B.temperature <= 0.0f) {
        const int best = block_argmax(lg, V, red);  // all threads must participate
        if (threadIdx.x == 0) B.codes[pos] = best;
        return;
    }

    // Sampled path. The 50-pass iterative top-k was ~140 us/position
    // (dominated by 150 block barriers); a bitonic sort of the whole
    // vocab in LDS does it in 66 barriers, and because multinomial
    // sampling only cares about the multiset of probabilities, the CDF
    // scan runs directly over the sorted array.
    // Sort in xs (free after the head's barrier) to avoid growing the
    // block's static shared footprint, which would cut co-resident blocks
    // per CU and slow the matvec phases in the fused kernel.
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
            uint64_t st = B.seed * 0x9E3779B97F4A7C15ull + (uint64_t) pos * 0xBF58476D1CE4E5B9ull + 1ull;
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
            uint64_t st = B.seed * 0x9E3779B97F4A7C15ull + (uint64_t) pos * 0xBF58476D1CE4E5B9ull + 1ull;
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

__device__ __forceinline__ void run_phase(int phase, const cp_params & P, const cp_weights & W, const cp_work & B,
                                          int il, int pos, float * xs, float * red) {
    switch (phase) {
        case PH_EMBED:     phase_embed(P, W, B, pos); break;
        case PH_QKV:       phase_qkv(P, W, B, il, xs, red); break;
        case PH_NORM_ROPE: phase_norm_rope(P, W, B, il, pos, xs, red); break;
        case PH_ATTN:      phase_attn(P, B, il, pos, xs, red); break;
        case PH_OPROJ:     phase_oproj(P, W, B, il, xs); break;
        case PH_GATE_UP:   phase_gate_up(P, W, B, il, xs, red); break;
        case PH_DOWN:      phase_down(P, W, B, il, xs); break;
        case PH_HEAD:      phase_head(P, W, B, pos, xs, red); break;
        case PH_SAMPLE:    phase_sample(P, B, pos, xs, red); break;
        default: break;
    }
}

// one launch per phase (debug driver)
__global__ void k_phase(int phase, cp_params P, cp_weights W, cp_work B, int il, int pos) {
    __shared__ float xs[MAX_VEC];
    __shared__ float red[32];
    run_phase(phase, P, W, B, il, pos, xs, red);
}

// the whole frame in one cooperative launch
__global__ void k_fused(cp_params P, cp_weights W, cp_work B, int n_pos) {
    __shared__ float xs[MAX_VEC];
    __shared__ float red[32];
    const unsigned nb = gridDim.x;
#define GBSYNC() do { if (!grid_barrier(B.bar, nb, B.spin_cap)) return; } while (0)
    for (int pos = 0; pos < n_pos; ++pos) {
        phase_embed(P, W, B, pos);
        GBSYNC();
        for (int il = 0; il < P.n_layer; ++il) {
            phase_qkv(P, W, B, il, xs, red);              GBSYNC();
            phase_norm_rope(P, W, B, il, pos, xs, red);   GBSYNC();
            phase_attn(P, B, il, pos, xs, red);           GBSYNC();
            phase_oproj(P, W, B, il, xs);                 GBSYNC();
            phase_gate_up(P, W, B, il, xs, red);          GBSYNC();
            phase_down(P, W, B, il, xs);                  GBSYNC();
        }
        phase_head(P, W, B, pos, xs, red);
        GBSYNC();
        phase_sample(P, B, pos, xs, red);
        GBSYNC();
    }
#undef GBSYNC
}

} // namespace

struct HipCodePredictor::impl {
    cp_params P;
    cp_weights W;
    cp_work B = {};
    float * d_hidden = nullptr;
    int32_t * d_codes = nullptr;
    float * d_step_embd = nullptr;   // device-assembled next-frame step embedding
    float * d_trailing = nullptr;    // staging for this frame's trailing text row
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    hipStream_t stream = nullptr;
    unsigned long long normal_spin_cap = 0;
    unsigned long long probe_spin_cap = 0;
};

// Assemble the next talker step embedding on-device:
//   out[h] = codec_embd[cb0][h] + sum_{cb=0..14} cp_embd[cb][code_cb][h] + trailing[h]
// Summation order matches the host generate() loop (codec row first, then the
// 15 code-predictor rows, then the trailing row) so the F32 result is
// bit-identical to the host path.
__global__ void k_assemble_step_embd(cp_params P, cp_weights W,
                                    const int32_t * codes, const float * trailing,
                                    float * out) {
    const int hidden = P.hidden;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden;
         h += gridDim.x * blockDim.x) {
        float v = __half2float(reinterpret_cast<const __half *>(W.talker_codec_embd)[(size_t)codes[0] * hidden + h]);
        #pragma unroll
        for (int cb = 0; cb < CP_N_HEADS_OUT; ++cb) {
            v += __half2float(reinterpret_cast<const __half *>(W.codec_embd[cb])[(size_t)codes[1 + cb] * hidden + h]);
        }
        v += trailing[h];
        out[h] = v;
    }
}

HipCodePredictor::HipCodePredictor() = default;
HipCodePredictor::~HipCodePredictor() { release(); }

bool HipCodePredictor::init(const cp_params & params, const cp_weights & weights, std::string * err) {
    release();
    if (params.n_layer > CP_MAX_LAYERS) { if (err) *err = "too many layers"; return false; }
    if (params.ff > MAX_VEC || params.n_head * params.head_dim > MAX_VEC || params.head_dim > MAX_HEAD_DIM) {
        if (err) *err = "shape exceeds kernel limits"; return false;
    }
    p_ = new impl();
    p_->P = params;
    p_->W = weights;
    const cp_params & P = params;
    const size_t kv = (size_t) P.n_kv_head * P.head_dim;
    CK(hipMalloc(&p_->B.x,      P.hidden * sizeof(float)));
    CK(hipMalloc(&p_->B.q,      P.n_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.k,      kv * sizeof(float)));
    CK(hipMalloc(&p_->B.v,      kv * sizeof(float)));
    CK(hipMalloc(&p_->B.kcache, P.n_layer * CP_MAX_POS * kv * sizeof(__half)));
    CK(hipMalloc(&p_->B.vcache, P.n_layer * CP_MAX_POS * kv * sizeof(__half)));
    CK(hipMalloc(&p_->B.attn,   P.n_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.h,      P.ff * sizeof(float)));
    CK(hipMalloc(&p_->B.logits, CP_N_HEADS_OUT * P.vocab * sizeof(float)));
    CK(hipMalloc(&p_->d_hidden, P.hidden * sizeof(float)));
    CK(hipMalloc(&p_->d_codes,  CP_MAX_POS * sizeof(int32_t)));
    CK(hipMalloc(&p_->d_step_embd, P.hidden * sizeof(float)));
    CK(hipMalloc(&p_->d_trailing,  P.hidden * sizeof(float)));
    p_->B.hidden_in = p_->d_hidden;
    p_->B.codes = p_->d_codes;
    CK(hipStreamCreateWithFlags(&p_->stream, hipStreamNonBlocking));
    CK(hipEventCreate(&p_->ev0));
    CK(hipEventCreate(&p_->ev1));

    // grid: as many co-resident blocks as the cooperative launch allows, up to 4/CU
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    if (!prop.cooperativeLaunch) { if (err) *err = "device does not support cooperative launch"; return false; }
    int per_cu = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_fused, BLOCK, 0));
    if (per_cu < 1) { if (err) *err = "k_fused cannot be resident"; return false; }
    int cap = 4;
    if (const char * e = getenv("QWEN3_TTS_HIP_BLOCKS_PER_CU")) cap = atoi(e);
    grid_blocks_ = prop.multiProcessorCount * (per_cu < cap ? per_cu : cap);

    CK(hipMalloc((void **) &p_->B.bar, 3 * sizeof(unsigned)));
    CK(hipMemset((void *) p_->B.bar, 0, 3 * sizeof(unsigned)));
    // ~0.5 s at a few ns per spin; legitimate stragglers are <1 ms.
    p_->normal_spin_cap = 50000000ull;
    if (const char * e = getenv("QWEN3_TTS_HIP_BARRIER_SPINS")) p_->normal_spin_cap = strtoull(e, nullptr, 10);
    p_->probe_spin_cap = p_->normal_spin_cap / 10;
    if (p_->probe_spin_cap < 1) p_->probe_spin_cap = 1;
    p_->B.spin_cap = p_->normal_spin_cap;
    return true;
}

void HipCodePredictor::set_probe_mode(bool tight) {
    if (!p_) return;
    p_->B.spin_cap = tight ? p_->probe_spin_cap : p_->normal_spin_cap;
}

void HipCodePredictor::release() {
    if (!p_) return;
    void * bufs[] = { p_->B.x, p_->B.q, p_->B.k, p_->B.v, p_->B.kcache, p_->B.vcache,
                      p_->B.attn, p_->B.h, p_->B.logits, p_->d_hidden, p_->d_codes,
                      p_->d_step_embd, p_->d_trailing, (void *) p_->B.bar };
    for (void * b : bufs) (void) hipFree(b);
    if (p_->ev0) (void) hipEventDestroy(p_->ev0);
    if (p_->ev1) (void) hipEventDestroy(p_->ev1);
    if (p_->stream) (void) hipStreamDestroy(p_->stream);
    delete p_;
    p_ = nullptr;
}

bool HipCodePredictor::run(const float * hidden, int32_t cb0_token, float temperature,
                          int32_t top_k, uint64_t seed, int32_t * codes_out, float * logits,
                          bool fused, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    const cp_params & P = p_->P;
    p_->B.temperature = temperature;
    p_->B.top_k = top_k;
    p_->B.seed = seed;
    p_->B.hidden_in = p_->d_hidden;
    CK(hipMemcpyAsync(p_->d_hidden, hidden, P.hidden * sizeof(float), hipMemcpyHostToDevice, p_->stream));
    CK(hipMemcpyAsync(p_->d_codes, &cb0_token, sizeof(int32_t), hipMemcpyHostToDevice, p_->stream));
    CK(hipEventRecord(p_->ev0, p_->stream));
    const int n_pos = CP_MAX_POS;
    if (fused) {
        CK(hipMemsetAsync((void *) p_->B.bar, 0, 3 * sizeof(unsigned), p_->stream));
        void * args[] = { (void *) &p_->P, (void *) &p_->W, (void *) &p_->B, (void *) &n_pos };
        CK(hipLaunchCooperativeKernel((const void *) k_fused, dim3(grid_blocks_), dim3(BLOCK), args, 0, p_->stream));
    } else {
        for (int pos = 0; pos < n_pos; ++pos) {
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_EMBED, p_->P, p_->W, p_->B, 0, pos);
            for (int il = 0; il < P.n_layer; ++il) {
                for (int ph = PH_QKV; ph <= PH_DOWN; ++ph) {
                    k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(ph, p_->P, p_->W, p_->B, il, pos);
                }
            }
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_HEAD, p_->P, p_->W, p_->B, 0, pos);
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_SAMPLE, p_->P, p_->W, p_->B, 0, pos);
        }
    }
    CK(hipEventRecord(p_->ev1, p_->stream));
    if (logits) {
        CK(hipMemcpyAsync(logits, p_->B.logits, CP_N_HEADS_OUT * P.vocab * sizeof(float), hipMemcpyDeviceToHost, p_->stream));
    }
    CK(hipStreamSynchronize(p_->stream));
    CK(hipGetLastError());
    if (fused) {
        unsigned hbar[3] = {0, 0, 0};
        CK(hipMemcpy(hbar, (const void *) p_->B.bar, sizeof(hbar), hipMemcpyDeviceToHost));
        if (hbar[2]) {
            if (err) *err = "grid barrier timed out (aborted): the cooperative grid could not run to completion; retry with fewer blocks per CU";
            return false;
        }
    }
    float ms = 0;
    CK(hipEventElapsedTime(&ms, p_->ev0, p_->ev1));
    last_run_us_ = ms * 1000.0f;
    if (codes_out) {
        CK(hipMemcpy(codes_out, p_->d_codes + 1, CP_N_HEADS_OUT * sizeof(int32_t), hipMemcpyDeviceToHost));
    }
    return true;
}

bool HipCodePredictor::run_device(const float * d_hidden, int32_t cb0_token, float temperature,
                                 int32_t top_k, uint64_t seed, int32_t * codes_out, float * logits,
                                 bool fused, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    if (!d_hidden) { if (err) *err = "run_device: null device hidden pointer"; return false; }
    const cp_params & P = p_->P;
    p_->B.temperature = temperature;
    p_->B.top_k = top_k;
    p_->B.seed = seed;
    // Read the talker's hidden straight off the device — no host round-trip.
    p_->B.hidden_in = d_hidden;
    CK(hipMemcpyAsync(p_->d_codes, &cb0_token, sizeof(int32_t), hipMemcpyHostToDevice, p_->stream));
    CK(hipEventRecord(p_->ev0, p_->stream));
    const int n_pos = CP_MAX_POS;
    if (fused) {
        CK(hipMemsetAsync((void *) p_->B.bar, 0, 3 * sizeof(unsigned), p_->stream));
        void * args[] = { (void *) &p_->P, (void *) &p_->W, (void *) &p_->B, (void *) &n_pos };
        CK(hipLaunchCooperativeKernel((const void *) k_fused, dim3(grid_blocks_), dim3(BLOCK), args, 0, p_->stream));
    } else {
        for (int pos = 0; pos < n_pos; ++pos) {
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_EMBED, p_->P, p_->W, p_->B, 0, pos);
            for (int il = 0; il < P.n_layer; ++il) {
                for (int ph = PH_QKV; ph <= PH_DOWN; ++ph) {
                    k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(ph, p_->P, p_->W, p_->B, il, pos);
                }
            }
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_HEAD, p_->P, p_->W, p_->B, 0, pos);
            k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_SAMPLE, p_->P, p_->W, p_->B, 0, pos);
        }
    }
    CK(hipEventRecord(p_->ev1, p_->stream));
    if (logits) {
        CK(hipMemcpyAsync(logits, p_->B.logits, CP_N_HEADS_OUT * P.vocab * sizeof(float), hipMemcpyDeviceToHost, p_->stream));
    }
    CK(hipStreamSynchronize(p_->stream));
    CK(hipGetLastError());
    if (fused) {
        unsigned hbar[3] = {0, 0, 0};
        CK(hipMemcpy(hbar, (const void *) p_->B.bar, sizeof(hbar), hipMemcpyDeviceToHost));
        if (hbar[2]) {
            if (err) *err = "grid barrier timed out (aborted): the cooperative grid could not run to completion; retry with fewer blocks per CU";
            return false;
        }
    }
    float ms = 0;
    CK(hipEventElapsedTime(&ms, p_->ev0, p_->ev1));
    last_run_us_ = ms * 1000.0f;
    if (codes_out) {
        CK(hipMemcpy(codes_out, p_->d_codes + 1, CP_N_HEADS_OUT * sizeof(int32_t), hipMemcpyDeviceToHost));
    }
    return true;
}

float * HipCodePredictor::device_step_embd() { return p_ ? p_->d_step_embd : nullptr; }

bool HipCodePredictor::assemble_step_embd(const float * trailing_row, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    if (!trailing_row) { if (err) *err = "assemble: null trailing row"; return false; }
    const cp_params & P = p_->P;
    // Stage this frame's trailing text row, then fold it with the codes
    // (already on device from the last run()) and the embedding tables.
    CK(hipMemcpyAsync(p_->d_trailing, trailing_row, P.hidden * sizeof(float),
                      hipMemcpyHostToDevice, p_->stream));
    k_assemble_step_embd<<<4, 256, 0, p_->stream>>>(p_->P, p_->W, p_->d_codes,
                                                   p_->d_trailing, p_->d_step_embd);
    CK(hipGetLastError());
    CK(hipStreamSynchronize(p_->stream));
    return true;
}

} // namespace qwen3_tts
