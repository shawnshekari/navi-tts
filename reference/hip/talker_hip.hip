// Fused HIP talker — see talker_hip.h.
//
// Phase structure for one decode step (position n_past), per layer:
//   A  rmsnorm(x)*attn_norm (every block, into LDS) -> q,k,v = W*xn
//   B  per-head rmsnorm(q,k) + NEOX rope, K/V -> external cache (f16) at row n_past
//   C  attention over positions 0..n_past, one block per q head
//   D  x += Wo * attn
//   E  rmsnorm(x)*ffn_norm -> h = silu(Wg*xn) * (Wu*xn)
//   F  x += Wd * h
// then rmsnorm(x)*output_norm -> hidden_out, logits = codec_head * xn,
// and on-device sampling of cb0 (suppression + repetition penalty +
// greedy/temperature-top-k). Same phase machinery as the fused code
// predictor; the differences are the external KV cache, the long
// attention window, and the sampling rules from generate().
//
// Matvecs partition output rows across all waves in the grid (one wave
// per row, 8 f16 per lane per iteration); the input vector is staged in
// LDS.

#include "talker_hip.h"
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
// Largest vector staged in LDS: ff (3072) and the padded vocab for the
// bitonic top-k (codec vocab 3072 -> next pow2 4096).
constexpr int MAX_VEC = 4096;
constexpr int MAX_HEAD_DIM = 128;

struct tl_work {
    float * x;          // [hidden]        residual stream
    float * q;          // [n_head*head_dim]
    float * k;          // [n_kv_head*head_dim]
    float * v;          // [n_kv_head*head_dim]
    float * attn;       // [n_head*head_dim]
    float * h;          // [ff]
    float * hidden_out; // [hidden]
    float * logits;     // [vocab]
    int32_t * cb0;      // [1] sampled codebook-0 token
    uint8_t * seen;     // [vocab] repetition-penalty set, persistent per request
    const uint16_t * kcache[TL_MAX_LAYERS];  // external, row stride n_kv_head*head_dim
    const uint16_t * vcache[TL_MAX_LAYERS];
    int32_t n_past;
    float temperature;
    int32_t top_k;
    float repetition_penalty;   // <= 0 disables
    int32_t eos_id;
    uint64_t seed;
    // Cooperative barrier state: [0]=count, [1]=sense, [2]=abort. Same
    // deadline-aware barrier as the code predictor: the occupancy API only
    // knows about this process, so a spin cap turns an unreschedulable
    // grid into a clean abort instead of a hung workstation.
    volatile unsigned * bar;
    unsigned long long spin_cap;
};

enum tl_phase : int {
    PH_QKV = 0,
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
// block_argmax come from hip/hip_dev.h (shared with the fused code
// predictor and the fused-frame kernel).

__device__ void phase_qkv(const tl_params & P, const tl_weights & W, const tl_work & B, int il, float * xs, float * red) {
    block_rmsnorm(B.x, W.attn_norm[il], P.hidden, P.eps, xs, red);
    grid_matvec<false>(W.wq[il], P.n_head * P.head_dim,    P.hidden, xs, B.q);
    grid_matvec<false>(W.wk[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.k);
    grid_matvec<false>(W.wv[il], P.n_kv_head * P.head_dim, P.hidden, xs, B.v);
}

// blocks [0, n_head): q heads; [n_head, n_head+n_kv_head): k heads (+cache);
// [.., +n_kv_head): v -> cache. Threads [0, head_dim) active.
__device__ void phase_norm_rope(const tl_params & P, const tl_weights & W, const tl_work & B, int il, float * xs, float * red) {
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
__device__ void phase_attn(const tl_params & P, const tl_work & B, int il, float * xs, float * red) {
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

__device__ void phase_oproj(const tl_params & P, const tl_weights & W, const tl_work & B, int il, float * xs) {
    block_stage(B.attn, P.n_head * P.head_dim, xs);
    grid_matvec<true>(W.wo[il], P.hidden, P.n_head * P.head_dim, xs, B.x);
}

__device__ void phase_gate_up(const tl_params & P, const tl_weights & W, const tl_work & B, int il, float * xs, float * red) {
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

__device__ void phase_down(const tl_params & P, const tl_weights & W, const tl_work & B, int il, float * xs) {
    block_stage(B.h, P.ff, xs);
    grid_matvec<true>(W.w_down[il], P.hidden, P.ff, xs, B.x);
}

__device__ void phase_head(const tl_params & P, const tl_weights & W, const tl_work & B, float * xs, float * red) {
    block_rmsnorm(B.x, W.output_norm, P.hidden, P.eps, xs, red);
    // The normalized hidden is the code predictor's input; stash it before
    // the head matvec reuses nothing (xs stays intact) but the caller may
    // not read back, so always copy to the device buffer.
    for (int i = threadIdx.x; i < P.hidden; i += blockDim.x) B.hidden_out[i] = xs[i];
    __syncthreads();
    grid_matvec<false>(W.codec_head, P.vocab, P.hidden, xs, B.logits);
}

// The logit the sampler sees for token i: suppression of
// [vocab - 1024, vocab) except eos, then the HuggingFace repetition
// penalty on tokens seen in earlier frames. Mirrors generate() exactly.
__device__ __forceinline__ float modified_logit(const tl_params & P, const tl_work & B, int i) {
    float v = B.logits[i];
    if (i >= P.vocab - 1024 && i != B.eos_id) return -INFINITY;
    if (B.repetition_penalty > 0.0f && B.seen[i]) {
        v = (v > 0.0f) ? v / B.repetition_penalty : v * B.repetition_penalty;
    }
    return v;
}

// One block (blockIdx.x == 0) samples cb0 and writes B.cb0[0]. Greedy if
// temperature <= 0, else temperature scaling, top-k threshold (bitonic
// sort), softmax, PCG32 multinomial seeded per call. The chosen token is
// added to the seen set unless it is EOS (the host stops before emitting
// it, matching generate()).
__device__ void phase_sample(const tl_params & P, const tl_work & B, float * xs, float * red) {
    if (blockIdx.x != 0) return;
    const int V = P.vocab;

    if (B.temperature <= 0.0f) {
        // Greedy over the modified logits: copy them into xs, then argmax.
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = modified_logit(P, B, i);
        __syncthreads();
        const int best = block_argmax(xs, V, red);
        if (threadIdx.x == 0) {
            B.cb0[0] = best;
            if (best != B.eos_id) B.seen[best] = 1u;
        }
        return;
    }

    // Sampled path. Bitonic sort of the scaled modified logits in xs,
    // padded to the next power of two (codec vocab 3072 -> 4096) with
    // -INF so the network is well-formed; the threshold and the CDF scan
    // only look at the first V entries.
    const int N = 4096;   // next pow2 >= vocab, == MAX_VEC
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        xs[i] = (i < V) ? modified_logit(P, B, i) / B.temperature : -INFINITY;
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
            const float v = modified_logit(P, B, i) / B.temperature;
            const float e = (v >= threshold) ? expf(v - mx) : 0.0f;
            xs[i] = e;
            ssum += e;
        }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.cb0[0] = chosen;
            if (chosen != B.eos_id) B.seen[chosen] = 1u;
        }
    } else {
        // temperature only, no top-k
        float maxl = -INFINITY;
        for (int i = threadIdx.x; i < V; i += blockDim.x) maxl = fmaxf(maxl, modified_logit(P, B, i) / B.temperature);
        maxl = block_sum(maxl, red);
        float ssum = 0.0f;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const float e = expf(modified_logit(P, B, i) / B.temperature - maxl);
            xs[i] = e;
            ssum += e;
        }
        ssum = block_sum(ssum, red);
        const float inv = 1.0f / ssum;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t st = B.seed * 0x9E3779B97F4A7C15ull + 0xBF58476D1CE4E5B9ull + 1ull;
            (void) pcg32_next(st); (void) pcg32_next(st); (void) pcg32_next(st);
            const float u = (float) (pcg32_next(st) >> 8) * (1.0f / 16777216.0f);  // [0,1)
            float cum = 0.0f;
            int chosen = V - 1;
            for (int i = 0; i < V; ++i) {
                cum += xs[i] * inv;
                if (cum >= u) { chosen = i; break; }
            }
            B.cb0[0] = chosen;
            if (chosen != B.eos_id) B.seen[chosen] = 1u;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void run_phase(int phase, const tl_params & P, const tl_weights & W, const tl_work & B,
                                          int il, float * xs, float * red) {
    switch (phase) {
        case PH_QKV:       phase_qkv(P, W, B, il, xs, red); break;
        case PH_NORM_ROPE: phase_norm_rope(P, W, B, il, xs, red); break;
        case PH_ATTN:      phase_attn(P, B, il, xs, red); break;
        case PH_OPROJ:     phase_oproj(P, W, B, il, xs); break;
        case PH_GATE_UP:   phase_gate_up(P, W, B, il, xs, red); break;
        case PH_DOWN:      phase_down(P, W, B, il, xs); break;
        case PH_HEAD:      phase_head(P, W, B, xs, red); break;
        case PH_SAMPLE:    phase_sample(P, B, xs, red); break;
        default: break;
    }
}

// one launch per phase (debug driver)
__global__ void k_phase(int phase, tl_params P, tl_weights W, tl_work B, int il) {
    __shared__ float xs[MAX_VEC];
    __shared__ float red[32];
    run_phase(phase, P, W, B, il, xs, red);
}

// the whole step in one cooperative launch
__global__ void k_fused(tl_params P, tl_weights W, tl_work B) {
    __shared__ float xs[MAX_VEC];
    __shared__ float red[32];
    const unsigned nb = gridDim.x;
#define GBSYNC() do { if (!grid_barrier(B.bar, nb, B.spin_cap)) return; } while (0)
    for (int il = 0; il < P.n_layer; ++il) {
        phase_qkv(P, W, B, il, xs, red);       GBSYNC();
        phase_norm_rope(P, W, B, il, xs, red); GBSYNC();
        phase_attn(P, B, il, xs, red);         GBSYNC();
        phase_oproj(P, W, B, il, xs);          GBSYNC();
        phase_gate_up(P, W, B, il, xs, red);   GBSYNC();
        phase_down(P, W, B, il, xs);           GBSYNC();
    }
    phase_head(P, W, B, xs, red);
    GBSYNC();
    phase_sample(P, B, xs, red);
    GBSYNC();
#undef GBSYNC
}

} // namespace

struct HipTalker::impl {
    tl_params P;
    tl_weights W;
    tl_work B = {};
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    hipStream_t stream = nullptr;
    unsigned long long normal_spin_cap = 0;
    unsigned long long probe_spin_cap = 0;
};

HipTalker::HipTalker() = default;
HipTalker::~HipTalker() { release(); }

bool HipTalker::init(const tl_params & params, const tl_weights & weights, std::string * err) {
    release();
    if (params.n_layer > TL_MAX_LAYERS) { if (err) *err = "too many layers"; return false; }
    if (params.ff > MAX_VEC || params.n_head * params.head_dim > MAX_VEC || params.head_dim > MAX_HEAD_DIM) {
        if (err) *err = "shape exceeds kernel limits"; return false;
    }
    if (params.vocab > TL_MAX_CTX) { if (err) *err = "vocab exceeds attention window"; return false; }
    p_ = new impl();
    p_->P = params;
    p_->W = weights;
    const tl_params & P = params;
    CK(hipMalloc(&p_->B.x,      P.hidden * sizeof(float)));
    CK(hipMalloc(&p_->B.q,      P.n_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.k,      P.n_kv_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.v,      P.n_kv_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.attn,   P.n_head * P.head_dim * sizeof(float)));
    CK(hipMalloc(&p_->B.h,      P.ff * sizeof(float)));
    CK(hipMalloc(&p_->B.hidden_out, P.hidden * sizeof(float)));
    CK(hipMalloc(&p_->B.logits, P.vocab * sizeof(float)));
    CK(hipMalloc(&p_->B.cb0,    sizeof(int32_t)));
    CK(hipMalloc((void **) &p_->B.seen, P.vocab * sizeof(uint8_t)));
    CK(hipMemset((void *) p_->B.seen, 0, P.vocab * sizeof(uint8_t)));
    d_hidden_ = p_->B.hidden_out;
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
    // Probe cap: 10x tighter so a self-heal probe on a still-busy GPU
    // gives up in ~50 ms rather than stalling the full ~0.5 s deadline.
    p_->probe_spin_cap = p_->normal_spin_cap / 10;
    if (p_->probe_spin_cap < 1) p_->probe_spin_cap = 1;
    p_->B.spin_cap = p_->normal_spin_cap;
    return true;
}

void HipTalker::set_probe_mode(bool tight) {
    if (!p_) return;
    p_->B.spin_cap = tight ? p_->probe_spin_cap : p_->normal_spin_cap;
}

void HipTalker::release() {
    if (!p_) return;
    void * bufs[] = { p_->B.x, p_->B.q, p_->B.k, p_->B.v, p_->B.attn, p_->B.h,
                      p_->B.hidden_out, p_->B.logits, (void *) p_->B.cb0, (void *) p_->B.seen,
                      (void *) p_->B.bar };
    for (void * b : bufs) (void) hipFree(b);
    if (p_->ev0) (void) hipEventDestroy(p_->ev0);
    if (p_->ev1) (void) hipEventDestroy(p_->ev1);
    if (p_->stream) (void) hipStreamDestroy(p_->stream);
    d_hidden_ = nullptr;
    delete p_;
    p_ = nullptr;
}

bool HipTalker::run(const float * step_embd, int32_t n_past, const tl_kv & kv,
                   float temperature, int32_t top_k, float repetition_penalty,
                   int32_t eos_id, uint64_t seed,
                   int32_t * cb0_out, float * hidden_out, float * logits_out,
                   bool fused, std::string * err) {
    return run_impl(step_embd, /*step_is_device=*/false, n_past, kv, temperature, top_k,
                    repetition_penalty, eos_id, seed, cb0_out, hidden_out, logits_out, fused, err);
}

bool HipTalker::run_device(const float * d_step_embd, int32_t n_past, const tl_kv & kv,
                          float temperature, int32_t top_k, float repetition_penalty,
                          int32_t eos_id, uint64_t seed,
                          int32_t * cb0_out, float * hidden_out, float * logits_out,
                          bool fused, std::string * err) {
    return run_impl(d_step_embd, /*step_is_device=*/true, n_past, kv, temperature, top_k,
                    repetition_penalty, eos_id, seed, cb0_out, hidden_out, logits_out, fused, err);
}

bool HipTalker::run_impl(const float * step_embd, bool step_is_device, int32_t n_past, const tl_kv & kv,
                        float temperature, int32_t top_k, float repetition_penalty,
                        int32_t eos_id, uint64_t seed,
                        int32_t * cb0_out, float * hidden_out, float * logits_out,
                        bool fused, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    if (n_past < 0 || n_past >= TL_MAX_CTX) { if (err) *err = "n_past out of range"; return false; }
    if (step_is_device && !step_embd) { if (err) *err = "run_device: null device step_embd"; return false; }
    const tl_params & P = p_->P;
    p_->B.n_past = n_past;
    p_->B.temperature = temperature;
    p_->B.top_k = top_k;
    p_->B.repetition_penalty = repetition_penalty;
    p_->B.eos_id = eos_id;
    p_->B.seed = seed;
    for (int il = 0; il < P.n_layer; ++il) {
        p_->B.kcache[il] = kv.k[il];
        p_->B.vcache[il] = kv.v[il];
        if (!kv.k[il] || !kv.v[il]) { if (err) *err = "kv cache pointer missing"; return false; }
    }
    // x starts as the step embedding; the residual stream is updated in place.
    // Device-resident chaining uses a D2D copy instead of crossing PCIe.
    CK(hipMemcpyAsync(p_->B.x, step_embd, P.hidden * sizeof(float),
                      step_is_device ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice,
                      p_->stream));
    CK(hipEventRecord(p_->ev0, p_->stream));
    if (fused) {
        CK(hipMemsetAsync((void *) p_->B.bar, 0, 3 * sizeof(unsigned), p_->stream));
        void * args[] = { (void *) &p_->P, (void *) &p_->W, (void *) &p_->B };
        CK(hipLaunchCooperativeKernel((const void *) k_fused, dim3(grid_blocks_), dim3(BLOCK), args, 0, p_->stream));
    } else {
        for (int il = 0; il < P.n_layer; ++il) {
            for (int ph = PH_QKV; ph <= PH_DOWN; ++ph) {
                k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(ph, p_->P, p_->W, p_->B, il);
            }
        }
        k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_HEAD, p_->P, p_->W, p_->B, 0);
        k_phase<<<grid_blocks_, BLOCK, 0, p_->stream>>>(PH_SAMPLE, p_->P, p_->W, p_->B, 0);
    }
    CK(hipEventRecord(p_->ev1, p_->stream));
    if (logits_out) {
        CK(hipMemcpyAsync(logits_out, p_->B.logits, P.vocab * sizeof(float), hipMemcpyDeviceToHost, p_->stream));
    }
    if (hidden_out) {
        CK(hipMemcpyAsync(hidden_out, p_->B.hidden_out, P.hidden * sizeof(float), hipMemcpyDeviceToHost, p_->stream));
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
    if (cb0_out) {
        CK(hipMemcpy(cb0_out, p_->B.cb0, sizeof(int32_t), hipMemcpyDeviceToHost));
    }
    return true;
}

bool HipTalker::fetch_hidden(float * host_out, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    CK(hipMemcpy(host_out, p_->B.hidden_out, p_->P.hidden * sizeof(float), hipMemcpyDeviceToHost));
    return true;
}

bool HipTalker::mark_seen(int32_t token, std::string * err) {
    if (!p_) { if (err) *err = "not initialised"; return false; }
    if (token < 0 || token >= p_->P.vocab) return true;
    const uint8_t one = 1u;
    CK(hipMemcpyAsync((void *) (p_->B.seen + token), &one, sizeof(uint8_t), hipMemcpyHostToDevice, p_->stream));
    CK(hipStreamSynchronize(p_->stream));
    return true;
}

uint8_t * HipTalker::device_seen() { return p_ ? (uint8_t *) p_->B.seen : nullptr; }

void HipTalker::reset_repetition() {
    if (!p_) return;
    (void) hipMemset((void *) p_->B.seen, 0, p_->P.vocab * sizeof(uint8_t));
}

} // namespace qwen3_tts
