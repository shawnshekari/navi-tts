#pragma once

// Shared HIP device helpers for the fused talker, fused code predictor and
// fused-frame kernels (docs/talker_fusion_handoff.md).
//
// These were duplicated verbatim between talker_hip.hip and
// code_pred_hip.hip; the fused talker+cp frame kernel must share the
// SAME deadline-aware grid_barrier (one bar buffer, one spin_cap across
// both sub-models' barrier schedules), so they live here once.
//
// Internal-linkage (static) so each TU that includes this header gets its
// own copy — safe to include from multiple .hip translation units.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace qwen3_tts {
namespace hipdev {

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize);
    return v;
}

// Block-wide sum; every thread gets the result. `red` is LDS scratch [32].
static __device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nw   = blockDim.x / warpSize;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    return red[0];
}

// Deadline-aware grid barrier (sense-reversing). Returns false once any
// block's wait exceeded spin_cap; every later barrier then short-circuits
// so the whole grid unwinds instead of deadlocking. Caller must zero the
// 3-word buffer before each cooperative launch.
static __device__ __forceinline__ bool grid_barrier(volatile unsigned * bar, unsigned nblocks,
                                                   unsigned long long spin_cap) {
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

static __device__ __forceinline__ float f32_from_bits(const uint16_t * p, int i) {
    return reinterpret_cast<const float *>(p)[i];
}

// out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i], x from global, out into LDS
static __device__ void block_rmsnorm(const float * x, const uint16_t * w_bits, int n, float eps,
                                     float * out, float * red) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    const float scale = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * f32_from_bits(w_bits, i);
    __syncthreads();
}

// Copy a global f32 vector into LDS.
static __device__ __forceinline__ void block_stage(const float * src, int n, float * dst) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}

static __device__ __forceinline__ float dot8(const uint4 & u, const float * xs) {
    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
    const float2 f0 = __half22float2(h2[0]);
    const float2 f1 = __half22float2(h2[1]);
    const float2 f2 = __half22float2(h2[2]);
    const float2 f3 = __half22float2(h2[3]);
    return f0.x * xs[0] + f0.y * xs[1] + f1.x * xs[2] + f1.y * xs[3]
         + f2.x * xs[4] + f2.y * xs[5] + f3.x * xs[6] + f3.y * xs[7];
}

// R rows of W (row-major f16) dotted with xs (f32, LDS) by one wave, reduced
// into acc[0..R). The R rows' loads are issued together so R*(n_cols/256)
// 16-byte loads are in flight per lane; one row at a time was latency-bound.
// Rows past n_rows contribute zero. n_cols % 8 == 0.
template <int R>
static __device__ __forceinline__ void wave_dot_rows(const uint16_t * W, int row0, int n_rows, int n_cols,
                                                     const float * xs, float * acc) {
    const int lane = threadIdx.x % warpSize;
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    #pragma unroll 2
    for (int c = lane * 8; c < n_cols; c += warpSize * 8) {
        uint4 u[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            u[r] = (row0 + r < n_rows)
                 ? *reinterpret_cast<const uint4 *>(W + (size_t) (row0 + r) * n_cols + c)
                 : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += dot8(u[r], xs + c);
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
}

static constexpr int MV_ROWS = 4;   // rows per wave per pass

// y[row] (=, or += when accumulate) W*xs over all rows, partitioned across the grid's waves
template <bool ACCUMULATE>
static __device__ void grid_matvec(const uint16_t * W, int n_rows, int n_cols, const float * xs, float * y) {
    const int lane  = threadIdx.x % warpSize;
    const int wpb   = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize;
    const int nwave = gridDim.x * wpb;
    for (int row0 = gwave * MV_ROWS; row0 < n_rows; row0 += nwave * MV_ROWS) {
        float acc[MV_ROWS];
        wave_dot_rows<MV_ROWS>(W, row0, n_rows, n_cols, xs, acc);
        if (lane < MV_ROWS && row0 + lane < n_rows) {
            // lane r holds acc[r] after the full-wave reduce (all lanes hold every value)
            float d = acc[0];
            #pragma unroll
            for (int r = 1; r < MV_ROWS; ++r) if (lane == r) d = acc[r];
            if (ACCUMULATE) y[row0 + lane] += d; else y[row0 + lane] = d;
        }
    }
}

static __device__ __forceinline__ uint32_t pcg32_next(uint64_t & s) {
    const uint64_t old = s;
    s = old * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t xs = (uint32_t) (((old >> 18u) ^ old) >> 27u);
    const uint32_t rot = (uint32_t) (old >> 59u);
    return (xs >> rot) | (xs << ((0u - rot) & 31u));
}

// Block-wide argmax over n elements of a global f32 array; thread 0 gets the
// index (lowest index wins ties). Scratch: red[0..] for values, red[16..]
// for indices (block 256 threads = 8 warps, fits).
static __device__ int block_argmax(const float * v, int n, float * red) {
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nw   = blockDim.x / warpSize;
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
    int * ri = (int *) (red + 16);
    __syncthreads();
    if (lane == 0) { rv[wid] = bv; ri[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < nw; ++w)
            if (rv[w] > rv[0] || (rv[w] == rv[0] && ri[w] < ri[0])) { rv[0] = rv[w]; ri[0] = ri[w]; }
    }
    __syncthreads();
    return ri[0];
}

} // namespace hipdev
} // namespace qwen3_tts
