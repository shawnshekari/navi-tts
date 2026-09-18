// The talker prefill's GEMMs: T = n_prefill rows (10 for the fixed prompt
// layout) against 1024-3072 wide f16 weights, 7 per layer x 28 layers. They
// ran through k_conv<K=1>, a 32-row tile: at T = 10 most of each block is
// padding and the 881 MB of weights streamed at ~60-130 GB/s.
//
// The constraint: each output's sum over ci must stay sequential (bit-exact
// with k_conv), so one lane owns a whole (t, co) dot product - no split-K,
// no cross-lane reduction. Three mappings, all memcmp'd against k_conv:
//   skinny   lane = t, wave = R channels, f16 tile in LDS, unpack per lane
//   skinny32 the same with the tile converted to f32 at staging
//   colwise  lane = channel, t in registers, per-lane row streaming
//   quad     lane quad per channel row (64 B coalesced), DPP exchange, t split
// 2026-09-18, XTX, per prefill (GEMMs only): k_conv 11.6-12.0 ms; skinny32
// R2 NW4 C512 5.2-5.6; quad 4.7-5.1; colwise 4.8-5.6. Every one sits at
// 150-240 GB/s. Probes: a pure stream at these grids does 450-730 GB/s, so
// the gap is inside the kernels - with T = 10 a GEMM has 1-2 waves per SIMD
// of dependent FMA chains waiting on LDS reads (ISA: 273 s_waitcnt per 2952
// instructions), and every attempt to hide that (deeper prefetch, register
// double-buffering, waves-per-eu hints) either spilled or changed nothing.
// Fusing q/k/v and gate/up would be worth ~15 %. skinny32 R2 NW4 C512 is
// what kernels.h k_gemm_skinny is: one instantiation per epilogue, any
// T <= 32, no intrinsics, for 10 % less than quad.
// Build (from bench/micro):
//   clang++ -x hip --offload-arch=gfx1100 --rocm-path=$ROCM -O3 -I../.. gemm_skinny.hip -o gemm_skinny -L$ROCM/lib -lamdhip64
// Run it briefly, the card is shared (CLAUDE.md).
#include "model/qwen3tts/kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>

using namespace navi::qwen3tts::kern;

// ---- skinny GEMM ----------------------------------------------------------
// Block: NW waves, each owning R consecutive output channels; lane t owns row
// t of the activations. Weights for the block's NW*R channels are staged in
// LDS CCH input channels at a time with one 16-byte load per thread (fully
// coalesced), then read back wave-uniform. PF = 1 prefetches the next chunk's
// loads into registers before computing the current one.
template <int R, int NW, int CCH, int PF, int PRO, int EPI>
__global__ void __launch_bounds__(NW * 32) k_skinny(ConvArgs a) {
    constexpr int NT = NW * 32, COB = NW * R;
    constexpr int LOADS = COB * CCH / 8;            // uint4 loads per chunk
    static_assert(LOADS % NT == 0, "chunk must be a whole number of loads per thread");
    constexpr int LPT = LOADS / NT;
    __shared__ __align__(16) unsigned short ws[COB][CCH];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int co0 = blockIdx.x * COB;
    const int t = lane;
    const bool live = t < a.T_in;
    const unsigned short * W = static_cast<const unsigned short *>(a.W);
    const float * xrow = a.x + static_cast<std::size_t>(live ? t : 0) * a.x_stride;
    float acc[R] = {};
    uint4 pre[LPT];
    auto load_chunk = [&](int ci0, uint4 * dst) {
#pragma unroll
        for (int l = 0; l < LPT; ++l) {
            const int i = tid + l * NT, row = i / (CCH / 8), c8 = (i % (CCH / 8)) * 8, co = co0 + row;
            dst[l] = (co < a.Cout) ? *reinterpret_cast<const uint4 *>(W + static_cast<std::size_t>(co) * a.Cin + ci0 + c8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto store_chunk = [&](const uint4 * src) {
#pragma unroll
        for (int l = 0; l < LPT; ++l) {
            const int i = tid + l * NT, row = i / (CCH / 8), c8 = (i % (CCH / 8)) * 8;
            *reinterpret_cast<uint4 *>(&ws[row][c8]) = src[l];
        }
    };
    if (PF) load_chunk(0, pre);
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CCH) {
        uint4 cur[LPT];
        if (PF) {
#pragma unroll
            for (int l = 0; l < LPT; ++l) cur[l] = pre[l];
            if (ci0 + CCH < a.Cin) load_chunk(ci0 + CCH, pre);
        } else {
            load_chunk(ci0, cur);
        }
        __syncthreads();   // everyone is done reading the previous chunk
        store_chunk(cur);
        __syncthreads();
        if (live) {
#pragma unroll 2
            for (int c = 0; c < CCH; c += 8) {
                float x[8];
                if (PRO == PRO_SILU_GATE) {
                    const float4 g0 = *reinterpret_cast<const float4 *>(xrow + ci0 + c), g1 = *reinterpret_cast<const float4 *>(xrow + ci0 + c + 4);
                    const float4 u0 = *reinterpret_cast<const float4 *>(xrow + a.Cin + ci0 + c), u1 = *reinterpret_cast<const float4 *>(xrow + a.Cin + ci0 + c + 4);
                    const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w}, u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
                    for (int k = 0; k < 8; ++k) x[k] = g[k] / (1.f + __expf(-g[k])) * u[k];
                } else {
                    const float4 x0 = *reinterpret_cast<const float4 *>(xrow + ci0 + c), x1 = *reinterpret_cast<const float4 *>(xrow + ci0 + c + 4);
                    x[0] = x0.x; x[1] = x0.y; x[2] = x0.z; x[3] = x0.w; x[4] = x1.x; x[5] = x1.y; x[6] = x1.z; x[7] = x1.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint4 w = *reinterpret_cast<const uint4 *>(&ws[wave * R + r][c]);
                    const unsigned wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const float wv = navi::dev::half_bits_to_float(static_cast<unsigned short>((k & 1) ? (wd[k / 2] >> 16) : (wd[k / 2] & 0xffffu)));
                        acc[r] += wv * x[k];
                    }
                }
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int co = co0 + wave * R + r;
        if (co >= a.Cout) continue;
        float v = acc[r];
        if (a.bias) v += a.bias[co];
        if (EPI == EPI_SILU) v = v / (1.f + __expf(-v));
        else if (EPI == EPI_RESIDUAL) v += a.residual[static_cast<std::size_t>(t) * a.Cout + co];
        a.y[static_cast<std::size_t>(t) * a.y_stride + co] = v;
    }
}


// ---- skinny GEMM, weights converted to f32 at staging ----------------------
// Same mapping; each thread converts the 8 halves it stages, so the inner
// loop is one 16-byte LDS read per 4 weights and the FMAs, no per-lane
// unpacking. PRO is not applied here: a gated input is prepared once by
// k_gate (same expression per element, so the same values).
template <int R, int NW, int CCH, int EPI, int MODE = 0>
__global__ void __launch_bounds__(NW * 32) k_skinny32(ConvArgs a) {
    constexpr int NT = NW * 32, COB = NW * R;
    constexpr int LOADS = COB * CCH / 8;
    static_assert(LOADS % NT == 0, "chunk must be a whole number of loads per thread");
    constexpr int LPT = LOADS / NT;
    __shared__ __align__(16) float ws[COB][CCH];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int co0 = blockIdx.x * COB;
    const int t = lane;
    const bool live = t < a.T_in;
    const unsigned short * W = static_cast<const unsigned short *>(a.W);
    const float * xrow = a.x + static_cast<std::size_t>(live ? t : 0) * a.x_stride;
    float acc[R] = {};
    uint4 pre[LPT];
    auto load_chunk = [&](int ci0, uint4 * dst) {
#pragma unroll
        for (int l = 0; l < LPT; ++l) {
            const int i = tid + l * NT, row = i / (CCH / 8), c8 = (i % (CCH / 8)) * 8, co = co0 + row;
            dst[l] = (co < a.Cout) ? *reinterpret_cast<const uint4 *>(W + static_cast<std::size_t>(co) * a.Cin + ci0 + c8) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    load_chunk(0, pre);
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CCH) {
        uint4 cur[LPT];
#pragma unroll
        for (int l = 0; l < LPT; ++l) cur[l] = pre[l];
        if (ci0 + CCH < a.Cin) load_chunk(ci0 + CCH, pre);
        __syncthreads();
#pragma unroll
        for (int l = 0; l < LPT; ++l) {
            const int i = tid + l * NT, row = i / (CCH / 8), c8 = (i % (CCH / 8)) * 8;
            const unsigned wd[4] = {cur[l].x, cur[l].y, cur[l].z, cur[l].w};
            float f[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = navi::dev::half_bits_to_float(static_cast<unsigned short>((k & 1) ? (wd[k / 2] >> 16) : (wd[k / 2] & 0xffffu)));
            *reinterpret_cast<float4 *>(&ws[row][c8]) = make_float4(f[0], f[1], f[2], f[3]);
            *reinterpret_cast<float4 *>(&ws[row][c8 + 4]) = make_float4(f[4], f[5], f[6], f[7]);
        }
        __syncthreads();
        if (live) {
#pragma unroll 2
            for (int c = 0; c < CCH; c += 8) {
                const float4 x0 = *reinterpret_cast<const float4 *>(xrow + ci0 + c), x1 = *reinterpret_cast<const float4 *>(xrow + ci0 + c + 4);
                const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float4 w0 = *reinterpret_cast<const float4 *>(&ws[wave * R + r][c]), w1 = *reinterpret_cast<const float4 *>(&ws[wave * R + r][c + 4]);
                    const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                    if (MODE == 1) { acc[r] += w[0] + x[0]; continue; }
#pragma unroll
                    for (int k = 0; k < 8; ++k) acc[r] += w[k] * x[k];
                }
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int co = co0 + wave * R + r;
        if (co >= a.Cout) continue;
        float v = acc[r];
        if (a.bias) v += a.bias[co];
        if (EPI == EPI_SILU) v = v / (1.f + __expf(-v));
        else if (EPI == EPI_RESIDUAL) v += a.residual[static_cast<std::size_t>(t) * a.Cout + co];
        a.y[static_cast<std::size_t>(t) * a.y_stride + co] = v;
    }
}


// ---- column-wise: lane = output channel, t in registers --------------------
// A wave streams 32/SPLIT weight rows, lane l owning row l % (32/SPLIT); the
// SPLIT lane groups load the same addresses (merged by the coalescer) and
// split the T activation rows between them, TPL each. Every lane converts a
// weight once and applies it to its TPL rows as independent chains, so the
// per-weight cost is ~TPL FMAs + 1 cvt spread over the wave, and the sum over
// ci is still sequential per output. Activations are staged in LDS CCH
// channels at a time and read as broadcasts. D 16-byte weight loads per lane
// in flight, software-pipelined.
template <int SPLIT, int TPL, int NW, int CCH, int D, int EPI>
__global__ void __launch_bounds__(NW * 32) k_colwise(ConvArgs a) {
    constexpr int NT = NW * 32, RPW = 32 / SPLIT, ROWS = NW * RPW, XP = CCH + 4;
    constexpr int TMAX = SPLIT * TPL;
    static_assert(CCH % (8 * D) == 0, "chunk must hold whole pipeline steps");
    __shared__ __align__(16) float xs[TMAX][XP];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int grp = lane / RPW, rl = lane % RPW;
    const int co = blockIdx.x * ROWS + wave * RPW + rl;
    const int t0 = grp * TPL;
    const unsigned short * W = static_cast<const unsigned short *>(a.W);
    const unsigned short * wrow = W + static_cast<std::size_t>(co < a.Cout ? co : 0) * a.Cin;
    float acc[TPL] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CCH) {
        __syncthreads();
        for (int i = tid; i < TMAX * (CCH / 4); i += NT) {
            const int t = i / (CCH / 4), c4 = (i % (CCH / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < a.T_in) v = *reinterpret_cast<const float4 *>(a.x + static_cast<std::size_t>(t) * a.x_stride + ci0 + c4);
            *reinterpret_cast<float4 *>(&xs[t][c4]) = v;
        }
        __syncthreads();
        uint4 pre[D], cur[D];
#pragma unroll
        for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + d * 8);
        for (int c = 0; c < CCH; c += 8 * D) {
#pragma unroll
            for (int d = 0; d < D; ++d) cur[d] = pre[d];
            if (c + 8 * D < CCH) {
#pragma unroll
                for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + c + 8 * D + d * 8);
            }
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const unsigned wd[4] = {cur[d].x, cur[d].y, cur[d].z, cur[d].w};
                float w[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) w[k] = navi::dev::half_bits_to_float(static_cast<unsigned short>((k & 1) ? (wd[k / 2] >> 16) : (wd[k / 2] & 0xffffu)));
#pragma unroll
                for (int tt = 0; tt < TPL; ++tt) {
                    const float4 x0 = *reinterpret_cast<const float4 *>(&xs[t0 + tt][c + d * 8]);
                    const float4 x1 = *reinterpret_cast<const float4 *>(&xs[t0 + tt][c + d * 8 + 4]);
                    acc[tt] += w[0] * x0.x; acc[tt] += w[1] * x0.y; acc[tt] += w[2] * x0.z; acc[tt] += w[3] * x0.w;
                    acc[tt] += w[4] * x1.x; acc[tt] += w[5] * x1.y; acc[tt] += w[6] * x1.z; acc[tt] += w[7] * x1.w;
                }
            }
        }
    }
    if (co >= a.Cout) return;
#pragma unroll
    for (int tt = 0; tt < TPL; ++tt) {
        const int t = t0 + tt;
        if (t >= a.T_in) continue;
        float v = acc[tt];
        if (a.bias) v += a.bias[co];
        if (EPI == EPI_SILU) v = v / (1.f + __expf(-v));
        else if (EPI == EPI_RESIDUAL) v += a.residual[static_cast<std::size_t>(t) * a.Cout + co];
        a.y[static_cast<std::size_t>(t) * a.y_stride + co] = v;
    }
}


// ---- quad: 8 rows per wave, a quad of lanes per row -----------------------
// Lane (row = lane/4, q = lane%4) loads piece q of each 64-byte row segment,
// so a wave-load touches 8 rows x 64 contiguous bytes (the per-lane-row form
// thrashed L0 at ~250 GB/s; this streams at ~450-530). The quad then swaps
// pieces with DPP quad-permutes so every lane holds the segment's 32 weights
// in ci order, converts them once, and applies them to its own TPL rows of
// t (t = q + 4i): one lane per (t, co), ci sequential. x staged in LDS per
// CCH chunk, read as broadcasts (4 distinct rows per instruction, padded to
// different banks). D segments per lane in flight.
__device__ __forceinline__ unsigned quad_bcast(unsigned v, int src) {
    // dpp quad_perm(src,src,src,src) = src * 0x55
    switch (src) {
    case 0: return __builtin_amdgcn_mov_dpp(v, 0x00, 0xf, 0xf, true);
    case 1: return __builtin_amdgcn_mov_dpp(v, 0x55, 0xf, 0xf, true);
    case 2: return __builtin_amdgcn_mov_dpp(v, 0xaa, 0xf, 0xf, true);
    default: return __builtin_amdgcn_mov_dpp(v, 0xff, 0xf, 0xf, true);
    }
}
template <int TPL, int NW, int CCH, int D, int EPI, int PIPE = 1, int WPE = 0>
__global__ void __launch_bounds__(NW * 32) __attribute__((amdgpu_waves_per_eu(WPE ? WPE : 1, WPE ? WPE : 16))) k_quad(ConvArgs a) {
    constexpr int NT = NW * 32, ROWS = NW * 8, XP = CCH + 4, TMAX = 4 * TPL;
    static_assert(CCH % (32 * D) == 0, "chunk must hold whole pipeline steps");
    __shared__ __align__(16) float xs[TMAX][XP];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int q = lane % 4, rl = lane / 4;
    const int co = blockIdx.x * ROWS + wave * 8 + rl;
    const unsigned short * W = static_cast<const unsigned short *>(a.W);
    const unsigned short * wrow = W + static_cast<std::size_t>(co < a.Cout ? co : 0) * a.Cin;
    float acc[TPL] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CCH) {
        __syncthreads();
        for (int i = tid; i < TMAX * (CCH / 4); i += NT) {
            const int t = i / (CCH / 4), c4 = (i % (CCH / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < a.T_in) v = *reinterpret_cast<const float4 *>(a.x + static_cast<std::size_t>(t) * a.x_stride + ci0 + c4);
            *reinterpret_cast<float4 *>(&xs[t][c4]) = v;
        }
        __syncthreads();
        uint4 pre[D], cur[D];
#pragma unroll
        for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + d * 32 + q * 8);
        for (int c = 0; c < CCH; c += 32 * D) {
#pragma unroll
            for (int d = 0; d < D; ++d) cur[d] = pre[d];
            if (PIPE && c + 32 * D < CCH) {
#pragma unroll
                for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + c + 32 * D + d * 32 + q * 8);
            } else if (PIPE && ci0 + CCH < a.Cin) {
#pragma unroll
                for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + CCH + d * 32 + q * 8);
            }
#pragma unroll
            for (int d = 0; d < D; ++d) {
                // convert own 8 weights, then gather the quad's 32 as floats in ci order
                const unsigned wd[4] = {cur[d].x, cur[d].y, cur[d].z, cur[d].w};
                float own[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) own[k] = navi::dev::half_bits_to_float(static_cast<unsigned short>((k & 1) ? (wd[k / 2] >> 16) : (wd[k / 2] & 0xffffu)));
                float w[32];
#pragma unroll
                for (int p = 0; p < 4; ++p)
#pragma unroll
                    for (int k = 0; k < 8; ++k) w[p * 8 + k] = __uint_as_float(quad_bcast(__float_as_uint(own[k]), p));
#pragma unroll
                for (int tt = 0; tt < TPL; ++tt) {
                    const int t = q + 4 * tt;
#pragma unroll
                    for (int k8 = 0; k8 < 4; ++k8) {
                        const float4 x0 = *reinterpret_cast<const float4 *>(&xs[t][c + d * 32 + k8 * 8]);
                        const float4 x1 = *reinterpret_cast<const float4 *>(&xs[t][c + d * 32 + k8 * 8 + 4]);
                        acc[tt] += w[k8 * 8 + 0] * x0.x; acc[tt] += w[k8 * 8 + 1] * x0.y; acc[tt] += w[k8 * 8 + 2] * x0.z; acc[tt] += w[k8 * 8 + 3] * x0.w;
                        acc[tt] += w[k8 * 8 + 4] * x1.x; acc[tt] += w[k8 * 8 + 5] * x1.y; acc[tt] += w[k8 * 8 + 6] * x1.z; acc[tt] += w[k8 * 8 + 7] * x1.w;
                    }
                }
            }
        }
    }
    if (co >= a.Cout) return;
#pragma unroll
    for (int tt = 0; tt < TPL; ++tt) {
        const int t = q + 4 * tt;
        if (t >= a.T_in) continue;
        float v = acc[tt];
        if (a.bias) v += a.bias[co];
        if (EPI == EPI_SILU) v = v / (1.f + __expf(-v));
        else if (EPI == EPI_RESIDUAL) v += a.residual[static_cast<std::size_t>(t) * a.Cout + co];
        a.y[static_cast<std::size_t>(t) * a.y_stride + co] = v;
    }
}


// ---- quad with explicit activation double-buffering -------------------------
// As k_quad, but the next segment's activation reads (TPL rows x 32 ci) are
// issued into registers before the current segment's FMAs, so the LDS latency
// is covered by the wave itself (the grid gives each SIMD only 1-2 waves).
template <int TPL, int NW, int CCH, int D, int EPI>
__global__ void __launch_bounds__(NW * 32) k_quad2(ConvArgs a) {
    constexpr int NT = NW * 32, ROWS = NW * 8, XP = CCH + 4, TMAX = 4 * TPL;
    static_assert(CCH % (32 * D) == 0, "chunk must hold whole pipeline steps");
    __shared__ __align__(16) float xs[TMAX][XP];
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int q = lane % 4, rl = lane / 4;
    const int co = blockIdx.x * ROWS + wave * 8 + rl;
    const unsigned short * W = static_cast<const unsigned short *>(a.W);
    const unsigned short * wrow = W + static_cast<std::size_t>(co < a.Cout ? co : 0) * a.Cin;
    float acc[TPL] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CCH) {
        __syncthreads();
        for (int i = tid; i < TMAX * (CCH / 4); i += NT) {
            const int t = i / (CCH / 4), c4 = (i % (CCH / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < a.T_in) v = *reinterpret_cast<const float4 *>(a.x + static_cast<std::size_t>(t) * a.x_stride + ci0 + c4);
            *reinterpret_cast<float4 *>(&xs[t][c4]) = v;
        }
        __syncthreads();
        uint4 pre[D], cur[D];
        float4 xn[TPL][8], xc[TPL][8];
        auto load_x = [&](int c, float4 (&dst)[TPL][8]) {
#pragma unroll
            for (int tt = 0; tt < TPL; ++tt)
#pragma unroll
                for (int k4 = 0; k4 < 8; ++k4) dst[tt][k4] = *reinterpret_cast<const float4 *>(&xs[q + 4 * tt][c + k4 * 4]);
        };
#pragma unroll
        for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + d * 32 + q * 8);
        load_x(0, xn);
        for (int c = 0; c < CCH; c += 32 * D) {
#pragma unroll
            for (int d = 0; d < D; ++d) cur[d] = pre[d];
            if (c + 32 * D < CCH) {
#pragma unroll
                for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + c + 32 * D + d * 32 + q * 8);
            } else if (ci0 + CCH < a.Cin) {
#pragma unroll
                for (int d = 0; d < D; ++d) pre[d] = *reinterpret_cast<const uint4 *>(wrow + ci0 + CCH + d * 32 + q * 8);
            }
#pragma unroll
            for (int d = 0; d < D; ++d) {
#pragma unroll
                for (int tt = 0; tt < TPL; ++tt)
#pragma unroll
                    for (int k4 = 0; k4 < 8; ++k4) xc[tt][k4] = xn[tt][k4];
                if (d + 1 < D) load_x(c + (d + 1) * 32, xn);
                else if (c + 32 * D < CCH) load_x(c + 32 * D, xn);
                const unsigned wd[4] = {cur[d].x, cur[d].y, cur[d].z, cur[d].w};
                float own[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) own[k] = navi::dev::half_bits_to_float(static_cast<unsigned short>((k & 1) ? (wd[k / 2] >> 16) : (wd[k / 2] & 0xffffu)));
                float w[32];
#pragma unroll
                for (int p = 0; p < 4; ++p)
#pragma unroll
                    for (int k = 0; k < 8; ++k) w[p * 8 + k] = __uint_as_float(quad_bcast(__float_as_uint(own[k]), p));
#pragma unroll
                for (int tt = 0; tt < TPL; ++tt)
#pragma unroll
                    for (int k4 = 0; k4 < 8; ++k4) {
                        acc[tt] += w[k4 * 4 + 0] * xc[tt][k4].x; acc[tt] += w[k4 * 4 + 1] * xc[tt][k4].y;
                        acc[tt] += w[k4 * 4 + 2] * xc[tt][k4].z; acc[tt] += w[k4 * 4 + 3] * xc[tt][k4].w;
                    }
            }
        }
    }
    if (co >= a.Cout) return;
#pragma unroll
    for (int tt = 0; tt < TPL; ++tt) {
        const int t = q + 4 * tt;
        if (t >= a.T_in) continue;
        float v = acc[tt];
        if (a.bias) v += a.bias[co];
        if (EPI == EPI_SILU) v = v / (1.f + __expf(-v));
        else if (EPI == EPI_RESIDUAL) v += a.residual[static_cast<std::size_t>(t) * a.Cout + co];
        a.y[static_cast<std::size_t>(t) * a.y_stride + co] = v;
    }
}

// silu(g) * u over [T][2C] -> [T][C], the gated GEMM's input, once
static __global__ void k_gate(const float * x, int C, int x_stride, float * y) {
    const int t = blockIdx.x;
    const float * row = x + static_cast<std::size_t>(t) * x_stride;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float g = row[c];
        y[static_cast<std::size_t>(t) * C + c] = g / (1.f + __expf(-g)) * row[C + c];
    }
}

// ---------------------------------------------------------------------------
struct Shape { const char * name; int T, Cin, Cout, pro, epi, per_prefill; };

static unsigned short f2h(float f) { _Float16 h = static_cast<_Float16>(f); unsigned short u; std::memcpy(&u, &h, 2); return u; }

int main() {
    // the talker's prefill at T = 10 (fixed prompt rows), one layer's seven
    // GEMMs x 28, the head, and the text projections at 25 text tokens
    std::vector<Shape> shapes = {
        {"q   1024->2048", 10, 1024, 2048, PRO_NONE, EPI_NONE, 28},
        {"k   1024->1024", 10, 1024, 1024, PRO_NONE, EPI_NONE, 56},
        {"qkv 1024->4096 fused", 10, 1024, 4096, PRO_NONE, EPI_NONE, 0},
        {"o   2048->1024 +res", 10, 2048, 1024, PRO_NONE, EPI_RESIDUAL, 28},
        {"g   1024->3072", 10, 1024, 3072, PRO_NONE, EPI_NONE, 56},
        {"gu  1024->6144 fused", 10, 1024, 6144, PRO_NONE, EPI_NONE, 0},
        {"d   3072->1024 gate", 10, 3072, 1024, PRO_SILU_GATE, EPI_RESIDUAL, 28},
        {"head 1024->3072", 10, 1024, 3072, PRO_NONE, EPI_NONE, 1},
        {"fc1 2048->2048 silu", 25, 2048, 2048, PRO_NONE, EPI_SILU, 1},
        {"fc2 2048->1024", 25, 2048, 1024, PRO_NONE, EPI_NONE, 1},
    };

    struct Variant { const char * name; } variants[] = {
        {"k_conv K=1 (current)"},
        {"skinny R2 NW8 C128 PF"},
        {"skinny32 R2 NW4 C512"},
        {"colwise S4 NW4 C256 D8"},
        {"quad NW4 C512 D8"},
    };
    constexpr int NV = sizeof(variants) / sizeof(variants[0]);
    double total[NV] = {}; bool ok[NV]; for (auto & o : ok) o = true;

    std::size_t maxW = 0, maxX = 0, maxY = 0;
    for (auto & s : shapes) { maxW = std::max(maxW, (std::size_t) s.Cin * s.Cout); maxX = std::max(maxX, (std::size_t) s.T * 2 * s.Cin); maxY = std::max(maxY, (std::size_t) s.T * s.Cout); }
    // rotate through 512 MB of weights so the Infinity Cache does not flatter the stream
    const std::size_t rot = (512u << 20) / 2 / maxW + 1;
    unsigned short * W; float * x, * y, * ref, * res, * bias, * gx;
    hipMalloc(&W, maxW * rot * 2); hipMalloc(&x, maxX * 4); hipMalloc(&y, maxY * 4); hipMalloc(&ref, maxY * 4); hipMalloc(&res, maxY * 4); hipMalloc(&bias, 4096 * 4); hipMalloc(&gx, maxX * 4);
    {
        unsigned s = 777u; auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.f - 0.5f; };
        std::vector<unsigned short> w(maxW * rot); for (auto & v : w) v = f2h(rnd() * 0.05f); hipMemcpy(W, w.data(), w.size() * 2, hipMemcpyHostToDevice);
        std::vector<float> f(maxX); for (auto & v : f) v = rnd(); hipMemcpy(x, f.data(), f.size() * 4, hipMemcpyHostToDevice);
        f.resize(maxY); for (auto & v : f) v = rnd(); hipMemcpy(res, f.data(), f.size() * 4, hipMemcpyHostToDevice);
        f.resize(4096); for (auto & v : f) v = rnd(); hipMemcpy(bias, f.data(), f.size() * 4, hipMemcpyHostToDevice);
    }

    for (const Shape & s : shapes) {
        printf("%s  T=%d\n", s.name, s.T);
        for (int v = 0; v < NV; ++v) {
            ConvArgs a{};
            a.W = W; a.bias = (s.epi == EPI_SILU) ? bias : nullptr; a.x = x; a.T_in = s.T; a.Cin = s.Cin; a.Cout = s.Cout; a.dil = 1;
            a.x_stride = (s.pro == PRO_SILU_GATE) ? 2 * s.Cin : s.Cin; a.phases = 1; a.y = y; a.y_stride = s.Cout; a.residual = res;
            std::function<void(ConvArgs &, int)> launch;
            const dim3 cgrid((s.T + TT - 1) / TT, (s.Cout + CT - 1) / CT, 1);
// TPL from T at compile time for the microbench: T=10 -> S1:10 S2:5 S4:3; T=25 -> S1:25 S2:13 S4:7
#define CW1(SS, TT_, NWW, CCC, DD) \
                if (s.epi == EPI_NONE) k_colwise<SS, TT_, NWW, CCC, DD, EPI_NONE><<<g, NWW * 32>>>(b); \
                else if (s.epi == EPI_RESIDUAL) k_colwise<SS, TT_, NWW, CCC, DD, EPI_RESIDUAL><<<g, NWW * 32>>>(b); \
                else k_colwise<SS, TT_, NWW, CCC, DD, EPI_SILU><<<g, NWW * 32>>>(b);
#define CW(SS, NWW, CCC, DD) \
            launch = [=](ConvArgs & a, int) { \
                ConvArgs b = a; \
                if (s.pro == PRO_SILU_GATE) { k_gate<<<s.T, 256>>>(a.x, s.Cin, a.x_stride, gx); b.x = gx; b.x_stride = s.Cin; } \
                const dim3 g((s.Cout + NWW * (32 / SS) - 1) / (NWW * (32 / SS))); \
                if (s.T == 10) { CW1(SS, (10 + SS - 1) / SS, NWW, CCC, DD) } else { CW1(SS, (25 + SS - 1) / SS, NWW, CCC, DD) } };
#define QD1(TT_, NWW, CCC, DD, WPE_) \
                if (s.epi == EPI_NONE) k_quad<TT_, NWW, CCC, DD, EPI_NONE, 1, WPE_><<<g, NWW * 32>>>(b); \
                else if (s.epi == EPI_RESIDUAL) k_quad<TT_, NWW, CCC, DD, EPI_RESIDUAL, 1, WPE_><<<g, NWW * 32>>>(b); \
                else k_quad<TT_, NWW, CCC, DD, EPI_SILU, 1, WPE_><<<g, NWW * 32>>>(b);
#define QD(NWW, CCC, DD) QDW(NWW, CCC, DD, 0)
#define QDW(NWW, CCC, DD, WPE_) \
            launch = [=](ConvArgs & a, int) { \
                ConvArgs b = a; \
                if (s.pro == PRO_SILU_GATE) { k_gate<<<s.T, 256>>>(a.x, s.Cin, a.x_stride, gx); b.x = gx; b.x_stride = s.Cin; } \
                const dim3 g((s.Cout + NWW * 8 - 1) / (NWW * 8)); \
                if (s.T == 10) { QD1(3, NWW, CCC, DD, WPE_) } else { QD1(7, NWW, CCC, DD, WPE_) } };
#define Q21(TT_, NWW, CCC, DD) \
                if (s.epi == EPI_NONE) k_quad2<TT_, NWW, CCC, DD, EPI_NONE><<<g, NWW * 32>>>(b); \
                else if (s.epi == EPI_RESIDUAL) k_quad2<TT_, NWW, CCC, DD, EPI_RESIDUAL><<<g, NWW * 32>>>(b); \
                else k_quad2<TT_, NWW, CCC, DD, EPI_SILU><<<g, NWW * 32>>>(b);
#define Q2(NWW, CCC, DD) \
            launch = [=](ConvArgs & a, int) { \
                ConvArgs b = a; \
                if (s.pro == PRO_SILU_GATE) { k_gate<<<s.T, 256>>>(a.x, s.Cin, a.x_stride, gx); b.x = gx; b.x_stride = s.Cin; } \
                const dim3 g((s.Cout + NWW * 8 - 1) / (NWW * 8)); \
                if (s.T == 10) { Q21(3, NWW, CCC, DD) } else { Q21(7, NWW, CCC, DD) } };
#define SK32(RR, NWW, CCC) \
            launch = [=](ConvArgs & a, int) { \
                ConvArgs b = a; \
                if (s.pro == PRO_SILU_GATE) { k_gate<<<s.T, 256>>>(a.x, s.Cin, a.x_stride, gx); b.x = gx; b.x_stride = s.Cin; } \
                const dim3 g((s.Cout + RR * NWW - 1) / (RR * NWW)); \
                if (s.epi == EPI_NONE) k_skinny32<RR, NWW, CCC, EPI_NONE><<<g, NWW * 32>>>(b); \
                else if (s.epi == EPI_RESIDUAL) k_skinny32<RR, NWW, CCC, EPI_RESIDUAL><<<g, NWW * 32>>>(b); \
                else k_skinny32<RR, NWW, CCC, EPI_SILU><<<g, NWW * 32>>>(b); };
#define SK(RR, NWW, CCC, PFF) \
            launch = [=](ConvArgs & a, int) { \
                const dim3 g((s.Cout + RR * NWW - 1) / (RR * NWW)); \
                if (s.pro == PRO_NONE && s.epi == EPI_NONE) k_skinny<RR, NWW, CCC, PFF, PRO_NONE, EPI_NONE><<<g, NWW * 32>>>(a); \
                else if (s.pro == PRO_NONE && s.epi == EPI_RESIDUAL) k_skinny<RR, NWW, CCC, PFF, PRO_NONE, EPI_RESIDUAL><<<g, NWW * 32>>>(a); \
                else if (s.pro == PRO_NONE && s.epi == EPI_SILU) k_skinny<RR, NWW, CCC, PFF, PRO_NONE, EPI_SILU><<<g, NWW * 32>>>(a); \
                else k_skinny<RR, NWW, CCC, PFF, PRO_SILU_GATE, EPI_RESIDUAL><<<g, NWW * 32>>>(a); };
            switch (v) {
            case 0: launch = [=](ConvArgs & a, int) {
                if (s.pro == PRO_NONE && s.epi == EPI_NONE) k_conv<unsigned short, 1, PRO_NONE, EPI_NONE, 1><<<cgrid, NTHREADS>>>(a);
                else if (s.pro == PRO_NONE && s.epi == EPI_RESIDUAL) k_conv<unsigned short, 1, PRO_NONE, EPI_RESIDUAL, 1><<<cgrid, NTHREADS>>>(a);
                else if (s.pro == PRO_NONE && s.epi == EPI_SILU) k_conv<unsigned short, 1, PRO_NONE, EPI_SILU, 1><<<cgrid, NTHREADS>>>(a);
                else k_conv<unsigned short, 1, PRO_SILU_GATE, EPI_RESIDUAL, 1><<<cgrid, NTHREADS>>>(a); }; break;
            case 1: SK(2, 8, 128, 1) break;
            case 2: SK32(2, 4, 512) break;
            case 3: CW(4, 4, 256, 8) break;
            case 4: QD(4, 512, 8) break;
            }
            // correctness on rotation slot 0
            hipMemset(y, 0, maxY * 4);
            launch(a, 0);
            if (hipDeviceSynchronize() != hipSuccess) { printf("  %-24s launch failed: %s\n", variants[v].name, hipGetErrorString(hipGetLastError())); ok[v] = false; continue; }
            const std::size_t ny = static_cast<std::size_t>(s.T) * s.Cout;
            if (v == 0) hipMemcpy(ref, y, ny * 4, hipMemcpyDeviceToDevice);
            else {
                std::vector<float> h(ny), r(ny);
                hipMemcpy(h.data(), y, ny * 4, hipMemcpyDeviceToHost); hipMemcpy(r.data(), ref, ny * 4, hipMemcpyDeviceToHost);
                if (std::memcmp(h.data(), r.data(), ny * 4) != 0) {
                    std::size_t first = 0; while (first < ny && h[first] == r[first]) ++first;
                    printf("  %-24s NOT bit-exact: first diff at %zu (%g vs %g)\n", variants[v].name, first, h[first], r[first]); ok[v] = false; continue;
                }
            }
            // timing over the rotation
            const int iters = 40;
            hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
            for (int i = 0; i < 4; ++i) { a.W = W + (i % rot) * maxW; launch(a, i); }
            hipEventRecord(e0);
            for (int i = 0; i < iters; ++i) { a.W = W + (i % rot) * maxW; launch(a, i); }
            hipEventRecord(e1); hipEventSynchronize(e1);
            float ms; hipEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1e3 / iters, bytes = 2.0 * s.Cin * s.Cout;
            printf("  %-24s %7.1f us  %6.0f GB/s\n", variants[v].name, us, bytes / us / 1e3);
            total[v] += us * s.per_prefill;
        }
    }
    printf("\nper prefill (28 layers + head + text projections), GEMMs only:\n");
    for (int v = 0; v < NV; ++v) printf("  %-24s %8.0f us%s\n", variants[v].name, total[v], ok[v] ? "" : "  (INVALID)");
    return 0;
}
