#pragma once

// Device kernels shared by the Qwen3-TTS modules (vocoder, talker prefill).
// Included from .hip translation units only.

#include "runtime/device/kernel_util.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace navi::qwen3tts::kern {

// ---------------------------------------------------------------------------
// The conv kernel. Causal Conv1d as an implicit GEMM over time-major f32
// activations [T][C] with f16 weights [Cout][Cin][K]:
//     y[t][co] = bias[co] + sum_{ci,j} W[co][ci][j] * x[t - (K-1-j)*dil][ci]
// Rows before t = 0 come from `hist` (the previous chunk's last H rows). K = 1
// is a plain GEMM (every Linear). A transposed conv with stride s and kernel
// k = s or 2s is `phases = s` of these with K = k/s on re-laid-out weights,
// each phase writing every s-th output row (host-side prep below).
// Tile: 32 time rows x 64 output channels per block, 4x2 per thread, input
// channels in chunks of 16 staged through LDS. A wave owns 4 rows and lane l
// owns channels l and l+32, so activation reads broadcast and weight reads
// and the store are contiguous across the wave.
//
// The weight tile is [channel][tap] with one dword of padding per row: CC*K
// is a multiple of 16, so the padded row stride is an odd number of dwords
// and lane l on row l hits bank (stride*l) mod 32 - all distinct, no
// conflicts. (The unpadded stride put every lane on one of two banks: a
// 16-way conflict on every weight load; bench/micro/conv7.hip.) f16 taps are
// read two per dword. Staging is vectorised (8-byte weight rows, float4
// activations) when the shapes and pointers allow, which every layer of this
// model does; the scalar path stays for the general case. DIL fixes the
// dilation at compile time so the compiler can share activation loads
// between taps (0 = runtime). The inner loop is deliberately not unrolled
// over the channel chunk for K > 1: fully unrolled it hoists hundreds of LDS
// loads, takes 150+ VGPRs, and runs slower.
//
// The per-output summation order is chunk, then channel, then tap, in every
// path - the kernel is bit-exact with its previous form.
// ---------------------------------------------------------------------------

constexpr int TT = 32, CT = 64, CC = 16, NTHREADS = 256, MAX_DIL = 9;

enum Pro { PRO_NONE = 0, PRO_SNAKE = 1, PRO_SILU_GATE = 2, PRO_ADD2 = 3 };
enum Epi { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2, EPI_SCALE_RESIDUAL = 3, EPI_CLAMP = 4, EPI_SILU = 5,
           EPI_RELU = 6, EPI_SIGMOID = 7, EPI_RELU_TANH = 8 };

template <class WT> __device__ __forceinline__ float wload(WT v);
template <> __device__ __forceinline__ float wload<unsigned short>(unsigned short v) { return dev::half_bits_to_float(v); }
template <> __device__ __forceinline__ float wload<float>(float v) { return v; }

struct ConvArgs {
    const void * W;             // [phases][Cout][Cin][K], f16 or f32
    const float * bias;         // [Cout] or null
    const float * x;            // [T_in][x_stride]
    const float * hist;         // [H][Cin] or null (causal convs)
    const float * x2;           // PRO_ADD2: second input, same stride, added to x before the conv
    int T_in, Cin, Cout, dil, x_stride, phases;
    int pad_left;               // (K-1)*dil for causal, (K-1)*dil/2 for "same"
    int reflect;                // out-of-range rows mirror (PyTorch padding_mode="reflect") instead of hist/zero
    float * y;                  // rows t*phases + phase, stride y_stride
    int y_stride;
    const float * snake_a;      // exp(alpha)         [Cin]
    const float * snake_b;      // 1/(exp(beta)+1e-9) [Cin]
    const float * residual;     // [T_out][Cout]
    const float * scale;        // [Cout]
};

template <class WT, int K, int PRO, int EPI, int DIL = 0>
__global__ void __launch_bounds__(NTHREADS) k_conv(ConvArgs a) {
    constexpr int WIN = TT + (K - 1) * MAX_DIL;
    constexpr int WROW = CC * K + 4 / static_cast<int>(sizeof(WT));   // + one dword: odd dword stride
    constexpr int EPQ = 8 / static_cast<int>(sizeof(WT));              // weight elements per 8-byte load
    constexpr int WQ = CC * K / EPQ;                                    // 8-byte loads per tile row
    constexpr int UC = K == 1 ? CC / 2 : 1;
    __shared__ __align__(16) float xs[WIN][CC];
    __shared__ __align__(16) WT ws[CT][WROW];

    const int phase = blockIdx.z;
    const int t0 = blockIdx.x * TT, co0 = blockIdx.y * CT;
    const int tid = threadIdx.x, lane = tid % 32, tl = (tid / 32) * 4;
    const int dil = DIL > 0 ? DIL : a.dil;
    const int H = (K - 1) * dil, win = TT + H;   // window rows; row 0 is t0 - pad_left
    const WT * Wp = static_cast<const WT *>(a.W) + static_cast<std::size_t>(phase) * a.Cout * a.Cin * K;
    const bool vec = (a.Cin % CC == 0) && ((a.Cin * K * static_cast<int>(sizeof(WT))) % 8 == 0) && (a.x_stride % 4 == 0)
        && ((reinterpret_cast<std::uintptr_t>(a.x) | reinterpret_cast<std::uintptr_t>(a.hist)
             | reinterpret_cast<std::uintptr_t>(a.x2) | reinterpret_cast<std::uintptr_t>(Wp)) & 15u) == 0;
    float acc[4][2] = {};

    for (int ci0 = 0; ci0 < a.Cin; ci0 += CC) {
        if (vec) {
            for (int i = tid; i < win * (CC / 4); i += NTHREADS) {
                const int r = i / (CC / 4), c4 = (i % (CC / 4)) * 4, ci = ci0 + c4;
                int t = t0 - a.pad_left + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (a.reflect) {
                    if (t < 0) t = -t;
                    if (t >= a.T_in) t = 2 * (a.T_in - 1) - t;
                }
                if (t >= 0 && t < a.T_in) {
                    const float * row = a.x + static_cast<std::size_t>(t) * a.x_stride;
                    if (PRO == PRO_SILU_GATE) {
                        const float4 g = *reinterpret_cast<const float4 *>(row + ci);
                        const float4 u = *reinterpret_cast<const float4 *>(row + a.Cin + ci);
                        v.x = g.x / (1.f + __expf(-g.x)) * u.x; v.y = g.y / (1.f + __expf(-g.y)) * u.y;
                        v.z = g.z / (1.f + __expf(-g.z)) * u.z; v.w = g.w / (1.f + __expf(-g.w)) * u.w;
                    } else if (PRO == PRO_ADD2) {
                        const float4 p = *reinterpret_cast<const float4 *>(row + ci);
                        const float4 q = *reinterpret_cast<const float4 *>(a.x2 + static_cast<std::size_t>(t) * a.x_stride + ci);
                        v.x = p.x + q.x; v.y = p.y + q.y; v.z = p.z + q.z; v.w = p.w + q.w;
                    } else {
                        v = *reinterpret_cast<const float4 *>(row + ci);
                    }
                } else if (t < 0 && a.hist) {
                    v = *reinterpret_cast<const float4 *>(a.hist + static_cast<std::size_t>(H + t) * a.Cin + ci);
                }
                if (PRO == PRO_SNAKE) {
                    float * e = reinterpret_cast<float *>(&v);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float s = __sinf(e[k] * a.snake_a[ci + k]);
                        e[k] += s * s * a.snake_b[ci + k];
                    }
                }
                *reinterpret_cast<float4 *>(&xs[r][c4]) = v;
            }
            for (int i = tid; i < CT * WQ; i += NTHREADS) {
                const int col = i / WQ, qi = i % WQ, co = co0 + col;
                uint2 w = make_uint2(0u, 0u);
                if (co < a.Cout) w = *reinterpret_cast<const uint2 *>(Wp + (static_cast<std::size_t>(co) * a.Cin + ci0) * K + qi * EPQ);
                unsigned * dst = reinterpret_cast<unsigned *>(&ws[col][qi * EPQ]);
                dst[0] = w.x; dst[1] = w.y;
            }
        } else {
            for (int i = tid; i < win * CC; i += NTHREADS) {
                const int r = i / CC, c = i % CC, ci = ci0 + c;
                int t = t0 - a.pad_left + r;
                float v = 0.f;
                if (ci < a.Cin) {
                    if (a.reflect) {
                        if (t < 0) t = -t;
                        if (t >= a.T_in) t = 2 * (a.T_in - 1) - t;
                    }
                    if (t >= 0 && t < a.T_in) {
                        const float * row = a.x + static_cast<std::size_t>(t) * a.x_stride;
                        if (PRO == PRO_SILU_GATE) {
                            const float g = row[ci];
                            v = g / (1.f + __expf(-g)) * row[a.Cin + ci];
                        } else if (PRO == PRO_ADD2) {
                            v = row[ci] + a.x2[static_cast<std::size_t>(t) * a.x_stride + ci];
                        } else {
                            v = row[ci];
                        }
                    } else if (t < 0 && a.hist) {
                        v = a.hist[static_cast<std::size_t>(H + t) * a.Cin + ci];
                    }
                    if (PRO == PRO_SNAKE) {
                        const float s = __sinf(v * a.snake_a[ci]);
                        v += s * s * a.snake_b[ci];
                    }
                }
                xs[r][c] = v;
            }
            for (int i = tid; i < CT * CC * K; i += NTHREADS) {
                const int col = i / (CC * K), rem = i % (CC * K);
                const int co = co0 + col, ci = ci0 + rem / K;
                ws[col][rem] = (co < a.Cout && ci < a.Cin)
                    ? Wp[(static_cast<std::size_t>(co) * a.Cin + ci) * K + rem % K] : static_cast<WT>(0);
            }
        }
        __syncthreads();
#pragma unroll UC
        for (int c = 0; c < CC; c += 2) {   // two channels = 2K taps per column, an even count
            unsigned wd[2][K];              // f16: 2K taps as K dwords
            float wf[2][2 * K];             // f32: read directly
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                if constexpr (sizeof(WT) == 2) {
#pragma unroll
                    for (int d = 0; d < K; ++d) wd[q][d] = *reinterpret_cast<const unsigned *>(&ws[lane + q * 32][c * K + 2 * d]);
                } else {
#pragma unroll
                    for (int m = 0; m < 2 * K; ++m) wf[q][m] = wload<WT>(ws[lane + q * 32][c * K + m]);
                }
            }
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    const int m = cc * K + j;
                    float w0, w1;
                    if constexpr (sizeof(WT) == 2) {
                        w0 = dev::half_bits_to_float(static_cast<unsigned short>((m & 1) ? (wd[0][m / 2] >> 16) : (wd[0][m / 2] & 0xffffu)));
                        w1 = dev::half_bits_to_float(static_cast<unsigned short>((m & 1) ? (wd[1][m / 2] >> 16) : (wd[1][m / 2] & 0xffffu)));
                    } else {
                        w0 = wf[0][m]; w1 = wf[1][m];
                    }
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const float xv = xs[tl + r + j * dil][c + cc];
                        acc[r][0] += w0 * xv;
                        acc[r][1] += w1 * xv;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int t = t0 + tl + r, co = co0 + lane + c * 32;
            if (t >= a.T_in || co >= a.Cout) continue;
            float v = acc[r][c];
            if (a.bias) v += a.bias[co];
            const std::size_t orow = static_cast<std::size_t>(t) * a.phases + phase;
            if (EPI == EPI_GELU) {
                v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
            } else if (EPI == EPI_RESIDUAL) {
                v += a.residual[orow * a.Cout + co];
            } else if (EPI == EPI_SCALE_RESIDUAL) {
                v = v * a.scale[co] + a.residual[orow * a.Cout + co];
            } else if (EPI == EPI_CLAMP) {
                v = fminf(fmaxf(v, -1.f), 1.f);
            } else if (EPI == EPI_SILU) {
                v = v / (1.f + __expf(-v));
            } else if (EPI == EPI_RELU) {
                v = fmaxf(v, 0.f);
            } else if (EPI == EPI_SIGMOID) {
                v = 1.f / (1.f + __expf(-v));
            } else if (EPI == EPI_RELU_TANH) {
                v = tanhf(fmaxf(v, 0.f));
            }
            a.y[orow * a.y_stride + co] = v;
        }
    }
}


static __global__ void k_rmsnorm(const float * x, const float * w, int C, float eps, float * y) {
    __shared__ float red[32];
    const float * row = x + static_cast<std::size_t>(blockIdx.x) * C;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) ss += row[c] * row[c];
    const float scale = rsqrtf(dev::block_sum(ss, red) / C + eps);
    float * out = y + static_cast<std::size_t>(blockIdx.x) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[c] = row[c] * scale * w[c];
}


} // namespace navi::qwen3tts::kern
