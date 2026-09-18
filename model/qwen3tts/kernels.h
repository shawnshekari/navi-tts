#pragma once

// Device kernels shared by the Qwen3-TTS modules (vocoder, talker prefill).
// Included from .hip translation units only.

#include "runtime/device/kernel_util.h"

#include <hip/hip_runtime.h>

#include <cstddef>

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
// channels in chunks of 16 staged through LDS.
// ---------------------------------------------------------------------------

constexpr int TT = 32, CT = 64, CC = 16, NTHREADS = 256, MAX_DIL = 9;

enum Pro { PRO_NONE = 0, PRO_SNAKE = 1, PRO_SILU_GATE = 2 };
enum Epi { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2, EPI_SCALE_RESIDUAL = 3, EPI_CLAMP = 4, EPI_SILU = 5 };

template <class WT> __device__ __forceinline__ float wload(WT v);
template <> __device__ __forceinline__ float wload<unsigned short>(unsigned short v) { return dev::half_bits_to_float(v); }
template <> __device__ __forceinline__ float wload<float>(float v) { return v; }

struct ConvArgs {
    const void * W;             // [phases][Cout][Cin][K], f16 or f32
    const float * bias;         // [Cout] or null
    const float * x;            // [T_in][x_stride]
    const float * hist;         // [H][Cin] or null
    int T_in, Cin, Cout, dil, x_stride, phases;
    float * y;                  // rows t*phases + phase, stride y_stride
    int y_stride;
    const float * snake_a;      // exp(alpha)         [Cin]
    const float * snake_b;      // 1/(exp(beta)+1e-9) [Cin]
    const float * residual;     // [T_out][Cout]
    const float * scale;        // [Cout]
};

template <class WT, int K, int PRO, int EPI>
__global__ void __launch_bounds__(NTHREADS) k_conv(ConvArgs a) {
    constexpr int WIN = TT + (K - 1) * MAX_DIL;
    __shared__ float xs[WIN][CC];
    __shared__ WT ws[CT][CC * K];

    const int phase = blockIdx.z;
    const int t0 = blockIdx.x * TT, co0 = blockIdx.y * CT;
    const int tid = threadIdx.x;
    const int tl = (tid / 32) * 4, cl = (tid % 32) * 2;
    const int H = (K - 1) * a.dil, win = TT + H;
    const WT * Wp = static_cast<const WT *>(a.W) + static_cast<std::size_t>(phase) * a.Cout * a.Cin * K;
    float acc[4][2] = {};

    for (int ci0 = 0; ci0 < a.Cin; ci0 += CC) {
        for (int i = tid; i < win * CC; i += NTHREADS) {
            const int r = i / CC, c = i % CC, ci = ci0 + c, t = t0 - H + r;
            float v = 0.f;
            if (ci < a.Cin) {
                if (t >= 0 && t < a.T_in) {
                    const float * row = a.x + static_cast<std::size_t>(t) * a.x_stride;
                    if (PRO == PRO_SILU_GATE) {
                        const float g = row[ci];
                        v = g / (1.f + __expf(-g)) * row[a.Cin + ci];
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
        __syncthreads();
#pragma unroll
        for (int c = 0; c < CC; ++c) {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                const float w0 = wload<WT>(ws[cl][c * K + j]);
                const float w1 = wload<WT>(ws[cl + 1][c * K + j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float xv = xs[tl + r + j * a.dil][c];
                    acc[r][0] += w0 * xv;
                    acc[r][1] += w1 * xv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int t = t0 + tl + r, co = co0 + cl + c;
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
