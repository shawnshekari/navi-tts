// The vocoder's k=7 conv (kernels.h k_conv) at its real shapes for one
// STREAM_BATCH chunk (16 frames): dec0 1024->1536 at 64 rows, then the four
// decoder blocks' three residual-unit convs each (768/384/192/96 wide at
// 512/2560/10240/30720 rows, dilations 1/3/9), and the 96->1 final conv.
// Variants of the thread tile (rows x channels per thread) and of the weight
// tile's LDS layout; every variant is checked bit-identical to the current
// kernel before it is timed (same per-output summation order: ci chunk, then
// ci, then tap). Build:
//   clang++ -x hip --offload-arch=gfx1100 --rocm-path=$ROCM -O3 -I../.. conv7.hip -o conv7 -L$ROCM/lib -lamdhip64
// Run it briefly, the card is shared.
// 2026-09-18, XTX, per 16-frame chunk (all variants bit-exact with k_base):
//   base 32x64 4x2 full unroll        9.4 ms   154 VGPRs; ws reads 16-way bank-conflicted
//   + transposed ws, no unroll        7.7      the unroll alone was worth ~1 ms
//   8x4 / 16x4 tiles, full unroll    12-47     spill to scratch (256 VGPRs)
//   v2: padded ws, paired taps, vector staging, runtime dil   5.2
//   v2 + compile-time dil, no unroll  4.3      -> kernels.h (staging 1.1, FMA 3.6 alone)
//   v2 64x128 8x4                     4.2      wins only on dil 9 at 96-192 wide; not worth
//                                              a second tile (~0.03 ms/frame)
#include "runtime/device/kernel_util.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>

namespace dev = navi::dev;

constexpr int TT = 32, CT = 64, CC = 16, NTHREADS = 256, MAX_DIL = 9;
enum Pro { PRO_NONE = 0, PRO_SNAKE = 1 };
enum Epi { EPI_NONE = 0, EPI_CLAMP = 4 };

__device__ __forceinline__ float wload(unsigned short v) { return dev::half_bits_to_float(v); }

struct ConvArgs {
    const unsigned short * W; const float * bias; const float * x; const float * hist;
    int T_in, Cin, Cout, dil, x_stride, phases, pad_left, reflect;
    float * y; int y_stride;
    const float * snake_a; const float * snake_b;
};

// ---- the current kernel, verbatim in the parts that matter -----------------
template <int K, int PRO, int EPI>
__global__ void __launch_bounds__(NTHREADS) k_base(ConvArgs a) {
    constexpr int WIN = TT + (K - 1) * MAX_DIL;
    __shared__ float xs[WIN][CC];
    __shared__ unsigned short ws[CT][CC * K];
    const int phase = blockIdx.z;
    const int t0 = blockIdx.x * TT, co0 = blockIdx.y * CT;
    const int tid = threadIdx.x;
    const int tl = (tid / 32) * 4, cl = (tid % 32) * 2;
    const int H = (K - 1) * a.dil, win = TT + H;
    const unsigned short * Wp = a.W + static_cast<std::size_t>(phase) * a.Cout * a.Cin * K;
    float acc[4][2] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CC) {
        for (int i = tid; i < win * CC; i += NTHREADS) {
            const int r = i / CC, c = i % CC, ci = ci0 + c;
            int t = t0 - a.pad_left + r;
            float v = 0.f;
            if (ci < a.Cin) {
                if (t >= 0 && t < a.T_in) v = a.x[static_cast<std::size_t>(t) * a.x_stride + ci];
                else if (t < 0 && a.hist) v = a.hist[static_cast<std::size_t>(H + t) * a.Cin + ci];
                if (PRO == PRO_SNAKE) { const float s = __sinf(v * a.snake_a[ci]); v += s * s * a.snake_b[ci]; }
            }
            xs[r][c] = v;
        }
        for (int i = tid; i < CT * CC * K; i += NTHREADS) {
            const int col = i / (CC * K), rem = i % (CC * K);
            const int co = co0 + col, ci = ci0 + rem / K;
            ws[col][rem] = (co < a.Cout && ci < a.Cin) ? Wp[(static_cast<std::size_t>(co) * a.Cin + ci) * K + rem % K] : static_cast<unsigned short>(0);
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < CC; ++c) {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                const float w0 = wload(ws[cl][c * K + j]);
                const float w1 = wload(ws[cl + 1][c * K + j]);
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
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int t = t0 + tl + r, co = co0 + cl + c;
            if (t >= a.T_in || co >= a.Cout) continue;
            float v = acc[r][c];
            if (a.bias) v += a.bias[co];
            if (EPI == EPI_CLAMP) v = fminf(fmaxf(v, -1.f), 1.f);
            a.y[(static_cast<std::size_t>(t) * a.phases + phase) * a.y_stride + co] = v;
        }
}

// ---- register-blocked: RT rows x CTh channels per thread, NW waves ---------
// Wave w owns rows [w*RT, (w+1)*RT); lane l owns channels l, l+32, ... so the
// weight reads are contiguous across the wave and the store is coalesced.
// The weight tile is staged [tap][channel] (+1 dword pad, conflict-free both
// ways) when WT_T, else the current [channel][tap] layout.
template <int K, int PRO, int EPI, int RT, int CTh, int NW, bool WT_T, int UC>
__global__ void __launch_bounds__(NW * 32) k_rb(ConvArgs a) {
    constexpr int NT = NW * 32, TTr = NW * RT, CTr = 32 * CTh;
    constexpr int WIN = TTr + (K - 1) * MAX_DIL;
    constexpr int CTP = CTr + 2;   // u16: one dword of pad
    __shared__ float xs[WIN][CC];
    __shared__ unsigned short ws[WT_T ? CC * K : CTr][WT_T ? CTP : CC * K];
    const int phase = blockIdx.z;
    const int t0 = blockIdx.x * TTr, co0 = blockIdx.y * CTr;
    const int tid = threadIdx.x, lane = tid % 32, tl = (tid / 32) * RT;
    const int H = (K - 1) * a.dil, win = TTr + H;
    const unsigned short * Wp = a.W + static_cast<std::size_t>(phase) * a.Cout * a.Cin * K;
    float acc[RT][CTh] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CC) {
        for (int i = tid; i < win * CC; i += NT) {
            const int r = i / CC, c = i % CC, ci = ci0 + c;
            int t = t0 - a.pad_left + r;
            float v = 0.f;
            if (ci < a.Cin) {
                if (t >= 0 && t < a.T_in) v = a.x[static_cast<std::size_t>(t) * a.x_stride + ci];
                else if (t < 0 && a.hist) v = a.hist[static_cast<std::size_t>(H + t) * a.Cin + ci];
                if (PRO == PRO_SNAKE) { const float s = __sinf(v * a.snake_a[ci]); v += s * s * a.snake_b[ci]; }
            }
            xs[r][c] = v;
        }
        for (int i = tid; i < CTr * CC * K; i += NT) {
            const int col = i / (CC * K), rem = i % (CC * K);
            const int co = co0 + col, ci = ci0 + rem / K;
            const unsigned short w = (co < a.Cout && ci < a.Cin) ? Wp[(static_cast<std::size_t>(co) * a.Cin + ci) * K + rem % K] : static_cast<unsigned short>(0);
            if (WT_T) ws[rem][col] = w; else ws[col][rem] = w;
        }
        __syncthreads();
#pragma unroll UC
        for (int c = 0; c < CC; ++c) {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                float w[CTh];
#pragma unroll
                for (int q = 0; q < CTh; ++q) w[q] = wload(WT_T ? ws[c * K + j][lane + q * 32] : ws[lane + q * 32][c * K + j]);
#pragma unroll
                for (int r = 0; r < RT; ++r) {
                    const float xv = xs[tl + r + j * a.dil][c];
#pragma unroll
                    for (int q = 0; q < CTh; ++q) acc[r][q] += w[q] * xv;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int q = 0; q < CTh; ++q) {
            const int t = t0 + tl + r, co = co0 + lane + q * 32;
            if (t >= a.T_in || co >= a.Cout) continue;
            float v = acc[r][q];
            if (a.bias) v += a.bias[co];
            if (EPI == EPI_CLAMP) v = fminf(fmaxf(v, -1.f), 1.f);
            a.y[(static_cast<std::size_t>(t) * a.phases + phase) * a.y_stride + co] = v;
        }
}


// ---- v2: same 4x2-or-more tile, cheaper staging, denser weight reads --------
// Weight tile stays [channel][tap] but with a 114-u16 (57-dword) row stride:
// odd dword stride -> lane l on row l hits bank (57 l) mod 32, all distinct,
// so the reads are conflict-free without transposing. A tile row is 112
// contiguous u16 in global memory, loaded as 28 qwords (Cin*K*2 is 8-byte
// aligned for every k=7 layer here) and stored as 2 dwords each. Taps are
// read two at a time as one dword. Activations stage as float4 (x_stride is a
// multiple of 4 for every layer here). DIL > 0 makes the dilation a constant
// so the compiler can share xs loads between taps that hit the same row.
// MODE 1 skips the FMA loop, MODE 2 skips staging (timing split only).
template <int K, int PRO, int EPI, int RT, int CTh, int NW, int DIL, int UC, int MODE>
__global__ void __launch_bounds__(NW * 32) k_v2(ConvArgs a) {
    static_assert(K == 7, "paired-tap loads assume K*CC even");
    constexpr int NT = NW * 32, TTr = NW * RT, CTr = 32 * CTh;
    constexpr int WIN = TTr + (K - 1) * MAX_DIL;
    constexpr int WROW = CC * K + 2;            // 114 u16 = 57 dwords, odd
    constexpr int WQ = CC * K / 4;               // 28 qwords per tile row
    __shared__ __align__(16) float xs[WIN][CC];
    __shared__ __align__(8) unsigned short ws[CTr][WROW];
    const int phase = blockIdx.z;
    const int t0 = blockIdx.x * TTr, co0 = blockIdx.y * CTr;
    const int tid = threadIdx.x, lane = tid % 32, tl = (tid / 32) * RT;
    const int dil = DIL > 0 ? DIL : a.dil;
    const int H = (K - 1) * dil, win = TTr + H;
    const unsigned short * Wp = a.W + static_cast<std::size_t>(phase) * a.Cout * a.Cin * K;
    float acc[RT][CTh] = {};
    for (int ci0 = 0; ci0 < a.Cin; ci0 += CC) {
        if (MODE != 2 || ci0 == 0) {   // MODE 2: stage once so the FMA loop reads defined data
            for (int i = tid; i < win * (CC / 4); i += NT) {
                const int r = i / (CC / 4), c4 = (i % (CC / 4)) * 4, ci = ci0 + c4;
                int t = t0 - a.pad_left + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ci < a.Cin) {
                    if (t >= 0 && t < a.T_in) v = *reinterpret_cast<const float4 *>(a.x + static_cast<std::size_t>(t) * a.x_stride + ci);
                    else if (t < 0 && a.hist) v = *reinterpret_cast<const float4 *>(a.hist + static_cast<std::size_t>(H + t) * a.Cin + ci);
                    if (PRO == PRO_SNAKE) {
                        float * e = reinterpret_cast<float *>(&v);
#pragma unroll
                        for (int k = 0; k < 4; ++k) { const float sn = __sinf(e[k] * a.snake_a[ci + k]); e[k] += sn * sn * a.snake_b[ci + k]; }
                    }
                }
                *reinterpret_cast<float4 *>(&xs[r][c4]) = v;
            }
            for (int i = tid; i < CTr * WQ; i += NT) {
                const int col = i / WQ, qi = i % WQ;
                const int co = co0 + col;
                uint2 w = make_uint2(0u, 0u);
                if (co < a.Cout) w = *reinterpret_cast<const uint2 *>(Wp + (static_cast<std::size_t>(co) * a.Cin + ci0) * K + qi * 4);
                unsigned * dst = reinterpret_cast<unsigned *>(&ws[col][qi * 4]);
                dst[0] = w.x; dst[1] = w.y;
            }
        }
        __syncthreads();
        if (MODE == 1) { acc[0][0] += xs[tl][lane % CC] + wload(ws[lane][tl % WROW]); }   // keep the staging live
        if (MODE != 1) {
#pragma unroll UC
            for (int c = 0; c < CC; c += 2) {
                unsigned wd[CTh][K];
#pragma unroll
                for (int q = 0; q < CTh; ++q)
#pragma unroll
                    for (int d = 0; d < K; ++d) wd[q][d] = *reinterpret_cast<const unsigned *>(&ws[lane + q * 32][c * K + 2 * d]);
#pragma unroll
                for (int cc = 0; cc < 2; ++cc)
#pragma unroll
                    for (int j = 0; j < K; ++j) {
                        const int m = cc * K + j;
                        float w[CTh];
#pragma unroll
                        for (int q = 0; q < CTh; ++q) w[q] = wload(static_cast<unsigned short>((m & 1) ? (wd[q][m / 2] >> 16) : (wd[q][m / 2] & 0xffffu)));
#pragma unroll
                        for (int r = 0; r < RT; ++r) {
                            const float xv = xs[tl + r + j * dil][c + cc];
#pragma unroll
                            for (int q = 0; q < CTh; ++q) acc[r][q] += w[q] * xv;
                        }
                    }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int q = 0; q < CTh; ++q) {
            const int t = t0 + tl + r, co = co0 + lane + q * 32;
            if (t >= a.T_in || co >= a.Cout) continue;
            float v = acc[r][q];
            if (a.bias) v += a.bias[co];
            if (EPI == EPI_CLAMP) v = fminf(fmaxf(v, -1.f), 1.f);
            a.y[(static_cast<std::size_t>(t) * a.phases + phase) * a.y_stride + co] = v;
        }
}

// ---------------------------------------------------------------------------
struct Shape { const char * name; int R, Cin, Cout, dil, pro, epi, count; };

static unsigned short f2h(float f) { _Float16 h = static_cast<_Float16>(f); unsigned short u; std::memcpy(&u, &h, 2); return u; }

struct Bufs {
    unsigned short * W; float * bias, * x, * hist, * y, * ref, * sa, * sb;
};

template <class Launch>
static bool run_variant(const char * name, const Shape & s, Bufs & b, ConvArgs & a, Launch launch, double & us_out, bool check) {
    hipMemset(b.y, 0, static_cast<std::size_t>(s.R) * s.Cout * 4);
    launch(a);
    if (hipDeviceSynchronize() != hipSuccess) { printf("  %-26s launch failed: %s\n", name, hipGetErrorString(hipGetLastError())); return false; }
    if (check) {
        std::vector<float> h(static_cast<std::size_t>(s.R) * s.Cout), r(h.size());
        hipMemcpy(h.data(), b.y, h.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(r.data(), b.ref, r.size() * 4, hipMemcpyDeviceToHost);
        if (std::memcmp(h.data(), r.data(), h.size() * 4) != 0) {
            std::size_t first = 0; while (first < h.size() && h[first] == r[first]) ++first;
            printf("  %-26s NOT bit-exact: first diff at %zu (%g vs %g)\n", name, first, h[first], r[first]);
            return false;
        }
    }
    const int iters = 20;
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    for (int i = 0; i < 3; ++i) launch(a);
    hipEventRecord(e0);
    for (int i = 0; i < iters; ++i) launch(a);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    us_out = ms * 1e3 / iters;
    const double flop = 2.0 * s.R * s.Cin * s.Cout * 7;
    printf("  %-26s %8.1f us  %6.1f TFLOP/s\n", name, us_out, flop / us_out / 1e6);
    return true;
}

#define BASE_CASE(PP, EE) \
    if (s.pro == PP && s.epi == EE) return [=](ConvArgs & a) { k_base<7, PP, EE><<<grid, NTHREADS>>>(a); };
#define RB_CASE(PP, EE, RT, CTh, NW, T, UC) \
    if (s.pro == PP && s.epi == EE) return [=](ConvArgs & a) { k_rb<7, PP, EE, RT, CTh, NW, T, UC><<<grid, NW * 32>>>(a); };
#define V2_CASE(PP, EE, RT, CTh, NW, D, UC, M) \
    if (s.pro == PP && s.epi == EE && (D == 0 || s.dil == D)) return [=](ConvArgs & a) { k_v2<7, PP, EE, RT, CTh, NW, D, UC, M><<<grid, NW * 32>>>(a); };
#define V2D(RT, CTh, NW, UC, M) V2_CASE(PRO_NONE, EPI_NONE, RT, CTh, NW, 1, UC, M) V2_CASE(PRO_SNAKE, EPI_NONE, RT, CTh, NW, 1, UC, M) V2_CASE(PRO_SNAKE, EPI_NONE, RT, CTh, NW, 3, UC, M) V2_CASE(PRO_SNAKE, EPI_NONE, RT, CTh, NW, 9, UC, M) V2_CASE(PRO_SNAKE, EPI_CLAMP, RT, CTh, NW, 1, UC, M)
#define V2R(RT, CTh, NW, UC, M) V2_CASE(PRO_NONE, EPI_NONE, RT, CTh, NW, 0, UC, M) V2_CASE(PRO_SNAKE, EPI_NONE, RT, CTh, NW, 0, UC, M) V2_CASE(PRO_SNAKE, EPI_CLAMP, RT, CTh, NW, 0, UC, M)
#define RB3(RT, CTh, NW, T, UC) RB_CASE(PRO_NONE, EPI_NONE, RT, CTh, NW, T, UC) RB_CASE(PRO_SNAKE, EPI_NONE, RT, CTh, NW, T, UC) RB_CASE(PRO_SNAKE, EPI_CLAMP, RT, CTh, NW, T, UC)

int main() {
    const int T = 16;   // STREAM_BATCH frames
    const int R0 = 4 * T;
    std::vector<Shape> shapes = {
        {"dec0 1024->1536", R0, 1024, 1536, 1, PRO_NONE, EPI_NONE, 1},
        {"b0 768 d1", R0 * 8, 768, 768, 1, PRO_SNAKE, EPI_NONE, 1},
        {"b0 768 d3", R0 * 8, 768, 768, 3, PRO_SNAKE, EPI_NONE, 1},
        {"b0 768 d9", R0 * 8, 768, 768, 9, PRO_SNAKE, EPI_NONE, 1},
        {"b1 384 d1", R0 * 40, 384, 384, 1, PRO_SNAKE, EPI_NONE, 1},
        {"b1 384 d3", R0 * 40, 384, 384, 3, PRO_SNAKE, EPI_NONE, 1},
        {"b1 384 d9", R0 * 40, 384, 384, 9, PRO_SNAKE, EPI_NONE, 1},
        {"b2 192 d1", R0 * 160, 192, 192, 1, PRO_SNAKE, EPI_NONE, 1},
        {"b2 192 d3", R0 * 160, 192, 192, 3, PRO_SNAKE, EPI_NONE, 1},
        {"b2 192 d9", R0 * 160, 192, 192, 9, PRO_SNAKE, EPI_NONE, 1},
        {"b3 96 d1", R0 * 480, 96, 96, 1, PRO_SNAKE, EPI_NONE, 1},
        {"b3 96 d3", R0 * 480, 96, 96, 3, PRO_SNAKE, EPI_NONE, 1},
        {"b3 96 d9", R0 * 480, 96, 96, 9, PRO_SNAKE, EPI_NONE, 1},
        {"final 96->1", R0 * 480, 96, 1, 1, PRO_SNAKE, EPI_CLAMP, 1},
    };
    struct Variant { const char * name; int tt, ct; } variants[] = {
        {"base 32x64 4x2 t256 u16", 32, 64},
        {"rb 32x64 4x2 t256 T u1", 32, 64},
        {"rb 64x128 8x4 t256 T u1", 64, 128},
        {"v2 32x64 4x2 rdil u1", 32, 64},
        {"v2 32x64 4x2 rdil u2", 32, 64},
        {"v2 32x64 4x2 cdil u1", 32, 64},
        {"v2 32x64 4x2 cdil u2", 32, 64},
        {"v2 32x64 4x2 cdil u4", 32, 64},
        {"v2 64x128 8x4 cdil u1", 64, 128},
        {"v2 32x128 4x4 cdil u1", 32, 128},
        {"v2 32x64 4x2 cdil u1 STAGE", 32, 64},
        {"v2 32x64 4x2 cdil u1 FMA", 32, 64},
    };
    constexpr int NV = sizeof(variants) / sizeof(variants[0]);
    double total[NV] = {};
    bool ok[NV]; for (auto & o : ok) o = true;

    std::size_t maxR = 0, maxW = 0, maxC = 0;
    for (auto & s : shapes) { maxR = std::max(maxR, (std::size_t) s.R); maxW = std::max(maxW, (std::size_t) s.Cin * s.Cout * 7); maxC = std::max(maxC, (std::size_t) std::max(s.Cin, s.Cout)); }
    Bufs b{};
    hipMalloc(&b.W, maxW * 2); hipMalloc(&b.bias, maxC * 4); hipMalloc(&b.x, maxR * maxC * 4); hipMalloc(&b.hist, 54 * maxC * 4);
    hipMalloc(&b.y, maxR * maxC * 4); hipMalloc(&b.ref, maxR * maxC * 4); hipMalloc(&b.sa, maxC * 4); hipMalloc(&b.sb, maxC * 4);
    {   // deterministic pseudo-random contents; values small enough that sums stay finite
        std::vector<unsigned short> w(maxW); std::vector<float> f(maxR * maxC);
        unsigned s = 12345u; auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.f - 0.5f; };
        for (auto & v : w) v = f2h(rnd() * 0.05f); hipMemcpy(b.W, w.data(), w.size() * 2, hipMemcpyHostToDevice);
        for (auto & v : f) v = rnd(); hipMemcpy(b.x, f.data(), f.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(b.hist, f.data(), 54 * maxC * 4, hipMemcpyHostToDevice);
        std::vector<float> c(maxC); for (auto & v : c) v = rnd(); hipMemcpy(b.bias, c.data(), maxC * 4, hipMemcpyHostToDevice);
        for (auto & v : c) v = 1.f + rnd(); hipMemcpy(b.sa, c.data(), maxC * 4, hipMemcpyHostToDevice);
        for (auto & v : c) v = 1.f + rnd(); hipMemcpy(b.sb, c.data(), maxC * 4, hipMemcpyHostToDevice);
    }

    for (const Shape & s : shapes) {
        ConvArgs a{};
        a.W = b.W; a.bias = b.bias; a.x = b.x; a.hist = b.hist; a.T_in = s.R; a.Cin = s.Cin; a.Cout = s.Cout; a.dil = s.dil;
        a.x_stride = s.Cin; a.phases = 1; a.pad_left = 6 * s.dil; a.reflect = 0; a.y = b.y; a.y_stride = s.Cout; a.snake_a = b.sa; a.snake_b = b.sb;
        printf("%s: %d rows, dil %d\n", s.name, s.R, s.dil);
        for (int v = 0; v < NV; ++v) {
            const dim3 grid((s.R + variants[v].tt - 1) / variants[v].tt, (s.Cout + variants[v].ct - 1) / variants[v].ct, 1);
            auto pick = [&]() -> std::function<void(ConvArgs &)> {
                switch (v) {
                case 0: BASE_CASE(PRO_NONE, EPI_NONE) BASE_CASE(PRO_SNAKE, EPI_NONE) BASE_CASE(PRO_SNAKE, EPI_CLAMP) break;
                case 1: RB3(4, 2, 8, true, 1) break;
                case 2: RB3(8, 4, 8, true, 1) break;
                case 3: V2R(4, 2, 8, 1, 0) break;
                case 4: V2R(4, 2, 8, 2, 0) break;
                case 5: V2D(4, 2, 8, 1, 0) break;
                case 6: V2D(4, 2, 8, 2, 0) break;
                case 7: V2D(4, 2, 8, 4, 0) break;
                case 8: V2D(8, 4, 8, 1, 0) break;
                case 9: V2D(4, 4, 8, 1, 0) break;
                case 10: V2D(4, 2, 8, 1, 1) break;
                case 11: V2D(4, 2, 8, 1, 2) break;
                }
                return nullptr;
            };
            auto launch = pick();
            if (!launch) { printf("  %-26s no instantiation\n", variants[v].name); ok[v] = false; continue; }
            double us = 0;
            if (v == 0) {
                if (!run_variant(variants[v].name, s, b, a, launch, us, false)) { ok[v] = false; continue; }
                hipMemcpy(b.ref, b.y, static_cast<std::size_t>(s.R) * s.Cout * 4, hipMemcpyDeviceToDevice);
            } else if (!run_variant(variants[v].name, s, b, a, launch, us, v < 10)) { ok[v] = false; continue; }
            total[v] += us * s.count;
        }
    }
    printf("\nper 16-frame chunk, the k=7 convs (dec0 + 4 blocks x 3 units + final):\n");
    for (int v = 0; v < NV; ++v) printf("  %-26s %8.1f us%s\n", variants[v].name, total[v], ok[v] ? "" : "  (INVALID)");
    return 0;
}
