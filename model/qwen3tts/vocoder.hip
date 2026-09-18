#include "model/qwen3tts/vocoder.h"

#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstring>

namespace navi::qwen3tts {

namespace {

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
enum Epi { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2, EPI_SCALE_RESIDUAL = 3, EPI_CLAMP = 4 };

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
            }
            a.y[orow * a.y_stride + co] = v;
        }
    }
}

// ConvNeXt front: depthwise causal conv (k=7) then LayerNorm over channels.
template <class WT>
__global__ void k_dwconv_ln(const WT * W, const float * bias, const float * x, const float * hist,
                            int T, int C, const float * ln_w, const float * ln_b, float eps, float * y) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const int per = (C + blockDim.x - 1) / blockDim.x;
    float v[8];
    float s = 0.f;
    for (int k = 0; k < per; ++k) {
        const int ch = threadIdx.x + k * blockDim.x;
        float acc = 0.f;
        if (ch < C) {
            acc = bias[ch];
            for (int j = 0; j < 7; ++j) {
                const int tt = t - 6 + j;
                const float xv = tt >= 0 ? x[static_cast<std::size_t>(tt) * C + ch]
                                         : hist[static_cast<std::size_t>(6 + tt) * C + ch];
                acc += wload<WT>(W[ch * 7 + j]) * xv;
            }
        }
        v[k] = acc;
        s += acc;
    }
    const float mean = dev::block_sum(s, red) / C;
    float ss = 0.f;
    for (int k = 0; k < per; ++k) {
        const int ch = threadIdx.x + k * blockDim.x;
        if (ch < C) { const float d = v[k] - mean; ss += d * d; }
    }
    const float inv = rsqrtf(dev::block_sum(ss, red) / C + eps);
    for (int k = 0; k < per; ++k) {
        const int ch = threadIdx.x + k * blockDim.x;
        if (ch < C) y[static_cast<std::size_t>(t) * C + ch] = (v[k] - mean) * inv * ln_w[ch] + ln_b[ch];
    }
}

__global__ void k_rmsnorm(const float * x, const float * w, int C, float eps, float * y) {
    __shared__ float red[32];
    const float * row = x + static_cast<std::size_t>(blockIdx.x) * C;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) ss += row[c] * row[c];
    const float scale = rsqrtf(dev::block_sum(ss, red) / C + eps);
    float * out = y + static_cast<std::size_t>(blockIdx.x) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[c] = row[c] * scale * w[c];
}

// qkv[T][3*H*D]: RoPE on q (in place) and k, append k and v to the ring caches.
__global__ void k_rope_append(float * qkv, int n_head, int head_dim, int pos0, int ring, double theta,
                              float * kc, float * vc) {
    const int t = blockIdx.x, pos = pos0 + t, slot = pos % ring;
    const int HD = n_head * head_dim, half = head_dim / 2;
    float * row = qkv + static_cast<std::size_t>(t) * 3 * HD;
    for (int i = threadIdx.x; i < n_head * half; i += blockDim.x) {
        const int h = i / half, p = i % half;
        const float inv_freq = static_cast<float>(1.0 / pow(theta, (2.0 * p) / head_dim));
        const float ang = static_cast<float>(pos) * inv_freq;
        const float cs = cosf(ang), sn = sinf(ang);
        float * q = row + h * head_dim;
        float * k = row + HD + h * head_dim;
        const float q0 = q[p], q1 = q[p + half];
        q[p] = q0 * cs - q1 * sn;
        q[p + half] = q1 * cs + q0 * sn;
        const float k0 = k[p], k1 = k[p + half];
        const float kr0 = k0 * cs - k1 * sn, kr1 = k1 * cs + k0 * sn;
        k[p] = kr0;
        k[p + half] = kr1;
        float * kd = kc + (static_cast<std::size_t>(slot) * n_head + h) * head_dim;
        kd[p] = kr0;
        kd[p + half] = kr1;
    }
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        vc[static_cast<std::size_t>(slot) * HD + i] = row[2 * HD + i];
    }
}

// Sliding-window causal attention over the ring caches: keys (pos-window, pos].
__global__ void k_attention(const float * qkv, int n_head, int head_dim, int pos0, int window, int ring,
                            const float * kc, const float * vc, float * out) {
    __shared__ float sc[128];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, pos = pos0 + t;
    const int HD = n_head * head_dim;
    const int k0 = pos - window + 1 > 0 ? pos - window + 1 : 0;
    const int n = pos - k0 + 1;
    const float * q = qkv + static_cast<std::size_t>(t) * 3 * HD + h * head_dim;
    const float scaling = rsqrtf(static_cast<float>(head_dim));
    float s = -INFINITY;
    if (static_cast<int>(threadIdx.x) < n) {
        const int slot = (k0 + threadIdx.x) % ring;
        const float * k = kc + (static_cast<std::size_t>(slot) * n_head + h) * head_dim;
        float d = 0.f;
        for (int i = 0; i < head_dim; ++i) d += q[i] * k[i];
        s = d * scaling;
    }
    const float m = dev::block_max(s, red);
    const float e = static_cast<int>(threadIdx.x) < n ? __expf(s - m) : 0.f;
    const float sum = dev::block_sum(e, red);
    sc[threadIdx.x] = e / sum;
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < head_dim) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            const int slot = (k0 + j) % ring;
            acc += sc[j] * vc[(static_cast<std::size_t>(slot) * n_head + h) * head_dim + threadIdx.x];
        }
        out[static_cast<std::size_t>(t) * HD + h * head_dim + threadIdx.x] = acc;
    }
}

// RVQ: e[t] = [ codebook_first[c0] | sum_q codebook_rest_q[c_q] ]  (f16 rows -> f32)
template <class WT>
__global__ void k_rvq(const int * codes, int n_q, const WT * cb_first, const WT * cb_rest,
                      int dim, int codebook_size, float * e) {
    const int t = blockIdx.x;
    const int * c = codes + t * n_q;
    for (int i = threadIdx.x; i < 2 * dim; i += blockDim.x) {
        float v;
        if (i < dim) {
            v = wload<WT>(cb_first[static_cast<std::size_t>(c[0]) * dim + i]);
        } else {
            v = 0.f;
            for (int q = 1; q < n_q; ++q) {
                v += wload<WT>(cb_rest[(static_cast<std::size_t>(q - 1) * codebook_size + c[q]) * dim + (i - dim)]);
            }
        }
        e[static_cast<std::size_t>(t) * 2 * dim + i] = v;
    }
}

// new_hist = last H rows of [hist ; x]
__global__ void k_hist_update(const float * hist, const float * x, int T, int H, int C, float * out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(H) * C) return;
    const int r = static_cast<int>(i / C), c = static_cast<int>(i % C);
    const int src = r + T;   // row index into [hist ; x]
    out[i] = src < H ? hist[static_cast<std::size_t>(src) * C + c] : x[static_cast<std::size_t>(src - H) * C + c];
}

// ---------------------------------------------------------------------------

struct Conv {
    const void * W = nullptr;
    const float * bias = nullptr;
    int Cin = 0, Cout = 0, K = 1, dil = 1, phases = 1;
    int H = 0;
    float * hist = nullptr, * hist2 = nullptr;
    const float * snake_a = nullptr, * snake_b = nullptr;
};

struct Layer {
    const float * ln1, * ln2, * attn_scale, * mlp_scale;
    Conv qkv, o, gateup, down;
    float * kc, * vc;
};

struct Unit { Conv c1, c2; };
struct Block { Conv up; Unit u[3]; int rate, Cin, Cout; };
struct ConvNeXt { const void * dw; const float * dw_bias, * ln_w, * ln_b, * gamma; float * hist, * hist2; Conv pw1, pw2; };

} // namespace

struct Vocoder::Impl {
    hipStream_t stream = nullptr;
    Params p;
    int Cmax = 0;
    int ring = 0;
    int pos0 = 0;
    std::vector<void *> owned;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    double last_ms = 0;
    Vocoder::StageHook hook;
    void snap(const char * name, const float * buf, int rows, int cols) {
        if (!hook) return;
        NAVI_HIP(hipStreamSynchronize(stream));
        hook(name, buf, rows, cols);
    }

    // weights / layers
    bool f32 = false;   // codec weights are f32 (general.codec_dtype)
    const void * cb_first = nullptr, * cb_rest = nullptr;
    Conv rvq_proj, pre_conv, in_proj, out_proj, dec0, final_conv;
    const float * final_norm = nullptr;
    std::vector<Layer> layers;
    Conv up_conv[2];
    ConvNeXt cnx[2];
    Block blocks[4];

    // scratch
    int * d_codes = nullptr;
    float * e = nullptr, * h = nullptr, * a = nullptr, * x = nullptr, * n = nullptr, * qkv = nullptr, * att = nullptr,
          * gu = nullptr, * u = nullptr, * A = nullptr, * B = nullptr, * G = nullptr, * d = nullptr, * pcm = nullptr;
    float * P[4] = {}, * Q[4] = {};
    int rows[4] = {};   // rows per Cmax frame at each block's output
    std::vector<float> h_pcm;

    template <class T> T * alloc(std::size_t n) {
        void * ptr = nullptr;
        NAVI_HIP(hipMalloc(&ptr, n * sizeof(T)));
        NAVI_HIP(hipMemset(ptr, 0, n * sizeof(T)));
        owned.push_back(ptr);
        return static_cast<T *>(ptr);
    }
    template <class T> T * upload(const std::vector<T> & v) {
        T * ptr = alloc<T>(v.size());
        NAVI_HIP(hipMemcpy(ptr, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
        return ptr;
    }

    void run_conv(Conv & c, int pro, int epi, const float * x_in, int T_in, int x_stride, float * y,
                  int y_stride, const float * residual = nullptr, const float * scale = nullptr) {
        ConvArgs a{};
        a.W = c.W; a.bias = c.bias; a.x = x_in; a.hist = c.hist; a.T_in = T_in; a.Cin = c.Cin; a.Cout = c.Cout;
        a.dil = c.dil; a.x_stride = x_stride; a.phases = c.phases; a.y = y; a.y_stride = y_stride;
        a.snake_a = c.snake_a; a.snake_b = c.snake_b; a.residual = residual; a.scale = scale;
        const dim3 grid((T_in + TT - 1) / TT, (c.Cout + CT - 1) / CT, c.phases);
        const dim3 block(NTHREADS);
#define NAVI_CONV_CASE(KK, PP, EE) \
        if (c.K == KK && pro == PP && epi == EE) { \
            if (f32) k_conv<float, KK, PP, EE><<<grid, block, 0, stream>>>(a); \
            else k_conv<unsigned short, KK, PP, EE><<<grid, block, 0, stream>>>(a); \
            goto launched; }
        NAVI_CONV_CASE(1, PRO_NONE, EPI_NONE)
        NAVI_CONV_CASE(1, PRO_NONE, EPI_GELU)
        NAVI_CONV_CASE(1, PRO_NONE, EPI_SCALE_RESIDUAL)
        NAVI_CONV_CASE(1, PRO_SILU_GATE, EPI_SCALE_RESIDUAL)
        NAVI_CONV_CASE(1, PRO_SNAKE, EPI_RESIDUAL)
        NAVI_CONV_CASE(2, PRO_SNAKE, EPI_NONE)
        NAVI_CONV_CASE(3, PRO_NONE, EPI_NONE)
        NAVI_CONV_CASE(7, PRO_NONE, EPI_NONE)
        NAVI_CONV_CASE(7, PRO_SNAKE, EPI_NONE)
        NAVI_CONV_CASE(7, PRO_SNAKE, EPI_CLAMP)
#undef NAVI_CONV_CASE
        fail("vocoder: no conv instantiation for K=" + std::to_string(c.K) + " pro=" + std::to_string(pro) +
             " epi=" + std::to_string(epi));
    launched:
        NAVI_HIP(hipGetLastError());
        if (c.hist) hist_update(c, x_in, T_in);
    }

    void hist_update(Conv & c, const float * x_in, int T_in) {
        const std::size_t n = static_cast<std::size_t>(c.H) * c.Cin;
        k_hist_update<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(c.hist, x_in, T_in, c.H, c.Cin, c.hist2);
        NAVI_HIP(hipGetLastError());
        std::swap(c.hist, c.hist2);
    }

    void decode_chunk(int T) {
        const auto & vp = p.vocoder;
        const int VH = vp.hidden, L = p.vq_latent, HD = vp.n_head * vp.head_dim;
        if (f32) k_rvq<float><<<T, 256, 0, stream>>>(d_codes, p.vq_n_q, static_cast<const float *>(cb_first), static_cast<const float *>(cb_rest), p.vq_codebook_dim / 2, p.vq_codebook_size, e);
        else k_rvq<unsigned short><<<T, 256, 0, stream>>>(d_codes, p.vq_n_q, static_cast<const unsigned short *>(cb_first), static_cast<const unsigned short *>(cb_rest), p.vq_codebook_dim / 2, p.vq_codebook_size, e);
        run_conv(rvq_proj, PRO_NONE, EPI_NONE, e, T, p.vq_codebook_dim, h, p.vq_codebook_dim);
        snap("rvq", h, T, p.vq_codebook_dim);
        run_conv(pre_conv, PRO_NONE, EPI_NONE, h, T, p.vq_codebook_dim, a, L);
        snap("pre_conv", a, T, L);
        run_conv(in_proj, PRO_NONE, EPI_NONE, a, T, L, x, VH);
        snap("in_proj", x, T, VH);
        int li = 0;
        for (Layer & ly : layers) {
            k_rmsnorm<<<T, 256, 0, stream>>>(x, ly.ln1, VH, vp.rms_eps, n);
            snap(("norm1_" + std::to_string(li)).c_str(), n, T, VH);
            run_conv(ly.qkv, PRO_NONE, EPI_NONE, n, T, VH, qkv, 3 * HD);
            snap(("qkv_" + std::to_string(li)).c_str(), qkv, T, 3 * HD);
            k_rope_append<<<T, 256, 0, stream>>>(qkv, vp.n_head, vp.head_dim, pos0, ring, vp.rope_theta, ly.kc, ly.vc);
            k_attention<<<dim3(T, vp.n_head), 128, 0, stream>>>(qkv, vp.n_head, vp.head_dim, pos0,
                                                                  p.vocoder_sliding_window, ring, ly.kc, ly.vc, att);
            snap(("attnraw" + std::to_string(li)).c_str(), att, T, HD);
            snap(("qkvrope_" + std::to_string(li)).c_str(), qkv, T, 3 * HD);
            run_conv(ly.o, PRO_NONE, EPI_SCALE_RESIDUAL, att, T, HD, x, VH, x, ly.attn_scale);
            snap(("xattn_" + std::to_string(li)).c_str(), x, T, VH);
            k_rmsnorm<<<T, 256, 0, stream>>>(x, ly.ln2, VH, vp.rms_eps, n);
            run_conv(ly.gateup, PRO_NONE, EPI_NONE, n, T, VH, gu, 2 * vp.ff);
            run_conv(ly.down, PRO_SILU_GATE, EPI_SCALE_RESIDUAL, gu, T, 2 * vp.ff, x, VH, x, ly.mlp_scale);
            snap(("layer" + std::to_string(li)).c_str(), x, T, VH);
            ++li;
        }
        k_rmsnorm<<<T, 256, 0, stream>>>(x, final_norm, VH, vp.rms_eps, n);
        run_conv(out_proj, PRO_NONE, EPI_NONE, n, T, VH, u, L);
        snap("pre_transformer", u, T, L);

        // upsample x2 x2: transposed conv (k = s) then ConvNeXt. Buffers A/B hold
        // up to 4*Cmax rows; each round: src -> dst (up), dst -> tmp (dw+LN),
        // tmp -> G (pw1), G -> dst (pw2 + residual dst).
        int R = T;
        float * src = u;
        for (int j = 0; j < 2; ++j) {
            float * dst = (j == 0) ? A : B;
            float * tmp = (j == 0) ? B : A;
            const int r = p.vocoder_upsampling_ratios[static_cast<std::size_t>(j)];
            run_conv(up_conv[j], PRO_NONE, EPI_NONE, src, R, L, dst, L);
            R *= r;
            snap(("up" + std::to_string(j) + "_t").c_str(), dst, R, L);
            ConvNeXt & cx = cnx[j];
            if (f32) k_dwconv_ln<float><<<R, 256, 0, stream>>>(static_cast<const float *>(cx.dw), cx.dw_bias, dst, cx.hist, R, L, cx.ln_w, cx.ln_b, 1e-6f, tmp);
            else k_dwconv_ln<unsigned short><<<R, 256, 0, stream>>>(static_cast<const unsigned short *>(cx.dw), cx.dw_bias, dst, cx.hist, R, L, cx.ln_w, cx.ln_b, 1e-6f, tmp);
            NAVI_HIP(hipGetLastError());
            snap(("up" + std::to_string(j) + "_ln").c_str(), tmp, R, L);
            {
                const std::size_t nh = static_cast<std::size_t>(6) * L;
                k_hist_update<<<static_cast<unsigned>((nh + 255) / 256), 256, 0, stream>>>(cx.hist, dst, R, 6, L, cx.hist2);
                std::swap(cx.hist, cx.hist2);
            }
            run_conv(cx.pw1, PRO_NONE, EPI_GELU, tmp, R, L, G, 4 * L);
            run_conv(cx.pw2, PRO_NONE, EPI_SCALE_RESIDUAL, G, R, 4 * L, dst, L, dst, cx.gamma);
            snap(("up" + std::to_string(j) + "_cnx").c_str(), dst, R, L);
            src = dst;
        }
        // src holds [R][L]
        run_conv(dec0, PRO_NONE, EPI_NONE, src, R, L, d, p.vocoder_decoder_dim);
        snap("dec0", d, R, p.vocoder_decoder_dim);
        const float * in = d;
        int width = p.vocoder_decoder_dim;
        for (int b = 0; b < 4; ++b) {
            Block & bk = blocks[b];
            run_conv(bk.up, PRO_SNAKE, EPI_NONE, in, R, width, P[b], bk.Cout);
            R *= bk.rate;
            width = bk.Cout;
            snap(("dec" + std::to_string(b + 1) + "_b1").c_str(), P[b], R, width);
            for (int k = 0; k < 3; ++k) {
                run_conv(bk.u[k].c1, PRO_SNAKE, EPI_NONE, P[b], R, width, Q[b], width);
                run_conv(bk.u[k].c2, PRO_SNAKE, EPI_RESIDUAL, Q[b], R, width, P[b], width, P[b]);
                snap(("dec" + std::to_string(b + 1) + "_b" + std::to_string(2 + k)).c_str(), P[b], R, width);
            }
            in = P[b];
        }
        run_conv(final_conv, PRO_SNAKE, EPI_CLAMP, in, R, width, pcm, 1);
        snap("pcm", pcm, R, 1);
        pos0 += T;
    }
};

// --- host-side weight preparation ------------------------------------------

namespace {

// Raw copy of a weight tensor in the file's own element type (f16 bits or f32).
template <class T>
std::vector<T> tensor_of(const NaviFile & f, const std::string & name, const std::vector<std::uint64_t> & dims) {
    const TensorInfo & t = f.get(name);
    const DType want = sizeof(T) == 2 ? DType::F16 : DType::F32;
    if (t.dtype != want || t.dims != dims) fail("vocoder: " + name + " has unexpected dtype/shape " + t.shape_str());
    std::vector<T> v(t.numel());
    std::memcpy(v.data(), f.data(t), t.nbytes);
    return v;
}

std::vector<float> f32_of(const NaviFile & f, const std::string & name) {
    const TensorInfo & t = f.get(name);
    if (t.dtype != DType::F32) fail("vocoder: " + name + " is not f32");
    std::vector<float> v(t.numel());
    std::memcpy(v.data(), f.data(t), t.nbytes);
    return v;
}

// ConvTranspose1d weight [Cin][Cout][k] -> [s][Cout][Cin][k/s]; within a phase,
// tap kk-1 is the current input row, tap kk-2 the previous one:
//   out[i*s + p] = sum_m W[ci][co][p + m*s] * x[i - m]
template <class T>
std::vector<T> transposed_phases(const std::vector<T> & w, int Cin, int Cout, int k, int s) {
    const int kk = k / s;
    std::vector<T> out(static_cast<std::size_t>(s) * Cout * Cin * kk);
    for (int ph = 0; ph < s; ++ph)
        for (int co = 0; co < Cout; ++co)
            for (int ci = 0; ci < Cin; ++ci)
                for (int tap = 0; tap < kk; ++tap) {
                    const int m = kk - 1 - tap;
                    out[((static_cast<std::size_t>(ph) * Cout + co) * Cin + ci) * kk + tap] =
                        w[(static_cast<std::size_t>(ci) * Cout + co) * k + ph + m * s];
                }
    return out;
}

template <class T>
std::vector<T> concat_rows(const std::vector<std::vector<T>> & parts) {
    std::vector<T> out;
    for (const auto & p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

// [Cout][Cin_a] ++ [Cout][Cin_b] along Cin
template <class T>
std::vector<T> concat_cols(const std::vector<T> & a, int ca, const std::vector<T> & b, int cb, int Cout) {
    std::vector<T> out(static_cast<std::size_t>(Cout) * (ca + cb));
    for (int co = 0; co < Cout; ++co) {
        std::memcpy(&out[static_cast<std::size_t>(co) * (ca + cb)], &a[static_cast<std::size_t>(co) * ca], ca * sizeof(T));
        std::memcpy(&out[static_cast<std::size_t>(co) * (ca + cb) + ca], &b[static_cast<std::size_t>(co) * cb], cb * sizeof(T));
    }
    return out;
}

const std::string DEC = "codec.decoder.";

bool ends_with(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

} // namespace

bool vocoder_owns_tensor(const std::string & name) {
    if (name.compare(0, DEC.size(), DEC) != 0) return false;
    const std::string rest = name.substr(DEC.size());
    if (rest.rfind("quantizer.", 0) == 0 && ends_with(rest, "output_proj.weight")) return true;
    if (rest.rfind("pre_transformer.layers.", 0) == 0 &&
        (ends_with(rest, "q_proj.weight") || ends_with(rest, "k_proj.weight") || ends_with(rest, "v_proj.weight") ||
         ends_with(rest, "gate_proj.weight") || ends_with(rest, "up_proj.weight"))) return true;
    if (rest.rfind("upsample.", 0) == 0 && ends_with(rest, ".0.conv.weight")) return true;
    if (rest.rfind("decoder.", 0) == 0 && ends_with(rest, ".block.1.conv.weight")) return true;
    // snake alpha/beta are re-uploaded as exp(alpha), 1/(exp(beta)+1e-9)
    if (ends_with(rest, ".alpha") || ends_with(rest, ".beta")) return true;
    return false;
}

std::unique_ptr<Vocoder> Vocoder::create(const Device & dev, const NaviFile & f, const DeviceWeights & w,
                                         const Params & p, int max_chunk_frames, int /*max_frames*/) {
    std::unique_ptr<Vocoder> v(new Vocoder());
    v->impl_ = std::make_unique<Impl>();
    Impl & I = *v->impl_;
    I.stream = dev.stream();
    I.p = p;
    I.Cmax = max_chunk_frames;
    const auto & vp = p.vocoder;
    const int L = p.vq_latent, VH = vp.hidden, CD = p.vq_codebook_dim, VQ = CD / 2, DD = p.vocoder_decoder_dim;
    if (p.vocoder_upsampling_ratios.size() != 2 || p.vocoder_upsample_rates.size() != 4) {
        fail("vocoder: expected 2 upsampling ratios and 4 upsample rates");
    }
    if (vp.n_head != vp.n_kv_head) fail("vocoder: GQA not supported in the pre-transformer");

    auto dw = [&](const std::string & name) { return w.get(name); };
    auto wp = [&](const std::string & name) { return dw(name).ptr; };
    I.f32 = f.get(DEC + "pre_conv.conv.weight").dtype == DType::F32;
    auto f32p = [&](const std::string & name) { return dw(name).as<float>(); };
    auto snake = [&](const std::string & prefix, Conv & c) {
        const auto alpha = f32_of(f, prefix + ".alpha"), beta = f32_of(f, prefix + ".beta");
        std::vector<float> ea(alpha.size()), ib(beta.size());
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            ea[i] = std::exp(alpha[i]);
            ib[i] = 1.f / (std::exp(beta[i]) + 1e-9f);
        }
        c.snake_a = I.upload(ea);
        c.snake_b = I.upload(ib);
    };
    auto with_hist = [&](Conv & c) {
        c.H = (c.K - 1) * c.dil;
        if (c.H > 0) {
            c.hist = I.alloc<float>(static_cast<std::size_t>(c.H) * c.Cin);
            c.hist2 = I.alloc<float>(static_cast<std::size_t>(c.H) * c.Cin);
        }
    };

    // RVQ (codebooks and projections are prepared per weight type below)
    I.cb_first = wp(DEC + "quantizer.rvq_first.vq.layers.0._codebook.codebook");
    I.rvq_proj.Cin = CD; I.rvq_proj.Cout = CD; I.rvq_proj.K = 1;
    I.pre_conv = Conv{wp(DEC + "pre_conv.conv.weight"), f32p(DEC + "pre_conv.conv.bias"), CD, L, 3, 1, 1};
    with_hist(I.pre_conv);
    I.in_proj = Conv{wp(DEC + "pre_transformer.input_proj.weight"), f32p(DEC + "pre_transformer.input_proj.bias"), L, VH, 1, 1, 1};
    I.out_proj = Conv{wp(DEC + "pre_transformer.output_proj.weight"), f32p(DEC + "pre_transformer.output_proj.bias"), VH, L, 1, 1, 1};
    I.final_norm = f32p(DEC + "pre_transformer.norm.weight");

    I.ring = 8;
    while (I.ring < p.vocoder_sliding_window + max_chunk_frames) I.ring *= 2;
    const int HD = vp.n_head * vp.head_dim;
    for (int l = 0; l < vp.n_layer; ++l) {
        const std::string lp = DEC + "pre_transformer.layers." + std::to_string(l) + ".";
        Layer ly{};
        ly.ln1 = f32p(lp + "input_layernorm.weight");
        ly.ln2 = f32p(lp + "post_attention_layernorm.weight");
        ly.attn_scale = f32p(lp + "self_attn_layer_scale.scale");
        ly.mlp_scale = f32p(lp + "mlp_layer_scale.scale");
        ly.qkv.Cin = VH; ly.qkv.Cout = 3 * HD; ly.qkv.K = 1;
        ly.o = Conv{wp(lp + "self_attn.o_proj.weight"), nullptr, HD, VH, 1, 1, 1};
        ly.gateup.Cin = VH; ly.gateup.Cout = 2 * vp.ff; ly.gateup.K = 1;
        ly.down = Conv{wp(lp + "mlp.down_proj.weight"), nullptr, vp.ff, VH, 1, 1, 1};
        ly.kc = I.alloc<float>(static_cast<std::size_t>(I.ring) * HD);
        ly.vc = I.alloc<float>(static_cast<std::size_t>(I.ring) * HD);
        I.layers.push_back(ly);
    }

    for (int j = 0; j < 2; ++j) {
        const int r = p.vocoder_upsampling_ratios[static_cast<std::size_t>(j)];
        const std::string up = DEC + "upsample." + std::to_string(j) + ".";
        Conv & c = I.up_conv[j];
        c.bias = f32p(up + "0.conv.bias");
        c.Cin = L; c.Cout = L; c.K = 1; c.phases = r;
        ConvNeXt & cx = I.cnx[j];
        cx.dw = wp(up + "1.dwconv.conv.weight");
        cx.dw_bias = f32p(up + "1.dwconv.conv.bias");
        cx.ln_w = f32p(up + "1.norm.weight");
        cx.ln_b = f32p(up + "1.norm.bias");
        cx.gamma = f32p(up + "1.gamma");
        cx.hist = I.alloc<float>(static_cast<std::size_t>(6) * L);
        cx.hist2 = I.alloc<float>(static_cast<std::size_t>(6) * L);
        cx.pw1 = Conv{wp(up + "1.pwconv1.weight"), f32p(up + "1.pwconv1.bias"), L, 4 * L, 1, 1, 1};
        cx.pw2 = Conv{wp(up + "1.pwconv2.weight"), f32p(up + "1.pwconv2.bias"), 4 * L, L, 1, 1, 1};
    }

    I.dec0 = Conv{wp(DEC + "decoder.0.conv.weight"), f32p(DEC + "decoder.0.conv.bias"), L, DD, 7, 1, 1};
    with_hist(I.dec0);
    int width = DD;
    int rows = 4;   // rows per frame after the two x2 upsamples
    for (int b = 0; b < 4; ++b) {
        Block & bk = I.blocks[b];
        bk.rate = p.vocoder_upsample_rates[static_cast<std::size_t>(b)];
        bk.Cin = width; bk.Cout = width / 2;
        const std::string bp = DEC + "decoder." + std::to_string(b + 1) + ".block.";
        bk.up.bias = f32p(bp + "1.conv.bias");
        bk.up.Cin = bk.Cin; bk.up.Cout = bk.Cout; bk.up.K = 2; bk.up.phases = bk.rate;
        with_hist(bk.up);
        snake(bp + "0", bk.up);
        const int dils[3] = {1, 3, 9};
        for (int k = 0; k < 3; ++k) {
            const std::string rp = bp + std::to_string(2 + k) + ".";
            Unit & un = bk.u[k];
            un.c1 = Conv{wp(rp + "conv1.conv.weight"), f32p(rp + "conv1.conv.bias"), bk.Cout, bk.Cout, 7, dils[k], 1};
            with_hist(un.c1);
            snake(rp + "act1", un.c1);
            un.c2 = Conv{wp(rp + "conv2.conv.weight"), f32p(rp + "conv2.conv.bias"), bk.Cout, bk.Cout, 1, 1, 1};
            snake(rp + "act2", un.c2);
        }
        rows *= bk.rate;
        I.rows[b] = rows;
        I.P[b] = I.alloc<float>(static_cast<std::size_t>(rows) * max_chunk_frames * bk.Cout);
        I.Q[b] = I.alloc<float>(static_cast<std::size_t>(rows) * max_chunk_frames * bk.Cout);
        width = bk.Cout;
    }
    if (rows != p.codec_upsample) fail("vocoder: upsample chain gives " + std::to_string(rows) + " samples/frame, config says " + std::to_string(p.codec_upsample));
    I.final_conv = Conv{wp(DEC + "decoder.6.conv.weight"), f32p(DEC + "decoder.6.conv.bias"), width, 1, 7, 1, 1};
    with_hist(I.final_conv);
    snake(DEC + "decoder.5", I.final_conv);

    // Weight-type-dependent preparation: packed codebooks, concatenated
    // projections, phase-major transposed convs. Same code for f16 and f32.
    auto u64 = [](int v) { return static_cast<std::uint64_t>(v); };
    auto prep = [&]<class T>() {
        std::vector<std::vector<T>> books;
        for (int q = 0; q < p.vq_n_q - p.vq_n_semantic; ++q) {
            books.push_back(tensor_of<T>(f, DEC + "quantizer.rvq_rest.vq.layers." + std::to_string(q) + "._codebook.codebook",
                                         {u64(p.vq_codebook_size), u64(VQ)}));
        }
        I.cb_rest = I.upload(concat_rows<T>(books));
        const auto wf = tensor_of<T>(f, DEC + "quantizer.rvq_first.output_proj.weight", {u64(CD), u64(VQ), 1});
        const auto wr = tensor_of<T>(f, DEC + "quantizer.rvq_rest.output_proj.weight", {u64(CD), u64(VQ), 1});
        I.rvq_proj.W = I.upload(concat_cols<T>(wf, VQ, wr, VQ, CD));
        for (int l = 0; l < vp.n_layer; ++l) {
            const std::string lp = DEC + "pre_transformer.layers." + std::to_string(l) + ".";
            const std::vector<std::uint64_t> qkv_dims{u64(HD), u64(VH)}, ff_dims{u64(vp.ff), u64(VH)};
            I.layers[static_cast<std::size_t>(l)].qkv.W = I.upload(concat_rows<T>({
                tensor_of<T>(f, lp + "self_attn.q_proj.weight", qkv_dims),
                tensor_of<T>(f, lp + "self_attn.k_proj.weight", qkv_dims),
                tensor_of<T>(f, lp + "self_attn.v_proj.weight", qkv_dims)}));
            I.layers[static_cast<std::size_t>(l)].gateup.W = I.upload(concat_rows<T>({
                tensor_of<T>(f, lp + "mlp.gate_proj.weight", ff_dims),
                tensor_of<T>(f, lp + "mlp.up_proj.weight", ff_dims)}));
        }
        for (int j = 0; j < 2; ++j) {
            const int r = p.vocoder_upsampling_ratios[static_cast<std::size_t>(j)];
            const auto wt = tensor_of<T>(f, DEC + "upsample." + std::to_string(j) + ".0.conv.weight", {u64(L), u64(L), u64(r)});
            I.up_conv[j].W = I.upload(transposed_phases<T>(wt, L, L, r, r));
        }
        for (int b = 0; b < 4; ++b) {
            Block & bk = I.blocks[b];
            const auto wt = tensor_of<T>(f, DEC + "decoder." + std::to_string(b + 1) + ".block.1.conv.weight",
                                         {u64(bk.Cin), u64(bk.Cout), u64(2 * bk.rate)});
            bk.up.W = I.upload(transposed_phases<T>(wt, bk.Cin, bk.Cout, 2 * bk.rate, bk.rate));
        }
    };
    if (I.f32) prep.template operator()<float>(); else prep.template operator()<unsigned short>();

    const std::size_t C = static_cast<std::size_t>(max_chunk_frames);
    I.d_codes = I.alloc<int>(C * p.vq_n_q);
    I.e = I.alloc<float>(C * CD); I.h = I.alloc<float>(C * CD); I.a = I.alloc<float>(C * L);
    I.x = I.alloc<float>(C * VH); I.n = I.alloc<float>(C * VH); I.qkv = I.alloc<float>(C * 3 * HD);
    I.att = I.alloc<float>(C * HD); I.gu = I.alloc<float>(C * 2 * vp.ff); I.u = I.alloc<float>(C * L);
    I.A = I.alloc<float>(4 * C * L); I.B = I.alloc<float>(4 * C * L); I.G = I.alloc<float>(4 * C * 4 * L);
    I.d = I.alloc<float>(4 * C * DD);
    I.pcm = I.alloc<float>(C * p.codec_upsample);
    I.h_pcm.resize(C * p.codec_upsample);
    NAVI_HIP(hipEventCreate(&I.ev0));
    NAVI_HIP(hipEventCreate(&I.ev1));
    v->reset();
    return v;
}

Vocoder::~Vocoder() {
    if (!impl_) return;
    if (impl_->ev0) (void) hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void) hipEventDestroy(impl_->ev1);
    for (void * p : impl_->owned) (void) hipFree(p);
}

void Vocoder::reset() {
    Impl & I = *impl_;
    I.pos0 = 0;
    auto zero = [&](float * ptr, std::size_t n) { if (ptr) NAVI_HIP(hipMemsetAsync(ptr, 0, n * sizeof(float), I.stream)); };
    auto zero_conv = [&](Conv & c) { zero(c.hist, static_cast<std::size_t>(c.H) * c.Cin); zero(c.hist2, static_cast<std::size_t>(c.H) * c.Cin); };
    zero_conv(I.pre_conv);
    zero_conv(I.dec0);
    zero_conv(I.final_conv);
    for (int j = 0; j < 2; ++j) { zero(I.cnx[j].hist, 6 * static_cast<std::size_t>(I.p.vq_latent)); zero(I.cnx[j].hist2, 6 * static_cast<std::size_t>(I.p.vq_latent)); }
    for (Block & b : I.blocks) { zero_conv(b.up); for (Unit & u : b.u) zero_conv(u.c1); }
    // KV rings need no clearing: attention only reads positions <= pos, all rewritten before use.
}

void Vocoder::decode(std::span<const std::int32_t> codes, int n_frames, std::vector<float> & out) {
    Impl & I = *impl_;
    if (static_cast<int>(codes.size()) < n_frames * I.p.vq_n_q) fail("vocoder: codes shorter than n_frames x n_q");
    NAVI_HIP(hipEventRecord(I.ev0, I.stream));
    for (int f0 = 0; f0 < n_frames; f0 += I.Cmax) {
        const int T = std::min(I.Cmax, n_frames - f0);
        NAVI_HIP(hipMemcpyAsync(I.d_codes, codes.data() + static_cast<std::size_t>(f0) * I.p.vq_n_q,
                                static_cast<std::size_t>(T) * I.p.vq_n_q * sizeof(int), hipMemcpyHostToDevice, I.stream));
        I.decode_chunk(T);
        const std::size_t ns = static_cast<std::size_t>(T) * I.p.codec_upsample;
        NAVI_HIP(hipMemcpyAsync(I.h_pcm.data(), I.pcm, ns * sizeof(float), hipMemcpyDeviceToHost, I.stream));
        NAVI_HIP(hipStreamSynchronize(I.stream));
        out.insert(out.end(), I.h_pcm.begin(), I.h_pcm.begin() + static_cast<std::ptrdiff_t>(ns));
    }
    NAVI_HIP(hipEventRecord(I.ev1, I.stream));
    NAVI_HIP(hipEventSynchronize(I.ev1));
    float ms = 0.f;
    NAVI_HIP(hipEventElapsedTime(&ms, I.ev0, I.ev1));
    I.last_ms = ms;
}

void Vocoder::set_stage_hook(StageHook hook) { impl_->hook = std::move(hook); }

int Vocoder::samples_per_frame() const { return impl_->p.codec_upsample; }
int Vocoder::max_chunk_frames() const { return impl_->Cmax; }
double Vocoder::last_ms() const { return impl_->last_ms; }

} // namespace navi::qwen3tts
