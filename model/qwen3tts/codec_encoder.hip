#include "model/qwen3tts/codec_encoder.h"

#include "model/qwen3tts/kernels.h"
#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace navi::qwen3tts {

namespace {

using namespace navi::qwen3tts::kern;

// Attention here is full causal over the whole clip, so a block keeps one score per key
// position in LDS. 1024 keys at 25 Hz is 40.96 s, past the 30 s clip cap; create() checks
// the clip against this rather than trusting the cap.
constexpr int ENC_MAX_KEYS = 1024;
constexpr int NT = 256;   // threads per block for the plain kernels below

// ---------------------------------------------------------------------------
// A strided causal conv - the one shape k_conv cannot do. Output row o reads input rows
// o * stride - pad_left + j for j < K, out of `hist` below zero and out of the right
// padding above T_in. The encoder's four downsampling convs and its 25 -> 12.5 Hz
// downsample are the only users; they run once per clone, and k_conv's LDS window would be
// 271 rows at stride 8. So this is the plain form: one thread per output channel, looping
// over the taps and the input channels, with the activation reads broadcasting across the
// block because every thread in it shares one window.
// ---------------------------------------------------------------------------
struct DownArgs {
    const float * W;        // [Cout][Cin][K] f32
    const float * bias;     // [Cout] or null
    const float * x;        // [T_in][Cin]
    const float * hist;     // [pad_left][Cin] or null
    int T_in, Cin, Cout, K, stride, pad_left, pad_right;
    int replicate;          // rows outside the input repeat the nearest one instead of being zero
    float * y;              // [T_out][Cout]
    int T_out;
};

template <int PRO>
__global__ void k_conv_down(DownArgs a) {
    const int o = blockIdx.x;
    if (o >= a.T_out) return;
    const int t0 = o * a.stride - a.pad_left;
    for (int co = threadIdx.x; co < a.Cout; co += blockDim.x) {
        float acc = a.bias ? a.bias[co] : 0.f;
        for (int ci = 0; ci < a.Cin; ++ci) {
            const float * w = a.W + (static_cast<std::size_t>(co) * a.Cin + ci) * a.K;
            for (int j = 0; j < a.K; ++j) {
                const int t = t0 + j;
                float v;
                if (t >= 0 && t < a.T_in) {
                    v = a.x[static_cast<std::size_t>(t) * a.Cin + ci];
                } else if (t < 0) {
                    if (a.hist) v = a.hist[static_cast<std::size_t>(a.pad_left + t) * a.Cin + ci];
                    else if (a.replicate) v = a.x[ci];                 // nothing before the clip
                    else v = 0.f;
                } else if (a.replicate) {
                    v = a.x[static_cast<std::size_t>(a.T_in - 1) * a.Cin + ci];
                } else {
                    v = 0.f;
                }
                if (PRO == PRO_ELU) v = v > 0.f ? v : __expf(v) - 1.f;
                acc += w[j] * v;
            }
        }
        a.y[static_cast<std::size_t>(o) * a.Cout + co] = acc;
    }
}

// Keep the last `n` rows of [previous history | this chunk] for the next chunk. Two
// buffers, swapped, because the source can be the buffer the next chunk overwrites.
__global__ void k_keep_tail(const float * x, int T_in, int Cin, int n, const float * prev, float * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * Cin) return;
    const int r = i / Cin, c = i % Cin;
    const int t = T_in - n + r;         // negative while the chunk is shorter than the history
    out[i] = t >= 0 ? x[static_cast<std::size_t>(t) * Cin + c]
                    : (prev ? prev[static_cast<std::size_t>(t + n) * Cin + c] : 0.f);
}

// LayerNorm over the channel axis, with bias - the encoder transformer's norm, and the only
// LayerNorm outside the vocoder's ConvNeXt blocks. Biased variance, as torch's.
__global__ void k_layernorm(const float * x, int C, const float * w, const float * b, float eps, float * y) {
    __shared__ float red[32];
    const float * row = x + static_cast<std::size_t>(blockIdx.x) * C;
    float * out = y + static_cast<std::size_t>(blockIdx.x) * C;
    float s = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) s += row[i];
    const float mean = dev::block_sum(s, red) / static_cast<float>(C);
    float v = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float d = row[i] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(dev::block_sum(v, red) / static_cast<float>(C) + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) out[i] = (row[i] - mean) * inv * w[i] + b[i];
}

// RoPE on q and k in place, rotate-half, position = row index (a clip starts at 0).
__global__ void k_rope(float * qkv, int n_head, int head_dim, double theta) {
    const int t = blockIdx.x;
    const int HD = n_head * head_dim, half = head_dim / 2;
    float * row = qkv + static_cast<std::size_t>(t) * 3 * HD;
    for (int i = threadIdx.x; i < n_head * half; i += blockDim.x) {
        const int h = i / half, p = i % half;
        const float inv_freq = static_cast<float>(1.0 / pow(theta, (2.0 * p) / head_dim));
        const float ang = static_cast<float>(t) * inv_freq;
        const float cs = cosf(ang), sn = sinf(ang);
        float * q = row + h * head_dim;
        float * k = row + HD + h * head_dim;
        const float q0 = q[p], q1 = q[p + half];
        q[p] = q0 * cs - q1 * sn;
        q[p + half] = q1 * cs + q0 * sn;
        const float k0 = k[p], k1 = k[p + half];
        k[p] = k0 * cs - k1 * sn;
        k[p + half] = k1 * cs + k0 * sn;
    }
}

// Full causal attention over the whole clip: row t attends over keys 0..t. The config
// carries sliding_window = 250 and it is NOT applied - transformers 4.57 ignores it for
// this model (measured: changing it moves the output by exactly 0, and a windowed form
// differs from the reference by 1.7e-2 rel-rms). One block per (row, head).
__global__ void k_attn_causal(const float * qkv, int n_head, int head_dim, float * out) {
    __shared__ float sc[ENC_MAX_KEYS];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, n = t + 1;
    const int HD = n_head * head_dim;
    const float * q = qkv + static_cast<std::size_t>(t) * 3 * HD + h * head_dim;
    const float scaling = rsqrtf(static_cast<float>(head_dim));

    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float * k = qkv + static_cast<std::size_t>(j) * 3 * HD + HD + h * head_dim;
        float d = 0.f;
        for (int i = 0; i < head_dim; ++i) d += q[i] * k[i];
        d *= scaling;
        sc[j] = d;
        m = fmaxf(m, d);
    }
    m = dev::block_max(m, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float ex = __expf(sc[j] - m);
        sc[j] = ex;
        sum += ex;
    }
    sum = dev::block_sum(sum, red);
    __syncthreads();
    const float rsum = 1.f / sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            acc += sc[j] * qkv[static_cast<std::size_t>(j) * 3 * HD + 2 * HD + h * head_dim + d];
        }
        out[static_cast<std::size_t>(t) * HD + h * head_dim + d] = acc * rsum;
    }
}

// Split RVQ encode, one block per frame: the semantic group's single codebook over its own
// projection, then the acoustic group's 15 over *its* projection - the acoustic group
// quantises the full latent, not the semantic residual - each subtracting the centroid it
// chose before the next one looks.
//
// The distance is the plain sum of squared differences rather than the cheaper
// ||c||^2 - 2x.c: this is a nearest-centroid search whose outcome can hinge on a 1e-5
// relative gap (tests/reference/README.md), so it is computed the way torch.cdist does.
// Ties go to the lowest index, as argmin does.
__global__ void k_rvq_encode(const float * lat_s, const float * lat_a, int dim, const float * books,
                             int n_codes, int n_books, int n_semantic, int * out_codes) {
    extern __shared__ float lds[];
    float * res = lds;                                      // [dim] the residual
    float * bval = lds + dim;                               // [blockDim.x]
    int * bidx = reinterpret_cast<int *>(bval + blockDim.x); // [blockDim.x]
    const int t = blockIdx.x;

    for (int b = 0; b < n_books; ++b) {
        if (b == 0 || b == n_semantic) {                    // each group starts from its projection
            const float * src = b < n_semantic ? lat_s : lat_a;
            for (int i = threadIdx.x; i < dim; i += blockDim.x) {
                res[i] = src[static_cast<std::size_t>(t) * dim + i];
            }
        }
        __syncthreads();
        const float * book = books + static_cast<std::size_t>(b) * n_codes * dim;
        float best = INFINITY;
        int best_i = 0;
        for (int c = threadIdx.x; c < n_codes; c += blockDim.x) {
            const float * row = book + static_cast<std::size_t>(c) * dim;
            float d = 0.f;
            for (int i = 0; i < dim; ++i) {
                const float e = res[i] - row[i];
                d += e * e;
            }
            if (d < best) { best = d; best_i = c; }         // c rises, so this keeps the first minimum
        }
        bval[threadIdx.x] = best;
        bidx[threadIdx.x] = best_i;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 1; i < static_cast<int>(blockDim.x); ++i) {
                if (bval[i] < best || (bval[i] == best && bidx[i] < best_i)) { best = bval[i]; best_i = bidx[i]; }
            }
            bidx[0] = best_i;
            out_codes[static_cast<std::size_t>(t) * n_books + b] = best_i;
        }
        __syncthreads();
        if (b + 1 < n_books && b + 1 != n_semantic) {
            const float * row = book + static_cast<std::size_t>(bidx[0]) * dim;
            for (int i = threadIdx.x; i < dim; i += blockDim.x) res[i] -= row[i];
        }
        __syncthreads();
    }
}

// Mimi's causal padding arithmetic (MimiConv1d): the left pad is kernel - stride, and the
// right pad is whatever makes the stride divide the length. Computed per conv from the FULL
// sequence length, which is what the reference does and what makes chunking exact - a chunk
// boundary at a multiple of the stack's total stride leaves the right pad where it belongs,
// at the end of the last chunk. Dilation is 1 throughout this stack.
struct Pad {
    int left = 0, right = 0, out = 0;
};

Pad causal_pad(int T_in, int K, int stride) {
    const int total = K - stride;
    const int num = T_in - K + total;
    const int n = num >= 0 ? (num + stride - 1) / stride : -((-num) / stride);   // ceil, signed
    Pad p;
    p.left = total;
    p.right = std::max(0, n * stride + K - total - T_in);
    p.out = (T_in + p.left + p.right - K) / stride + 1;
    return p;
}

struct Conv {
    const float * W = nullptr;
    const float * bias = nullptr;
    int Cin = 0, Cout = 0, K = 1, stride = 1;
    int replicate = 0;                              // only the 25 -> 12.5 Hz downsample
    float * hist = nullptr, * hist2 = nullptr;      // [pad.left][Cin], double-buffered
    Pad pad;                                        // from the clip's length, per encode()
};

struct XfLayer {
    const float * ln1_w = nullptr, * ln1_b = nullptr, * ln2_w = nullptr, * ln2_b = nullptr;
    const float * attn_scale = nullptr, * mlp_scale = nullptr;
    Conv q, k, v, o, fc1, fc2;                      // K = 1 convs are plain GEMMs
};

// One chunk of the conv stack's time axis.
struct Slice {
    int off = 0, len = 0;
    bool last = false;
};

std::vector<Slice> slices(int T, int chunk) {
    std::vector<Slice> out;
    for (int off = 0; off < T; off += chunk) {
        Slice s;
        s.off = off;
        s.len = std::min(chunk, T - off);
        s.last = off + s.len >= T;
        out.push_back(s);
    }
    return out;
}

} // namespace

struct CodecEncoder::Impl {
    hipStream_t stream = nullptr;
    Params p;
    Params::CodecEncoder e;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    double last_ms = 0;
    CodecEncoder::StageHook hook;
    int chunk = 0, default_chunk = 0;
    int max_samples = 0, max_rows25 = 0, max_frames = 0;
    std::vector<void *> owned;

    // the conv stack in execution order
    struct Unit { Conv c1, c2; };
    struct Level { Unit unit; Conv down; int Cin = 0, Cout = 0, stride = 0; };
    Conv in_conv;
    std::vector<Level> levels;
    Conv out_conv;
    std::vector<XfLayer> xf;
    Conv down;                        // 25 Hz -> 12.5 Hz, the replicate-padded one
    Conv proj_s, proj_a;              // the quantizer groups' input projections
    const float * books = nullptr;    // [n_books][n_codes][dim], semantic first

    // scratch
    std::vector<float *> a, b, t1;    // per level: input/output and the unit's inner buffer
    float * d_wav = nullptr, * x25 = nullptr, * norm = nullptr, * qkv = nullptr, * att = nullptr, * mlp = nullptr;
    float * lat = nullptr, * ls = nullptr, * la = nullptr;
    int * d_codes = nullptr;

    template <class T> T * alloc(std::size_t n) {
        void * ptr = nullptr;
        NAVI_HIP(hipMalloc(&ptr, n * sizeof(T)));
        NAVI_HIP(hipMemset(ptr, 0, n * sizeof(T)));
        owned.push_back(ptr);
        return static_cast<T *>(ptr);
    }

    void snap(const char * name, const float * buf, int rows, int cols) {
        if (!hook) return;
        NAVI_HIP(hipStreamSynchronize(stream));
        hook(name, buf, rows, cols);
    }

    // `pro` is applied as rows are staged, history included, so the history buffers hold
    // pre-activation rows (kernels.h).
    // `y_stride` is 0 for a plain [T][Cout] output; the transformer's q/k/v write into one
    // [T][3 * HD] row each, which is the only caller that needs it.
    void run_conv(const Conv & c, int pro, int epi, const float * x, int T_in, int T_out, int pad_right,
                  float * y, const float * residual = nullptr, const float * scale = nullptr,
                  int y_stride = 0) {
        if (c.stride > 1 || c.replicate) {
            DownArgs d{};
            d.W = c.W; d.bias = c.bias; d.x = x; d.hist = c.hist;
            d.T_in = T_in; d.Cin = c.Cin; d.Cout = c.Cout; d.K = c.K; d.stride = c.stride;
            d.pad_left = c.pad.left; d.pad_right = pad_right; d.replicate = c.replicate;
            d.y = y; d.T_out = T_out;
            if (y_stride && y_stride != c.Cout) fail("codec encoder: a strided conv cannot pack its output");
            if (pro == PRO_ELU) k_conv_down<PRO_ELU><<<static_cast<unsigned>(T_out), NT, 0, stream>>>(d);
            else if (pro == PRO_NONE) k_conv_down<PRO_NONE><<<static_cast<unsigned>(T_out), NT, 0, stream>>>(d);
            else fail("codec encoder: k_conv_down has no instantiation for pro=" + std::to_string(pro));
            NAVI_HIP(hipGetLastError());
            return;
        }
        ConvArgs ar{};
        ar.W = c.W; ar.bias = c.bias; ar.x = x; ar.hist = c.hist;
        ar.T_in = T_in; ar.Cin = c.Cin; ar.Cout = c.Cout; ar.dil = 1; ar.x_stride = c.Cin;
        ar.phases = 1; ar.pad_left = c.pad.left; ar.reflect = 0;
        ar.y = y; ar.y_stride = y_stride ? y_stride : c.Cout; ar.residual = residual; ar.scale = scale;
        // k_conv writes one output row per input row, so these two must agree; only the
        // strided path above produces a different count.
        if (T_out != T_in) fail("codec encoder: a stride-1 conv was asked for a different row count");
        const dim3 grid(static_cast<unsigned>((T_in + TT - 1) / TT), static_cast<unsigned>((c.Cout + CT - 1) / CT), 1);
#define NAVI_ENC_CONV(KK, PP, EE) \
        if (c.K == KK && pro == PP && epi == EE) { \
            k_conv<float, KK, PP, EE, 1><<<grid, dim3(NTHREADS), 0, stream>>>(ar); goto launched; }
        NAVI_ENC_CONV(1, PRO_NONE, EPI_NONE)
        NAVI_ENC_CONV(1, PRO_NONE, EPI_GELU)
        NAVI_ENC_CONV(1, PRO_NONE, EPI_SCALE_RESIDUAL)
        NAVI_ENC_CONV(1, PRO_ELU, EPI_RESIDUAL)
        NAVI_ENC_CONV(3, PRO_ELU, EPI_NONE)
        NAVI_ENC_CONV(3, PRO_NONE, EPI_NONE)
        NAVI_ENC_CONV(7, PRO_NONE, EPI_NONE)
#undef NAVI_ENC_CONV
        fail("codec encoder: no conv instantiation for K=" + std::to_string(c.K) + " pro=" + std::to_string(pro)
             + " epi=" + std::to_string(epi));
    launched:
        NAVI_HIP(hipGetLastError());
    }

    void keep_tail(Conv & c, const float * x, int T_in) {
        if (c.pad.left <= 0 || !c.hist) return;
        const int n = c.pad.left * c.Cin;
        k_keep_tail<<<static_cast<unsigned>((n + NT - 1) / NT), NT, 0, stream>>>(x, T_in, c.Cin, c.pad.left,
                                                                                c.hist, c.hist2);
        NAVI_HIP(hipGetLastError());
        std::swap(c.hist, c.hist2);
    }

    void clear_hist(Conv & c) {
        if (!c.hist) return;
        const std::size_t n = static_cast<std::size_t>(c.pad.left) * c.Cin * sizeof(float);
        NAVI_HIP(hipMemsetAsync(c.hist, 0, n, stream));
        NAVI_HIP(hipMemsetAsync(c.hist2, 0, n, stream));
    }

    // Every stage's full-sequence row count and padding, from the clip's length. Nothing
    // here assumes a relation between the rates: 25 Hz rows and 12.5 Hz frames differ by a
    // ceil, not by a factor of two (375 rows -> 188 frames).
    void plan(int T_samples) {
        in_conv.pad = causal_pad(T_samples, in_conv.K, 1);
        int T = in_conv.pad.out;
        for (Level & l : levels) {
            l.unit.c1.pad = causal_pad(T, l.unit.c1.K, 1);
            l.unit.c2.pad = causal_pad(T, l.unit.c2.K, 1);
            l.down.pad = causal_pad(T, l.down.K, l.down.stride);
            T = l.down.pad.out;
        }
        out_conv.pad = causal_pad(T, out_conv.K, 1);
        down.pad = causal_pad(out_conv.pad.out, down.K, down.stride);
    }
};

std::unique_ptr<CodecEncoder> CodecEncoder::create(const Device & dev, const NaviFile & file,
                                                   const DeviceWeights & w, const Params & p,
                                                   double max_seconds, int chunk_frames) {
    (void) file;
    if (!p.has_codec_encoder) {
        fail("this model has no codec encoder, so it cannot make reference codes for in-context "
             "cloning (tools/convert.py --with-encoder adds it)");
    }
    std::unique_ptr<CodecEncoder> ce(new CodecEncoder());
    ce->impl_ = std::make_unique<Impl>();
    Impl & I = *ce->impl_;
    I.stream = dev.stream();
    I.p = p;
    I.e = p.codec_encoder;
    const Params::CodecEncoder & e = I.e;
    const int L = e.latent, HD = e.n_head * e.head_dim;
    const std::string pre = "codec.encoder.";
    // Every kernel here reads f32, and the converter stores this component f32 for a
    // measured reason (docs/model.md "Codec encoder"). A blob whose encoder is some other
    // dtype is refused by name rather than reinterpreted: `as<float>()` only casts the
    // pointer, so an f16 tensor would be read as half as many wrong values and off its end.
    auto f32 = [&](const std::string & n) {
        const DeviceTensor & t = w.get(n);
        if (t.info.dtype != DType::F32) {
            fail("codec encoder: " + n + " is " + std::string(dtype_name(t.info.dtype))
                 + " and every kernel in this component reads f32; convert with --with-encoder, which "
                 "stores it f32 (tools/convert.py says why)");
        }
        return t.as<float>();
    };

    // the conv stack, at the layer indices params.cpp lists
    int idx = 0, dim = e.num_filters;
    I.in_conv.W = f32(pre + "encoder.layers.0.conv.weight");
    I.in_conv.bias = f32(pre + "encoder.layers.0.conv.bias");
    I.in_conv.Cin = 1; I.in_conv.Cout = dim; I.in_conv.K = e.kernel_size;
    ++idx;
    for (std::size_t r = e.ratios.size(); r-- > 0;) {        // the encoder walks them reversed
        Impl::Level l;
        l.Cin = dim;
        l.stride = e.ratios[r];
        const int inner = dim / e.compress;
        const std::string b = pre + "encoder.layers." + std::to_string(idx) + ".block.";
        l.unit.c1.W = f32(b + "1.conv.weight"); l.unit.c1.bias = f32(b + "1.conv.bias");
        l.unit.c1.Cin = dim; l.unit.c1.Cout = inner; l.unit.c1.K = e.residual_kernel_size;
        l.unit.c2.W = f32(b + "3.conv.weight"); l.unit.c2.bias = f32(b + "3.conv.bias");
        l.unit.c2.Cin = inner; l.unit.c2.Cout = dim; l.unit.c2.K = 1;
        idx += 2;                                            // the unit, then an ELU with no tensors
        const std::string c = pre + "encoder.layers." + std::to_string(idx) + ".conv.";
        l.down.W = f32(c + "weight"); l.down.bias = f32(c + "bias");
        l.down.Cin = dim; l.down.Cout = dim * 2; l.down.K = l.stride * 2; l.down.stride = l.stride;
        ++idx;
        dim *= 2;
        l.Cout = dim;
        I.levels.push_back(l);
    }
    ++idx;                                                   // the last ELU
    const std::string last = pre + "encoder.layers." + std::to_string(idx) + ".conv.";
    I.out_conv.W = f32(last + "weight"); I.out_conv.bias = f32(last + "bias");
    I.out_conv.Cin = dim; I.out_conv.Cout = L; I.out_conv.K = e.last_kernel_size;

    for (int l = 0; l < e.n_layer; ++l) {
        const std::string t = pre + "encoder_transformer.layers." + std::to_string(l) + ".";
        XfLayer x{};
        x.ln1_w = f32(t + "input_layernorm.weight");
        x.ln1_b = f32(t + "input_layernorm.bias");
        x.ln2_w = f32(t + "post_attention_layernorm.weight");
        x.ln2_b = f32(t + "post_attention_layernorm.bias");
        x.attn_scale = f32(t + "self_attn_layer_scale.scale");
        x.mlp_scale = f32(t + "mlp_layer_scale.scale");
        const auto lin = [&](const char * n, int Cin, int Cout) {
            Conv c;
            c.W = f32(t + n);
            c.Cin = Cin; c.Cout = Cout; c.K = 1;
            return c;
        };
        x.q = lin("self_attn.q_proj.weight", L, HD);
        x.k = lin("self_attn.k_proj.weight", L, HD);
        x.v = lin("self_attn.v_proj.weight", L, HD);
        x.o = lin("self_attn.o_proj.weight", HD, L);
        x.fc1 = lin("mlp.fc1.weight", L, e.ff);
        x.fc2 = lin("mlp.fc2.weight", e.ff, L);
        I.xf.push_back(x);
    }

    I.down.W = f32(pre + "downsample.conv.weight");
    I.down.Cin = L; I.down.Cout = L;
    I.down.stride = e.downsample_rate;
    for (int r : e.ratios) I.down.stride /= r;
    I.down.K = I.down.stride * 2;
    I.down.replicate = 1;        // the one conv in this model that pads by repeating a row

    const std::string sem = pre + "quantizer.semantic_residual_vector_quantizer.";
    const std::string aco = pre + "quantizer.acoustic_residual_vector_quantizer.";
    I.proj_s.W = f32(sem + "input_proj.weight"); I.proj_s.Cin = L; I.proj_s.Cout = e.vq_dim; I.proj_s.K = 1;
    I.proj_a.W = f32(aco + "input_proj.weight"); I.proj_a.Cin = L; I.proj_a.Cout = e.vq_dim; I.proj_a.K = 1;

    // The codebooks in one buffer, semantic first, so the RVQ kernel indexes them by book.
    // They are adjacent in the arena in that order as the converter writes them; copied if
    // a future layout is not.
    {
        std::vector<const float *> rows;
        rows.push_back(f32(sem + "layers.0.codebook.codebook"));
        for (int i = 0; i + 1 < e.n_valid_q; ++i) {
            rows.push_back(f32(aco + "layers." + std::to_string(i) + ".codebook.codebook"));
        }
        const std::size_t per = static_cast<std::size_t>(e.codebook_size) * e.codebook_dim;
        bool adjacent = true;
        for (std::size_t i = 1; i < rows.size(); ++i) adjacent = adjacent && rows[i] == rows[i - 1] + per;
        if (adjacent) {
            I.books = rows[0];
        } else {   // d_codebooks() is how a test finds this buffer instead of the arena

            float * dst = I.alloc<float>(per * rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                NAVI_HIP(hipMemcpy(dst + i * per, rows[i], per * sizeof(float), hipMemcpyDeviceToDevice));
            }
            I.books = dst;
        }
    }

    // --- scratch, for the longest clip this encoder accepts ------------------
    I.max_samples = static_cast<int>(max_seconds * e.sample_rate);
    I.max_samples -= I.max_samples % e.downsample_rate;                  // whole frames
    if (I.max_samples < e.downsample_rate) fail("codec encoder: max_seconds is shorter than one frame");
    I.max_frames = I.max_samples / e.downsample_rate;
    I.plan(I.max_samples);
    I.max_rows25 = I.out_conv.pad.out;
    if (I.max_rows25 > ENC_MAX_KEYS) {
        fail("codec encoder: " + std::to_string(I.max_rows25) + " transformer positions for "
             + std::to_string(max_seconds) + " s is past the " + std::to_string(ENC_MAX_KEYS)
             + " its attention keeps scores for");
    }
    // A chunk is a whole number of frames, so every stage's chunk length is a whole number
    // of rows at every rate.
    if (chunk_frames < 1) fail("codec encoder: chunk_frames must be at least 1");
    I.default_chunk = std::min(chunk_frames, I.max_frames);
    const int chunk_rows = I.default_chunk * e.downsample_rate;

    I.d_wav = I.alloc<float>(static_cast<std::size_t>(I.max_samples));
    {
        int rows = chunk_rows, width = e.num_filters;
        for (std::size_t i = 0; i <= I.levels.size(); ++i) {
            I.a.push_back(I.alloc<float>(static_cast<std::size_t>(rows) * width));
            if (i < I.levels.size()) {
                I.b.push_back(I.alloc<float>(static_cast<std::size_t>(rows) * width));
                I.t1.push_back(I.alloc<float>(static_cast<std::size_t>(rows) * (width / e.compress)));
                rows /= I.levels[i].stride;
                width = I.levels[i].Cout;
            }
        }
    }
    const auto R = static_cast<std::size_t>(I.max_rows25);
    I.x25 = I.alloc<float>(R * L);
    I.norm = I.alloc<float>(R * L);
    I.qkv = I.alloc<float>(R * 3 * HD);
    I.att = I.alloc<float>(R * HD);
    I.mlp = I.alloc<float>(R * static_cast<std::size_t>(e.ff));
    I.lat = I.alloc<float>(static_cast<std::size_t>(I.max_frames) * L);
    I.ls = I.alloc<float>(static_cast<std::size_t>(I.max_frames) * e.vq_dim);
    I.la = I.alloc<float>(static_cast<std::size_t>(I.max_frames) * e.vq_dim);
    I.d_codes = I.alloc<int>(static_cast<std::size_t>(I.max_frames) * e.n_valid_q);

    const auto hist = [&](Conv & c) {
        if (c.pad.left <= 0) return;
        const std::size_t n = static_cast<std::size_t>(c.pad.left) * c.Cin;
        c.hist = I.alloc<float>(n);
        c.hist2 = I.alloc<float>(n);
    };
    hist(I.in_conv);
    for (Impl::Level & l : I.levels) { hist(l.unit.c1); hist(l.unit.c2); hist(l.down); }
    hist(I.out_conv);
    // `down` runs over the whole 25 Hz sequence, never chunked, so it has no history: its
    // left padding repeats row 0, which is what replicate padding means at a sequence start.

    const std::size_t rvq_lds = static_cast<std::size_t>(e.vq_dim) * sizeof(float)
                                + NT * sizeof(float) + NT * sizeof(int);
    if (rvq_lds > dev.info().lds_per_block) fail("codec encoder: the RVQ step needs more LDS than this device has");
    NAVI_HIP(hipEventCreate(&I.ev0));
    NAVI_HIP(hipEventCreate(&I.ev1));
    return ce;
}

CodecEncoder::~CodecEncoder() {
    if (!impl_) return;
    for (void * ptr : impl_->owned) (void) hipFree(ptr);
    if (impl_->ev0) (void) hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void) hipEventDestroy(impl_->ev1);
}

int CodecEncoder::max_samples() const { return impl_->max_samples; }
int CodecEncoder::codebooks() const { return impl_->e.n_valid_q; }
const float * CodecEncoder::d_codebooks() const { return impl_->books; }
double CodecEncoder::last_ms() const { return impl_->last_ms; }
void CodecEncoder::set_stage_hook(StageHook hook) { impl_->hook = std::move(hook); }
void CodecEncoder::set_chunk_frames(int frames) {
    if (frames > impl_->default_chunk) {
        fail("codec encoder: this encoder's scratch is sized for " + std::to_string(impl_->default_chunk)
             + "-frame chunks, not " + std::to_string(frames) + " (create() takes the size)");
    }
    impl_->chunk = frames < 0 ? 0 : frames;
}
int CodecEncoder::chunk_frames() const { return impl_->chunk > 0 ? impl_->chunk : impl_->default_chunk; }

int CodecEncoder::frames_for(int n_samples) const {
    const int d = impl_->e.downsample_rate;
    return (n_samples + d - 1) / d;
}

std::vector<std::int32_t> CodecEncoder::encode(std::span<const float> wav, int * n_frames_out) {
    Impl & I = *impl_;
    const Params::CodecEncoder & e = I.e;
    const int L = e.latent, HD = e.n_head * e.head_dim, G = e.n_valid_q;
    const int T = static_cast<int>(wav.size());
    if (T <= 0) fail("codec encoder: empty audio");
    if (T > I.max_samples) {
        fail("codec encoder: " + std::to_string(T) + " samples is longer than the "
             + std::to_string(I.max_samples) + " (" + std::to_string(I.max_samples / e.sample_rate)
             + " s) this encoder was built for");
    }
    for (float v : wav) {
        if (!std::isfinite(v)) fail("codec encoder: the audio is not finite");
    }
    const int frames = frames_for(T);
    I.plan(T);
    const int T25 = I.out_conv.pad.out;
    if (I.down.pad.out != frames) {
        fail("codec encoder: the stage lengths give " + std::to_string(I.down.pad.out)
             + " frames where the frame rate says " + std::to_string(frames));
    }
    if (T25 > ENC_MAX_KEYS) {
        fail("codec encoder: " + std::to_string(T25) + " transformer positions is past the "
             + std::to_string(ENC_MAX_KEYS) + " its attention keeps scores for");
    }

    NAVI_HIP(hipEventRecord(I.ev0, I.stream));
    NAVI_HIP(hipMemcpyAsync(I.d_wav, wav.data(), static_cast<std::size_t>(T) * sizeof(float),
                            hipMemcpyHostToDevice, I.stream));

    // --- the conv stack, chunked in time -----------------------------------
    const int n_lvl = static_cast<int>(I.levels.size());
    const int chunk = chunk_frames() * e.downsample_rate;
    I.clear_hist(I.in_conv);
    for (Impl::Level & l : I.levels) { I.clear_hist(l.unit.c1); I.clear_hist(l.unit.c2); I.clear_hist(l.down); }
    I.clear_hist(I.out_conv);

    // full-sequence row counts per stage, so the last chunk's output counts come from the
    // plan rather than from a division that does not divide
    std::vector<int> full(static_cast<std::size_t>(n_lvl) + 1);
    full[0] = I.in_conv.pad.out;
    for (int i = 0; i < n_lvl; ++i) full[static_cast<std::size_t>(i) + 1] = I.levels[static_cast<std::size_t>(i)].down.pad.out;

    for (const Slice & s : slices(T, chunk)) {
        int off = s.off, len = s.len;
        I.run_conv(I.in_conv, PRO_NONE, EPI_NONE, I.d_wav + off, len, len,
                   s.last ? I.in_conv.pad.right : 0, I.a[0]);
        I.keep_tail(I.in_conv, I.d_wav + off, len);

        for (int li = 0; li < n_lvl; ++li) {
            Impl::Level & l = I.levels[static_cast<std::size_t>(li)];
            const auto u = static_cast<std::size_t>(li);
            // the residual unit: x + conv_k1(ELU(conv_k3(ELU(x))))
            I.run_conv(l.unit.c1, PRO_ELU, EPI_NONE, I.a[u], len, len, 0, I.t1[u]);
            I.keep_tail(l.unit.c1, I.a[u], len);
            I.run_conv(l.unit.c2, PRO_ELU, EPI_RESIDUAL, I.t1[u], len, len, 0, I.b[u], I.a[u]);
            I.keep_tail(l.unit.c2, I.t1[u], len);
            // then an ELU and the strided conv
            const int next_off = off / l.stride;
            const int right = s.last ? l.down.pad.right : 0;
            const int next_len = s.last ? full[u + 1] - next_off : len / l.stride;
            if (next_len * l.stride != len + right) {
                fail("codec encoder: chunk arithmetic is off at level " + std::to_string(li) + ": "
                     + std::to_string(next_len) + " x " + std::to_string(l.stride) + " != "
                     + std::to_string(len) + " + " + std::to_string(right));
            }
            I.run_conv(l.down, PRO_ELU, EPI_NONE, I.b[u], len, next_len, right, I.a[u + 1]);
            I.keep_tail(l.down, I.b[u], len);
            off = next_off;
            len = next_len;
        }
        // the output conv lands straight in the 25 Hz sequence
        I.run_conv(I.out_conv, PRO_ELU, EPI_NONE, I.a[static_cast<std::size_t>(n_lvl)], len, len, 0,
                   I.x25 + static_cast<std::size_t>(off) * L);
        I.keep_tail(I.out_conv, I.a[static_cast<std::size_t>(n_lvl)], len);
        if (s.last && off + len != T25) {
            fail("codec encoder: the chunks produced " + std::to_string(off + len) + " rows at 25 Hz, the plan says "
                 + std::to_string(T25));
        }
    }
    I.snap("conv", I.x25, T25, L);

    // --- the transformer, once over the whole sequence ----------------------
    // No KV cache, no window, no cross-chunk state: the sequence is 1.5 MB at the clip cap,
    // so chunking it would buy nothing and cost exactness.
    for (const XfLayer & x : I.xf) {
        k_layernorm<<<static_cast<unsigned>(T25), NT, 0, I.stream>>>(I.x25, L, x.ln1_w, x.ln1_b, e.norm_eps, I.norm);
        NAVI_HIP(hipGetLastError());
        I.run_conv(x.q, PRO_NONE, EPI_NONE, I.norm, T25, T25, 0, I.qkv, nullptr, nullptr, 3 * HD);
        I.run_conv(x.k, PRO_NONE, EPI_NONE, I.norm, T25, T25, 0, I.qkv + HD, nullptr, nullptr, 3 * HD);
        I.run_conv(x.v, PRO_NONE, EPI_NONE, I.norm, T25, T25, 0, I.qkv + 2 * HD, nullptr, nullptr, 3 * HD);
        k_rope<<<static_cast<unsigned>(T25), NT, 0, I.stream>>>(I.qkv, e.n_head, e.head_dim, e.rope_theta);
        NAVI_HIP(hipGetLastError());
        k_attn_causal<<<dim3(static_cast<unsigned>(T25), static_cast<unsigned>(e.n_head)), NT, 0, I.stream>>>(
            I.qkv, e.n_head, e.head_dim, I.att);
        NAVI_HIP(hipGetLastError());
        I.run_conv(x.o, PRO_NONE, EPI_SCALE_RESIDUAL, I.att, T25, T25, 0, I.x25, I.x25, x.attn_scale);
        k_layernorm<<<static_cast<unsigned>(T25), NT, 0, I.stream>>>(I.x25, L, x.ln2_w, x.ln2_b, e.norm_eps, I.norm);
        NAVI_HIP(hipGetLastError());
        I.run_conv(x.fc1, PRO_NONE, EPI_GELU, I.norm, T25, T25, 0, I.mlp);
        I.run_conv(x.fc2, PRO_NONE, EPI_SCALE_RESIDUAL, I.mlp, T25, T25, 0, I.x25, I.x25, x.mlp_scale);
    }
    I.snap("xf", I.x25, T25, L);

    // --- 25 Hz -> 12.5 Hz, then the RVQ -------------------------------------
    I.run_conv(I.down, PRO_NONE, EPI_NONE, I.x25, T25, frames, I.down.pad.right, I.lat);
    I.snap("latents", I.lat, frames, L);
    I.run_conv(I.proj_s, PRO_NONE, EPI_NONE, I.lat, frames, frames, 0, I.ls);
    I.run_conv(I.proj_a, PRO_NONE, EPI_NONE, I.lat, frames, frames, 0, I.la);
    // The two quantiser inputs, so a test that sees a code mismatch can replay the cascade
    // on the host and say whether it was a near-tie or an error.
    I.snap("proj_s", I.ls, frames, e.vq_dim);
    I.snap("proj_a", I.la, frames, e.vq_dim);
    const std::size_t rvq_lds = static_cast<std::size_t>(e.vq_dim) * sizeof(float)
                                + NT * sizeof(float) + NT * sizeof(int);
    k_rvq_encode<<<static_cast<unsigned>(frames), NT, rvq_lds, I.stream>>>(
        I.ls, I.la, e.vq_dim, I.books, e.codebook_size, G, e.n_semantic, I.d_codes);
    NAVI_HIP(hipGetLastError());

    std::vector<std::int32_t> codes(static_cast<std::size_t>(frames) * G);
    NAVI_HIP(hipMemcpyAsync(codes.data(), I.d_codes, codes.size() * sizeof(int), hipMemcpyDeviceToHost, I.stream));
    NAVI_HIP(hipEventRecord(I.ev1, I.stream));
    NAVI_HIP(hipStreamSynchronize(I.stream));
    float ms = 0.f;
    NAVI_HIP(hipEventElapsedTime(&ms, I.ev0, I.ev1));
    I.last_ms = ms;
    if (n_frames_out) *n_frames_out = frames;
    return codes;
}

} // namespace navi::qwen3tts
