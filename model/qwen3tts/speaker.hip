#include "model/qwen3tts/speaker.h"

#include "model/qwen3tts/kernels.h"
#include "runtime/common/error.h"
#include "runtime/common/hip_check.h"
#include "runtime/device/kernel_util.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace navi::qwen3tts {

namespace {

using namespace navi::qwen3tts::kern;

// --- host mel front end ------------------------------------------------------

// In-place iterative radix-2 FFT, n a power of two.
void fft(std::vector<std::complex<float>> & a) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / static_cast<double>(len);
        const std::complex<float> wl(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.f, 0.f);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax): slaney mel scale, slaney norm, float32.
std::vector<float> mel_filterbank(int sr, int n_fft, int n_mels, double fmin, double fmax) {
    auto hz_to_mel = [](double f) {
        const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
        return f >= min_log_hz ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
    };
    auto mel_to_hz = [](double m) {
        const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
        return m >= min_log_mel ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
    };
    const int n_bins = n_fft / 2 + 1;
    std::vector<double> fftfreqs(n_bins), mel_f(n_mels + 2);
    for (int i = 0; i < n_bins; ++i) fftfreqs[i] = static_cast<double>(sr) / 2.0 * i / (n_bins - 1);
    const double m0 = hz_to_mel(fmin), m1 = hz_to_mel(fmax);
    for (int i = 0; i < n_mels + 2; ++i) mel_f[i] = mel_to_hz(m0 + (m1 - m0) * i / (n_mels + 1));
    std::vector<float> w(static_cast<std::size_t>(n_mels) * n_bins);
    for (int i = 0; i < n_mels; ++i) {
        const double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (int k = 0; k < n_bins; ++k) {
            const double lower = (fftfreqs[k] - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            const double upper = (mel_f[i + 2] - fftfreqs[k]) / (mel_f[i + 2] - mel_f[i + 1]);
            w[static_cast<std::size_t>(i) * n_bins + k] = static_cast<float>(std::max(0.0, std::min(lower, upper)) * enorm);
        }
    }
    return w;
}

// --- device kernels -------------------------------------------------------------

// mean over time per channel: out[c] = mean_t x[t][c]
__global__ void k_time_mean(const float * x, int T, int C, float * out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = 0.f;
    for (int t = 0; t < T; ++t) s += x[static_cast<std::size_t>(t) * C + c];
    out[c] = s / T;
}

// y[t][c] = x[t][c] * g[c] + r[t][c]   (SE scaling + the block's residual)
__global__ void k_scale_add(const float * x, const float * g, const float * r, int T, int C, float * y) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(T) * C) return;
    y[i] = x[i] * g[i % C] + r[i];
}

// ASP statistics: per channel, weighted mean and std over time with weights w[t][c]
// (or uniform 1/T when w == null). out[c] = mean, out[C + c] = std.
__global__ void k_weighted_stats(const float * x, const float * w, int T, int C, float * out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = 0.f;
    for (int t = 0; t < T; ++t) m += (w ? w[static_cast<std::size_t>(t) * C + c] : 1.f / T) * x[static_cast<std::size_t>(t) * C + c];
    float v = 0.f;
    for (int t = 0; t < T; ++t) {
        const float d = x[static_cast<std::size_t>(t) * C + c] - m;
        v += (w ? w[static_cast<std::size_t>(t) * C + c] : 1.f / T) * d * d;
    }
    out[c] = m;
    out[C + c] = sqrtf(fmaxf(v, 1e-12f));
}

// cat[t] = [x[t] | mean | std]  (C each)
__global__ void k_cat_stats(const float * x, const float * stats, int C, float * cat) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < 3 * C; i += blockDim.x) {
        cat[static_cast<std::size_t>(t) * 3 * C + i] = i < C ? x[static_cast<std::size_t>(t) * C + i] : stats[i - C];
    }
}

// softmax over time, per channel, in place: a[t][c]
__global__ void k_time_softmax(float * a, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float m = -INFINITY;
    for (int t = 0; t < T; ++t) m = fmaxf(m, a[static_cast<std::size_t>(t) * C + c]);
    float s = 0.f;
    for (int t = 0; t < T; ++t) { const float e = __expf(a[static_cast<std::size_t>(t) * C + c] - m); a[static_cast<std::size_t>(t) * C + c] = e; s += e; }
    for (int t = 0; t < T; ++t) a[static_cast<std::size_t>(t) * C + c] /= s;
}

struct Conv { const unsigned short * W; const float * b; int Cin, Cout, K, dil; };

} // namespace

struct SpeakerEncoder::Impl {
    hipStream_t stream = nullptr;
    Params p;
    int max_frames = 0;
    std::vector<void *> owned;
    hipEvent_t ev0 = nullptr, ev1 = nullptr;
    double last_ms = 0;

    Conv block0;
    struct SeRes2 { Conv tdnn1, res[7], tdnn2, se1, se2; int dil; } blocks[3];
    Conv mfa, asp_tdnn, asp_conv, fc;
    int C = 512, R = 64, SE = 128, MFA = 1536, ASP = 128, scale = 8;

    float * mel = nullptr, * a = nullptr, * b = nullptr, * c = nullptr, * cat3 = nullptr, * g = nullptr, * gse = nullptr,
          * stats = nullptr, * att = nullptr, * attw = nullptr, * pooled = nullptr, * emb = nullptr, * mean = nullptr;

    template <class T> T * alloc(std::size_t n) {
        void * ptr = nullptr;
        NAVI_HIP(hipMalloc(&ptr, n * sizeof(T)));
        NAVI_HIP(hipMemset(ptr, 0, n * sizeof(T)));
        owned.push_back(ptr);
        return static_cast<T *>(ptr);
    }

    // "same" reflect-padded conv over [T][x_stride] -> [T][y_stride]
    void conv(const Conv & cv, int pro, int epi, const float * x, int T, int x_stride, float * y, int y_stride, const float * x2 = nullptr) {
        ConvArgs a{};
        a.W = cv.W; a.bias = cv.b; a.x = x; a.x2 = x2; a.T_in = T; a.Cin = cv.Cin; a.Cout = cv.Cout; a.dil = cv.dil;
        a.x_stride = x_stride; a.phases = 1; a.y = y; a.y_stride = y_stride;
        a.pad_left = (cv.K - 1) * cv.dil / 2; a.reflect = 1;
        const dim3 grid((T + TT - 1) / TT, (cv.Cout + CT - 1) / CT, 1), block(NTHREADS);
#define NAVI_SPK_CASE(KK, PP, EE) \
        if (cv.K == KK && pro == PP && epi == EE) { k_conv<unsigned short, KK, PP, EE><<<grid, block, 0, stream>>>(a); NAVI_HIP(hipGetLastError()); return; }
        NAVI_SPK_CASE(5, PRO_NONE, EPI_RELU)
        NAVI_SPK_CASE(3, PRO_NONE, EPI_RELU)
        NAVI_SPK_CASE(3, PRO_ADD2, EPI_RELU)
        NAVI_SPK_CASE(1, PRO_NONE, EPI_RELU)
        NAVI_SPK_CASE(1, PRO_NONE, EPI_NONE)
        NAVI_SPK_CASE(1, PRO_NONE, EPI_SIGMOID)
        NAVI_SPK_CASE(1, PRO_NONE, EPI_RELU_TANH)
#undef NAVI_SPK_CASE
        fail("speaker: no conv instantiation");
    }

    void run(int T) {
        NAVI_HIP(hipEventRecord(ev0, stream));
        // block 0: mel [T][128] -> a [T][512]
        conv(block0, PRO_NONE, EPI_RELU, mel, T, p.mel_num_mels, a, C);
        const float * in = a;
        for (int bi = 0; bi < 3; ++bi) {
            SeRes2 & bk = blocks[bi];
            float * out = cat3 + static_cast<std::size_t>(bi) * C;   // this block's output column slice of [T][1536]
            conv(bk.tdnn1, PRO_NONE, EPI_RELU, in, T, C, b, C);
            // res2net over 8 slices of R channels: part 0 copied, part 1 = tdnn(part 1), part i = tdnn(part i + out_{i-1})
            NAVI_HIP(hipMemcpy2DAsync(c, C * sizeof(float), b, C * sizeof(float), R * sizeof(float), T, hipMemcpyDeviceToDevice, stream));
            for (int i = 1; i < scale; ++i) {
                const float * xi = b + i * R;
                float * yi = c + i * R;
                if (i == 1) conv(bk.res[0], PRO_NONE, EPI_RELU, xi, T, C, yi, C);
                else conv(bk.res[i - 1], PRO_ADD2, EPI_RELU, xi, T, C, yi, C, c + (i - 1) * R);
            }
            conv(bk.tdnn2, PRO_NONE, EPI_RELU, c, T, C, b, C);
            // squeeze-excitation on the time mean
            k_time_mean<<<(C + 255) / 256, 256, 0, stream>>>(b, T, C, mean);
            conv(bk.se1, PRO_NONE, EPI_RELU, mean, 1, C, g, SE);
            conv(bk.se2, PRO_NONE, EPI_SIGMOID, g, 1, SE, gse, C);
            const std::size_t n = static_cast<std::size_t>(T) * C;
            // out = b * gse + residual(in); written into the MFA concat buffer (stride 1536) via a temp
            k_scale_add<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(b, gse, in, T, C, a2(bi));
            NAVI_HIP(hipMemcpy2DAsync(out, MFA * sizeof(float), a2(bi), C * sizeof(float), C * sizeof(float), T, hipMemcpyDeviceToDevice, stream));
            in = a2(bi);
        }
        // MFA: [T][1536] -> [T][1536]
        conv(mfa, PRO_NONE, EPI_RELU, cat3, T, MFA, att, MFA);
        // ASP: stats -> [x | mean | std] -> tdnn(4608->128)+relu -> tanh -> conv(128->1536) -> softmax over t -> weighted stats
        k_weighted_stats<<<(MFA + 255) / 256, 256, 0, stream>>>(att, nullptr, T, MFA, stats);
        k_cat_stats<<<T, 256, 0, stream>>>(att, stats, MFA, catx);
        conv(asp_tdnn, PRO_NONE, EPI_RELU_TANH, catx, T, 3 * MFA, g2, ASP);
        conv(asp_conv, PRO_NONE, EPI_NONE, g2, T, ASP, attw, MFA);
        k_time_softmax<<<(MFA + 255) / 256, 256, 0, stream>>>(attw, T, MFA);
        k_weighted_stats<<<(MFA + 255) / 256, 256, 0, stream>>>(att, attw, T, MFA, pooled);
        conv(fc, PRO_NONE, EPI_NONE, pooled, 1, 2 * MFA, emb, p.spk_dim);
        NAVI_HIP(hipGetLastError());
        NAVI_HIP(hipEventRecord(ev1, stream));
    }
    float * abuf[3] = {};
    float * a2(int i) { return abuf[i]; }
    float * catx = nullptr, * g2 = nullptr;
};

std::unique_ptr<SpeakerEncoder> SpeakerEncoder::create(const Device & dev, const DeviceWeights & w, const Params & p, double max_seconds) {
    std::unique_ptr<SpeakerEncoder> s(new SpeakerEncoder());
    s->impl_ = std::make_unique<Impl>();
    Impl & I = *s->impl_;
    I.stream = dev.stream();
    I.p = p;
    I.max_frames = static_cast<int>(max_seconds * p.spk_sample_rate / p.mel_hop) + 2;
    auto f16p = [&](const std::string & n) { return w.get(n).as<unsigned short>(); };
    auto f32p = [&](const std::string & n) { return w.get(n).as<float>(); };
    auto cv = [&](const std::string & n, int cin, int cout, int k, int dil) {
        return Conv{f16p(n + ".weight"), f32p(n + ".bias"), cin, cout, k, dil};
    };
    const std::string S = "speaker_encoder.";
    I.block0 = cv(S + "blocks.0.conv", p.mel_num_mels, I.C, 5, 1);
    const int dils[3] = {2, 3, 4};
    for (int b = 0; b < 3; ++b) {
        const std::string bp = S + "blocks." + std::to_string(b + 1) + ".";
        Impl::SeRes2 & bk = I.blocks[b];
        bk.dil = dils[b];
        bk.tdnn1 = cv(bp + "tdnn1.conv", I.C, I.C, 1, 1);
        for (int r = 0; r < I.scale - 1; ++r) bk.res[r] = cv(bp + "res2net_block.blocks." + std::to_string(r) + ".conv", I.R, I.R, 3, dils[b]);
        bk.tdnn2 = cv(bp + "tdnn2.conv", I.C, I.C, 1, 1);
        bk.se1 = cv(bp + "se_block.conv1", I.C, I.SE, 1, 1);
        bk.se2 = cv(bp + "se_block.conv2", I.SE, I.C, 1, 1);
    }
    I.mfa = cv(S + "mfa.conv", I.MFA, I.MFA, 1, 1);
    I.asp_tdnn = cv(S + "asp.tdnn.conv", 3 * I.MFA, I.ASP, 1, 1);
    I.asp_conv = cv(S + "asp.conv", I.ASP, I.MFA, 1, 1);
    I.fc = cv(S + "fc", 2 * I.MFA, p.spk_dim, 1, 1);

    const std::size_t T = static_cast<std::size_t>(I.max_frames);
    I.mel = I.alloc<float>(T * p.mel_num_mels);
    I.a = I.alloc<float>(T * I.C); I.b = I.alloc<float>(T * I.C); I.c = I.alloc<float>(T * I.C);
    for (int i = 0; i < 3; ++i) I.abuf[i] = I.alloc<float>(T * I.C);
    I.cat3 = I.alloc<float>(T * I.MFA); I.att = I.alloc<float>(T * I.MFA); I.attw = I.alloc<float>(T * I.MFA);
    I.catx = I.alloc<float>(T * 3 * I.MFA); I.g2 = I.alloc<float>(T * I.ASP);
    I.mean = I.alloc<float>(I.C); I.g = I.alloc<float>(I.SE); I.gse = I.alloc<float>(I.C);
    I.stats = I.alloc<float>(2 * I.MFA); I.pooled = I.alloc<float>(2 * I.MFA); I.emb = I.alloc<float>(p.spk_dim);
    NAVI_HIP(hipEventCreate(&I.ev0));
    NAVI_HIP(hipEventCreate(&I.ev1));
    return s;
}

SpeakerEncoder::~SpeakerEncoder() {
    if (!impl_) return;
    if (impl_->ev0) (void) hipEventDestroy(impl_->ev0);
    if (impl_->ev1) (void) hipEventDestroy(impl_->ev1);
    for (void * ptr : impl_->owned) (void) hipFree(ptr);
}

std::vector<float> SpeakerEncoder::mel(std::span<const float> wav, const Params & p, int * n_frames_out) {
    const int n_fft = p.mel_n_fft, hop = p.mel_hop, win = p.mel_win, n_mels = p.mel_num_mels, n_bins = n_fft / 2 + 1;
    if (win != n_fft) fail("speaker: win_size != n_fft not supported");
    const int pad = (n_fft - hop) / 2;
    const std::size_t n = wav.size();
    if (n < static_cast<std::size_t>(pad + 1)) fail("speaker: reference audio too short");
    // reflect pad
    std::vector<float> y(n + 2 * pad);
    for (std::size_t i = 0; i < y.size(); ++i) {
        long idx = static_cast<long>(i) - pad;
        if (idx < 0) idx = -idx;
        if (idx >= static_cast<long>(n)) idx = 2 * static_cast<long>(n) - 2 - idx;
        y[i] = wav[static_cast<std::size_t>(idx)];
    }
    const int T = static_cast<int>((y.size() - n_fft) / hop) + 1;
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; ++i) hann[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * i / n_fft));   // periodic
    const auto fb = mel_filterbank(p.spk_sample_rate, n_fft, n_mels, p.mel_fmin, p.mel_fmax);
    std::vector<float> out(static_cast<std::size_t>(n_mels) * T);
    std::vector<std::complex<float>> frame(n_fft);
    std::vector<float> mag(n_bins);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n_fft; ++i) frame[i] = std::complex<float>(y[static_cast<std::size_t>(t) * hop + i] * hann[i], 0.f);
        fft(frame);
        for (int k = 0; k < n_bins; ++k) mag[k] = std::sqrt(std::norm(frame[k]) + 1e-9f);
        for (int m = 0; m < n_mels; ++m) {
            float s = 0.f;
            const float * row = &fb[static_cast<std::size_t>(m) * n_bins];
            for (int k = 0; k < n_bins; ++k) s += row[k] * mag[k];
            out[static_cast<std::size_t>(m) * T + t] = std::log(std::max(s, 1e-5f));
        }
    }
    if (n_frames_out) *n_frames_out = T;
    return out;
}

std::vector<float> SpeakerEncoder::embed_mel(std::span<const float> mel_in, int T) {
    Impl & I = *impl_;
    if (T > I.max_frames) fail("speaker: reference audio longer than the encoder's max_seconds");
    // [n_mels][T] -> time-major [T][n_mels]
    std::vector<float> tm(static_cast<std::size_t>(T) * I.p.mel_num_mels);
    for (int m = 0; m < I.p.mel_num_mels; ++m)
        for (int t = 0; t < T; ++t) tm[static_cast<std::size_t>(t) * I.p.mel_num_mels + m] = mel_in[static_cast<std::size_t>(m) * T + t];
    NAVI_HIP(hipMemcpyAsync(I.mel, tm.data(), tm.size() * sizeof(float), hipMemcpyHostToDevice, I.stream));
    I.run(T);
    std::vector<float> out(static_cast<std::size_t>(I.p.spk_dim));
    NAVI_HIP(hipMemcpyAsync(out.data(), I.emb, out.size() * sizeof(float), hipMemcpyDeviceToHost, I.stream));
    NAVI_HIP(hipStreamSynchronize(I.stream));
    float ms = 0.f;
    NAVI_HIP(hipEventElapsedTime(&ms, I.ev0, I.ev1));
    I.last_ms = ms;
    return out;
}

std::vector<float> SpeakerEncoder::embed(std::span<const float> wav) {
    int T = 0;
    const auto m = mel(wav, impl_->p, &T);
    return embed_mel(m, T);
}

double SpeakerEncoder::last_ms() const { return impl_->last_ms; }

} // namespace navi::qwen3tts
