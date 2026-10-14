// Codec encoder gate (DESIGN 6): voice_1.wav -> 16 codes per 12.5 Hz frame, against the
// PyTorch dumps, stage by stage so a failure says *where*.
//   conv     vs ref_conv.npy     the SEANet conv stack at 25 Hz
//   xf       vs ref_xf.npy       after the 8-layer transformer (full causal, no window)
//   latents  vs ref_latents.npy  after the 25 -> 12.5 Hz downsample, the last float
//   codes    vs ref_codes.npy    EXACT, cb0 included
//
// Codes are a nearest-centroid search, so they are gated exactly and a mismatch is
// reported with both centroids' distances rather than tolerated: a tie and an error look
// the same in a pass/fail, and only one of them is acceptable. The float stages carry the
// numerical claim.
//
// Also checked: a chunked encode equals a one-shot one (the conv stack threads each conv's
// causal history across chunks); encoding twice gives the same codes (nothing is carried);
// and a negative control - moving a centroid the reference actually selected must make the
// codes gate fail, or the gate proves nothing.
//
// Usage: test_codec_encoder <model.navi> <reference dir>

#include "model/qwen3tts/codec_encoder.h"
#include "model/qwen3tts/params.h"
#include "runtime/audio/wav.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Cmp {
    double max_abs = 0, ref_max = 0, rel_rms = 0;
    bool finite = true;
};

// Both sides are checked for finiteness, and so are the derived metrics: a NaN anywhere
// makes rel_rms a NaN, every comparison against which is false, so it would pass a bare
// threshold test unnoticed.
Cmp compare(const std::vector<float> & got, const std::vector<float> & want) {
    Cmp c;
    double se = 0, ss = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double x = got[i], y = want[i];
        if (!std::isfinite(x) || !std::isfinite(y)) c.finite = false;
        const double d = std::fabs(x - y);
        c.max_abs = std::max(c.max_abs, d);
        c.ref_max = std::max(c.ref_max, std::fabs(y));
        se += d * d;
        ss += y * y;
    }
    c.rel_rms = std::sqrt(se / (ss > 0 ? ss : 1e-30));
    if (!std::isfinite(c.rel_rms) || !std::isfinite(c.max_abs)) c.finite = false;
    return c;
}

bool report(const char * name, const Cmp & c, double gate) {
    const bool ok = c.finite && c.rel_rms < gate;
    std::printf("  %-8s max|d| %.3e (ref max %.3e)  rel-rms %.2e  gate %.0e  %s\n", name, c.max_abs, c.ref_max,
                c.rel_rms, gate, ok ? "ok" : (c.finite ? "OVER THE GATE" : "NON-FINITE"));
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        if (!params.has_codec_encoder) {
            std::printf("this blob has no codec encoder (convert.py --with-encoder): nothing to gate\n");
            return 0;
        }
        const auto & e = params.codec_encoder;
        const auto wav = navi::audio::read_wav(ref + "/voice_1.wav");
        if (wav.sample_rate != e.sample_rate) { std::printf("FAIL: voice_1.wav is %d Hz\n", wav.sample_rate); return 1; }

        const auto conv_npy = navi::Npy::load(ref + "/ref_conv.npy");
        const auto xf_npy = navi::Npy::load(ref + "/ref_xf.npy");
        const auto lat_npy = navi::Npy::load(ref + "/ref_latents.npy");
        const auto codes_npy = navi::Npy::load(ref + "/ref_codes.npy");
        const auto want_conv = conv_npy.as_f32();
        const auto want_xf = xf_npy.as_f32();
        const auto want_lat = lat_npy.as_f32();
        const auto want_codes = codes_npy.as_i32();
        const int T25 = static_cast<int>(conv_npy.shape[0]);
        const int frames = static_cast<int>(codes_npy.shape[0]);
        const int G = static_cast<int>(codes_npy.shape[1]);

        navi::Device dev = navi::Device::open();
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) {
            return t.name.rfind("codec.encoder.", 0) == 0;
        });
        auto enc = navi::qwen3tts::CodecEncoder::create(dev, file, w, params);
        file.close();

        // --- the stages ----------------------------------------------------
        std::vector<Cmp> stages;
        std::vector<std::string> names;
        std::vector<float> got_lat, got_proj_s, got_proj_a;   // kept for the diagnostics below
        const auto grab = [](const float * buf, int rows, int cols, std::vector<float> & into) {
            into.resize(static_cast<std::size_t>(rows) * cols);
            if (hipMemcpy(into.data(), buf, into.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
                throw std::runtime_error("stage readback failed");
            }
        };
        enc->set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
            if (name == "proj_s") { grab(buf, rows, cols, got_proj_s); return; }
            if (name == "proj_a") { grab(buf, rows, cols, got_proj_a); return; }
            std::vector<float> host;
            grab(buf, rows, cols, host);
            if (name == "latents") got_lat = host;
            const std::vector<float> * want = name == "conv" ? &want_conv : (name == "xf" ? &want_xf : &want_lat);
            if (host.size() != want->size()) {
                std::printf("  %-8s SHAPE %d x %d, reference %zu floats\n", name.c_str(), rows, cols, want->size());
                stages.push_back(Cmp{0, 0, 1e9, true});
            } else {
                stages.push_back(compare(host, *want));
            }
            names.push_back(name);
        });

        int n_frames = 0;
        const auto codes = enc->encode(wav.pcm, &n_frames);
        const double ms = enc->last_ms();
        std::printf("codec encoder: %zu samples -> %d rows at 25 Hz -> %d frames x %d codebooks  (%.1f ms, %d-frame chunks)\n",
                    wav.pcm.size(), T25, n_frames, G, ms, enc->chunk_frames());
        if (n_frames != frames) { std::printf("FAIL: %d frames, reference %d\n", n_frames, frames); return 1; }

        // The gates are the measured f32 values' own order of magnitude, as the prefill
        // gate's 5e-3 was: this is f32 weights against an f32 reference, so what is left is
        // accumulation order.
        bool stages_ok = stages.size() == 3;
        if (!stages_ok) std::printf("FAIL: %zu stages reported, expected 3\n", stages.size());
        for (std::size_t i = 0; i < stages.size(); ++i) {
            stages_ok = report(names[i].c_str(), stages[i], 1e-4) && stages_ok;
        }

        // Say whether a code mismatch is a near-tie or an error, because a pass/fail cannot
        // tell them apart and only one of them is acceptable. Only the FIRST mismatch in a
        // frame is informative: after it the residual differs, so every later codebook of
        // that frame is a consequence, not a finding. The residual is rebuilt from the
        // group's projected input minus the centroids the OTHER side chose before it.
        const auto explain = [&](int t, int cb, const std::vector<std::int32_t> & mine_codes,
                                 const std::vector<std::int32_t> & ref_codes_in,
                                 const std::vector<float> & proj_s, const std::vector<float> & proj_a) {
            const int dim = e.vq_dim;
            const int mine = mine_codes[static_cast<std::size_t>(t) * G + cb];
            const int theirs = ref_codes_in[static_cast<std::size_t>(t) * G + cb];
            std::printf("    frame %d codebook %d: engine %d, reference %d\n", t, cb, mine, theirs);
            const std::vector<float> & proj = cb == 0 ? proj_s : proj_a;
            std::vector<float> books(static_cast<std::size_t>(G) * e.codebook_size * dim);
            if (proj.size() < static_cast<std::size_t>(t + 1) * dim
                || hipMemcpy(books.data(), enc->d_codebooks(), books.size() * sizeof(float),
                             hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("    (could not read back the quantiser input to explain it)\n");
                return;
            }
            const auto row = [&](int book, int code) {
                return books.data() + (static_cast<std::size_t>(book) * e.codebook_size + code) * dim;
            };
            std::vector<double> res(static_cast<std::size_t>(dim));
            for (int i = 0; i < dim; ++i) res[static_cast<std::size_t>(i)] = proj[static_cast<std::size_t>(t) * dim + i];
            for (int b = (cb == 0 ? 0 : e.n_semantic); b < cb; ++b) {
                const float * c = row(b, ref_codes_in[static_cast<std::size_t>(t) * G + b]);
                for (int i = 0; i < dim; ++i) res[static_cast<std::size_t>(i)] -= c[i];
            }
            const auto dist = [&](int code) {
                const float * c = row(cb, code);
                double d = 0;
                for (int i = 0; i < dim; ++i) {
                    const double x = res[static_cast<std::size_t>(i)] - c[i];
                    d += x * x;
                }
                return std::sqrt(d);
            };
            const double dm = dist(mine), dt = dist(theirs);
            const double gap = std::fabs(dm - dt) / std::max(dm, 1e-9);
            std::printf("    distance from the quantiser's input: engine's centroid %.6f, the other's %.6f, "
                        "gap %.2e relative -> %s\n", dm, dt, gap,
                        gap < 1e-4 ? "a near-tie, the two are the same distance away to within 1e-4"
                                   : "NOT a tie: one of them is measurably further");
        };

        // --- the codes, exactly --------------------------------------------
        int bad = 0, bad_cb0 = 0, first_bad_frame = -1, first_bad_cb = -1;
        for (int t = 0; t < frames; ++t) {
            for (int g = 0; g < G; ++g) {
                const std::size_t i = static_cast<std::size_t>(t) * G + g;
                if (codes[i] != want_codes[i]) {
                    ++bad;
                    if (g == 0) ++bad_cb0;
                    if (first_bad_frame < 0) { first_bad_frame = t; first_bad_cb = g; }
                }
            }
        }
        std::printf("  codes    %d of %d differ from the reference%s\n", bad, frames * G,
                    bad ? "" : " (exact, cb0 included)");
        if (bad) explain(first_bad_frame, first_bad_cb, codes, want_codes, got_proj_s, got_proj_a);
        const bool codes_ok = bad == 0;

        // --- chunked == one-shot -------------------------------------------
        // One shot needs its own encoder: the scratch is sized for the chunk, so a bigger
        // chunk is a bigger build, not a setter (and it is ~230 MB at this clip's length).
        // Identical codes are necessary and not sufficient: the chunking is a float-level
        // claim about every conv's history, and a difference could hide inside a codebook
        // cell. So the latents are compared as well, at the gate the plan set.
        const auto lat_of = [&](navi::qwen3tts::CodecEncoder & c, std::vector<float> & into) {
            c.set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
                if (name == "latents") grab(buf, rows, cols, into);
            });
            const auto out = c.encode(wav.pcm);
            c.set_stage_hook(nullptr);
            return out;
        };
        std::vector<float> lat_again, lat5, lat_one;
        const auto again = lat_of(*enc, lat_again);
        const bool repeat_ok = again == codes && lat_again == got_lat;
        enc->set_chunk_frames(5);                    // a chunk the clip is not a multiple of
        const auto chunked5 = lat_of(*enc, lat5);
        enc->set_chunk_frames(0);
        std::vector<std::int32_t> one_shot;
        {
            navi::NaviFile f2 = navi::NaviFile::open(argv[1]);
            auto whole = navi::qwen3tts::CodecEncoder::create(dev, f2, w, params, 30.0, frames);
            f2.close();
            one_shot = lat_of(*whole, lat_one);
        }
        const Cmp c5 = compare(lat5, got_lat), c1 = compare(lat_one, got_lat);
        const bool chunk_ok = one_shot == codes && chunked5 == codes
                              && c5.finite && c5.rel_rms < 1e-6 && c1.finite && c1.rel_rms < 1e-6;
        std::printf("  chunks   %d frames (default), 5 frames, and one shot (%d frames): %s "
                    "(latents rel-rms %.1e and %.1e, gate 1e-6); encoding twice: %s\n",
                    enc->chunk_frames(), frames, one_shot == codes && chunked5 == codes ? "identical codes" : "DIFFER",
                    c5.rel_rms, c1.rel_rms, repeat_ok ? "identical, latents included" : "DIFFERS");

        // --- the unaligned length, which is what real references look like ----
        // 360,000 samples is exactly 187.5 frames, and every conv's right "extra" pad comes
        // out zero. Truncating by 777 makes all of them non-zero and stops the strides
        // dividing the chunk lengths - the path any real reference takes (a 20 s clip is
        // 249.75 frames).
        bool odd_ok = true;
        {
            const auto want_odd = navi::Npy::load(ref + "/ref_codes_odd.npy").as_i32();
            const auto want_odd_lat = navi::Npy::load(ref + "/ref_latents_odd.npy");
            const std::size_t n = wav.pcm.size() - 777;
            const std::span<const float> odd(wav.pcm.data(), n);
            std::vector<float> odd_lat;
            enc->set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
                if (name == "latents") grab(buf, rows, cols, odd_lat);
            });
            int odd_frames = 0;
            const auto odd_codes = enc->encode(odd, &odd_frames);
            enc->set_stage_hook(nullptr);
            enc->set_chunk_frames(5);
            const auto odd_chunked = enc->encode(odd);
            enc->set_chunk_frames(0);
            const int bad_odd = [&] {
                int k = 0;
                for (std::size_t i = 0; i < odd_codes.size() && i < want_odd.size(); ++i) {
                    if (odd_codes[i] != want_odd[i]) ++k;
                }
                return k;
            }();
            const Cmp lo = odd_lat.size() == want_odd_lat.as_f32().size()
                ? compare(odd_lat, want_odd_lat.as_f32()) : Cmp{0, 0, 1e9, true};
            std::printf("  odd      %zu samples -> %d frames (reference %zu): %d codes differ, "
                        "latents rel-rms %.2e, 5-frame chunks %s\n",
                        n, odd_frames, want_odd.size() / static_cast<std::size_t>(G), bad_odd, lo.rel_rms,
                        odd_chunked == odd_codes ? "identical" : "DIFFER");
            odd_ok = bad_odd == 0 && odd_codes.size() == want_odd.size() && lo.finite && lo.rel_rms < 1e-4
                     && odd_chunked == odd_codes;
        }

        // --- negative control ----------------------------------------------
        // Move a centroid the reference actually selected far away. The argmin cannot
        // return it any more, so the codes MUST change; if they do not, the gate above is
        // not reading what it claims to read.
        // It has to be the buffer the kernel reads: the encoder copies the codebooks when the
        // arena does not hold them adjacent, and writing to the arena then changes nothing -
        // which is how this control first reported UNCHANGED while the encoder was correct.
        const int ctl_frame = frames / 2;
        const int ctl_code = want_codes[static_cast<std::size_t>(ctl_frame) * G];
        float * row = const_cast<float *>(enc->d_codebooks())
                      + static_cast<std::size_t>(ctl_code) * e.codebook_dim;
        std::vector<float> saved(static_cast<std::size_t>(e.codebook_dim));
        if (hipMemcpy(saved.data(), row, saved.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
            throw std::runtime_error("control readback failed");
        }
        std::vector<float> moved = saved;
        for (float & v : moved) v += 1e3f;
        if (hipMemcpy(row, moved.data(), moved.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("control write failed");
        }
        std::vector<float> ctl_proj_s, ctl_proj_a;
        enc->set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
            if (name == "proj_s") grab(buf, rows, cols, ctl_proj_s);
            if (name == "proj_a") grab(buf, rows, cols, ctl_proj_a);
        });
        const auto perturbed = enc->encode(wav.pcm);
        enc->set_stage_hook(nullptr);
        const bool control_ok_codes = perturbed != codes;
        std::printf("  control  centroid %d of cb0 (selected at frame %d) moved by 1e3: codes %s\n",
                    ctl_code, ctl_frame, control_ok_codes ? "changed (the gate has teeth)" : "UNCHANGED (!)");
        // The control is also the only chance to see the mismatch diagnostic run, and its
        // cause is known: a centroid 1e3 away cannot be a tie. If it reports one, the
        // diagnostic is lying and the real gate's explanation could not be trusted either.
        if (control_ok_codes) {
            int ct = -1, ccb = -1;
            for (std::size_t i = 0; i < perturbed.size() && ct < 0; ++i) {
                if (perturbed[i] != codes[i]) { ct = static_cast<int>(i) / G; ccb = static_cast<int>(i) % G; }
            }
            explain(ct, ccb, perturbed, codes, ctl_proj_s, ctl_proj_a);
        }
        if (hipMemcpy(row, saved.data(), saved.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("control restore failed");
        }
        const auto restored = enc->encode(wav.pcm);
        const bool control_ok = control_ok_codes && restored == codes;
        std::printf("  control  restored: %s\n", restored == codes ? "identical codes" : "DIFFER (!)");

        const bool ok = stages_ok && codes_ok && chunk_ok && repeat_ok && odd_ok && control_ok;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
