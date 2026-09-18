#include "runtime/bench/bench.h"

#include "model/qwen3tts/graph.h"
#include "model/qwen3tts/params.h"
#include "runtime/audio/wav.h"
#include "runtime/voices/store.h"
#include "navi/build_info.h"
#include "runtime/common/error.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace navi {

namespace {

std::string json_escape(const std::string & s) {
    std::string o;
    for (const char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

struct Json {
    std::ostringstream ss;
    bool first = true;
    void key(const char * k) { ss << (first ? "" : ", ") << '"' << k << "\": "; first = false; }
    void str(const char * k, const std::string & v) { key(k); ss << '"' << json_escape(v) << '"'; }
    void num(const char * k, double v) { key(k); ss << std::setprecision(6) << v; }
    void u64(const char * k, std::uint64_t v) { key(k); ss << v; }
    template <class T> void opt(const char * k, const std::optional<T> & v) {
        key(k);
        if (!v) { ss << "null"; return; }
        if constexpr (std::is_same_v<T, std::string>) ss << '"' << json_escape(*v) << '"';
        else ss << std::setprecision(6) << *v;
    }
};

std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

} // namespace

std::string BenchResult::to_json() const {
    Json j;
    j.ss << '{';
    j.str("ts", timestamp_utc);
    j.str("git", git_hash);
    j.str("gfx", gfx);
    j.str("rocm", rocm);
    j.str("hip_runtime", hip_runtime);
    j.str("device", device_name);
    j.u64("multiprocessors", multiprocessors);
    j.str("model", model_name);
    j.str("dtype", model_dtype);
    j.u64("weight_bytes", weight_bytes);
    j.num("load_map_ms", load_map_ms);
    j.num("load_upload_ms", load_upload_ms);
    j.num("upload_gbps", upload_gbps);
    j.opt("prefill_ms", prefill_ms);
    j.opt("frame_ms", frame_ms);
    j.opt("talker_ms", talker_ms);
    j.opt("cp_ms", cp_ms);
    j.opt("vocoder_ms_per_frame", vocoder_ms_per_frame);
    j.opt("ttfa_ms", ttfa_ms);
    j.opt("rtf", rtf);
    j.opt("n_frames", n_frames);
    j.opt("wav_sha256", wav_sha256);
    j.str("text", text);
    j.u64("seed", seed);
    j.ss << '}';
    return j.ss.str();
}

BenchResult run_bench(const Device & dev, const BenchOptions & opt) {
    using clock = std::chrono::steady_clock;
    BenchResult r;
    r.timestamp_utc  = utc_now();
    r.git_hash       = NAVI_GIT_HASH;
    r.gfx            = dev.info().arch;
    r.rocm           = NAVI_ROCM_VERSION;
    r.hip_runtime    = std::to_string(dev.info().hip_runtime_version);
    r.device_name    = dev.info().name;
    r.multiprocessors = dev.info().multiprocessors;
    r.text           = opt.text;
    r.seed           = opt.seed;

    const auto t0 = clock::now();
    NaviFile file = NaviFile::open(opt.model_path);
    const auto params = qwen3tts::read_params(file);
    const auto v = qwen3tts::validate(file, params);
    if (!v.ok()) fail(opt.model_path + ": tensor table does not match the model (see `navi-tts info --model`)");
    r.model_name  = params.name;
    r.model_dtype = params.dtype;
    const auto t1 = clock::now();
    r.load_map_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    file.close();
    auto graph = qwen3tts::Graph::load(dev, opt.model_path);
    const auto & w = graph->weights();
    r.load_upload_ms = w.stats().seconds * 1e3;
    r.weight_bytes   = w.stats().bytes;
    r.upload_gbps    = w.stats().seconds > 0 ? (w.stats().bytes / 1e9) / w.stats().seconds : 0.0;

    // The utterance, exactly as the server runs it: the model's sampling
    // defaults, the reference voice, the fixed seed.
    const auto ref = audio::to_rate(audio::read_wav(opt.voice_wav), params.spk_sample_rate);
    const std::vector<float> speaker = graph->embed_speaker(ref.pcm);
    qwen3tts::SynthRequest sr;
    sr.text = opt.text;
    sr.speaker = speaker;
    sr.seed = opt.seed;
    sr.sampling.temperature = params.temperature;
    sr.sampling.top_k = params.top_k;
    sr.sampling.repetition_penalty = params.repetition_penalty;
    sr.sampling.cp_temperature = params.cp_temperature;
    sr.sampling.cp_top_k = params.cp_top_k;
    sr.vocoder_batch = opt.vocoder_batch;
    std::vector<float> pcm;
    for (int i = 0; i < opt.warmup; ++i) graph->synth(sr, pcm);
    const qwen3tts::SynthStats st = graph->synth(sr, pcm);
    r.prefill_ms = st.prefill_ms;
    r.n_frames = st.n_frames;
    r.frame_ms = st.n_frames ? st.frames_ms / st.n_frames : 0.0;
    r.vocoder_ms_per_frame = st.n_frames ? st.vocoder_ms / st.n_frames : 0.0;
    r.ttfa_ms = st.ttfa_ms;
    r.rtf = st.rtf;
    const auto bytes = audio::wav_bytes(audio::to_s16(pcm), params.codec_sample_rate);
    r.wav_sha256 = voices::Store::sha256_hex(bytes);
    return r;
}

} // namespace navi
