// navi-tts: one binary. Subcommands:
//   info   [--model FILE] [--upload]   device report; with a model, the tensor table check
//   bench  --model FILE [--out FILE]   the harness (DESIGN 7); JSON line to stdout or appended to --out
//   serve  --model FILE [--voices DIR]  the HTTP front (DESIGN 3)
//   voices --voices DIR list|add|rm     the persistent voice store, offline (one-time imports)

#include "model/qwen3tts/graph.h"
#include "model/qwen3tts/params.h"
#include "runtime/audio/wav.h"
#include "runtime/common/npy.h"
#include "runtime/server/server.h"
#include "runtime/voices/store.h"
#include "navi/build_info.h"
#include "runtime/bench/bench.h"
#include "runtime/common/error.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string cmd;
    std::string model;
    std::string out;
    std::string text;
    std::string speaker;        // .npy [hidden] embedding
    std::string voice;          // reference WAV (24 kHz mono) to clone
    std::string language = "english";
    std::uint64_t seed = 0;
    int max_frames = 600;
    bool greedy = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    int xtts_port = 0;
    std::string xtts_fallback_male = "default", xtts_fallback_female = "default";
    std::string voices_dir;     // default: $XDG_DATA_HOME/navi-tts/voices
    std::string default_voice;
    std::string model_id;
    std::string name;           // voices add
    std::string ref_text;
    std::vector<std::string> rest;   // positional: voices <action> [id]
    bool upload = false;
    bool warmup = true;
    bool verbose = false;
    int repeats = 1;
};

int usage(const char * argv0) {
    std::fprintf(stderr,
        "navi-tts %s (%s, ROCm %s, %s)\n"
        "usage: %s info  [--model FILE] [--upload] [-v]\n"
        "       %s bench --model FILE [--out results.jsonl] [--repeats N] [--voice REF.wav] [--seed N]\n"
        "       %s synth --model FILE --text TEXT (--voice REF.wav | --speaker EMB.npy) --out out.wav [--seed N] [--language L] [--max-frames N] [--greedy]\n"
        "       %s serve --model FILE [--voices DIR] [--default-voice ID] [--model-id ID] [--host 127.0.0.1] [--port 8080] [--xtts-port 8020] [--xtts-fallback-male ID] [--xtts-fallback-female ID] [--max-frames 600] [--no-warmup] [-V]\n"
        "       %s voices [--voices DIR] list\n"
        "       %s voices [--voices DIR] add --model FILE --name NAME --voice REF.wav [--ref-text TEXT]\n"
        "       %s voices [--voices DIR] rm ID\n"
        "  --voices defaults to $XDG_DATA_HOME/navi-tts/voices (~/.local/share/navi-tts/voices)\n",
        NAVI_GIT_HASH, NAVI_GPU_ARCHS, NAVI_ROCM_VERSION, NAVI_BUILD_TYPE, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
    return 2;
}

Args parse(int argc, char ** argv) {
    Args a;
    if (argc < 2) return a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) navi::fail(std::string(flag) + " needs a value");
            return argv[++i];
        };
        if (s == "--model" || s == "-m") a.model = next("--model");
        else if (s == "--out") a.out = next("--out");
        else if (s == "--repeats") a.repeats = std::stoi(next("--repeats"));
        else if (s == "--text") a.text = next("--text");
        else if (s == "--speaker") a.speaker = next("--speaker");
        else if (s == "--voice") a.voice = next("--voice");
        else if (s == "--language") a.language = next("--language");
        else if (s == "--seed") a.seed = std::stoull(next("--seed"));
        else if (s == "--max-frames") a.max_frames = std::stoi(next("--max-frames"));
        else if (s == "--greedy") a.greedy = true;
        else if (s == "--host") a.host = next("--host");
        else if (s == "--port") a.port = std::stoi(next("--port"));
        else if (s == "--xtts-port") a.xtts_port = std::stoi(next("--xtts-port"));
        else if (s == "--xtts-fallback-male") a.xtts_fallback_male = next("--xtts-fallback-male");
        else if (s == "--xtts-fallback-female") a.xtts_fallback_female = next("--xtts-fallback-female");
        else if (s == "--upload") a.upload = true;
        else if (s == "--no-warmup") a.warmup = false;
        else if (s == "--voices") a.voices_dir = next("--voices");
        else if (s == "--default-voice") a.default_voice = next("--default-voice");
        else if (s == "--model-id") a.model_id = next("--model-id");
        else if (s == "--name") a.name = next("--name");
        else if (s == "--ref-text") a.ref_text = next("--ref-text");
        else if (s == "-v" || s == "-V" || s == "--verbose") a.verbose = true;
        else if (!s.empty() && s[0] != '-') a.rest.push_back(s);
        else navi::fail("unknown argument " + s);
    }
    if (a.voices_dir.empty()) {
        const char * xdg = std::getenv("XDG_DATA_HOME");
        const char * home = std::getenv("HOME");
        if (xdg && *xdg) a.voices_dir = std::string(xdg) + "/navi-tts/voices";
        else if (home && *home) a.voices_dir = std::string(home) + "/.local/share/navi-tts/voices";
        else a.voices_dir = "voices";
    }
    if (a.model_id.empty() && !a.model.empty()) {
        std::string base = a.model.substr(a.model.find_last_of('/') == std::string::npos ? 0 : a.model.find_last_of('/') + 1);
        if (base.size() > 5 && base.compare(base.size() - 5, 5, ".navi") == 0) base.resize(base.size() - 5);
        a.model_id = base;
    }
    return a;
}

// Reference WAV -> mono PCM at the speaker encoder's rate.
navi::audio::Wav read_reference(const std::string & path, const navi::qwen3tts::Params & p) {
    return navi::audio::to_rate(navi::audio::read_wav(path), p.spk_sample_rate);
}

void print_device(const navi::DeviceInfo & d, bool selected) {
    std::printf("device %d%s: %s\n", d.index, selected ? " (selected)" : "", d.name.c_str());
    std::printf("  arch              %s\n", d.arch.c_str());
    std::printf("  multiprocessors   %d  (HIP multiProcessorCount; WGPs on RDNA3 - the cooperative grid size)\n",
                d.multiprocessors);
    std::printf("  wave size         %d\n", d.warp_size);
    std::printf("  cooperative       %s\n", d.cooperative_launch ? "yes" : "no");
    std::printf("  threads/block     %d,  LDS/block %zu KiB,  regs/block %d\n",
                d.max_threads_per_block, d.lds_per_block / 1024, d.regs_per_block);
    std::printf("  clocks            core %d MHz, mem %d MHz, bus %d bit, L2 %zu MiB\n",
                d.clock_khz / 1000, d.mem_clock_khz / 1000, d.mem_bus_width, d.l2_bytes >> 20);
    std::printf("  vram              %.2f GiB total, %.2f GiB free\n",
                d.vram_total / 1073741824.0, d.vram_free / 1073741824.0);
    std::printf("  hip               runtime %d, driver %d\n", d.hip_runtime_version, d.hip_driver_version);
}

int cmd_info(const Args & a) {
    std::printf("navi-tts %s  built for %s  ROCm %s (needs >= %s)  %s\n", NAVI_GIT_HASH, NAVI_GPU_ARCHS,
                NAVI_ROCM_VERSION, NAVI_ROCM_MIN_VERSION, NAVI_BUILD_TYPE);
    const auto devs = navi::enumerate_devices();
    if (devs.empty()) std::printf("no HIP devices visible\n");
    navi::Device dev = navi::Device::open();   // throws with the refusal message
    for (const auto & d : devs) print_device(d.index == dev.info().index ? dev.info() : d, d.index == dev.info().index);
    if (a.model.empty()) return 0;

    navi::NaviFile file = navi::NaviFile::open(a.model);
    std::printf("\nmodel %s\n", a.model.c_str());
    std::printf("  size              %.3f GB, data at +%llu\n", file.file_size() / 1e9,
                static_cast<unsigned long long>(file.data_offset()));
    std::printf("  kv                %zu entries\n", file.kv().size());
    std::printf("  tensors           %zu\n", file.tensors().size());
    if (a.verbose) {
        for (const auto & [k, v] : file.kv()) {
            std::printf("    %-48s ", k.c_str());
            switch (v.type) {
                case navi::KV::Type::I64: std::printf("%lld\n", static_cast<long long>(v.i64)); break;
                case navi::KV::Type::F64: std::printf("%g\n", v.f64); break;
                case navi::KV::Type::STR: std::printf("\"%s\"\n", v.str.c_str()); break;
                case navi::KV::Type::I64_ARRAY: {
                    std::printf("[");
                    for (std::size_t i = 0; i < v.arr.size(); ++i)
                        std::printf("%s%lld", i ? ", " : "", static_cast<long long>(v.arr[i]));
                    std::printf("]\n");
                    break;
                }
            }
        }
        for (const auto & t : file.tensors()) {
            std::printf("    %-72s %-5s %s\n", t.name.c_str(), navi::dtype_name(t.dtype), t.shape_str().c_str());
        }
    }

    const auto params = navi::qwen3tts::read_params(file);
    std::printf("  arch              qwen3-tts  %s  (%s)\n", params.name.c_str(), params.dtype.c_str());
    std::printf("  talker            %d layers, hidden %d, %d/%d heads x %d, ff %d, codec vocab %d, text vocab %d\n",
                params.talker.n_layer, params.talker.hidden, params.talker.n_head, params.talker.n_kv_head,
                params.talker.head_dim, params.talker.ff, params.talker.vocab, params.text_vocab);
    std::printf("  code predictor    %d layers, hidden %d, %d/%d heads x %d, ff %d, %d codebooks x %d\n",
                params.cp.n_layer, params.cp.hidden, params.cp.n_head, params.cp.n_kv_head, params.cp.head_dim,
                params.cp.ff, params.n_code_groups, params.cp.vocab);
    std::printf("  vocoder           %d layers, hidden %d, latent %d, %d x %d codebooks, %d samples/frame @ %d Hz%s\n",
                params.vocoder.n_layer, params.vocoder.hidden, params.vq_latent, params.vq_n_q,
                params.vq_codebook_size, params.codec_upsample, params.codec_sample_rate,
                params.has_codec_encoder ? ", encoder included" : "");
    std::printf("  languages        ");
    for (const auto & [k, v] : params.language_ids) std::printf(" %s=%d", k.c_str(), v);
    std::printf("\n");

    const auto v = navi::qwen3tts::validate(file, params);
    std::printf("  tensor table      %s: %zu missing, %zu shape mismatches, %zu unused\n",
                v.ok() ? "ok" : "FAILED", v.missing.size(), v.shape_mismatch.size(), v.unused.size());
    for (const auto & m : v.missing) std::printf("    missing   %s\n", m.c_str());
    for (const auto & m : v.shape_mismatch) std::printf("    mismatch  %s\n", m.c_str());
    for (const auto & m : v.unused) std::printf("    unused    %s\n", m.c_str());
    if (!v.ok()) return 1;

    if (a.upload) {
        navi::DeviceWeights w = navi::DeviceWeights::upload(file);
        file.close();
        const auto & s = w.stats();
        std::printf("  upload            %zu tensors, %.3f GB in %.0f ms (%.1f GB/s), arena %.3f GB\n",
                    s.n_tensors, s.bytes / 1e9, s.seconds * 1e3, (s.bytes / 1e9) / s.seconds, s.arena_bytes / 1e9);
        dev.refresh_memory();
        std::printf("  vram after        %.2f GiB free\n", dev.info().vram_free / 1073741824.0);
    }
    return 0;
}

int cmd_bench(const Args & a) {
    if (a.model.empty()) navi::fail("bench needs --model");
    navi::Device dev = navi::Device::open();
    navi::BenchOptions opt;
    opt.model_path = a.model;
    opt.repeats = a.repeats;
    if (!a.voice.empty()) opt.voice_wav = a.voice;
    if (!a.text.empty()) opt.text = a.text;
    if (a.seed) opt.seed = a.seed;
    for (int i = 0; i < a.repeats; ++i) {
        const navi::BenchResult r = navi::run_bench(dev, opt);
        const std::string line = r.to_json();
        if (a.out.empty()) {
            std::printf("%s\n", line.c_str());
        } else {
            std::ofstream f(a.out, std::ios::app);
            if (!f) navi::fail("cannot open " + a.out + " for append");
            f << line << '\n';
            std::fprintf(stderr, "appended to %s: %d frames, prefill %.1f ms, %.2f ms/frame, vocoder %.2f ms/frame, ttfa %.0f ms, RTF %.3f, sha %s\n",
                         a.out.c_str(), r.n_frames.value_or(0), r.prefill_ms.value_or(0), r.frame_ms.value_or(0),
                         r.vocoder_ms_per_frame.value_or(0), r.ttfa_ms.value_or(0), r.rtf.value_or(0),
                         r.wav_sha256.value_or("-").substr(0, 12).c_str());
        }
    }
    return 0;
}

int cmd_synth(const Args & a) {
    if (a.model.empty() || a.text.empty() || (a.speaker.empty() && a.voice.empty()) || a.out.empty()) {
        navi::fail("synth needs --model, --text, --voice or --speaker, and --out");
    }
    navi::Device dev = navi::Device::open();
    const auto t0 = std::chrono::steady_clock::now();
    auto graph = navi::qwen3tts::Graph::load(dev, a.model);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> spk;
    if (!a.voice.empty()) {
        spk = graph->embed_speaker(read_reference(a.voice, graph->params()).pcm);
    } else {
        spk = navi::Npy::load(a.speaker).as_f32();
    }

    navi::qwen3tts::SynthRequest req;
    req.text = a.text;
    req.language = a.language;
    req.speaker = spk;
    req.seed = a.seed;
    req.max_frames = a.max_frames;
    if (a.greedy) { req.sampling.temperature = 0.f; req.sampling.cp_temperature = 0.f; }
    std::vector<float> pcm;
    const auto st = graph->synth(req, pcm);
    navi::audio::write_wav(a.out, pcm, graph->params().codec_sample_rate);
    std::fprintf(stderr,
                 "load %.0f ms | %d tokens, %d frames (%s) | prefill %.1f ms, frames %.1f ms (%.2f ms/frame), "
                 "vocoder %.1f ms | ttfa %.0f ms | %.2f s audio in %.0f ms, RTF %.3f | %s\n",
                 load_ms, st.n_tokens, st.n_frames, st.eos ? "eos" : "cap", st.prefill_ms, st.frames_ms,
                 st.n_frames ? st.frames_ms / st.n_frames : 0.0, st.vocoder_ms, st.ttfa_ms, st.audio_s, st.total_ms,
                 st.rtf, a.out.c_str());
    return 0;
}

int cmd_serve(const Args & a) {
    if (a.model.empty()) navi::fail("serve needs --model");
    navi::Device dev = navi::Device::open();
    const auto t0 = std::chrono::steady_clock::now();
    auto graph = navi::qwen3tts::Graph::load(dev, a.model);
    std::fprintf(stderr, "navi-tts %s: %s loaded on %s (%s) in %.0f ms\n", NAVI_GIT_HASH, a.model.c_str(),
                 dev.info().name.c_str(), dev.info().arch.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    navi::server::Options opt;
    opt.host = a.host;
    opt.port = a.port;
    opt.xtts_port = a.xtts_port;
    opt.xtts_fallback_male = a.xtts_fallback_male;
    opt.xtts_fallback_female = a.xtts_fallback_female;
    opt.verbose = a.verbose;
    opt.max_audio_tokens = a.max_frames;
    opt.warmup = a.warmup;
    opt.model_id = a.model_id;
    opt.default_voice = a.default_voice;
    navi::voices::Store store = navi::voices::Store::open(a.voices_dir, graph->params().talker.hidden);
    return navi::server::run(*graph, store, opt);
}

int cmd_voices(const Args & a) {
    const std::string action = a.rest.empty() ? "list" : a.rest[0];
    if (action == "list") {
        navi::voices::Store store = navi::voices::Store::open(a.voices_dir, 0);   // no model: any embedding size
        for (const auto & v : store.all()) {
            std::printf("%-28s %-28s %5.1f s  %zu-d  %s  %s\n", v->id.c_str(), v->name.c_str(), v->sample_seconds,
                        v->embedding.size(), v->created.c_str(), v->model.c_str());
        }
        return 0;
    }
    if (action == "rm") {
        if (a.rest.size() < 2) navi::fail("voices rm needs an id");
        navi::voices::Store store = navi::voices::Store::open(a.voices_dir, 0);
        if (!store.remove(a.rest[1])) navi::fail("no such voice " + a.rest[1]);
        std::printf("removed %s\n", a.rest[1].c_str());
        return 0;
    }
    if (action == "add") {
        if (a.model.empty() || a.name.empty() || a.voice.empty()) navi::fail("voices add needs --model, --name and --voice");
        navi::Device dev = navi::Device::open();
        auto graph = navi::qwen3tts::Graph::load(dev, a.model);
        navi::voices::Store store = navi::voices::Store::open(a.voices_dir, graph->params().talker.hidden);
        std::ifstream f(a.voice, std::ios::binary);
        if (!f) navi::fail("cannot read " + a.voice);
        const std::string bytes((std::istreambuf_iterator<char>(f)), {});
        const std::span<const std::uint8_t> sample(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
        navi::voices::Voice v;
        v.id = navi::voices::Store::id_for(a.name);
        v.name = a.name;
        v.ref_text = a.ref_text;
        v.model = a.model_id;
        v.sample_sha256 = navi::voices::Store::sha256_hex(sample);
        if (auto cur = store.find(v.id); cur && cur->sample_sha256 == v.sample_sha256) {
            std::printf("%s: unchanged\n", v.id.c_str());
            return 0;
        }
        const auto wav = navi::audio::to_rate(navi::audio::parse_wav(sample), graph->params().spk_sample_rate);
        v.sample_seconds = static_cast<double>(wav.pcm.size()) / wav.sample_rate;
        const bool replaced = store.find(v.id) != nullptr;
        v.embedding = graph->embed_speaker(wav.pcm);
        store.put(std::move(v), sample);
        std::printf("%s: %s (%.1f s of audio) -> %s\n", navi::voices::Store::id_for(a.name).c_str(),
                    replaced ? "replaced" : "added", static_cast<double>(wav.pcm.size()) / wav.sample_rate, a.voices_dir.c_str());
        return 0;
    }
    navi::fail("voices: unknown action " + action + " (list|add|rm)");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Args a = parse(argc, argv);
        if (a.cmd == "info") return cmd_info(a);
        if (a.cmd == "bench") return cmd_bench(a);
        if (a.cmd == "synth") return cmd_synth(a);
        if (a.cmd == "serve") return cmd_serve(a);
        if (a.cmd == "voices") return cmd_voices(a);
        return usage(argv[0]);
    } catch (const navi::Error & e) {
        std::fprintf(stderr, "navi-tts: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "navi-tts: %s\n", e.what());
        return 1;
    }
}
