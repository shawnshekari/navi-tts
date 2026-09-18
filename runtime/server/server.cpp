#include "runtime/server/server.h"

#include "runtime/audio/wav.h"
#include "runtime/common/error.h"
#include "runtime/common/json.h"

#include "third_party/cpp-httplib/httplib.h"

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <thread>

namespace navi::server {

namespace {

// The single serving thread: jobs run one at a time, in order.
class Worker {
public:
    Worker() : thread_([this] { loop(); }) {}
    ~Worker() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        thread_.join();
    }
    // Runs `job` on the worker and waits for it. Exceptions propagate to the caller.
    void run(const std::function<void()> & job) {
        std::exception_ptr err;
        bool done = false;
        std::mutex dm;
        std::condition_variable dcv;
        {
            std::lock_guard<std::mutex> l(m_);
            q_.push_back([&] {
                try { job(); } catch (...) { err = std::current_exception(); }
                { std::lock_guard<std::mutex> dl(dm); done = true; }
                dcv.notify_one();
            });
        }
        cv_.notify_one();
        std::unique_lock<std::mutex> dl(dm);
        dcv.wait(dl, [&] { return done; });
        if (err) std::rethrow_exception(err);
    }

private:
    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                job = std::move(q_.front());
                q_.pop_front();
            }
            job();
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    bool stop_ = false;
    std::thread thread_;
};

void send_error(httplib::Response & res, int status, const std::string & message, const char * type = "invalid_request_error") {
    Json err;
    Json inner;
    inner.set("message", message);
    inner.set("type", type);
    err.set("error", inner);
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

const std::map<std::string, std::string> LANG_CODES = {
    {"en", "english"}, {"zh", "chinese"}, {"de", "german"}, {"it", "italian"}, {"pt", "portuguese"},
    {"es", "spanish"}, {"ja", "japanese"}, {"ko", "korean"}, {"fr", "french"}, {"ru", "russian"},
};

// "" for auto; false if the language is not one the model knows
bool resolve_language(const std::string & in, const qwen3tts::Params & p, std::string & out) {
    if (in.empty() || in == "auto") { out.clear(); return true; }
    std::string s;
    for (const char c : in) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (p.language_ids.count(s)) { out = s; return true; }
    const auto it = LANG_CODES.find(s);
    if (it != LANG_CODES.end() && p.language_ids.count(it->second)) { out = it->second; return true; }
    return false;
}

httplib::Server * g_server = nullptr;
void on_signal(int) { if (g_server) g_server->stop(); }

} // namespace

int run(qwen3tts::Graph & graph, voices::Store & store, const Options & opt) {
    const qwen3tts::Params & params = graph.params();
    httplib::Server svr;
    Worker worker;
    std::mt19937_64 seed_rng{std::random_device{}()};
    std::mutex rng_mutex;
    auto random_seed = [&] { std::lock_guard<std::mutex> l(rng_mutex); return seed_rng(); };

    // "default" (and an empty voice) -> the configured default, else voice_1, else the first id.
    auto resolve_voice = [&](const std::string & id) -> voices::VoicePtr {
        if (!id.empty() && id != "default") return store.find(id);
        if (!opt.default_voice.empty()) return store.find(opt.default_voice);
        if (auto v = store.find("voice_1")) return v;
        const auto ids = store.ids();
        return ids.empty() ? nullptr : store.find(ids.front());
    };

    // Clone `sample` under `name`: same name + same bytes is a no-op, a new
    // sample under an existing name replaces it. Runs the encoder on the worker.
    struct CloneResult { voices::VoicePtr voice; const char * status; };
    auto clone = [&](const std::string & name, std::span<const std::uint8_t> sample, const std::string & ref_text) -> CloneResult {
        const std::string id = voices::Store::id_for(name);
        const std::string sha = voices::Store::sha256_hex(sample);
        if (auto v = store.find(id); v && v->sample_sha256 == sha) return {v, "exists"};
        const auto wav = audio::to_rate(audio::parse_wav(sample), params.spk_sample_rate);
        if (wav.pcm.size() < static_cast<std::size_t>(params.spk_sample_rate) / 2) throw std::invalid_argument("reference audio is shorter than 0.5 s");
        voices::Voice v;
        v.id = id;
        v.name = name;
        v.ref_text = ref_text;
        v.model = opt.model_id;
        v.sample_sha256 = sha;
        v.sample_seconds = static_cast<double>(wav.pcm.size()) / wav.sample_rate;
        const bool replaced = store.find(id) != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        worker.run([&] { v.embedding = graph.embed_speaker(wav.pcm); });
        auto ptr = store.put(std::move(v), sample);
        if (opt.verbose) {
            std::fprintf(stderr, "voice %s%s: %.1f s of audio, %.0f ms\n", id.c_str(), replaced ? " (replaced)" : "",
                         ptr->sample_seconds,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        return {ptr, replaced ? "replaced" : "created"};
    };

    svr.set_payload_max_length(64u << 20);
    svr.set_read_timeout(60);
    svr.set_write_timeout(120);

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        Json j;
        j.set("status", "ok");
        j.set("model", opt.model_id);
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        Json m;
        m.set("id", opt.model_id); m.set("object", "model"); m.set("owned_by", "navi-tts");
        Json j;
        j.set("object", "list");
        j.set("data", Json::Array{m});
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/v1/audio/languages", [&](const httplib::Request &, httplib::Response & res) {
        Json::Array langs;
        for (const auto & [code, name] : LANG_CODES) {
            const auto it = params.language_ids.find(name);
            if (it == params.language_ids.end()) continue;
            Json l;
            l.set("code", code); l.set("name", name); l.set("id", it->second);
            langs.push_back(l);
        }
        Json j;
        j.set("languages", langs);
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        Json::Array list;
        if (resolve_voice("default")) list.push_back(Json("default"));
        for (const auto & id : store.ids()) list.push_back(Json(id));
        Json j;
        j.set(opt.model_id, list);
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/audio/voices", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            std::string name, ref_text;
            if (req.has_param("name")) name = req.get_param_value("name");
            if (req.form.has_field("name")) name = req.form.get_field("name");
            if (req.form.has_field("ref_text")) ref_text = req.form.get_field("ref_text");
            httplib::FormData file;
            bool have_file = false;
            for (const char * field : {"audio_sample", "audio", "file", "wav_file"}) {
                if (req.form.has_file(field)) { file = req.form.get_file(field); have_file = true; break; }
            }
            if (!have_file) { send_error(res, 400, "multipart field 'audio_sample' (or 'audio') with the reference WAV is required"); return; }
            if (name.empty()) name = file.filename.empty() ? "voice" : file.filename;
            const auto r = clone(name, std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t *>(file.content.data()), file.content.size()), ref_text);
            Json j;
            j.set("id", r.voice->id); j.set("name", r.voice->name); j.set("status", r.status);
            res.set_content(j.dump(), "application/json");
        } catch (const navi::Error & e) {
            send_error(res, 500, e.what(), "engine_error");
        } catch (const std::exception & e) {
            send_error(res, 400, e.what());
        }
    });

    svr.Delete(R"(/v1/audio/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        if (!store.remove(req.matches[1])) { send_error(res, 404, "no such voice"); return; }
        Json j;
        j.set("deleted", true);
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        Json body;
        try { body = Json::parse(req.body); } catch (const std::exception &) { send_error(res, 400, "invalid JSON"); return; }
        if (!body.is_object() || !body.find("input") || !body.get("input").is_string()) { send_error(res, 400, "'input' is required"); return; }
        try {
            qwen3tts::SynthRequest sr;
            sr.text = body.get("input").as_string();
            if (sr.text.empty()) { send_error(res, 400, "'input' is empty"); return; }
            if (static_cast<int>(sr.text.size()) > opt.max_input_chars) {
                send_error(res, 400, "'input' exceeds " + std::to_string(opt.max_input_chars) + " characters"); return;
            }
            if (!resolve_language(body.str_or("language", "en"), params, sr.language)) {
                send_error(res, 400, "unknown language '" + body.str_or("language", "") + "'"); return;
            }
            const std::string voice_id = body.str_or("voice", "default");
            const voices::VoicePtr voice = resolve_voice(voice_id);   // held for the request: a replace/delete cannot free it
            if (!voice) {
                send_error(res, 400, store.size() == 0 ? "no voices registered: POST /v1/audio/voices a reference WAV first"
                                                       : "unknown voice '" + voice_id + "'");
                return;
            }
            sr.speaker = voice->embedding;
            sr.seed = body.find("seed") ? static_cast<std::uint64_t>(body.get("seed").as_int()) : random_seed();
            sr.max_frames = static_cast<int>(body.int_or("max_audio_tokens", opt.max_audio_tokens));
            if (sr.max_frames < 1 || sr.max_frames > 4096) { send_error(res, 400, "'max_audio_tokens' out of range"); return; }
            sr.sampling.temperature = static_cast<float>(body.num_or("temperature", params.temperature));
            sr.sampling.top_k = static_cast<int>(body.int_or("top_k", params.top_k));
            sr.sampling.repetition_penalty = static_cast<float>(body.num_or("repetition_penalty", params.repetition_penalty));
            sr.sampling.cp_temperature = params.cp_temperature;
            sr.sampling.cp_top_k = params.cp_top_k;

            std::vector<float> pcm;
            qwen3tts::SynthStats st;
            worker.run([&] { st = graph.synth(sr, pcm); });
            const auto bytes = audio::wav_bytes(audio::to_s16(pcm), params.codec_sample_rate);
            res.set_content(reinterpret_cast<const char *>(bytes.data()), bytes.size(), "audio/wav");
            if (opt.verbose) {
                std::fprintf(stderr,
                             "speech %s seed %llu: %d tokens, %d frames (%s), prefill %.1f frames %.1f (%.2f/frame) vocoder %.1f | "
                             "%.2f s audio in %.0f ms, RTF %.3f\n",
                             voice_id.c_str(), static_cast<unsigned long long>(sr.seed), st.n_tokens, st.n_frames,
                             st.eos ? "eos" : "cap", st.prefill_ms, st.frames_ms, st.n_frames ? st.frames_ms / st.n_frames : 0.0,
                             st.vocoder_ms, st.audio_s, st.total_ms, st.rtf);
            }
        } catch (const navi::Error & e) {
            send_error(res, 500, e.what(), "engine_error");
        } catch (const std::exception & e) {
            send_error(res, 400, e.what());
        }
    });

    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "navi-tts serving %s on http://%s:%d, %zu voice(s) in %s\n", opt.model_id.c_str(), opt.host.c_str(),
                 opt.port, store.size(), store.dir().c_str());
    const bool ok = svr.listen(opt.host, opt.port);
    g_server = nullptr;
    if (!ok) { std::fprintf(stderr, "navi-tts: cannot listen on %s:%d\n", opt.host.c_str(), opt.port); return 1; }
    return 0;
}

} // namespace navi::server
