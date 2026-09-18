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

struct Voice {
    std::string name;
    std::vector<float> embedding;
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

int run(qwen3tts::Graph & graph, const Options & opt) {
    const qwen3tts::Params & params = graph.params();
    httplib::Server svr;
    Worker worker;
    std::mutex voices_mutex;
    std::map<std::string, Voice> voices;
    int next_voice = 1;
    std::mt19937_64 seed_rng{std::random_device{}()};

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
        Json::Array list{Json("default")};
        {
            std::lock_guard<std::mutex> l(voices_mutex);
            for (const auto & [id, v] : voices) list.push_back(Json(id));
        }
        Json j;
        j.set(opt.model_id, list);
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/audio/voices", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            std::string name;
            if (req.has_param("name")) name = req.get_param_value("name");
            if (req.form.has_field("name")) name = req.form.get_field("name");
            httplib::FormData file;
            bool have_file = false;
            for (const char * field : {"audio_sample", "audio", "file", "wav_file"}) {
                if (req.form.has_file(field)) { file = req.form.get_file(field); have_file = true; break; }
            }
            if (!have_file) { send_error(res, 400, "multipart field 'audio_sample' (or 'audio') with the reference WAV is required"); return; }
            if (name.empty()) name = file.filename.empty() ? "voice" : file.filename;
            const auto wav = audio::parse_wav(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t *>(file.content.data()), file.content.size()));
            if (wav.sample_rate != params.spk_sample_rate) {
                send_error(res, 400, "reference audio is " + std::to_string(wav.sample_rate) + " Hz; " +
                                     std::to_string(params.spk_sample_rate) + " Hz mono is required");
                return;
            }
            std::vector<float> emb;
            const auto t0 = std::chrono::steady_clock::now();
            worker.run([&] { emb = graph.embed_speaker(wav.pcm); });
            std::string id;
            {
                std::lock_guard<std::mutex> l(voices_mutex);
                id = "voice_" + std::to_string(next_voice++);
                voices[id] = Voice{name, std::move(emb)};
            }
            if (opt.verbose) {
                std::fprintf(stderr, "voice %s '%s': %.1f s of audio, %.0f ms\n", id.c_str(), name.c_str(),
                             static_cast<double>(wav.pcm.size()) / wav.sample_rate,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            Json j;
            j.set("id", id); j.set("name", name);
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception & e) {
            send_error(res, 400, e.what());
        }
    });

    svr.Delete(R"(/v1/audio/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> l(voices_mutex);
        if (voices.erase(req.matches[1]) == 0) { send_error(res, 404, "no such voice"); return; }
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
            {
                std::lock_guard<std::mutex> l(voices_mutex);
                auto it = voices.find(voice_id);
                if (it == voices.end() && voice_id == "default" && !voices.empty()) it = voices.begin();
                if (it == voices.end()) {
                    send_error(res, 400, voices.empty() ? "no voices registered: POST /v1/audio/voices a reference WAV first"
                                                        : "unknown voice '" + voice_id + "'");
                    return;
                }
                sr.speaker = it->second.embedding;   // the store is append-only while serving; the span stays valid
            }
            sr.seed = body.find("seed") ? static_cast<std::uint64_t>(body.get("seed").as_int()) : seed_rng();
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
    std::fprintf(stderr, "navi-tts serving %s on http://%s:%d\n", opt.model_id.c_str(), opt.host.c_str(), opt.port);
    const bool ok = svr.listen(opt.host, opt.port);
    g_server = nullptr;
    if (!ok) { std::fprintf(stderr, "navi-tts: cannot listen on %s:%d\n", opt.host.c_str(), opt.port); return 1; }
    return 0;
}

} // namespace navi::server
