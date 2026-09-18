#include "runtime/server/server.h"

#include "runtime/audio/wav.h"
#include "runtime/common/error.h"
#include "runtime/common/json.h"

#include "navi/build_info.h"
#include "third_party/cpp-httplib/httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <initializer_list>
#include <iterator>
#include <thread>
#include <vector>

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
    // Queues `job` and returns; the job must handle its own exceptions.
    void post(std::function<void()> job) {
        { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(job)); }
        cv_.notify_one();
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

// Streaming batch sizes (frames; 12.5 frames/s). Measured on the XTX: first
// audio 67 ms at 4 frames vs 102 at 8; vocoder 1.17 ms/frame at 16 vs 1.56 at 8.
constexpr int STREAM_FIRST_BATCH = 4;
constexpr int STREAM_BATCH = 16;

// The bounded queue between the worker (producer: one vocoder batch at a
// time) and the response's content provider (consumer). If the consumer
// stops taking - the client stopped reading - the producer's push times out
// and the request fails; the GPU worker never waits on a socket.
class PcmQueue {
public:
    static constexpr std::size_t CAPACITY = 16;              // batches; 16 x 8 frames = ~10 s of audio
    static constexpr std::chrono::seconds PUSH_TIMEOUT{10};

    // Producer. false: the consumer is gone or stalled; the producer must stop.
    bool push(std::vector<std::int16_t> chunk) {
        std::unique_lock<std::mutex> l(m_);
        if (!cv_.wait_for(l, PUSH_TIMEOUT, [&] { return aborted_ || q_.size() < CAPACITY; })) { aborted_ = true; return false; }
        if (aborted_) return false;
        q_.push_back(std::move(chunk));
        cv_.notify_all();
        return true;
    }
    // Producer, once: no more chunks. `error` empty on success.
    void finish(std::string error) {
        std::lock_guard<std::mutex> l(m_);
        done_ = true;
        error_ = std::move(error);
        cv_.notify_all();
    }
    // Consumer. false with an empty chunk: the stream is over (see error()).
    bool pop(std::vector<std::int16_t> & chunk) {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return done_ || !q_.empty(); });
        if (q_.empty()) return false;
        chunk = std::move(q_.front());
        q_.pop_front();
        cv_.notify_all();
        return true;
    }
    // Consumer: waits until the first chunk or the end; true if audio is coming.
    bool wait_first() {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return done_ || !q_.empty(); });
        return !q_.empty();
    }
    // Consumer: the client went away.
    void abort() {
        std::lock_guard<std::mutex> l(m_);
        aborted_ = true;
        q_.clear();
        cv_.notify_all();
    }
    bool aborted() const { std::lock_guard<std::mutex> l(m_); return aborted_; }
    std::string error() const { std::lock_guard<std::mutex> l(m_); return error_; }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::vector<std::int16_t>> q_;
    bool done_ = false, aborted_ = false;
    std::string error_;
};

std::string base64(const void * data, std::size_t n) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto * p = static_cast<const std::uint8_t *>(data);
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (std::size_t i = 0; i < n; i += 3) {
        const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (i + 1 < n ? std::uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < n ? tbl[v & 63] : '=';
    }
    return out;
}

// GET /metrics (DESIGN 3.1, 7): llama.cpp's shape - `tts:` counters and
// gauges in the text exposition format - so the existing panels carry over.
// Bumped once per request where the stats land; no dependency.
struct Metrics {
    std::atomic<std::uint64_t> requests_total{0}, requests_processing{0}, clones_total{0};
    std::atomic<std::uint64_t> prompt_tokens_total{0}, predicted_tokens_total{0}, frame_cap_hits_total{0};
    std::atomic<double> prompt_seconds_total{0}, predicted_seconds_total{0}, audio_seconds_total{0};
    std::atomic<double> tokenize_seconds_total{0}, encode_seconds_total{0}, generate_seconds_total{0}, vocode_seconds_total{0};
    std::atomic<double> request_seconds_total{0}, clone_seconds_total{0};
    // Histograms for the two numbers that matter (DESIGN 7): p50/p99 come out of
    // histogram_quantile; the _last gauges are the instantaneous readout.
    struct Histogram {
        static constexpr std::size_t MAX = 12;
        const double * bounds; std::size_t n;
        std::atomic<std::uint64_t> bucket[MAX + 1]{};
        std::atomic<double> sum{0}, last{0};
        std::atomic<std::uint64_t> count{0};
        template <std::size_t N> explicit Histogram(const double (&b)[N]) : bounds(b), n(N) { static_assert(N <= MAX); }
        void observe(double v) {
            std::size_t b = 0;
            while (b < n && v > bounds[b]) ++b;
            bucket[b]++;
            add(sum, v);
            last.store(v);
            count++;
        }
    };
    static constexpr double TTFA_BUCKETS[] = {0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0};
    static constexpr double RTF_BUCKETS[] = {0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0};
    Histogram ttfa{TTFA_BUCKETS}, rtf{RTF_BUCKETS};
    std::map<std::string, std::atomic<std::uint64_t>> errors;   // by stage; keys fixed at construction

    // request: 4xx; generate: a failed synth; stream: the client went away
    Metrics() { for (const char * st : {"request", "clone", "generate", "stream"}) errors[st] = 0; }

    static void add(std::atomic<double> & a, double v) {
        double cur = a.load();
        while (!a.compare_exchange_weak(cur, cur + v)) {}
    }
    void observe(const qwen3tts::SynthStats & st) {
        requests_total++;
        prompt_tokens_total += static_cast<std::uint64_t>(std::max(0, st.n_tokens));
        predicted_tokens_total += static_cast<std::uint64_t>(std::max(0, st.n_frames));
        if (st.hit_cap) frame_cap_hits_total++;
        add(prompt_seconds_total, st.prefill_ms / 1e3);
        add(predicted_seconds_total, st.frames_ms / 1e3);
        add(audio_seconds_total, st.audio_s);
        add(tokenize_seconds_total, st.tokenize_ms / 1e3);
        add(encode_seconds_total, (st.prefill_ms - st.tokenize_ms) / 1e3);
        add(generate_seconds_total, st.frames_ms / 1e3);
        add(vocode_seconds_total, st.vocoder_ms / 1e3);
        add(request_seconds_total, st.total_ms / 1e3);
        if (st.ttfa_ms > 0) ttfa.observe(st.ttfa_ms / 1e3);
        if (st.rtf > 0) rtf.observe(st.rtf);
    }
    void error(const std::string & stage) {
        const auto it = errors.find(stage);
        if (it != errors.end()) it->second++;
    }

    std::string render(const std::string & model, std::size_t voices) const {
        std::string o;
        auto line = [&](const char * name, const char * type, const char * help, const std::string & value, const std::string & labels = "") {
            o += "# HELP tts:"; o += name; o += ' '; o += help; o += '\n';
            o += "# TYPE tts:"; o += name; o += ' '; o += type; o += '\n';
            o += "tts:"; o += name; o += labels; o += ' '; o += value; o += '\n';
        };
        auto u = [](std::uint64_t v) { return std::to_string(v); };
        auto d = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.6g", v); return std::string(b); };
        line("model_info", "gauge", "The loaded model.", "1", "{model=\"" + model + "\",git=\"" NAVI_GIT_HASH "\",gfx=\"" NAVI_GPU_ARCHS "\"}");
        line("voices", "gauge", "Voices in the store.", u(voices));
        line("requests_total", "counter", "Speech requests completed.", u(requests_total));
        line("requests_processing", "gauge", "Speech requests in flight (queued on the worker or streaming).", u(requests_processing));
        line("clones_total", "counter", "Voice clones computed (a re-registration of the same sample is not one).", u(clones_total));
        line("clone_seconds_total", "counter", "Time in the speaker encoder.", d(clone_seconds_total));
        line("prompt_tokens_total", "counter", "Text tokens in.", u(prompt_tokens_total));
        line("prompt_seconds_total", "counter", "Time in tokenize + prefill.", d(prompt_seconds_total));
        line("predicted_tokens_total", "counter", "Codec frames out (12.5 Hz).", u(predicted_tokens_total));
        line("predicted_seconds_total", "counter", "Time in the frame loop.", d(predicted_seconds_total));
        line("audio_seconds_total", "counter", "Audio produced; rate(request_seconds)/rate(audio_seconds) is the RTF.", d(audio_seconds_total));
        line("request_seconds_total", "counter", "Wall time of speech requests.", d(request_seconds_total));
        line("tokenize_seconds_total", "counter", "Stage: BPE.", d(tokenize_seconds_total));
        line("encode_seconds_total", "counter", "Stage: prompt build + prefill.", d(encode_seconds_total));
        line("generate_seconds_total", "counter", "Stage: frame kernel (talker + code predictor).", d(generate_seconds_total));
        line("vocode_seconds_total", "counter", "Stage: vocoder.", d(vocode_seconds_total));
        line("frame_cap_hits_total", "counter", "Requests that hit max_audio_tokens without EOS.", u(frame_cap_hits_total));
        o += "# HELP tts:errors_total Failed requests by stage.\n# TYPE tts:errors_total counter\n";
        for (const auto & [stage, n] : errors) o += "tts:errors_total{stage=\"" + stage + "\"} " + u(n) + '\n';
        auto histogram = [&](const char * name, const char * help, const Histogram & h) {
            o += "# HELP tts:"; o += name; o += ' '; o += help; o += "\n# TYPE tts:"; o += name; o += " histogram\n";
            std::uint64_t cum = 0;
            for (std::size_t b = 0; b < h.n; ++b) {
                cum += h.bucket[b];
                o += "tts:" + std::string(name) + "_bucket{le=\"" + d(h.bounds[b]) + "\"} " + u(cum) + '\n';
            }
            cum += h.bucket[h.n];
            o += "tts:" + std::string(name) + "_bucket{le=\"+Inf\"} " + u(cum) + '\n';
            o += "tts:" + std::string(name) + "_sum " + d(h.sum) + '\n';
            o += "tts:" + std::string(name) + "_count " + u(h.count) + '\n';
        };
        histogram("ttfa_seconds", "Time to first audio (prefill + first frames + one vocoder batch).", ttfa);
        histogram("rtf", "Real-time factor per request (wall / audio seconds).", rtf);
        line("ttfa_seconds_last", "gauge", "Time to first audio of the last request.", d(ttfa.last));
        line("rtf_last", "gauge", "RTF of the last request.", d(rtf.last));
        return o;
    }
};

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

std::vector<httplib::Server *> g_servers;
void on_signal(int) { for (httplib::Server * s : g_servers) s->stop(); }

std::span<const std::uint8_t> bytes_of(const std::string & s) {
    return {reinterpret_cast<const std::uint8_t *>(s.data()), s.size()};
}

bool is_riff(const std::string & s) { return s.size() >= 12 && s.compare(0, 4, "RIFF") == 0; }

// Everything the handlers share. One instance; the routes are installed on
// each listening server.
class Service {
public:
    Service(qwen3tts::Graph & graph, voices::Store & store, const Options & opt)
        : graph_(graph), store_(store), opt_(opt), params_(graph.params()), seed_rng_(std::random_device{}()) {}

    void install(httplib::Server & svr) {
        svr.set_payload_max_length(64u << 20);
        svr.set_read_timeout(60);
        svr.set_write_timeout(30);   // loopback: a socket that blocks this long is a dead client
        install_openai(svr);
        install_xtts(svr);
    }

private:
    // --- shared -----------------------------------------------------------
    std::uint64_t random_seed() { std::lock_guard<std::mutex> l(rng_mutex_); return seed_rng_(); }

    // "default" (and an empty voice) -> the configured default, else voice_1, else the first id.
    voices::VoicePtr resolve_voice(const std::string & id) {
        if (!id.empty() && id != "default") return store_.find(id);
        if (!opt_.default_voice.empty()) return store_.find(opt_.default_voice);
        if (auto v = store_.find("voice_1")) return v;
        const auto ids = store_.ids();
        return ids.empty() ? nullptr : store_.find(ids.front());
    }

    // Clone `sample` under `name`: same name + same bytes is a no-op, a new
    // sample under an existing name replaces it. Runs the encoder on the worker.
    struct CloneResult { voices::VoicePtr voice; const char * status; };
    CloneResult clone(const std::string & name, std::span<const std::uint8_t> sample, const std::string & ref_text) {
        const std::string id = voices::Store::id_for(name);
        const std::string sha = voices::Store::sha256_hex(sample);
        if (auto v = store_.find(id); v && v->sample_sha256 == sha) return {v, "exists"};
        const auto wav = audio::to_rate(audio::parse_wav(sample), params_.spk_sample_rate);
        if (wav.pcm.size() < static_cast<std::size_t>(params_.spk_sample_rate) / 2) throw std::invalid_argument("reference audio is shorter than 0.5 s");
        voices::Voice v;
        v.id = id;
        v.name = name;
        v.ref_text = ref_text;
        v.model = opt_.model_id;
        v.sample_sha256 = sha;
        v.sample_seconds = static_cast<double>(wav.pcm.size()) / wav.sample_rate;
        const bool replaced = store_.find(id) != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        worker_.run([&] { v.embedding = graph_.embed_speaker(wav.pcm); });
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        metrics_.clones_total++;
        Metrics::add(metrics_.clone_seconds_total, ms / 1e3);
        auto ptr = store_.put(std::move(v), sample);
        if (opt_.verbose) std::fprintf(stderr, "voice %s%s: %.1f s of audio, %.0f ms\n", id.c_str(), replaced ? " (replaced)" : "", ptr->sample_seconds, ms);
        return {ptr, replaced ? "replaced" : "created"};
    }

    // A validated request: what both dialects reduce to.
    struct Speech {
        qwen3tts::SynthRequest sr;
        voices::VoicePtr voice;      // held for the request: a replace/delete cannot free the embedding
        std::string voice_id;        // as the client named it, for the log
    };

    // Fills `sp` from the common fields or reports why not (status, message).
    bool prepare(Speech & sp, const std::string & text, const std::string & voice_id, const std::string & language,
                 httplib::Response & res) {
        const bool ok = prepare_(sp, text, voice_id, language, res);
        if (!ok) metrics_.error("request");
        return ok;
    }
    bool prepare_(Speech & sp, const std::string & text, const std::string & voice_id, const std::string & language,
                  httplib::Response & res) {
        sp.sr.text = text;
        sp.voice_id = voice_id;
        if (sp.sr.text.empty()) { send_error(res, 400, "'input' is empty"); return false; }
        if (static_cast<int>(sp.sr.text.size()) > opt_.max_input_chars) {
            send_error(res, 400, "'input' exceeds " + std::to_string(opt_.max_input_chars) + " characters"); return false;
        }
        if (!resolve_language(language, params_, sp.sr.language)) { send_error(res, 400, "unknown language '" + language + "'"); return false; }
        sp.voice = resolve_voice(voice_id);
        if (!sp.voice) {
            send_error(res, 400, store_.size() == 0 ? "no voices registered: POST /v1/audio/voices a reference WAV first"
                                                    : "unknown voice '" + voice_id + "'");
            return false;
        }
        sp.sr.speaker = sp.voice->embedding;
        sp.sr.max_frames = opt_.max_audio_tokens;
        sp.sr.sampling.temperature = params_.temperature;
        sp.sr.sampling.top_k = params_.top_k;
        sp.sr.sampling.repetition_penalty = params_.repetition_penalty;
        sp.sr.sampling.cp_temperature = params_.cp_temperature;
        sp.sr.sampling.cp_top_k = params_.cp_top_k;
        return true;
    }

    // Runs the request on the worker and answers with the whole body: a WAV, or raw s16le.
    void speak_wav(const Speech & sp, bool raw_pcm, httplib::Response & res) {
        std::vector<float> pcm;
        qwen3tts::SynthStats st;
        metrics_.requests_processing++;
        try {
            worker_.run([&] { st = graph_.synth(sp.sr, pcm); });
        } catch (...) {
            metrics_.requests_processing--;
            metrics_.error("generate");
            throw;
        }
        metrics_.requests_processing--;
        metrics_.observe(st);
        const auto s16 = audio::to_s16(pcm);
        if (raw_pcm) {
            res.set_content(reinterpret_cast<const char *>(s16.data()), s16.size() * 2, "audio/pcm");
        } else {
            const auto bytes = audio::wav_bytes(s16, params_.codec_sample_rate);
            res.set_content(reinterpret_cast<const char *>(bytes.data()), bytes.size(), "audio/wav");
        }
        log_speech(sp, st);
    }

    // Streams the request: the worker pushes each vocoder batch as s16 into a
    // bounded queue, the content provider drains it. Headers go out once the
    // first batch (or the failure) is known, so an early error is still an
    // HTTP status and the headers' arrival is the time to first audio.
    //   audio: chunked body, WAV with a streaming header (0xFFFFFFFF sizes) or raw s16le
    //   sse:   text/event-stream, speech.audio.delta {audio: base64 s16le} ..., speech.audio.done {navi: stats}
    void speak_stream(std::shared_ptr<Speech> sp, bool sse, bool raw_pcm, httplib::Response & res) {
        auto q = std::make_shared<PcmQueue>();
        auto st = std::make_shared<qwen3tts::SynthStats>();
        metrics_.requests_processing++;
        worker_.post([this, sp, q, st] {
            std::string error;
            try {
                *st = graph_.synth(sp->sr, [&](std::span<const float> pcm) {
                    if (!q->push(audio::to_s16(pcm))) fail("client stopped reading");
                });
            } catch (const std::exception & e) {
                error = e.what();
            }
            q->finish(error);
            metrics_.requests_processing--;
            if (error.empty()) { metrics_.observe(*st); log_speech(*sp, *st); }
            else {
                metrics_.error(q->aborted() ? "stream" : "generate");
                if (opt_.verbose) std::fprintf(stderr, "speech %s: failed: %s\n", sp->voice_id.c_str(), error.c_str());
            }
        });
        if (!q->wait_first()) {
            const std::string err = q->error();
            send_error(res, 500, err.empty() ? "no audio produced" : err, "engine_error");
            return;
        }
        const int rate = params_.codec_sample_rate;
        auto provider = [q, st, sse, raw_pcm, rate, first = true](std::size_t, httplib::DataSink & sink) mutable {
            std::vector<std::int16_t> chunk;
            std::string out;
            if (first && !sse && !raw_pcm) {
                const auto hdr = audio::wav_header(rate, audio::WAV_STREAMING);
                out.assign(reinterpret_cast<const char *>(hdr.data()), hdr.size());
            }
            first = false;
            if (q->pop(chunk)) {
                if (sse) {
                    Json ev;
                    ev.set("type", "speech.audio.delta");
                    ev.set("audio", base64(chunk.data(), chunk.size() * 2));
                    out += "data: " + ev.dump() + "\n\n";
                } else {
                    out.append(reinterpret_cast<const char *>(chunk.data()), chunk.size() * 2);
                }
                if (!sink.write(out.data(), out.size())) { q->abort(); return false; }
                return true;
            }
            if (sse) {
                Json ev;
                ev.set("type", q->error().empty() ? "speech.audio.done" : "error");
                if (!q->error().empty()) ev.set("error", q->error());
                Json usage;
                usage.set("input_tokens", st->n_tokens); usage.set("output_tokens", st->n_frames);
                usage.set("total_tokens", st->n_tokens + st->n_frames);
                ev.set("usage", usage);
                Json navi;
                navi.set("frames", st->n_frames); navi.set("eos", st->eos); navi.set("audio_seconds", st->audio_s);
                navi.set("prefill_ms", st->prefill_ms); navi.set("frames_ms", st->frames_ms); navi.set("vocoder_ms", st->vocoder_ms);
                navi.set("ttfa_ms", st->ttfa_ms); navi.set("total_ms", st->total_ms); navi.set("rtf", st->rtf);
                ev.set("navi", navi);
                out += "data: " + ev.dump() + "\n\n";
                if (!out.empty() && !sink.write(out.data(), out.size())) { q->abort(); return false; }
            } else if (!out.empty() && !sink.write(out.data(), out.size())) {
                q->abort(); return false;
            }
            sink.done();
            return true;
        };
        res.set_chunked_content_provider(sse ? "text/event-stream" : raw_pcm ? "audio/pcm" : "audio/wav", provider,
                                         [q](bool) { q->abort(); });
        if (sse) res.set_header("Cache-Control", "no-cache");
    }

    void log_speech(const Speech & sp, const qwen3tts::SynthStats & st) {
        if (!opt_.verbose) return;
        std::fprintf(stderr,
                     "speech %s seed %llu: %d tokens, %d frames (%s), prefill %.1f frames %.1f (%.2f/frame) vocoder %.1f | "
                     "%.2f s audio in %.0f ms, RTF %.3f\n",
                     sp.voice_id.c_str(), static_cast<unsigned long long>(sp.sr.seed), st.n_tokens, st.n_frames,
                     st.eos ? "eos" : "cap", st.prefill_ms, st.frames_ms, st.n_frames ? st.frames_ms / st.n_frames : 0.0,
                     st.vocoder_ms, st.audio_s, st.total_ms, st.rtf);
    }

    // --- OpenAI dialect (DESIGN 3.1) ---------------------------------------
    void install_openai(httplib::Server & svr) {
        svr.Get("/metrics", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content(metrics_.render(opt_.model_id, store_.size()), "text/plain; version=0.0.4");
        });

        svr.Get("/health", [this](const httplib::Request &, httplib::Response & res) {
            Json j;
            j.set("status", "ok");
            j.set("model", opt_.model_id);
            j.set("voices", static_cast<std::int64_t>(store_.size()));
            res.set_content(j.dump(), "application/json");
        });

        svr.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            Json m;
            m.set("id", opt_.model_id); m.set("object", "model"); m.set("owned_by", "navi-tts");
            Json j;
            j.set("object", "list");
            j.set("data", Json::Array{m});
            res.set_content(j.dump(), "application/json");
        });

        svr.Get("/v1/audio/languages", [this](const httplib::Request &, httplib::Response & res) {
            Json::Array langs;
            for (const auto & [code, name] : LANG_CODES) {
                const auto it = params_.language_ids.find(name);
                if (it == params_.language_ids.end()) continue;
                Json l;
                l.set("code", code); l.set("name", name); l.set("id", it->second);
                langs.push_back(l);
            }
            Json j;
            j.set("languages", langs);
            res.set_content(j.dump(), "application/json");
        });

        svr.Get("/v1/audio/voices", [this](const httplib::Request &, httplib::Response & res) {
            Json::Array list;
            if (resolve_voice("default")) list.push_back(Json("default"));
            for (const auto & id : store_.ids()) list.push_back(Json(id));
            Json j;
            j.set(opt_.model_id, list);
            res.set_content(j.dump(), "application/json");
        });

        svr.Post("/v1/audio/voices", [this](const httplib::Request & req, httplib::Response & res) {
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
                const auto r = clone(name, bytes_of(file.content), ref_text);
                Json j;
                j.set("id", r.voice->id); j.set("name", r.voice->name); j.set("status", r.status);
                res.set_content(j.dump(), "application/json");
            } catch (const navi::Error & e) {
                metrics_.error("clone");
                send_error(res, 500, e.what(), "engine_error");
            } catch (const std::exception & e) {
                metrics_.error("clone");
                send_error(res, 400, e.what());
            }
        });

        svr.Delete(R"(/v1/audio/voices/(.+))", [this](const httplib::Request & req, httplib::Response & res) {
            if (!store_.remove(req.matches[1])) { send_error(res, 404, "no such voice"); return; }
            Json j;
            j.set("deleted", true);
            res.set_content(j.dump(), "application/json");
        });

        svr.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            Json body;
            try { body = Json::parse(req.body); } catch (const std::exception &) { send_error(res, 400, "invalid JSON"); return; }
            if (!body.is_object() || !body.find("input") || !body.get("input").is_string()) { send_error(res, 400, "'input' is required"); return; }
            try {
                auto sp = std::make_shared<Speech>();
                if (!prepare(*sp, body.get("input").as_string(), body.str_or("voice", "default"), body.str_or("language", "en"), res)) return;
                sp->sr.seed = body.find("seed") ? static_cast<std::uint64_t>(body.get("seed").as_int()) : random_seed();
                sp->sr.max_frames = static_cast<int>(body.int_or("max_audio_tokens", opt_.max_audio_tokens));
                if (sp->sr.max_frames < 1 || sp->sr.max_frames > 4096) { send_error(res, 400, "'max_audio_tokens' out of range"); return; }
                sp->sr.sampling.temperature = static_cast<float>(body.num_or("temperature", params_.temperature));
                sp->sr.sampling.top_k = static_cast<int>(body.int_or("top_k", params_.top_k));
                sp->sr.sampling.repetition_penalty = static_cast<float>(body.num_or("repetition_penalty", params_.repetition_penalty));
                const std::string fmt = body.str_or("response_format", "wav");
                if (fmt != "wav" && fmt != "pcm") { send_error(res, 400, "'response_format' must be wav or pcm"); return; }
                const std::string stream = body.str_or("stream_format", "");
                if (stream == "audio" || stream == "sse") {
                    // Batching does not change the PCM, only when it arrives: a small first
                    // batch for time to first audio, then bigger ones for the vocoder's sake.
                    sp->sr.vocoder_batch = static_cast<int>(body.int_or("stream_batch_size", STREAM_BATCH));
                    if (sp->sr.vocoder_batch < 1 || sp->sr.vocoder_batch > 64) { send_error(res, 400, "'stream_batch_size' out of range (1-64)"); return; }
                    sp->sr.first_batch = std::min(STREAM_FIRST_BATCH, sp->sr.vocoder_batch);
                    speak_stream(sp, stream == "sse", fmt == "pcm", res);
                } else if (stream.empty()) {
                    sp->sr.vocoder_batch = 0;   // whole body: the vocoder's max batch is the cheapest
                    speak_wav(*sp, fmt == "pcm", res);
                } else { send_error(res, 400, "'stream_format' must be audio or sse"); return; }
            } catch (const navi::Error & e) {
                send_error(res, 500, e.what(), "engine_error");
            } catch (const std::exception & e) {
                send_error(res, 400, e.what());
            }
        });
    }

    // --- XTTS dialect (DESIGN 3.1): what SkyrimNet's XTTSInterface sends ----
    // Speakers are Skyrim voice types (malecommoner, femaleeventoned, ...),
    // cloned on first use from a sample SkyrimNet uploads with the request.
    // The voice id is the name; an unknown name falls back by gender prefix.

    // One fixed seed per speaker, so an NPC sounds the same line to line.
    static std::uint64_t stable_seed(const std::string & speaker) {
        const std::string h = voices::Store::sha256_hex(bytes_of(speaker));
        return std::stoull(h.substr(0, 8), nullptr, 16);
    }

    voices::VoicePtr resolve_xtts_speaker(const std::string & speaker, std::string & used) {
        if (!speaker.empty()) {
            used = voices::Store::id_for(speaker);
            if (auto v = store_.find(used)) return v;
        }
        std::string low;
        for (const char c : speaker) low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        used = low.rfind("male", 0) == 0 ? opt_.xtts_fallback_male : opt_.xtts_fallback_female;
        return resolve_voice(used);
    }

    void xtts_error(httplib::Response & res, int status, const std::string & message) {
        Json j;
        j.set("error", message);
        res.status = status;
        res.set_content(j.dump(), "application/json");
    }

    void install_xtts(httplib::Server & svr) {
        auto ok = [this](const httplib::Request &, httplib::Response & res) {
            Json j;
            j.set("status", "ok"); j.set("model", opt_.model_id);
            j.set("voices", static_cast<std::int64_t>(store_.size()));
            res.set_content(j.dump(), "application/json");
        };
        for (const char * p : {"/", "/docs", "/get_folders", "/get_models_list", "/sample", "/get_tts_settings"}) svr.Get(p, ok);

        // /speakers is a list of names, /speakers_list a map; both cheap.
        svr.Get("/speakers", [this](const httplib::Request &, httplib::Response & res) {
            Json::Array names;
            for (const auto & id : store_.ids()) names.push_back(Json(id));
            res.set_content(Json(names).dump(), "application/json");
        });
        svr.Get("/speakers_list", [this](const httplib::Request &, httplib::Response & res) {
            Json j = Json(Json::Object{});
            for (const auto & id : store_.ids()) {
                Json e;
                e.set("speaker_name", id);
                j.set(id, e);
            }
            res.set_content(j.dump(), "application/json");
        });
        svr.Get("/languages", [this](const httplib::Request &, httplib::Response & res) {
            Json::Array codes;
            for (const auto & [code, name] : LANG_CODES) if (params_.language_ids.count(name)) codes.push_back(Json(code));
            res.set_content(Json(codes).dump(), "application/json");
        });

        // {text, speaker_wav (= the name), language} -> audio/wav. JSON or form fields.
        auto tts = [this](const httplib::Request & req, httplib::Response & res) {
            std::string text, speaker, language;
            auto field = [&](std::initializer_list<const char *> keys, const Json * body) -> std::string {
                for (const char * k : keys) {
                    if (body && body->is_object() && body->find(k) && body->get(k).is_string()) return body->get(k).as_string();
                    if (req.form.has_field(k)) return req.form.get_field(k);
                    if (req.has_param(k)) return req.get_param_value(k);
                }
                return "";
            };
            Json body;
            bool json = false;
            try { body = Json::parse(req.body); json = body.is_object(); } catch (const std::exception &) {}
            const Json * b = json ? &body : nullptr;
            text = field({"text", "input"}, b);
            speaker = field({"speaker_wav", "speaker", "speaker_name", "voice"}, b);
            language = field({"language"}, b);
            if (language.empty()) language = "en";
            language = language.substr(0, language.find('-'));   // en-US -> en
            if (text.find_first_not_of(" \t\r\n") == std::string::npos) { xtts_error(res, 400, "empty text"); return; }
            try {
                Speech sp;
                std::string used;
                const voices::VoicePtr voice = resolve_xtts_speaker(speaker, used);
                if (!voice) { xtts_error(res, 400, "no voice for speaker '" + speaker + "' and no fallback registered"); return; }
                if (!prepare(sp, text, used, language, res)) return;
                sp.sr.seed = stable_seed(speaker.empty() ? used : speaker);
                speak_wav(sp, false, res);
            } catch (const navi::Error & e) {
                xtts_error(res, 500, e.what());
            } catch (const std::exception & e) {
                xtts_error(res, 400, e.what());
            }
        };
        for (const char * p : {"/tts_to_audio", "/tts_to_audio/", "/tts_stream", "/tts_stream/", "/tts_to_file", "/tts_to_file/"}) svr.Post(p, tts);

        // multipart wav_file + speaker_name (or a raw RIFF body with the name in the query) -> clone under the name
        auto clone_speaker = [this](const httplib::Request & req, httplib::Response & res) {
            std::string wav, name, ref_text;
            for (const char * k : {"wav_file", "wav", "file", "audio", "audio_sample", "speaker_wav", "sample"}) {
                if (req.form.has_file(k) && is_riff(req.form.get_file(k).content)) { wav = req.form.get_file(k).content; break; }
            }
            if (wav.empty()) {
                for (const auto & [k, f] : req.form.files) if (is_riff(f.content)) { wav = f.content; break; }
            }
            if (wav.empty() && is_riff(req.body)) wav = req.body;
            for (const char * k : {"speaker_name", "name", "speaker", "voice_name"}) {
                if (req.form.has_field(k)) { name = req.form.get_field(k); break; }
                if (req.has_param(k)) { name = req.get_param_value(k); break; }
            }
            for (const char * k : {"ref_text", "text"}) if (req.form.has_field(k)) { ref_text = req.form.get_field(k); break; }
            if (wav.empty() || name.empty()) {
                if (opt_.verbose) std::fprintf(stderr, "xtts clone missing %s (%s)\n", wav.empty() ? "wav" : "name", req.path.c_str());
                xtts_error(res, 400, "need a wav file and a speaker name");
                return;
            }
            try {
                const auto r = clone(name, bytes_of(wav), ref_text);
                Json j;
                j.set("status", "ok"); j.set("speaker_name", name); j.set("name", name);
                j.set("voice_id", r.voice->id); j.set("latents", r.voice->id); j.set("message", r.status);
                res.set_content(j.dump(), "application/json");
            } catch (const navi::Error & e) {
                metrics_.error("clone");
                xtts_error(res, 500, e.what());
            } catch (const std::exception & e) {
                metrics_.error("clone");
                xtts_error(res, 400, e.what());
            }
        };
        for (const char * p : {"/create_and_store_latents", "/create_and_store_latents/", "/clone_speaker", "/set_speaker", "/upload_speaker"}) svr.Post(p, clone_speaker);
    }

    qwen3tts::Graph & graph_;
    voices::Store & store_;
    const Options & opt_;
    const qwen3tts::Params & params_;
    Worker worker_;
    Metrics metrics_;
    std::mt19937_64 seed_rng_;
    std::mutex rng_mutex_;
};

} // namespace

// Every stage once, so the first real request runs at steady state: kernel
// and buffer first-touch happen here, not on an utterance somebody waits for.
void warm_up(qwen3tts::Graph & graph, const voices::Store & store, bool verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    const qwen3tts::Params & p = graph.params();
    const std::vector<float> silence(static_cast<std::size_t>(p.spk_sample_rate) * 2, 0.f);
    std::vector<float> speaker = graph.embed_speaker(silence);
    if (const auto ids = store.ids(); !ids.empty()) speaker = store.find(ids.front())->embedding;
    qwen3tts::SynthRequest sr;
    sr.text = "Warm up.";
    sr.speaker = speaker;
    sr.max_frames = 16;
    sr.sampling.temperature = p.temperature;
    sr.sampling.top_k = p.top_k;
    sr.sampling.repetition_penalty = p.repetition_penalty;
    sr.sampling.cp_temperature = p.cp_temperature;
    sr.sampling.cp_top_k = p.cp_top_k;
    std::vector<float> pcm;
    const auto st = graph.synth(sr, pcm);
    if (verbose) {
        std::fprintf(stderr, "warm-up: %d frames, prefill %.1f ms, %.2f ms/frame, vocoder %.1f ms; %.0f ms total\n", st.n_frames,
                     st.prefill_ms, st.n_frames ? st.frames_ms / st.n_frames : 0.0, st.vocoder_ms,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}

int run(qwen3tts::Graph & graph, voices::Store & store, const Options & opt) {
    if (opt.warmup) warm_up(graph, store, opt.verbose);
    Service service(graph, store, opt);
    httplib::Server svr, xtts;
    service.install(svr);
    g_servers = {&svr};
    std::thread xtts_thread;
    if (opt.xtts_port > 0) {
        service.install(xtts);
        g_servers.push_back(&xtts);
        if (!xtts.bind_to_port(opt.host, opt.xtts_port)) {
            std::fprintf(stderr, "navi-tts: cannot listen on %s:%d (xtts)\n", opt.host.c_str(), opt.xtts_port);
            return 1;
        }
        xtts_thread = std::thread([&] { xtts.listen_after_bind(); });
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "navi-tts serving %s on http://%s:%d%s, %zu voice(s) in %s\n", opt.model_id.c_str(), opt.host.c_str(),
                 opt.port, opt.xtts_port > 0 ? (" and :" + std::to_string(opt.xtts_port) + " (xtts)").c_str() : "",
                 store.size(), store.dir().c_str());
    const bool ok = svr.listen(opt.host, opt.port);
    if (opt.xtts_port > 0) { xtts.stop(); xtts_thread.join(); }
    g_servers.clear();
    if (!ok) { std::fprintf(stderr, "navi-tts: cannot listen on %s:%d\n", opt.host.c_str(), opt.port); return 1; }
    return 0;
}

} // namespace navi::server
