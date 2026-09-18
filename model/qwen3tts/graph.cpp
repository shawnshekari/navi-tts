#include "model/qwen3tts/graph.h"

#include "runtime/common/error.h"
#include "runtime/weights/navi_file.h"

#include <chrono>

namespace navi::qwen3tts {

struct Graph::Impl {
    Params params;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<DeviceWeights> weights;
    std::unique_ptr<Talker> talker;
    std::unique_ptr<Frame> frame;
    std::unique_ptr<Vocoder> vocoder;
    std::unique_ptr<SpeakerEncoder> speaker;
    std::vector<std::int32_t> codes;
    std::vector<float> pcm;
};

std::unique_ptr<Graph> Graph::load(const Device & dev, const std::string & navi_path, const FrameOptions & fopt, int vocoder_chunk) {
    std::unique_ptr<Graph> g(new Graph());
    g->impl_ = std::make_unique<Impl>();
    Impl & I = *g->impl_;
    NaviFile file = NaviFile::open(navi_path);
    I.params = read_params(file);
    const Validation v = validate(file, I.params);
    if (!v.ok()) fail(navi_path + ": tensor table does not match the model (navi-tts info --model)");
    I.tokenizer = std::make_unique<Tokenizer>(Tokenizer::from_navi(file));
    I.weights = std::make_unique<DeviceWeights>(DeviceWeights::upload(file, [](const TensorInfo & t) {
        if (t.dtype == DType::U8) return false;
        if (t.name.rfind("speaker_encoder.", 0) == 0) return true;
        if (t.name.rfind("codec.encoder.", 0) == 0) return false;   // ICL encoder: not used
        return !vocoder_owns_tensor(t.name);
    }));
    I.talker = Talker::create(dev, *I.weights, I.params);
    I.frame = Frame::create(dev, *I.weights, I.params, fopt);
    I.vocoder = Vocoder::create(dev, file, *I.weights, I.params, vocoder_chunk);
    I.speaker = SpeakerEncoder::create(dev, *I.weights, I.params);
    file.close();
    return g;
}

Graph::~Graph() = default;

const Params & Graph::params() const { return impl_->params; }
const Tokenizer & Graph::tokenizer() const { return *impl_->tokenizer; }
const DeviceWeights & Graph::weights() const { return *impl_->weights; }
std::vector<float> Graph::embed_speaker(std::span<const float> wav) { return impl_->speaker->embed(wav); }

SynthStats Graph::synth(const SynthRequest & req, const std::function<void(std::span<const float>)> & on_pcm) {
    using clock = std::chrono::steady_clock;
    Impl & I = *impl_;
    const Params & p = I.params;
    SynthStats st;
    const auto t0 = clock::now();
    auto ms_since = [&](clock::time_point a) { return std::chrono::duration<double, std::milli>(clock::now() - a).count(); };

    const std::string prompt_text = "<|im_start|>assistant\n" + req.text + "<|im_end|>\n<|im_start|>assistant\n";
    const auto ids = I.tokenizer->encode(prompt_text);
    st.n_tokens = static_cast<int>(ids.size()) - 8;
    if (st.n_tokens < 1) fail("empty text");

    const Prompt & pr = I.talker->build_prompt(ids, req.language, req.speaker);
    I.talker->prefill(pr);
    st.prefill_ms = ms_since(t0);

    I.frame->reset();
    I.vocoder->reset();
    I.codes.clear();
    const int H = p.talker.hidden;
    const int batch = req.vocoder_batch > 0 ? std::min(req.vocoder_batch, I.vocoder->max_chunk_frames()) : I.vocoder->max_chunk_frames();
    int pending = 0;   // frames decoded by the talker but not yet vocoded
    bool first_audio = true;
    auto flush = [&](bool final) {
        if (pending == 0 || (!final && pending < batch)) return;
        const auto tv = clock::now();
        I.pcm.clear();
        const std::size_t off = I.codes.size() - static_cast<std::size_t>(pending) * 16;
        I.vocoder->decode(std::span<const std::int32_t>(I.codes.data() + off, static_cast<std::size_t>(pending) * 16), pending, I.pcm);
        st.vocoder_ms += ms_since(tv);
        pending = 0;
        if (first_audio) { st.ttfa_ms = ms_since(t0); first_audio = false; }
        on_pcm(I.pcm);
    };

    const auto tf = clock::now();
    for (int i = 0; i < req.max_frames; ++i) {
        const float * trailing = i < pr.n_trailing ? pr.trailing + static_cast<std::size_t>(i) * H : pr.tts_pad;
        FrameResult r = i == 0
            ? I.frame->run_first(I.talker->d_hidden(), I.talker->d_logits(), trailing, req.sampling, req.seed, i)
            : I.frame->run(I.talker->kv(), pr.n_prefill + i - 1, trailing, req.sampling, req.seed, i);
        if (!r.ok) fail("frame " + std::to_string(i) + ": " + r.error);
        st.frames_ms += r.us / 1000.0;
        if (r.eos) { st.eos = true; break; }
        I.codes.insert(I.codes.end(), r.codes, r.codes + 16);
        ++st.n_frames;
        ++pending;
        flush(false);
    }
    st.hit_cap = !st.eos;
    (void) tf;
    flush(true);
    st.total_ms = ms_since(t0);
    st.audio_s = static_cast<double>(st.n_frames) * p.codec_upsample / p.codec_sample_rate;
    st.rtf = st.audio_s > 0 ? st.total_ms / 1000.0 / st.audio_s : 0.0;
    return st;
}

SynthStats Graph::synth(const SynthRequest & req, std::vector<float> & pcm) {
    pcm.clear();
    return synth(req, [&](std::span<const float> chunk) { pcm.insert(pcm.end(), chunk.begin(), chunk.end()); });
}

} // namespace navi::qwen3tts
