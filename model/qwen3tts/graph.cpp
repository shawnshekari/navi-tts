#include "model/qwen3tts/graph.h"

#include "runtime/common/error.h"
#include "runtime/weights/navi_file.h"

#include <algorithm>
#include <cstdio>
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
    std::unique_ptr<CodecEncoder> codec_encoder;   // null unless the blob carries one
    GraphOptions gopt;
    std::vector<std::int32_t> codes;
    std::vector<float> pcm;
};

std::unique_ptr<Graph> Graph::load(const Device & dev, const std::string & navi_path, const FrameOptions & fopt,
                                   int vocoder_chunk, const GraphOptions & gopt) {
    std::unique_ptr<Graph> g(new Graph());
    g->impl_ = std::make_unique<Impl>();
    Impl & I = *g->impl_;
    NaviFile file = NaviFile::open(navi_path);
    I.params = read_params(file);
    const Validation v = validate(file, I.params);
    if (!v.ok()) fail(navi_path + ": tensor table does not match the model (navi-tts info --model)");
    I.gopt = gopt;
    I.tokenizer = std::make_unique<Tokenizer>(Tokenizer::from_navi(file));
    const bool with_encoder = I.params.has_codec_encoder;
    I.weights = std::make_unique<DeviceWeights>(DeviceWeights::upload(file, [&](const TensorInfo & t) {
        if (t.dtype == DType::U8) return false;
        if (t.name.rfind("speaker_encoder.", 0) == 0) return true;
        if (t.name.rfind("codec.encoder.", 0) == 0) return with_encoder;   // 190 MB, once per clone
        return !vocoder_owns_tensor(t.name);
    }, Talker::fused_groups(I.params)));
    I.talker = Talker::create(dev, *I.weights, I.params, gopt.max_prompt_tokens, gopt.max_positions,
                              gopt.max_ref_frames);
    I.frame = Frame::create(dev, *I.weights, I.params, fopt);
    I.vocoder = Vocoder::create(dev, file, *I.weights, I.params, vocoder_chunk);
    I.speaker = SpeakerEncoder::create(dev, *I.weights, I.params);
    // The encoder is what *makes* reference codes; a blob without it still serves voices
    // whose codes were made by one that has it. So a blob whose encoder this build cannot
    // use (the wrong dtype, say) loses in-context *cloning* and nothing else: it is reported
    // and the model still serves, rather than refusing to load over a component that only
    // one operation needs. An explicit ICL request then fails with the encoder's own message.
    if (with_encoder) {
        try {
            I.codec_encoder = CodecEncoder::create(dev, file, *I.weights, I.params);
        } catch (const std::exception & ex) {
            std::fprintf(stderr, "%s: the codec encoder will not load, so this model cannot make reference "
                                 "codes for in-context cloning (it still serves voices that have them): %s\n",
                         navi_path.c_str(), ex.what());
        }
    }
    file.close();
    return g;
}

Graph::~Graph() = default;

const Params & Graph::params() const { return impl_->params; }
const Tokenizer & Graph::tokenizer() const { return *impl_->tokenizer; }
const DeviceWeights & Graph::weights() const { return *impl_->weights; }
std::vector<float> Graph::embed_speaker(std::span<const float> wav) { return impl_->speaker->embed(wav); }
bool Graph::has_codec_encoder() const { return impl_->codec_encoder != nullptr; }
int Graph::max_ref_frames() const { return impl_->gopt.max_ref_frames; }

std::vector<std::int32_t> Graph::encode_reference(std::span<const float> wav, int * n_frames) {
    Impl & I = *impl_;
    if (!I.codec_encoder) {
        fail("this model has no codec encoder, so it cannot make reference codes for in-context "
             "cloning (tools/convert.py --with-encoder adds it)");
    }
    return I.codec_encoder->encode(wav, n_frames);
}

SynthStats Graph::synth(const SynthRequest & req, const std::function<void(std::span<const float>)> & on_pcm) {
    using clock = std::chrono::steady_clock;
    Impl & I = *impl_;
    const Params & p = I.params;
    SynthStats st;
    const auto t0 = clock::now();
    auto ms_since = [&](clock::time_point a) { return std::chrono::duration<double, std::milli>(clock::now() - a).count(); };

    const std::string prompt_text = "<|im_start|>assistant\n" + req.text + "<|im_end|>\n<|im_start|>assistant\n";
    const auto ids = I.tokenizer->encode(prompt_text);
    st.tokenize_ms = ms_since(t0);
    st.n_tokens = static_cast<int>(ids.size()) - 8;
    if (st.n_tokens < 1) fail("empty text");

    // In-context cloning when the request carries both halves of a reference. The
    // transcript is wrapped as its own assistant turn, which is the form build_prompt
    // slices; the codes come from a codec encoder, here or on another box.
    std::vector<std::int32_t> ref_ids;
    IclRef icl;
    const bool use_icl = req.ref_frames > 0 && !req.ref_text.empty();
    if (use_icl) {
        ref_ids = I.tokenizer->encode("<|im_start|>assistant\n" + req.ref_text + "<|im_end|>\n");
        icl.ids = ref_ids;
        icl.codes = req.ref_codes;
        icl.n_frames = req.ref_frames;
    } else if (req.ref_frames > 0 || !req.ref_text.empty()) {
        // Half a reference is not a usable prompt; say which half is missing rather than
        // quietly cloning by embedding alone.
        fail(req.ref_frames > 0 ? "in-context cloning needs the reference transcript as well as its codes"
                                : "in-context cloning needs the reference's codes as well as its transcript");
    }
    const Prompt & pr = I.talker->build_prompt(ids, req.language, req.speaker, use_icl ? &icl : nullptr);
    st.prefill_rows = pr.n_prefill;
    st.icl = use_icl;
    // Frame i attends at position n_prefill + i - 1, so max_pos - n_prefill + 1 frames fit;
    // the cap below is one short of that, deliberately, rather than cut fine. Clamping
    // instead of refusing keeps the contract that max_frames is a cap whose being reached
    // is reported as hit_cap (DESIGN 2) - a request that would have ended on EOS long
    // before it should not be refused for asking for more than fits.
    const int room = I.talker->kv().max_pos - pr.n_prefill;
    st.frame_cap = std::min(req.max_frames, room);
    if (st.frame_cap < 1) {
        fail("the prompt is " + std::to_string(pr.n_prefill) + " positions and the KV cache holds "
             + std::to_string(I.talker->kv().max_pos) + ": no room to generate");
    }
    I.talker->prefill(pr);
    st.prefill_ms = ms_since(t0);

    I.frame->reset();
    I.vocoder->reset();
    I.codes.clear();
    const int H = p.talker.hidden;
    const int batch = req.vocoder_batch > 0 ? std::min(req.vocoder_batch, I.vocoder->max_chunk_frames()) : I.vocoder->max_chunk_frames();
    const int first = req.first_batch > 0 ? std::min(req.first_batch, batch) : batch;
    int pending = 0;   // frames decoded by the talker but not yet vocoded
    bool first_audio = true;
    auto flush = [&](bool final) {
        if (pending == 0 || (!final && pending < (first_audio ? first : batch))) return;
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
    for (int i = 0; i < st.frame_cap; ++i) {
        const float * trailing = i < pr.n_trailing ? pr.trailing + static_cast<std::size_t>(i) * H : pr.tts_pad;
        FrameResult r = i == 0
            ? I.frame->run_first(I.talker->d_hidden(), I.talker->d_logits(), trailing, req.sampling, req.seed, i)
            : I.frame->run(I.talker->kv(), pr.n_prefill + i - 1, trailing, req.sampling, req.seed, i);
        if (!r.ok) fail("frame " + std::to_string(i) + ": " + r.error);
        st.frames_ms += r.us / 1000.0;
        if (req.frame_ms) req.frame_ms->push_back(r.us / 1000.0);
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
