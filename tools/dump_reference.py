#!/usr/bin/env python3
"""PyTorch reference dumps for the parity gates (docs/DESIGN.md 6).

    uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-0.6B-Base --out ../tests/reference

Runs the HF model on the CPU in float32 (the shared GPU stays untouched, and
f32 on CPU is the reference we want) for a fixed text and reference voice, and
writes what each engine stage is checked against:

  text_ids.npy            token ids of the assistant-formatted prompt (BPE gate)
  tokenizer_cases.json    more strings -> ids for the tokenizer tests
  speaker_mels.npy        [n_mels, T] mel input to the speaker encoder
  speaker_embedding.npy   [speaker enc_dim] (speaker encoder gate)
  ref_conv.npy            [T25, 512] the codec encoder's SEANet conv stack output (25 Hz)
  ref_xf.npy              [T25, 512] after its 8-layer transformer (full causal: the config's
                          sliding_window is not applied by transformers 4.57, measured)
  ref_latents.npy         [T, 512] after the stride-2 downsample - the last float before any
                          argmin, and the encoder's float gate. T = ceil(T25 / 2).
  ref_codes.npy           [T, 16] the codec encoder's codes for the reference audio (ICL)
  ref_codes_odd.npy       the same for the clip truncated by 777 samples, so its length is not a
  ref_latents_odd.npy     multiple of the frame size: every conv then gets a non-zero right pad
                          and the strides stop dividing the chunk lengths. Real references are
                          this shape (a 20 s clip is 249.75 frames), and the aligned clip
                          never takes that path.
  ref_codes_f16w.npy      the same codes with every >=2-D encoder weight rounded to f16 the way
                          convert.py stores the rest of the model. NOT a gate: it documents why
                          the encoder is stored f32 (manifest ref_codes_f16w_diff). An RVQ is a
                          cascade of argmins with no error averaging, so a tie in one codebook
                          changes every later codebook of that frame.
  prefill_embeds.npy      [T, talker hidden] the talker prefill input (embeddings, after text_projection)
  prefill_mask.npy        [T] attention mask (all ones at batch one)
  prefill_position_ids.npy [3, T] M-RoPE positions
  trailing_text.npy       [n, talker hidden] the per-frame trailing text rows (text tokens[1:] + tts_eos)
  tts_pad_embed.npy       [talker hidden] the row added once trailing text is exhausted
  prefill_hidden.npy      [talker hidden] post-norm talker hidden at the last prefill position
  prefill_logits.npy      [3072] codec_head logits at that position
  cp_inputs.npy           [16, cp hidden] the code predictor's 16 inputs for greedy frame 0, after
                          its small_to_mtp_projection. The predictor's embeddings are stored at the
                          TALKER's width and projected down to its own when the two differ (1.7B);
                          when they are equal (0.6B) that stage is nn.Identity() and this is the raw
                          embeddings. Gates the engine's cp_embed, projection stage included.
  prefill_kv.npz          k_{l}, v_{l}: [T, n_kv_head, head_dim] after RoPE, per talker layer

  The ICL (in-context cloning) set below conditions on the reference audio's codes plus a
  transcript, instead of the speaker embedding alone. Its ref_text is a FIXED STRING that does
  not describe voice_1.wav (no transcript exists for it): parity does not depend on it being
  true, only on it being the same string the engine is given. Nothing in this set is a listen
  reference - see the README.

  icl_ref_text_ids.npy    token ids of the reference transcript wrapped as its own assistant
                          turn ("<|im_start|>assistant\n{ref}<|im_end|>\n"); the engine is given
                          these and uses ids[3:-2]
  icl_prefill_embeds.npy  [T_icl, talker hidden] the ICL prefill input. T_icl = 9 + 1 + T_ref:
                          the 9 prefix rows of the embedding-only prompt, then codec_bos, then
                          one row per reference frame (16 code embeddings summed at the TALKER's
                          width) with the reference transcript + the target text + tts_eos
                          summed in on the text side, right-padded with tts_pad.
  icl_trailing_text.npy   [n, talker hidden] its per-frame trailing text rows: the text that did
                          not fit in the block, else a single tts_pad row (the usual case).
  icl_prefill_hidden.npy  [talker hidden] post-norm hidden at its last prefill position
  icl_prefill_logits.npy  [3072] codec_head logits there
  icl_greedy_codes.npy    [n_frames, 16] the ICL greedy run (the same code-for-code gate as
                          greedy_codes, and the same warning: it is not speech)
  icl_frame0_codes.npy    [16] its first frame
  icl_sampled_codes.npy   a seeded sampled ICL run. METADATA ONLY: the engine's sampler is not
                          torch's (DESIGN 1), so its frame count is a sanity comparison, not a
                          target, and its mismatched transcript makes it useless to listen to.
  icl_short_prefill_embeds.npy / icl_short_trailing_text.npy
                          the same prompt built from only the first icl_short_ref_frames frames
                          of the reference, so the target text overflows the codec block and the
                          trailing text is non-empty - the other branch of the alignment.
  greedy_codes.npy        [n_frames, 16] argmax run with the default repetition penalty (the
                          talker / code-predictor gate). Argmax decoding collapses into a loop of
                          silence codes after ~10 frames, so it is capped and never used as audio.
  frame0_codes.npy        [16] its first frame
  sampled_codes.npy       [n_frames, 16] a seeded sampled run with the model's default sampling:
                          real speech, the fixed code sequence for the vocoder gate
  sampled_pcm.npy         [n_samples] f32 decode of sampled_codes; sampled.wav is the same as s16
  sampled_pcm_f16w.npy    the same decode with every >=2-D decoder weight rounded to f16 (what the
                          engine's weights are): the kernel-exactness gate. sampled_pcm.npy vs this
                          is the quantisation budget, not something the engine can close.
  vocoder16_pcm.npy       decode of the first 16 frames of sampled_codes only
  manifest.json           what produced all of the above
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BENCH_TEXT = "The quick brown fox jumps over the lazy dog, and the bench text stays fixed."
# The ICL reference transcript. voice_1.wav has no transcript and this box has no ASR, so this
# is a fixed string that does not describe it - fine for parity (the engine gets the same
# string), useless for listening, and labelled as such in the manifest and the README.
REF_TEXT = "This is the reference recording used by the parity dumps."
TOKENIZER_CASES = [
    "Hello, world!",
    "It's 3.5:1 at 10:45 PM on 2026-09-17.",
    "naïve café — résumé… “quotes” and 'apostrophes'",
    "    leading spaces and\ttabs\nnewlines",
    "日本語のテキストと中文混合 текст",
    "<|im_start|>assistant\nplain<|im_end|>\n",
    "x" * 300,
    # NFC: decomposed input must tokenize like its composed form
    "cafe\u0301 A\u030a ngstro\u0308m e\u0327\u0301 \u1112\u1161\u11ab\uae00 \u212b \u2126",
    "café Å ångström ȩ́ 한글 Å Ω",
    # case-aware split, CamelCase, all caps, mixed
    "HelloWorld CamelCase HTTPServer XMLHttpRequest iPhone macOS ALLCAPS lowerUPPER ΑΒΓαβγ ÉCOLE École",
    # the [\r\n/]* tail on the punctuation alternative
    "a/b/c path/to/file.txt http://x.y/z ...///  -/\n/x",
    # whitespace shapes
    "\r\n", "a\r\nb", "a\n\n\nb", "a \n b", "a\n \n  b", "trailing   ", "  ", " ", "\t\t", "a\u00a0b \u3000c\u2003d",
    "line1\nline2\n", "x\n\n", " \n\n ", "\n  x", "  \n",
    # numbers
    "3.14159 1,000,000 ١٢٣ ๑๒๓ Ⅻ ½ 2²",
    # punctuation runs and symbols
    "!!! ??? #hashtag @user $100 €50 50% 100°C (a) [b] {c} <d> a=b+c*d/e",
    "\"quoted\" 'single' `back` «guillemets» 「括弧」",
    # apostrophes and contractions, case-insensitive
    "don't DON'T Don'T can't've it's I'm we'll they'd o'clock rock'n'roll ’smart’",
    # emoji, ZWJ, skin tone, flags, non-BMP
    "👍🏽 👨‍👩‍👧 🇺🇸 🏳️‍🌈 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 😀😀😀",
    # marks and letterless marks
    "\u0301 1\u0301 a\u0301\u0301 \u05d0\u05b8 \u0e01\u0e34 \u0928\u094d\u0924\u0947",
    # control characters, unusual whitespace, DEL
    "a\x01b\x1cc\x7fd\x85e",
    # code-ish
    "def f(x):\n    return x**2  # comment\n",
    "if (a && b) { c(); }\n",
    # long words and repeats
    "a" * 100, "ab" * 100, "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "😀" * 40, "  " * 20 + "x",
    # special-token-like text that is not an added token, and the real ones mid-text
    "<|im_start|> <|notatoken|> <tool_call>x</tool_call> <|endoftext|><|im_end|>",
    "assistant\n", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    # the household TTS queue's typical outputs
    "Build finished: 3 warnings, 0 errors. Running tests...",
    "Ok — I've updated tts_queue.py (lines 42–58) and restarted the service.",
    "Yes.", "No", "", " ",
]


def save(path: Path, arr: np.ndarray) -> None:
    """np.save, but always C-contiguous.

    A transposed torch tensor's .numpy() is a Fortran-order view, and np.save records that
    in the header - which the engine's Npy reader refuses, rightly: nothing in this repo
    wants two layouts. Several dumps are transposes, so every save goes through here.
    """
    np.save(path, np.ascontiguousarray(arr))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wav(path: Path, sr_expected: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise SystemExit(f"{path}: {sr} Hz, need {sr_expected} (ffmpeg -ar {sr_expected} -ac 1)")
    return wav.mean(axis=1).astype(np.float32)


def encoder_stages(stm, wav: np.ndarray) -> dict:
    """Walk the codec encoder stage by stage, so a parity failure localises to a stage.

    The stages are the engine's: the SEANet conv stack (causal convs, ELU, residual units,
    four strided downsamples) at 25 Hz, the 8-layer transformer, the stride-2 downsample to
    12.5 Hz, then the split RVQ's argmin cascade. Activations come back **time-major**
    [T][C], which is the layout the engine's conv kernel works in.
    """
    enc = stm.encoder
    v = torch.from_numpy(wav).reshape(1, 1, -1)
    conv = enc.encoder(v)                                             # [1, 512, T25]
    xf = enc.encoder_transformer(conv.transpose(1, 2))[0].transpose(1, 2)
    lat = enc.downsample(xf)                                          # [1, 512, T]
    codes = enc.quantizer.encode(lat, stm.encoder_valid_num_quantizers)[:, 0]   # [16, T]

    def tc(x):
        return x[0].transpose(0, 1).contiguous().numpy().astype(np.float32)

    return dict(conv=tc(conv), xf=tc(xf), latents=tc(lat),
                codes=codes.transpose(0, 1).contiguous().numpy().astype(np.int32))


def f16_round_encoder(stm) -> int:
    """Round the codec encoder the way convert.py stores weights: >=2-D tensors to f16, 1-D
    left alone, and the codebooks divided in f32 first (embed_sum / clamp(cluster_usage)) and
    only then rounded. Destructive - call it last."""
    enc = stm.encoder
    n = 0
    for _, p in enc.named_parameters():
        if p.dim() >= 2:
            p.copy_(p.to(torch.float16).to(torch.float32))
            n += 1
    for rvq in (enc.quantizer.semantic_residual_vector_quantizer,
                enc.quantizer.acoustic_residual_vector_quantizer):
        for layer in rvq.layers:
            cb = layer.codebook
            book = cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]
            cb._embed = book.to(torch.float16).to(torch.float32)
            n += 1
    return n


def write_vocoder_refs(model, sampled: np.ndarray, out: Path) -> np.ndarray:
    """sampled_pcm.npy / sampled.wav (f32 weights), vocoder16_pcm.npy, and sampled_pcm_f16w.npy."""
    import copy
    st = model.speech_tokenizer
    codes = torch.from_numpy(sampled).long()
    wavs, fs = st.decode([{"audio_codes": codes}])
    pcm = np.asarray(wavs[0], dtype=np.float32)
    save(out / "sampled_pcm.npy", pcm)
    sf.write(out / "sampled.wav", pcm, fs, subtype="PCM_16")
    wavs16, _ = st.decode([{"audio_codes": codes[:16]}])
    save(out / "vocoder16_pcm.npy", np.asarray(wavs16[0], dtype=np.float32))

    # f16-rounded weights, exactly as tools/convert.py stores them: every >=2-D
    # tensor, with the EMA codebooks baked first and then rounded.
    dec = copy.deepcopy(st.model.decoder)
    for name, p in dec.named_parameters():
        if p.ndim >= 2 and "embedding_sum" not in name:
            p.data = p.data.half().float()
    for m in dec.modules():
        if hasattr(m, "embedding_sum"):
            usage = m.cluster_usage.clamp(min=m.epsilon)[:, None]
            m.embedding_sum.data = (m.embedding_sum / usage).half().float() * usage
    pcm16w = dec(codes.T.unsqueeze(0))[0, 0].numpy().astype(np.float32)
    save(out / "sampled_pcm_f16w.npy", pcm16w)
    return pcm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="HF model directory")
    ap.add_argument("--out", type=Path, required=True, help="output directory (tests/reference)")
    ap.add_argument("--voice", type=Path, default=None, help="reference wav, 24 kHz mono (default <out>/voice_1.wav)")
    ap.add_argument("--text", default=BENCH_TEXT)
    ap.add_argument("--language", default="english")
    ap.add_argument("--max-frames", type=int, default=32, help="cap for the greedy run")
    ap.add_argument("--sampled-max-frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2, help="torch seed for the sampled run")
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="the model's default; pure argmax (1.0) locks into a 2-frame loop and never emits EOS")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ref-text", default=REF_TEXT,
                    help="the ICL reference transcript. The default is a fixed string that does NOT "
                         "describe voice_1.wav; parity does not depend on it being true.")
    ap.add_argument("--icl-short-frames", type=int, default=8,
                    help="reference frames for the second ICL case, short enough that the target "
                         "text overflows the codec block (the other alignment branch)")
    ap.add_argument("--tokenizer-only", action="store_true", help="only text_ids.npy and tokenizer_cases.json")
    ap.add_argument("--vocoder-only", action="store_true", help="only re-decode the existing sampled_codes.npy")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    voice = a.voice or (out / "voice_1.wav")

    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(str(a.model), device_map="cpu", dtype=torch.float32)
    model = tts.model
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    cfg = model.config
    tcfg = cfg.talker_config

    if a.vocoder_only:
        sampled = np.load(out / "sampled_codes.npy")
        write_vocoder_refs(model, sampled, out)
        return

    # --- tokenizer ---------------------------------------------------------
    prompt_text = tts._build_assistant_text(a.text)
    text_ids = tts._tokenize_texts([prompt_text])[0]          # [1, 3 + T + 5]
    save(out / "text_ids.npy", text_ids[0].numpy().astype(np.int32))
    cases = {s: tts._tokenize_texts([s])[0][0].tolist() for s in TOKENIZER_CASES + [a.text]}
    (out / "tokenizer_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))

    if a.tokenizer_only:
        print(f"wrote {len(cases)} tokenizer cases", file=sys.stderr)
        return

    # --- speaker encoder -----------------------------------------------------
    sr = model.speaker_encoder_sample_rate
    wav = load_wav(voice, sr)
    mels = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=sr,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000)   # [1, n_mels, T]
    save(out / "speaker_mels.npy", mels[0].numpy().astype(np.float32))
    spk = model.extract_speaker_embedding(audio=wav, sr=sr)                  # [1024]
    save(out / "speaker_embedding.npy", spk.numpy().astype(np.float32))
    print(f"speaker embedding: {tuple(spk.shape)}, mels {tuple(mels.shape)}", file=sys.stderr)

    # --- codec encoder: the reference codes, and every stage behind them --------
    stm = model.speech_tokenizer.model                      # Qwen3TTSTokenizerV2Model
    ref_codes = model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0]   # [T, 16], authoritative
    save(out / "ref_codes.npy", ref_codes.numpy().astype(np.int32))
    stages = encoder_stages(stm, wav)
    if not np.array_equal(stages["codes"], ref_codes.numpy()):
        raise SystemExit("the stage walk does not reproduce speech_tokenizer.encode's codes")
    save(out / "ref_conv.npy", stages["conv"])
    save(out / "ref_xf.npy", stages["xf"])
    save(out / "ref_latents.npy", stages["latents"])
    # The same clip truncated to a length that is NOT a multiple of the frame size, which is
    # the usual case for real audio (a 20 s clip is 479,520 samples = 249.75 frames).
    # Every conv then gets a non-zero right "extra" pad computed from the full
    # length, and the strides stop dividing the chunk lengths - a path the aligned clip never
    # takes. Codes and latents only; the engine has the full set for the aligned case.
    odd = wav[: wav.shape[0] - 777]
    odd_stages = encoder_stages(stm, odd)
    save(out / "ref_codes_odd.npy", odd_stages["codes"])
    save(out / "ref_latents_odd.npy", odd_stages["latents"])
    print(f"unaligned case: {odd.shape[0]} samples -> {odd_stages['conv'].shape[0]} rows at 25 Hz -> "
          f"{odd_stages['codes'].shape[0]} frames", file=sys.stderr)
    print(f"codec encoder: {wav.shape[0]} samples -> {stages['conv'].shape[0]} rows at 25 Hz -> "
          f"{ref_codes.shape[0]} frames x {ref_codes.shape[1]} codebooks", file=sys.stderr)

    # --- prompt assembly, captured from the model's own generate --------------
    captured = {}
    talker = model.talker
    orig_generate = talker.generate

    def capture_generate(*args, **kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].clone()
        captured["attention_mask"] = kwargs["attention_mask"].clone()
        captured["trailing_text_hidden"] = kwargs["trailing_text_hidden"].clone()
        captured["tts_pad_embed"] = kwargs["tts_pad_embed"].clone()
        captured["suppress_tokens"] = list(kwargs.get("suppress_tokens", []))
        captured["result"] = orig_generate(*args, **kwargs)
        return captured["result"]

    talker.generate = capture_generate
    try:
        prompt = dict(ref_code=[None], ref_spk_embedding=[spk], x_vector_only_mode=[True], icl_mode=[False])
        t1 = time.time()
        codes_list, hidden_list = model.generate(
            input_ids=[text_ids], ref_ids=None, voice_clone_prompt=prompt, languages=[a.language],
            non_streaming_mode=False, max_new_tokens=a.max_frames,
            do_sample=False, subtalker_dosample=False, repetition_penalty=a.repetition_penalty,
            top_k=1, subtalker_top_k=1,
        )
        print(f"greedy run: {codes_list[0].shape[0]} frames in {time.time() - t1:.1f}s", file=sys.stderr)
    finally:
        talker.generate = orig_generate

    embeds = captured["inputs_embeds"]           # [1, T, 1024]
    mask = captured["attention_mask"]            # [1, T]
    trailing = captured["trailing_text_hidden"]  # [1, n, 1024]
    pad = captured["tts_pad_embed"]              # [1, 1, 1024]
    save(out / "prefill_embeds.npy", embeds[0].numpy().astype(np.float32))
    save(out / "prefill_mask.npy", mask[0].numpy().astype(np.int32))
    save(out / "trailing_text.npy", trailing[0].numpy().astype(np.float32))
    save(out / "tts_pad_embed.npy", pad[0, 0].numpy().astype(np.float32))

    greedy = codes_list[0].numpy().astype(np.int32)          # [n_frames, 16]
    save(out / "greedy_codes.npy", greedy)
    save(out / "frame0_codes.npy", greedy[0])

    # --- prefill on its own: hidden, logits, KV, positions ----------------------
    talker.rope_deltas = None
    pre = talker(inputs_embeds=embeds, attention_mask=mask, use_cache=True, output_hidden_states=True)
    hidden_last = pre.past_hidden[0, -1]                     # post-norm, what feeds codec_head and the cp
    logits_last = pre.logits[0, -1]
    save(out / "prefill_hidden.npy", hidden_last.numpy().astype(np.float32))
    save(out / "prefill_logits.npy", logits_last.numpy().astype(np.float32))
    position_ids, _ = talker.get_rope_index(mask)
    save(out / "prefill_position_ids.npy", position_ids[:, 0].numpy().astype(np.int32))

    # --- the code predictor's inputs for frame 0 (the projection gate) ----------
    # Assembled the way Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward
    # does: the talker's hidden at position 0, the TALKER's codec_embedding of codebook 0
    # at position 1, then step p's fed-back code through the predictor's own table p-2
    # (HF embeds generation step g with get_input_embeddings()[g-1], and at position p the
    # step is g = p-1). All at the talker's width, then the projection.
    cpm = talker.code_predictor
    cp_rows = [hidden_last.unsqueeze(0), talker.get_input_embeddings()(torch.tensor([int(greedy[0, 0])]))]
    for pos in range(2, tcfg.num_code_groups):
        cp_rows.append(cpm.get_input_embeddings()[pos - 2](torch.tensor([int(greedy[0, pos - 1])])))
    cp_inputs = cpm.small_to_mtp_projection(torch.cat(cp_rows, dim=0))
    save(out / "cp_inputs.npy", cp_inputs.numpy().astype(np.float32))
    print(f"cp inputs: {tuple(cp_inputs.shape)}, projection "
          f"{type(cpm.small_to_mtp_projection).__name__}", file=sys.stderr)
    kv = {}
    cache = pre.past_key_values
    for l in range(tcfg.num_hidden_layers):
        if hasattr(cache, "layers"):
            k, v = cache.layers[l].keys, cache.layers[l].values
        else:
            k, v = cache[l]
        kv[f"k_{l}"] = k[0].permute(1, 0, 2).contiguous().numpy().astype(np.float32)   # [T, n_kv, hd]
        kv[f"v_{l}"] = v[0].permute(1, 0, 2).contiguous().numpy().astype(np.float32)
    np.savez(out / "prefill_kv.npz", **kv)

    # self-check: greedy cb0 with the suppress list equals the generate's first code
    masked = logits_last.clone()
    masked[captured["suppress_tokens"]] = -float("inf")
    cb0 = int(masked.argmax())
    if cb0 != int(greedy[0, 0]):
        print(f"WARNING: prefill argmax {cb0} != generate frame0 cb0 {greedy[0, 0]}", file=sys.stderr)

    # --- ICL: the same prompt capture, conditioned on the reference's codes -----
    # ref_text is wrapped as its own assistant turn and sliced [3:-2]; HF then lays the
    # reference transcript, the target text and tts_eos along the reference's frames.
    ref_text_ids = tts._tokenize_texts([tts._build_ref_text(a.ref_text)])[0]
    save(out / "icl_ref_text_ids.npy", ref_text_ids[0].numpy().astype(np.int32))
    icl_lens = {}
    for tag, frames in (("icl", int(ref_codes.shape[0])), ("icl_short", a.icl_short_frames)):
        codes_in = ref_codes[:frames]
        icl_prompt = dict(ref_code=[codes_in], ref_spk_embedding=[spk],
                          x_vector_only_mode=[False], icl_mode=[True])
        captured.clear()
        talker.generate = capture_generate
        try:
            t4 = time.time()
            icl_codes_list, _ = model.generate(
                input_ids=[text_ids], ref_ids=[ref_text_ids], voice_clone_prompt=icl_prompt,
                languages=[a.language], non_streaming_mode=False,
                max_new_tokens=a.max_frames if tag == "icl" else 2,
                do_sample=False, subtalker_dosample=False, repetition_penalty=a.repetition_penalty,
                top_k=1, subtalker_top_k=1,
            )
        finally:
            talker.generate = orig_generate
        icl_embeds = captured["inputs_embeds"]
        icl_mask = captured["attention_mask"]
        icl_trailing = captured["trailing_text_hidden"]
        save(out / f"{tag}_prefill_embeds.npy", icl_embeds[0].numpy().astype(np.float32))
        save(out / f"{tag}_trailing_text.npy", icl_trailing[0].numpy().astype(np.float32))
        icl_lens[tag] = dict(ref_frames=frames, prefill_len=int(icl_embeds.shape[1]),
                             trailing_len=int(icl_trailing.shape[1]))
        print(f"{tag}: {frames} reference frames -> {icl_embeds.shape[1]} prefill rows, "
              f"{icl_trailing.shape[1]} trailing, {icl_codes_list[0].shape[0]} greedy frames "
              f"in {time.time() - t4:.1f}s", file=sys.stderr)
        if tag != "icl":
            continue
        icl_greedy = icl_codes_list[0].numpy().astype(np.int32)
        save(out / "icl_greedy_codes.npy", icl_greedy)
        save(out / "icl_frame0_codes.npy", icl_greedy[0])
        icl_lens[tag]["greedy_frames"] = int(icl_greedy.shape[0])
        talker.rope_deltas = None
        icl_pre = talker(inputs_embeds=icl_embeds, attention_mask=icl_mask, use_cache=True,
                         output_hidden_states=True)   # same call as the embedding-only prefill; its KV is not dumped
        save(out / "icl_prefill_hidden.npy", icl_pre.past_hidden[0, -1].numpy().astype(np.float32))
        save(out / "icl_prefill_logits.npy", icl_pre.logits[0, -1].numpy().astype(np.float32))
        torch.manual_seed(a.seed)
        t5 = time.time()
        icl_sampled_list, _ = model.generate(
            input_ids=[text_ids], ref_ids=[ref_text_ids], voice_clone_prompt=icl_prompt,
            languages=[a.language], non_streaming_mode=False, max_new_tokens=a.sampled_max_frames)
        icl_sampled = icl_sampled_list[0].numpy().astype(np.int32)
        save(out / "icl_sampled_codes.npy", icl_sampled)
        icl_lens[tag]["sampled_frames"] = int(icl_sampled.shape[0])
        print(f"icl sampled run: {icl_sampled.shape[0]} frames in {time.time() - t5:.1f}s",
              file=sys.stderr)

    # --- sampled run: real speech for the vocoder gate ------------------------
    torch.manual_seed(a.seed)
    t2 = time.time()
    sampled_list, _ = model.generate(
        input_ids=[text_ids], ref_ids=None, voice_clone_prompt=prompt, languages=[a.language],
        non_streaming_mode=False, max_new_tokens=a.sampled_max_frames,
    )
    sampled = sampled_list[0].numpy().astype(np.int32)
    save(out / "sampled_codes.npy", sampled)
    print(f"sampled run: {sampled.shape[0]} frames in {time.time() - t2:.1f}s", file=sys.stderr)

    # --- vocoder ----------------------------------------------------------------
    t3 = time.time()
    pcm = write_vocoder_refs(model, sampled, out)
    print(f"vocoder: {pcm.shape[0]} samples ({pcm.shape[0] / 24000:.2f}s) in {time.time() - t3:.1f}s", file=sys.stderr)

    # --- what f16 encoder weights would do (documentation, not a gate) ----------
    n16 = f16_round_encoder(stm)
    codes_f16w = model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0].numpy().astype(np.int32)
    save(out / "ref_codes_f16w.npy", codes_f16w)
    f16w_diff = int((codes_f16w != ref_codes.numpy()).sum())
    print(f"f16 encoder weights ({n16} tensors): {f16w_diff} of {ref_codes.numel()} codes differ",
          file=sys.stderr)

    import importlib.metadata
    import transformers
    manifest = {
        "model": a.model.name,
        "model_config_sha256": sha256(a.model / "config.json"),
        "safetensors_sha256": sha256(a.model / "model.safetensors"),
        "codec_safetensors_sha256": sha256(a.model / "speech_tokenizer" / "model.safetensors"),
        "text": a.text,
        "language": a.language,
        "voice_wav": voice.name,
        "voice_wav_sha256": sha256(voice),
        "voice_samples": int(wav.shape[0]),
        "sample_rate": sr,
        "prompt_text": prompt_text,
        "n_text_ids": int(text_ids.shape[1]),
        "prefill_len": int(embeds.shape[1]),
        "trailing_len": int(trailing.shape[1]),
        "talker_hidden": int(tcfg.hidden_size),
        "cp_hidden": int(tcfg.code_predictor_config.hidden_size),
        "cp_projection": type(model.talker.code_predictor.small_to_mtp_projection).__name__,
        "greedy_frames": int(greedy.shape[0]),
        "ref_frames": int(ref_codes.shape[0]),
        "ref_conv_rows": int(stages["conv"].shape[0]),
        "ref_odd_samples": int(wav.shape[0] - 777),
        "ref_odd_frames": int(odd_stages["codes"].shape[0]),
        "ref_codes_f16w_diff": f16w_diff,
        "ref_codes_f16w_note": "documentation, not a gate: f16 encoder weights move the ICL "
                               "prefill hidden by 1.3e-2 rel-rms, so the engine stores the "
                               "encoder f32 while the decoder stays f16 (DESIGN 6)",
        "icl_ref_text": a.ref_text,
        "icl_ref_text_note": "a FIXED STRING, not a transcription of the reference audio: no "
                             "transcript exists for it. Parity does not depend on it being true. "
                             "Nothing in the icl_* set is a listen reference.",
        "icl_n_ref_text_ids": int(ref_text_ids.shape[1]),
        "icl": icl_lens["icl"],
        "icl_short": icl_lens["icl_short"],
        "sampled_frames": int(sampled.shape[0]),
        "sampled_samples": int(pcm.shape[0]),
        "sampled_seed": a.seed,
        "suppress_tokens": [captured["suppress_tokens"][0], captured["suppress_tokens"][-1], "range, minus codec_eos"],
        "greedy": {"do_sample": False, "subtalker_dosample": False, "repetition_penalty": a.repetition_penalty,
                   "max_new_tokens": a.max_frames, "min_new_tokens": 2, "non_streaming_mode": False,
                   "audio": "NOT SPEECH. Argmax decoding of this model collapses into a loop of silence codes "
                            "after ~10 frames and never emits EOS (with or without the repetition penalty). "
                            "Compare greedy_codes code-for-code; never decode it, never listen to it. "
                            "The audible reference is the sampled run."},
        "sampled": {"note": "model.generate defaults (generation_config.json), torch.manual_seed(sampled_seed)",
                    "max_new_tokens": a.sampled_max_frames},
        "dtype": "float32", "device": "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__, "qwen_tts": importlib.metadata.version("qwen-tts"),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
