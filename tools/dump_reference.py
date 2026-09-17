#!/usr/bin/env python3
"""PyTorch reference dumps for the parity gates (docs/DESIGN.md 6).

    uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-0.6B-Base --out ../tests/reference

Runs the HF model on the CPU in float32 (the shared GPU stays untouched, and
f32 on CPU is the reference we want) for a fixed text and reference voice, and
writes what each engine stage is checked against:

  text_ids.npy            token ids of the assistant-formatted prompt (BPE gate)
  tokenizer_cases.json    more strings -> ids for the tokenizer tests
  speaker_mels.npy        [n_mels, T] mel input to the speaker encoder
  speaker_embedding.npy   [1024] (speaker encoder gate)
  ref_codes.npy           [T, 16] codec encoder output for the reference audio (ICL, later)
  prefill_embeds.npy      [T, 1024] the talker prefill input (embeddings, after text_projection)
  prefill_mask.npy        [T] attention mask (all ones at batch one)
  prefill_position_ids.npy [3, T] M-RoPE positions
  trailing_text.npy       [n, 1024] the per-frame trailing text rows (text tokens[1:] + tts_eos)
  tts_pad_embed.npy       [1024] the row added once trailing text is exhausted
  prefill_hidden.npy      [1024] post-norm talker hidden at the last prefill position
  prefill_logits.npy      [3072] codec_head logits at that position
  prefill_kv.npz          k_{l}, v_{l}: [T, n_kv_head, head_dim] after RoPE, per talker layer
  greedy_codes.npy        [n_frames, 16] argmax run with the default repetition penalty (the
                          talker / code-predictor gate). Argmax decoding collapses into a loop of
                          silence codes after ~10 frames, so it is capped and never used as audio.
  frame0_codes.npy        [16] its first frame
  sampled_codes.npy       [n_frames, 16] a seeded sampled run with the model's default sampling:
                          real speech, the fixed code sequence for the vocoder gate
  sampled_pcm.npy         [n_samples] f32 decode of sampled_codes; sampled.wav is the same as s16
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
    # the TTS-Player queue's typical outputs
    "Build finished: 3 warnings, 0 errors. Running tests...",
    "Ok — I've updated tts_queue.py (lines 42–58) and restarted the service.",
    "Yes.", "No", "", " ",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wav(path: Path, sr_expected: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_expected:
        raise SystemExit(f"{path}: {sr} Hz, need {sr_expected} (ffmpeg -ar {sr_expected} -ac 1)")
    return wav.mean(axis=1).astype(np.float32)


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
    ap.add_argument("--tokenizer-only", action="store_true", help="only text_ids.npy and tokenizer_cases.json")
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

    # --- tokenizer ---------------------------------------------------------
    prompt_text = tts._build_assistant_text(a.text)
    text_ids = tts._tokenize_texts([prompt_text])[0]          # [1, 3 + T + 5]
    np.save(out / "text_ids.npy", text_ids[0].numpy().astype(np.int32))
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
    np.save(out / "speaker_mels.npy", mels[0].numpy().astype(np.float32))
    spk = model.extract_speaker_embedding(audio=wav, sr=sr)                  # [1024]
    np.save(out / "speaker_embedding.npy", spk.numpy().astype(np.float32))
    print(f"speaker embedding: {tuple(spk.shape)}, mels {tuple(mels.shape)}", file=sys.stderr)

    # --- codec encoder (reference codes; ICL is OPEN, kept for later) ----------
    enc = model.speech_tokenizer.encode(wav, sr=sr)
    ref_codes = enc.audio_codes[0]
    np.save(out / "ref_codes.npy", ref_codes.numpy().astype(np.int32))

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
    np.save(out / "prefill_embeds.npy", embeds[0].numpy().astype(np.float32))
    np.save(out / "prefill_mask.npy", mask[0].numpy().astype(np.int32))
    np.save(out / "trailing_text.npy", trailing[0].numpy().astype(np.float32))
    np.save(out / "tts_pad_embed.npy", pad[0, 0].numpy().astype(np.float32))

    greedy = codes_list[0].numpy().astype(np.int32)          # [n_frames, 16]
    np.save(out / "greedy_codes.npy", greedy)
    np.save(out / "frame0_codes.npy", greedy[0])

    # --- prefill on its own: hidden, logits, KV, positions ----------------------
    talker.rope_deltas = None
    pre = talker(inputs_embeds=embeds, attention_mask=mask, use_cache=True, output_hidden_states=True)
    hidden_last = pre.past_hidden[0, -1]                     # post-norm, what feeds codec_head and the cp
    logits_last = pre.logits[0, -1]
    np.save(out / "prefill_hidden.npy", hidden_last.numpy().astype(np.float32))
    np.save(out / "prefill_logits.npy", logits_last.numpy().astype(np.float32))
    position_ids, _ = talker.get_rope_index(mask)
    np.save(out / "prefill_position_ids.npy", position_ids[:, 0].numpy().astype(np.int32))
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

    # --- sampled run: real speech for the vocoder gate ------------------------
    torch.manual_seed(a.seed)
    t2 = time.time()
    sampled_list, _ = model.generate(
        input_ids=[text_ids], ref_ids=None, voice_clone_prompt=prompt, languages=[a.language],
        non_streaming_mode=False, max_new_tokens=a.sampled_max_frames,
    )
    sampled = sampled_list[0].numpy().astype(np.int32)
    np.save(out / "sampled_codes.npy", sampled)
    print(f"sampled run: {sampled.shape[0]} frames in {time.time() - t2:.1f}s", file=sys.stderr)

    # --- vocoder ----------------------------------------------------------------
    t3 = time.time()
    wavs, fs = model.speech_tokenizer.decode([{"audio_codes": torch.from_numpy(sampled).long()}])
    pcm = np.asarray(wavs[0], dtype=np.float32)
    np.save(out / "sampled_pcm.npy", pcm)
    sf.write(out / "sampled.wav", pcm, fs, subtype="PCM_16")
    wavs16, _ = model.speech_tokenizer.decode([{"audio_codes": torch.from_numpy(sampled[:16]).long()}])
    np.save(out / "vocoder16_pcm.npy", np.asarray(wavs16[0], dtype=np.float32))
    print(f"vocoder: {pcm.shape[0]} samples @ {fs} Hz ({pcm.shape[0] / fs:.2f}s) in {time.time() - t3:.1f}s", file=sys.stderr)

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
        "greedy_frames": int(greedy.shape[0]),
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
