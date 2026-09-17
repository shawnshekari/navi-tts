# Qwen3-TTS-12Hz-0.6B-Base: what the engine needs to know

Re-derived at M0 from the HF release (`config.json`, `speech_tokenizer/config.json`,
`generation_config.json`, the safetensors headers) and the model's own
`qwen-tts` package (the reference implementation the dumps come from). Every
number here is also in the `.navi` KV section (`docs/navi-format.md`), which is
what the engine actually reads; this file is the human copy.

## Components

| Component | Params | Tensors | Prefix |
|---|---|---|---|
| Text embedding + projection | 151936 × 2048 table, MLP 2048→2048→1024 | 5 | `talker.model.text_embedding`, `talker.text_projection.*` |
| Talker | 28 layers, 1024 hidden | 311 | `talker.model.*`, `talker.codec_head` |
| Code predictor | 5 layers, 1024 hidden, 15 heads | 86 | `talker.code_predictor.*` |
| Speaker encoder | ECAPA-TDNN → 1024 | 76 | `speaker_encoder.*` |
| Codec decoder (vocoder) | 8-layer transformer + convnet | 261 (after baking codebooks) | `codec.decoder.*` |
| Codec encoder | not converted by default (ICL cloning is OPEN) | — | `codec.encoder.*` |

Safetensors dtypes: the main checkpoint is BF16 throughout, the codec F32.
Converted: F16 for ≥2-D tensors, F32 for 1-D. No value in the BF16 checkpoint
exceeds the F16 range.

## Talker

Qwen3 decoder (RMSNorm, SwiGLU, GQA, QK-norm), same shape as Qwen3-0.6B.

| | |
|---|---|
| layers | 28 |
| hidden | 1024 |
| heads / KV heads / head_dim | 16 / 8 / 128 (q_proj is 2048 × 1024; o_proj 1024 × 2048) |
| intermediate | 3072 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1e6, M-RoPE `mrope_section [24, 20, 20]`, interleaved |
| codec vocab | 3072 (`codec_embedding`, `codec_head`) |
| text vocab / hidden | 151936 / 2048 (projected to 1024 by `text_projection`: fc1 2048→2048, SiLU, fc2 2048→1024, both with bias) |
| max positions | 32768 |

Per layer: `input_layernorm`, `self_attn.{q,k,v,o}_proj` (no bias),
`self_attn.{q,k}_norm` (128), `post_attention_layernorm`, `mlp.{gate,up,down}_proj`.
Then `talker.model.norm` and `talker.codec_head` (3072 × 1024, no bias).

**M-RoPE at batch one is plain RoPE.** `get_rope_index` builds three identical
position rows (`cumsum(mask) − 1`); the interleaved layout only decides which of
the three rows each rotary frequency takes its angle from, so with identical
rows it is standard RoPE (rotate-half pairing, dim `i` with `i + 64`, the usual
`1e6^(-2i/128)` frequencies). Positions continue counting through generation
(`n_past`), one per frame.

### Codec-side control ids (in the 3072 codec vocab)

| id | |
|---|---|
| 0–2047 | codebook-0 acoustic codes |
| 2048 | `codec_pad` |
| 2049 | `codec_bos` |
| 2050 | `codec_eos` |
| 2054 | `codec_think` |
| 2055 | `codec_nothink` |
| 2056 | `codec_think_bos` |
| 2057 | `codec_think_eos` |
| language ids | chinese 2055, english 2050, german 2053, italian 2070, portuguese 2071, spanish 2054, japanese 2058, korean 2064, french 2061, russian 2069 |

(Yes: `english` shares the id 2050 with `codec_eos`, and `chinese` shares 2055 with
`codec_nothink`. They live in different positions of the prompt, so nothing is
ambiguous; but no id below 2048+1024 is free.) The Base model has no built-in
speakers (`spk_id` is empty): every voice is a cloned speaker embedding.

### Text-side special ids (in the 151936 text vocab)

| id | token |
|---|---|
| 151643 | `<\|endoftext\|>` |
| 151644 | `<\|im_start\|>` |
| 151645 | `<\|im_end\|>` |
| 77091 | `assistant` |
| 198 | `\n` |
| 151671 | `<tts_pad>` |
| 151672 | `<tts_text_bos>` |
| 151673 | `<tts_text_eod>` (the config calls it `tts_eos_token_id`) |

The tokenizer is the Qwen2/Qwen3 byte-level BPE (`vocab.json` 151643 entries,
`merges.txt`, plus 33 added tokens 151643–151675 from `tokenizer_config.json`).
`add_prefix_space` and `add_bos_token` are off.

### Prompt layout (speaker-embedding cloning, streaming text mode)

The text is wrapped as `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
and tokenised; call the ids `ids`, so the text tokens are `ids[3:-5]`. All text
rows go through `text_embedding` then `text_projection`; codec rows come from
`codec_embedding`; the speaker embedding is used as-is (1024-d, same space as
the codec embedding). Each prefill row is the **sum** of a text-side row and a
codec-side row:

| pos | text side | codec side |
|---|---|---|
| 0–2 | `<\|im_start\|>`, `assistant`, `\n` | (none) |
| 3 | tts_pad | `codec_think` |
| 4 | tts_pad | `codec_think_bos` |
| 5 | tts_pad | language id |
| 6 | tts_pad | `codec_think_eos` |
| 7 | tts_pad | speaker embedding |
| 8 | tts_bos | `codec_pad` |
| 9 | first text token `ids[3]` | `codec_bos` |

With `language: auto` the think block is `[codec_nothink, codec_think_bos, codec_think_eos]`
(three rows, no language). For the bench text the prefill is 10 rows.

**Trailing text.** The remaining text tokens `ids[4:-5]` followed by `tts_eos`,
projected, are the per-frame text rows: frame `i`'s input is the frame's summed
code embedding plus `trailing[i]`, or plus the `tts_pad` row once the text is
exhausted. (`tests/reference/trailing_text.npy`, `tts_pad_embed.npy`.)

### One frame

1. Talker step on the input row at position `n_past` → post-norm hidden `h` (1024) → `codec_head` → 3072 logits.
2. Suppress ids 2048–3071 except `codec_eos` (2050); repetition penalty 1.05 over the codes emitted so far; sample cb0 (temperature 0.9, top-k 50, top-p 1.0 by default). The model's generate also forces at least 2 frames before EOS.
3. cb0 == `codec_eos` ends the utterance (the frame is not emitted).
4. Code predictor: a 2-row prefill `[h, talker.codec_embedding(cb0)]` at positions 0 and 1 gives codebook 1 from `lm_head[0]`; then for `i = 1..14` the row `cp.codec_embedding[i-1](code_i)` at position `i + 1` gives codebook `i + 1` from `lm_head[i]`. 16 positions per frame, causal, its own KV cache reset per frame; plain RoPE. Sampling per step with the `subtalker_*` defaults (0.9 / 50 / 1.0). (For a model whose cp hidden differs from the talker's, a `small_to_mtp_projection` Linear sits in front; it is Identity for the 0.6B and absent from its checkpoint.)
5. Next talker input = `codec_embedding(cb0) + Σ_i cp.codec_embedding[i](code_{i+1})` + the trailing text row.

The talker's hidden fed to the code predictor is the **post-final-norm** hidden
(`tests/reference/prefill_hidden.npy` is that vector at the last prefill position).

## Code predictor

| | |
|---|---|
| layers | 5 (same layer structure as the talker, with QK-norm) |
| hidden / heads / KV heads / head_dim / intermediate | 1024 / 16 / 8 / 128 / 3072 |
| rms_norm_eps, rope_theta | 1e-6, 1e6 (plain RoPE, positions 0..16 within a frame) |
| per codebook 1..15 | `model.codec_embedding.{i}.weight` 2048 × 1024, `lm_head.{i}.weight` 2048 × 1024 |
| final norm | `model.norm.weight` |

## Speaker encoder

ECAPA-TDNN over a 128-bin log-mel spectrogram of 24 kHz audio: `n_fft 1024,
hop 256, win 1024, fmin 0, fmax 12000` (fixed in the model code, not in
`config.json`; stored under `speaker.mel.*`). Channels `[512, 512, 512, 512, 1536]`,
kernels `[5, 3, 3, 3, 1]`, dilations `[1, 2, 3, 4, 1]` (the class defaults,
stored under `speaker.enc_*`). `blocks.0` TDNN 128→512 k=5 (conv with
"same" reflect padding + ReLU); three SE-Res2Net blocks (`tdnn1` 512→512 k=1,
Res2Net scale 8 = seven 64→64 k=3 dilated convs, `tdnn2` 512→512 k=1,
squeeze-excitation 512→128→512); `mfa` TDNN over the three block outputs
concatenated, 1536→1536; attentive statistics pooling (`asp.tdnn`
4608→128 on `[x, mean, std]`, tanh, `asp.conv` 128→1536 → softmax weights →
weighted mean and std); `fc` 3072→1024 k=1. Output 1024-d, used directly as a
prompt row. Runs once per clone. Reference: `tests/reference/speaker_mels.npy`
→ `speaker_embedding.npy`.

## Codec decoder (vocoder)

12.5 Hz frames, 1920 samples per frame at 24 kHz (`decode_upsample_rate`).

1. **RVQ decode.** 16 codebooks of 2048 × 256: `rvq_first` (1 semantic) and
   `rvq_rest` (15 acoustic), each with `input_proj` (encode only) and
   `output_proj` 256→512 (1-wide conv, no bias). Per group: sum the codebook
   rows, then `output_proj`; add the two groups → 512-d per frame. The
   checkpoint stores EMA state; the converter bakes
   `codebook = embedding_sum / max(cluster_usage, 1e-5)`.
2. `pre_conv` causal conv 512→1024, k=3.
3. **pre_transformer**: `input_proj` 1024→512 (bias), 8 layers (hidden 512, 16
   heads × 64, no GQA, no QK-norm, intermediate 1024, RMSNorm eps 1e-5, RoPE
   θ=1e4, sliding window 72, LayerScale on both branches), `norm`,
   `output_proj` 512→1024 (bias).
4. **upsample** ×2 ×2: causal transposed conv 1024→1024 (k=stride=2) then a
   ConvNeXt block (depthwise k=7, LayerNorm, pw 1024→4096→1024, `gamma`).
5. **decoder**: causal conv 1024→1536 k=7; four blocks with rates 8, 5, 4, 3 —
   each: SnakeBeta, transposed conv halving the width (k=2·rate, stride=rate,
   output trimmed by k−stride on the right), three residual units with
   dilations 1, 3, 9 (SnakeBeta, causal conv k=7, SnakeBeta, conv k=1, plus
   the residual); then SnakeBeta(96) and causal conv 96→1 k=7. Output clamped
   to [-1, 1]. SnakeBeta is `x + sin²(x·e^α) / (e^β + 1e-9)` per channel.
   "Causal" convs left-pad by `(k−1)·dilation − (stride−1)`.

Total upsampling 2·2·8·5·4·3 = 1920. The reference chunked decode uses a
left context of 25 frames and trims `context × 1920` samples; the engine's
streaming design threads state instead (DESIGN 5.2).

## Generation defaults (`generation_config.json`)

`temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05`, and the same
for the code predictor (`subtalker_*`). Stored under `gen.*`.

## Reference dumps

`tools/dump_reference.py` → `tests/reference/` (manifest.json lists every
file, the text, the voice sample's sha256 and the package versions). Two
runs of the bench text with `voice_1.wav`:

- **greedy** (argmax, repetition penalty 1.05, 32-frame cap): the talker and
  code-predictor parity target. Argmax decoding of this model collapses into a
  loop of silence codes after ~10 frames and never emits EOS, so the greedy
  sequence is only compared code-for-code, never listened to.
- **sampled** (the model's defaults, `torch.manual_seed(2)`, 72 frames): real
  speech; its codes and f32 PCM are the vocoder gate and the listen reference.
