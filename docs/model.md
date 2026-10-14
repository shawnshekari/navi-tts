# Qwen3-TTS-12Hz Base: what the engine needs to know

Re-derived at M0 from the HF release (`config.json`, `speech_tokenizer/config.json`,
`generation_config.json`, the safetensors headers) and the model's own
`qwen-tts` package (the reference implementation the dumps come from). Every
number here is also in the `.navi` KV section (`docs/navi-format.md`), which is
what the engine actually reads; this file is the human copy.

Shapes below are the **0.6B-Base**, the model the engine shipped with. The
**1.7B-Base** is the same architecture with a wider talker, supported since
2026-10-13; the two differ in exactly four numbers, and only one of them needs
engine code rather than configuration:

| | 0.6B-Base | 1.7B-Base |
|---|---|---|
| talker hidden | 1024 | **2048** |
| talker intermediate | 3072 | **6144** |
| speaker encoder output | 1024 | **2048** |
| code predictor hidden | 1024 | 1024 |
| `small_to_mtp_projection` | absent (Identity) | **2048 -> 1024, with bias** |

Everything else - the code predictor's layer count and shapes, the codec, the
whole vocoder, the tokenizer, the special ids, the language ids - is identical.
See "Code predictor" for the projection, which is the one place the widths stop
being interchangeable.

Switching between the two is a different model file and nothing else, with one
operational catch: a **stored voice holds an embedding of its model's width**, so
the voice store skips voices whose width does not match the loaded model (it says
so on stderr). Re-clone them from their reference audio after a switch
(`navi-tts voices add`, or `POST /v1/audio/voices`); the stored
`sample.wav` makes that a local operation.

## Components

| Component | Params | Tensors | Prefix |
|---|---|---|---|
| Text embedding + projection | 151936 × 2048 table, MLP 2048→2048→1024 | 5 | `talker.model.text_embedding`, `talker.text_projection.*` |
| Talker | 28 layers, 1024 hidden | 311 | `talker.model.*`, `talker.codec_head` |
| Code predictor | 5 layers, 1024 hidden, 15 heads | 86 (88 on the 1.7B, with the projection) | `talker.code_predictor.*` |
| Speaker encoder | ECAPA-TDNN → 1024 | 76 | `speaker_encoder.*` |
| Codec decoder (vocoder) | 8-layer transformer + convnet | 261 (after baking codebooks) | `codec.decoder.*` |
| Codec encoder | 8-layer transformer + conv stack + split RVQ, f32 | 143 (only with `--with-encoder`) | `codec.encoder.*` |

Safetensors dtypes: the main checkpoint is BF16 throughout, the codec F32.
Converted: F16 for ≥2-D tensors, F32 for 1-D. Every BF16 value is exactly
representable in F16 (8-bit mantissa into 11, same exponent range in
practice; none exceed 65504), so the talker, code predictor and speaker
encoder weights in the `.navi` are bit-identical to the checkpoint - the
engine's only error sources there are f16 KV storage and accumulation order.
The codec is the one component that loses precision at F16 (DESIGN 6).

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
`add_prefix_space` and `add_bos_token` are off. The reference dumps come from
HF's **fast** tokenizer (the model directory has no `tokenizer.json`; transformers
4.57 converts the slow one), whose pipeline is what `model/qwen3tts/tokenizer`
reproduces exactly:

1. Added tokens (all 33, special or not; none with lstrip/rstrip/normalized)
   are cut out of the raw text first, leftmost-longest.
2. Each remaining segment is **NFC**-normalised.
3. Pre-tokenisation with the Qwen2 fast pattern, ordered alternation:
   `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+`
   | `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*`
   | `\p{N}` | ` ?[^\s\p{L}\p{N}]+[\r\n/]*` | `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`.
   Note what this is *not*: no `'s|'t|…` contraction alternative (an
   apostrophe is the optional prefix of the letter run: `It's` → `It`, `'s`),
   CamelCase splits at the case change, and the punctuation alternative
   swallows trailing `/` as well as CR/LF.
4. Byte-level BPE over each piece's UTF-8 bytes, merging one pair at a time in
   rank order (HF's heap semantics, which differ from "merge every occurrence
   of the best pair" on some inputs).

Decoding returns the NFC form of what was encoded.

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


### Prompt layout (in-context cloning)

In-context cloning (ICL) conditions on a reference clip's **codes** plus its
**transcript**, on top of the speaker embedding. The first nine rows are the same as
above. What changes is row 9: instead of one row (first text token + `codec_bos`),
there is a block of `1 + T_ref` rows, where `T_ref` is the reference's
frame count at 12.5 Hz:

| | codec side | text side |
|---|---|---|
| block row 0 | `codec_bos` | reference transcript token 0 |
| block row j | reference frame `j-1` | the next text token |

The codec side of a frame row is the **sum of its 16 code embeddings at the talker's
width** - codebook 0 from `talker.model.codec_embedding`, codebooks 1..15 from the
code predictor's tables, exactly the rows `assemble_step` sums to build a frame's
next input (see "Code predictor" for why they are stored at that width).

The text side is one sequence - **the reference transcript, then the target text, then
`tts_eos`** - laid along those rows and right-padded with `tts_pad` if it is
shorter. Whatever does not fit becomes the per-frame trailing text, exactly as in the
embedding-only form. So a long reference (the usual case) leaves a single
`tts_pad` trailing row, with the whole text already in the prompt; a short one
leaves real text to feed frame by frame. The reference transcript is wrapped as its own
assistant turn (`<|im_start|>assistant\n{ref}<|im_end|>\n`) and sliced `[3:-2]`,
while the target text keeps its `[3:-5]` slice.

Sizes, measured: the 15 s parity clip is 188 frames, so 198 prefill rows against the
embedding-only 10; a 20 s reference is 250 frames and 260 rows.

**What this costs is not confined to the prefill.** The talker attends over
`n_prefill + frame_index` keys, so a 250-frame reference adds 250 keys to *every*
frame, not just to the one-off prompt pass. `bench --ref-text` reports per-position
frame times for that reason (`docs/tuning.md`).

A voice carries its codes with it: the codec is the same checkpoint in every model of
this family (same safetensors sha256 for the 0.6B and the 1.7B), so **reference codes
survive a model switch even though the speaker embedding does not**, and a blob without
the encoder can still serve a voice whose codes were made by one that has it. It just
cannot make new ones.

### One frame

1. Talker step on the input row at position `n_past` → post-norm hidden `h` (1024) → `codec_head` → 3072 logits.
2. Suppress ids 2048–3071 except `codec_eos` (2050); repetition penalty 1.05 over the codes emitted so far; sample cb0 (temperature 0.9, top-k 50, top-p 1.0 by default). The model's generate also forces at least 2 frames before EOS.
3. cb0 == `codec_eos` ends the utterance (the frame is not emitted).
4. Code predictor: a 2-row prefill `[h, talker.codec_embedding(cb0)]` at positions 0 and 1 gives codebook 1 from `lm_head[0]`; then for `i = 1..14` the row `cp.codec_embedding[i-1](code_i)` at position `i + 1` gives codebook `i + 1` from `lm_head[i]`. 16 positions per frame, causal, its own KV cache reset per frame; plain RoPE. Sampling per step with the `subtalker_*` defaults (0.9 / 50 / 1.0). Every one of those rows is at the talker's width and goes through `small_to_mtp_projection` first when the two widths differ (the 1.7B); for the 0.6B that stage is Identity and absent from the checkpoint. See "Code predictor".
5. Next talker input = `codec_embedding(cb0) + Σ_i cp.codec_embedding[i](code_{i+1})` + the trailing text row.

The talker's hidden fed to the code predictor is the **post-final-norm** hidden
(`tests/reference/prefill_hidden.npy` is that vector at the last prefill position).


## Code predictor

| | |
|---|---|
| layers | 5 (same layer structure as the talker, with QK-norm) |
| hidden / heads / KV heads / head_dim / intermediate | 1024 / 16 / 8 / 128 / 3072 |
| rms_norm_eps, rope_theta | 1e-6, 1e6 (plain RoPE, positions 0..16 within a frame) |
| per codebook 1..15 | `model.codec_embedding.{i}.weight` 2048 × **talker hidden**, `lm_head.{i}.weight` 2048 × cp hidden |
| final norm | `model.norm.weight` |

### The two widths, and `small_to_mtp_projection`

The predictor's 15 input embedding tables are stored at the **talker's** width,
not its own. That is not a quirk of the checkpoint: HF constructs the predictor
with `embedding_dim = talker_config.hidden_size` because those same rows do
double duty - they are also what the talker's next frame input is built from
(`codec_hiddens.sum(1)` in the talker's forward, `assemble_step` here).

So the predictor's input arrives at the talker's width and has to reach its own:

- **widths equal (0.6B, both 1024):** nothing to do. HF's stage is
  `nn.Identity()` and the two tensors are absent from the checkpoint.
- **widths differ (1.7B, 2048 -> 1024):** `small_to_mtp_projection`, a Linear
  with bias, applied to **every** one of the 16 positions - the talker's hidden at
  position 0, the talker's codebook-0 embedding at 1, and each fed-back code after
  that. Not only the prefill rows.

In the engine (`model/qwen3tts/frame.hip`, `cp_embed`) each block stages
the talker-width row into its own LDS and the projection is one grid matvec, with
the bias folded into its write site. It needs no extra barrier, because every block
can name the source row by itself: `codes[]` and the talker's hidden are both
written before the barrier that already precedes the phase. `assemble_step` is
the mirror image and works at the talker's width.

A blob whose projection is missing or the wrong shape when the widths differ is
refused at load by `validate()`, not in a kernel.

`tests/reference-1.7b/cp_inputs.npy` is the float-level gate on this stage; the
codes gate does not cover it (see that directory's README for the measurement that
shows why).

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

## Codec encoder

Only in a blob converted with `--with-encoder`, and only used to *make* reference
codes for in-context cloning: 24 kHz audio in, 16 codes per 12.5 Hz frame out. It runs
once per clone, like the speaker encoder, never per frame.

It is a **Mimi** encoder with the decoder halves nulled out, 47.6 M values in the 143
tensors the engine keeps:

1. **SEANet conv stack**, causal, ELU, identity-shortcut residual units:
   `conv 1->64 k7`, then for each ratio in `[4, 5, 6, 8]` (the config's
   `upsampling_ratios` reversed) a residual unit
   `x + conv_k1(ELU(conv_k3(ELU(x))))` with inner width `dim/2`, an ELU, and a
   downsampling `conv dim->2*dim` with `k = 2*stride`; then an ELU and
   `conv 1024->512 k3`. Total stride 960, so 25 Hz.
2. **8-layer transformer**, hidden 512, 8 heads x 64, **LayerNorm with bias** (eps 1e-5 -
   the only LayerNorms outside the vocoder's ConvNeXt blocks), MLP 512->2048->512 with no
   bias and exact (erf) GELU, LayerScale on both branches, RoPE theta=1e4.
3. **`downsample`**: `conv 512->512 k4 stride 2`, no bias. 25 Hz -> 12.5 Hz.
4. **Split RVQ encode**: `input_proj` 512->256 per group, then nearest centroid with
   residual subtraction - codebook 0 from the 1-quantizer *semantic* group, codebooks
   1..15 from the first 15 of the *acoustic* group's 31. The acoustic group quantises the
   **full** latent, not the semantic residual. Codebooks are
   `embed_sum / clamp(cluster_usage, 1e-5)`, baked by the converter.

Three facts that are not visible in the config, each measured on this box:

- **The attention is full causal.** The encoder config says `sliding_window: 250` and
  transformers 4.57.3 never applies it: setting it to 8, 64 or `None` changes the
  output by exactly 0.0, and a hand-rolled full-causal layer matches the module at
  2.4e-7 rel-rms where a windowed one is off by 1.7e-2. The dumps record the transformers
  version; a release that started honouring the window would change the codes.
- **`downsample` pads by replicating a row**, not with zeros, on both sides - the
  one conv in this model that does (`pad_mode="replicate"` in its constructor).
  Every other conv left-pads with zeros, plus a right "extra" pad computed from the
  length so the stride divides it. That extra pad is zero whenever the length is a
  multiple of 1920, and non-zero for the 15 s parity clip (360,000 samples = 187.5
  frames -> 188, which costs one replicated row at 25 Hz: 375 rows -> 188 frames is a
  ceil, not a halving).
- **The weights are stored f32** while the decoder's stay f16. Not a style choice: the
  output is 16 nearest-centroid searches per frame, a cascade with no error averaging,
  so f16 weights change **38 of 3008** codes on the parity clip and move the ICL
  prefill's hidden state by **1.27e-2 rel-rms** - 36x the talker prefill gate. A 1.2e-5
  relative tie in codebook 6 flips that code and every later codebook of the frame then
  diverges by much more. cb0, the semantic code, never flipped at any precision tried.
  The cost is 190 MB of VRAM and 0 ms per frame. The engine's encoder kernels read f32
  only and refuse any other dtype by tensor name, rather than quietly reinterpreting it.
  (`tests/reference/ref_codes_f16w.npy` is that measurement.)

In the engine (`model/qwen3tts/codec_encoder.hip`) the conv stack is chunked in time
with each conv's causal history threaded across chunks - a chunked encode equals a
one-shot one, which is a gate - because its first level is 184 MB at the 30 s clip cap.
Everything after it runs once over the whole sequence: 375 rows x 512 is 1.5 MB, so the
transformer needs no KV cache and no cross-chunk state. Measured: all 3008 codes exact
against the reference, stages within 7.5e-6 rel-rms, 606 ms for a 15 s clip.

Reference: `tests/reference/ref_conv.npy`, `ref_xf.npy`,
`ref_latents.npy`, `ref_codes.npy`.

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
