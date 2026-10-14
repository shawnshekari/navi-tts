# tests/reference/

PyTorch reference dumps for the parity gates (DESIGN 6), produced by
`tools/dump_reference.py` on the CPU in float32. `manifest.json` lists the
text, the voice sample's sha256, the model checksums and the package versions;
the docstring of the script and `docs/model.md` describe every file.

`voice_1.wav` is the production `voice_1` reference sample (15 s, 24 kHz mono).

**Mixed provenance, deliberately.** The `icl_*` and `ref_{conv,xf,latents,codes_f16w}` files
were produced later than the rest of this directory, after the tools venv moved from torch
2.14.0 to 2.14.1 (and python 3.13.7 to 3.13.9) - `manifest.json`'s `icl_tooling` records
which. Every discrete output came out byte-identical across the two versions (codes, token
ids, mels); the float stages moved by ~1e-6 rel-rms, and the 9 prefix rows that
`icl_prefill_embeds.npy` shares with `prefill_embeds.npy` differ by 2.0e-07 rel-rms
(9.5e-07 max). The older files were **not** regenerated on purpose: the vocoder's
kernel-exactness gate is 2e-5 max and regenerating its reference would have moved it 6e-6 -
a quarter of the budget - for no gain. Regenerate the whole directory only with a reason, and
re-measure the gates if you do.

**Two runs, two purposes - do not mix them up:**

- `greedy_codes.npy` / `frame0_codes.npy` (argmax, 31 frames): the talker and
  code-predictor gate, compared code-for-code. **It is not speech.** Argmax
  decoding of this model collapses into a loop of silence codes after ~10
  frames and never emits EOS, with or without the repetition penalty. An engine
  that decodes it and plays silence is behaving correctly; there is no bug to
  find there.
- `sampled_codes.npy` / `sampled_pcm.npy` / `sampled.wav` (model-default
  sampling, `torch.manual_seed(2)`, 72 frames = 138,240 samples = 5.76 s): the
  vocoder gate and the only thing to listen to. Confirmed by ear 2026-09-17:
  it is voice_1 saying the bench text.
- `sampled_pcm_f16w.npy`: the same decode with the decoder's weights rounded
  to f16 the way `tools/convert.py` stores them. The engine must match this
  within 2e-5 (kernel exactness; it does at 9.5e-6 / 114 dB). Against the f32
  `sampled_pcm.npy` the same output is 3.5e-3 max / 65 dB SNR - that gap is
  weight quantisation, identical in torch, and not something the kernels can
  close; DESIGN 6's "within 1e-4" would need f32 vocoder weights.

`cp_inputs.npy` is the code predictor's 16 inputs for greedy frame 0, after its
`small_to_mtp_projection`. On this model the talker and the predictor are both
1024 wide, so that stage is `nn.Identity()` and this is the raw embeddings -
still a gate on which row `cp_embed` picks per position. The projecting path is
exercised by `../reference-1.7b/`, whose talker is 2048.


## The codec encoder and the ICL set

The encoder is what turns reference audio into the codes an in-context clone is
conditioned on. It is dumped stage by stage so a parity failure localises instead of
just saying "the codes are wrong":

- `ref_conv.npy` [375, 512] after the SEANet conv stack, at 25 Hz
- `ref_xf.npy` [375, 512] after its 8-layer transformer
- `ref_latents.npy` [188, 512] after the stride-2 downsample - the last float before
  any argmin, and the encoder's float gate. Note 188 = ceil(375 / 2): the lengths are not a
  clean factor of two, and the odd row costs one replicated row of right padding.
- `ref_codes.npy` [188, 16] the codes themselves, the exact-match gate
- `ref_codes_f16w.npy` the same codes with every >=2-D encoder weight rounded to f16,
  the way `convert.py` stores the rest of the model. **Not a gate.** It is the
  measurement that put the encoder at f32: **38 of 3008 codes change**, and that moves the ICL
  prefill's hidden state by 1.27e-2 rel-rms - 36x the talker prefill gate. An RVQ is a cascade
  of argmins with no error averaging, so a 1.2e-5 relative tie in codebook 6 flips that code
  and every later codebook of the frame then diverges by much more. cb0, the semantic code,
  never flipped in any precision tried.

The ICL prompt set (`icl_*`) is the same capture as the embedding-only one, with the
reference's codes and transcript in the prompt: `icl_prefill_embeds.npy` is 198 rows
= the 9 prefix rows + `codec_bos` + one row per reference frame, and
`icl_trailing_text.npy` is the single `tts_pad` row that is left once the text
has been laid along those frames. `icl_short_*` is the same prompt built from only the
first 8 reference frames, so the text overflows the block and the trailing text is 20 rows -
the other branch of the alignment, which the long case never reaches.

**Nothing in the ICL set is a listen reference, and its transcript is not true.**
`voice_1.wav` has no transcript (and this box has no ASR), so the dumps pair it with a
fixed string that does not describe it. Parity does not care - the gate asks whether the engine
builds the same prompt from the same inputs - but the audio such a prompt produces is a model
reading one text while conditioned on a recording of another. `icl_sampled_codes.npy`
exists for its frame count and provenance only (and it is short for exactly that reason); there
is deliberately no `icl_sampled.wav`. Listening and quality work uses a real
audio/transcript pair, not this one.

Regenerate (about two minutes on this box's CPU):

    cd tools && uv sync && uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-0.6B-Base --out ../tests/reference
