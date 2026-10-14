# tests/reference-1.7b/

The same PyTorch reference dumps as `tests/reference/`, for
**Qwen3-TTS-12Hz-1.7B-Base**, produced on the Strix Halo box 2026-10-13 by
`tools/dump_reference.py` on the CPU in float32. `manifest.json` lists the text,
the voice sample's sha256, the model checksums and the package versions.

Why a second set: the 1.7B's talker is 2048 wide where its code predictor is
1024, so it is the only model here that exercises the predictor's
`small_to_mtp_projection` stage. The 0.6B's two widths are equal, which makes
that stage `nn.Identity()` and leaves the kernel's projecting path untested.

Files that are byte-identical to the 0.6B set are **symlinks** into
`../reference/` rather than copies: `voice_1.wav` (same reference sample),
`speaker_mels.npy` (the mel front end is fixed, not per model),
`tokenizer_cases.json` and `text_ids.npy` (same Qwen3 BPE; verified 49/49 exact
against this blob's own embedded tokenizer), and `prefill_mask.npy` /
`prefill_position_ids.npy` (same prompt shape). Everything else differs and is
its own file. Byte equality was checked with `cmp`, not assumed.

The codec is the same checkpoint in both models (same safetensors sha256), so the four
encoder files - `ref_conv.npy`, `ref_xf.npy`, `ref_latents.npy`, `ref_codes_f16w.npy` - are
symlinks as well, `cmp`-verified like the rest. The `icl_*` files are **not**: they carry the
talker's width, which is the whole point of this directory. See `../reference/README.md` for
what the encoder and ICL files are and why the ICL set must not be listened to.

`cp_inputs.npy` is the gate for the projection: the predictor's 16 inputs for
greedy frame 0 **after** `small_to_mtp_projection`. The codes gate does not
stand in for it - measured on this box, putting a 1% error in the projection's
bias still left greedy frames 0 and 1 matching 16/16, which is what
`test_frame`'s code gate asks for, while the float check caught it at 1.39e-2
rel-rms against its 5e-3 threshold.

Measured against this set (gfx1151, ROCm 10.1.0, same with
`HIP_LAUNCH_BLOCKING=1`):

- greedy codes: **31 of 31 frames match all 16 codes** (the 0.6B diverges at
  frame 29; a late greedy divergence is a near-tie flipping under f16 KV, not a
  bug - see `../reference/README.md`)
- cp inputs: worst of the 16 positions 1.90e-4 rel-rms
- prefill: hidden 3.56e-4, logits 2.06e-4 rel-rms; greedy cb0 exact
- speaker embedding (2048-d): cosine 1.000000, 2.28e-6 rel-rms
- vocoder: 114.0 dB against the f16-weight reference, 65.5 dB against f32 -
  the same figures as the 0.6B, which is expected: the codec is the same weights
- sampled run: 83 frames in torch, 87 in the engine, bit-identical run to run
- ICL prompt: 198 prefill rows (9 + codec_bos + 188 reference frames), 1 trailing row

The greedy run is **not speech** here either; the audible reference is
`sampled.wav` (6.64 s).

Regenerate (about a minute on this box's CPU):

    cd tools && uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-1.7B-Base --out ../tests/reference-1.7b
