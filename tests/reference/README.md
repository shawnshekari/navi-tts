# tests/reference/

PyTorch reference dumps for the parity gates (DESIGN 6), produced by
`tools/dump_reference.py` on the CPU in float32. `manifest.json` lists the
text, the voice sample's sha256, the model checksums and the package versions;
the docstring of the script and `docs/model.md` describe every file.

`voice_1.wav` is the production `voice_1` reference sample (10 s, 24 kHz mono).

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

Regenerate (about a minute on the workstation CPU):

    cd tools && uv sync && uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-0.6B-Base --out ../tests/reference
