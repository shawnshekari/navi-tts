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
  vocoder gate and the only thing to listen to.

Regenerate (about a minute on the workstation CPU):

    cd tools && uv sync && uv run dump_reference.py --model ../models/Qwen3-TTS-12Hz-0.6B-Base --out ../tests/reference
