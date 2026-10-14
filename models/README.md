# models/

Weights live here; everything except this file is git-ignored (multi-GB).

- `Qwen3-TTS-12Hz-0.6B-Base/` - the HF release as downloaded: `config.json`,
  `model.safetensors` (talker + code predictor + speaker encoder), the Qwen3
  BPE tokenizer (`vocab.json`, `merges.txt`, `tokenizer_config.json`), and
  `speech_tokenizer/` (the 12 Hz codec: encoder, decoder/vocoder, its
  `config.json`). Source of truth for `tools/convert.py` and `docs/model.md`.
- `Qwen3-TTS-12Hz-1.7B-Base/` - the same layout, wider talker (2048 against
  1024). Supported since 2026-10-13; converts and loads with no code change, and
  has its own parity dumps in `tests/reference-1.7b/`. Its code predictor needs
  the `small_to_mtp_projection` stage (`docs/model.md`).
- `*.navi` - converted blobs the engine loads, produced by `tools/convert.py`.
  `--with-encoder` includes the codec audio encoder (+190 MB, f32 - `docs/model.md`
  says why it is not f16), which is what makes the reference codes for in-context
  cloning. A blob without it clones by speaker embedding only; it can still serve a
  voice whose codes were made by a blob that has it.

Re-fetch with `huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base` (or
`...-1.7B-Base`) if lost.

Weights here (and the `.navi` blobs converted from them) are governed by the
Qwen Community License 1.0, not by the repository's AGPL-3.0 code license.
