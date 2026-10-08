# models/

Weights live here; everything except this file is git-ignored (multi-GB).

- `Qwen3-TTS-12Hz-0.6B-Base/` - the HF release as downloaded: `config.json`,
  `model.safetensors` (talker + code predictor + speaker encoder), the Qwen3
  BPE tokenizer (`vocab.json`, `merges.txt`, `tokenizer_config.json`), and
  `speech_tokenizer/` (the 12 Hz codec: encoder, decoder/vocoder, its
  `config.json`). Source of truth for `tools/convert.py` and `docs/model.md`.
- `*.navi` - converted blobs the engine loads, produced by `tools/convert.py`.

Re-fetch with `huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base` if lost.

Weights here (and the `.navi` blobs converted from them) are governed by the
Qwen Community License 1.0, not by the repository's AGPL-3.0 code license.
