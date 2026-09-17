# The `.navi` weight file

One file per model: a header, a key/value table of hyperparameters, a tensor
table, then page-aligned tensor data. Written by `tools/convert.py`, read by
`runtime/weights`. The engine reads whatever dtype the table says; the
converter is where a fine-tune or a quantised variant plugs in (DESIGN 5.3).

All integers are little-endian. Version 1.

## Layout

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `NAVI` |
| 4 | u32 | format version (1) |
| 8 | u64 | `data_offset`: start of the tensor data region, multiple of 4096 |
| 16 | u32 | number of KV entries |
| 20 | u32 | number of tensor entries |
| 24 | u64 | total file size (sanity check) |
| 32 | … | KV entries, then tensor entries |

`string` := u32 byte length, then that many UTF-8 bytes (no terminator).

KV entry := `string` key, u8 type, value:

| type | value |
|---|---|
| 0 `I64` | i64 |
| 1 `F64` | f64 |
| 2 `STR` | `string` |
| 3 `I64_ARRAY` | u32 count, then count × i64 |

Tensor entry := `string` name, u8 dtype, u8 ndim, ndim × u64 dims (row-major,
outermost first, as PyTorch prints them), u64 offset relative to
`data_offset` (multiple of 4096), u64 byte length.

| dtype | |
|---|---|
| 0 | F32 |
| 1 | F16 |
| 2 | BF16 |
| 3 | I32 |
| 4 | U8 (raw bytes, e.g. tokenizer files) |

The file is `mmap`ed at start; each tensor is uploaded straight from the
mapping, then the mapping is dropped. Nothing in the header needs a parser
beyond `memcpy`.

## Keys written for Qwen3-TTS

`docs/model.md` lists the values. Namespaces:

- `general.*` — `arch` (`qwen3-tts`), `name`, `dtype`, `converter`, source file names.
- `text.*` — the text-side special token ids (`im_start`, `im_end`, `tts_bos`, `tts_eos`, `tts_pad`, …).
- `talker.*` — the talker transformer hyperparameters, its M-RoPE section, the codec-side control ids (`codec_bos`, `codec_eos`, `codec_pad`, think/nothink), and `talker.language.<name>` ids.
- `cp.*` — the code predictor hyperparameters.
- `speaker.*` — speaker encoder output dim, sample rate, and the mel front-end parameters.
- `codec.*`, `codec.decoder.*` (and `codec.encoder.*` when converted with `--with-encoder`).
- `gen.*` — the model's default sampling parameters from `generation_config.json`.

## Tensors

Names are the HF names verbatim for the main checkpoint (`talker.*`,
`speaker_encoder.*`); codec tensors get a `codec.` prefix. Three tensors are
not weights: `tokenizer.vocab.json`, `tokenizer.merges.txt` and
`tokenizer.config.json` are the tokenizer files as U8 bytes, so one file is
the whole model.

Conversion rules (the only transformations applied):

- Tensors with two or more dims are stored in the target dtype (F16 by
  default); one-dimensional tensors (norm weights, biases, `alpha`/`beta`,
  layer scales) stay F32.
- The codec's EMA codebooks (`embedding_sum`, `cluster_usage`) are baked into
  the effective codebook `embedding_sum / max(cluster_usage, 1e-5)` and stored
  as `…vq.layers.N.codebook` `[2048, 256]`; the pair is dropped.
- Bookkeeping tensors (`…codebook.initialized`) are dropped.
- The codec encoder (ICL cloning, DESIGN 9 OPEN) is left out unless
  `--with-encoder` is given.
