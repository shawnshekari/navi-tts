# bench/

`results.jsonl` is the record the README numbers are generated from. One JSON
line per `navi-tts bench` run, appended by the harness itself:

    ./build/xtx/navi-tts bench --model models/qwen3-tts-0.6b-f16.navi --out bench/results.jsonl

Only append from a clean checkout (`git` field without `-dirty`) with
`tts-engine` stopped and the card otherwise quiet (CLAUDE.md); numbers taken
under contention or with weights spilled to GTT are not comparable.

Fields: `ts`, `git`, `gfx`, `rocm`, `hip_runtime`, `device`, `multiprocessors`,
`model`, `dtype`, `weight_bytes`, `load_map_ms`, `load_upload_ms`,
`upload_gbps`, then the stage numbers `prefill_ms`, `frame_ms`, `talker_ms`,
`cp_ms`, `vocoder_ms_per_frame`, `ttfa_ms`, `rtf`, `n_frames`, `wav_sha256`
(`null` until the stage exists - M1), and the fixed `text` / `seed`.

The bench text and seed are the same ones `tools/dump_reference.py` uses
(`tests/reference/manifest.json`), so a `wav_sha256` change is either an
intended arithmetic change or a bug (DESIGN 6.3).
