# bench/

`results.jsonl` is the record the README numbers are generated from. One JSON
line per `navi-tts bench` run, appended by the harness itself:

    ./build/xtx/navi-tts bench --model models/qwen3-tts-0.6b-f16.navi --out bench/results.jsonl

Only append from a clean checkout (`git` field without `-dirty`) with
`navi-tts` (and `tts-queue`) stopped and the card otherwise quiet; numbers taken
under contention or with weights spilled to GTT are not comparable (on Strix
Halo everything is GTT by design - the 512 MiB carve-out - so that check does not apply).

Fields: `ts`, `git`, `gfx`, `rocm`, `hip_runtime`, `device`, `multiprocessors`,
`model`, `dtype`, `weight_bytes`, `load_map_ms`, `load_upload_ms`,
`upload_gbps`, then the stage numbers `prefill_ms`, `frame_ms`, `talker_ms`,
`cp_ms`, `vocoder_ms_per_frame`, `ttfa_ms`, `rtf`, `n_frames`, `prefill_rows`,
`wav_sha256` (`talker_ms` / `cp_ms` stay `null`: one launch per frame, not
split), and the fixed `text` / `voice` / `seed`.

A run with `--ref-text` is in-context cloning and carries `clone_mode: "icl"`,
`ref_frames`, and `frame_ms_at` - the frame time at positions 10/50/100/200,
because an in-context prompt adds its reference's frames to *every* frame's
attention and a mean over utterances of different lengths would hide that. A
position the run never reached is absent rather than zero. Those rows measure a
different thing from the standard protocol, so `tools/bench_table.py` leaves
them (and any row with a non-default `voice`) out of the README table and
`docs/tuning.md` compares them instead.

`tools/bench_table.py` regenerates the README table from this file; never type
numbers into the README. The utterance runs with
the streaming batch configuration (`STREAM_FIRST_BATCH` then `STREAM_BATCH`,
graph.h) so `ttfa_ms` is what a streaming client sees; `--batch N` measures a
fixed batch instead (32 = a whole-body request, the queue's case).

The bench text and seed are the same ones `tools/dump_reference.py` uses
(`tests/reference/manifest.json`), so a `wav_sha256` change is either an
intended arithmetic change or a bug (DESIGN 6.3).

`micro/` holds the standalone HIP microbenchmarks behind kernel decisions
(matvec shape, sampling). They are not built by CMake; each file says how.
