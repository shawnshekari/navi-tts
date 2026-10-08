# navi-tts

AMD-first TTS engine: RX 7900 XTX (gfx1100) first, Strix Halo (gfx1151) later.
HIP only. Read `docs/DESIGN.md` before changing anything; `HISTORY.md` says why.

## Rules that are not up for re-litigation

- **No ggml, no CPU/Vulkan/CUDA path, no fallback path.** A failed kernel fails
  the request cleanly. Never add a "temporary" software path.
- **No code from either upstream** (predict-woo/qwen3-tts.cpp, khimaros/qwen3-tts.cpp).
  Not source, not docs, not scripts - re-derive facts, don't copy text. The
  frozen fork (`khimaros/qwen3-tts.cpp`) is read-only reference; the only files
  from it that belong here are already in `reference/` (the author's own work).
- **`reference/` is never built or included.** Draw from it, then write it here.
- **Decisions marked DECIDED in `docs/DESIGN.md` stand.** Items marked OPEN need
  the user's call before code depends on them.

## Toolchain (workstation)

- ROCm 10.1.0, therock tarball: `~/tools/therock-tarball/install`.
  `LD_LIBRARY_PATH=~/tools/therock-tarball/install/lib:~/tools/therock-tarball/install/llvm/lib` at runtime.
- Compiler: the ROCm clang for host AND device - `clang++` on PATH is
  `~/tools/therock-tarball/install/llvm/bin/clang++` (AMD clang 23). Call it
  directly (`-x hip --offload-arch=gfx1100`), not the `hipcc` wrapper. Never mix
  in system g++ for host objects. See `docs/reference/toolchain.md` for the
  compiler-rt gotcha if linking fails.
- CMake 3.28 + Ninja. Presets in `CMakePresets.json` (`xtx`; `strix` at M3).
  Host flags: C++20, `-O3 -march=znver3 -flto`; no `-ffast-math` until the parity
  gates pass without it.
- CPU: Ryzen 7 5800X3D (Zen 3, AVX2/FMA, no AVX-512, 96 MB L3). 125 GB RAM.
- Python only for `tools/` (converter, reference dumps): `uv`, never a system pip.

## Model files

- HF weights: `models/Qwen3-TTS-12Hz-0.6B-Base/` (copied from the fork; see `models/README.md`)
  (`config.json`, `model.safetensors`, `vocab.json`, `merges.txt`,
  `speech_tokenizer/`). Source of truth for `tools/convert.py` and `docs/model.md`.
- Output `.navi` blobs go in `models/` too. Everything under `models/` except its README is git-ignored.

## The live GPU is shared - do not break production

- `tts-engine.service` (the old fork) is what the user hears right now, and
  `llama-server` + `embedding-server` hold most of the VRAM. Do **not** stop or
  restart any of them without asking. `Before=llama-server.service` ordering in
  the old unit is deliberate (VRAM first); keep it in `navi-tts.service`.
- Benchmarks are only valid with `tts-engine` stopped and the card otherwise
  quiet; ask first, and restore it after. Check residency with
  `grep drm-resident /proc/<pid>/fdinfo/*` - GTT spill means numbers are 2x off.
- Contention testing uses `reference/tests/gpu_saturate.hip` as the hog, on
  purpose and briefly.
- A cooperative kernel must never be able to hang the box: every barrier has a
  spin cap and an abort flag from day one, before the kernel does real work.

## How to verify

- Parity gates and reference dumps: `docs/DESIGN.md` §6. Same text + seed must
  be bit-exact run to run; `cmp` the WAV.
- `navi-tts bench` (M0+) emits JSON; append to `bench/results.jsonl` with the git
  hash. README numbers are generated from that file, never typed.
- Serving contract: `docs/DESIGN.md` §3. Clients are the household TTS
  queue, SkyrimNet, and direct OpenAI-dialect clients; none may
  need changes at cutover.

## Commits

Small, focused, conventional-commit style (`feat:`, `fix:`, `perf:`, `docs:`,
`bench:`). A `perf:` commit states the before/after numbers in its body.
