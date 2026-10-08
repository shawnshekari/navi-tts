# History

Where navi-tts came from: the discrete-GPU and HIP work done on a fork of
[khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp) (itself on
[predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp)) between
June and September 2026, before both upstreams went quiet and this project was
started from scratch. navi-tts shares no code with that fork; this file keeps the
reasoning - what was tried, what it measured, what broke - so it doesn't have to be
rediscovered. The fork itself was a frozen local clone (hashes below are
its), 26 commits past upstream `0c8b2ba`.

## The arc

Everything ran on an RX 7900 XTX (gfx1100, 96 CU) and was measured as real-time
factor (RTF = render time / audio length) and milliseconds per 12 Hz frame.

| Step | RTF | Where it came from |
|---|---|---|
| Vulkan baseline, vocoder on CPU | 0.89 | the fork as inherited |
| Vocoder weights kept on the GPU (F16 → F32 cast for the transposed convs) | 0.32 | `b11fd22`, `cebfbd3` |
| Persistent code-predictor graphs, no per-run up-cast | 0.24 (Vulkan) | `9e3950c` |
| O(K) `conv_transpose_1d` in ggml (own ggml fork) | 0.205 | `e2597f1`, `22f25a0` |
| Fused cooperative code predictor with on-device sampling | – | `5800f57` |
| Fused cooperative talker with on-device cb0 sampling (~19% end-to-end) | ~0.13 | `d411e57` |
| One fused talker+cp kernel, one cooperative launch per frame | – | `c1b977a` |
| Permanent latch regression under contention | 0.21 | `dd39236` |
| Self-healing re-armable latch | – | `09580fd` |
| Stale-hidden fix (runaway root cause), 1 block/CU | 0.12, 14.7 ms/frame | `1bf3904` |

Lessons that carry into navi-tts, independent of ggml:

- **Batch-one decode is bandwidth- and launch-bound.** Every big win was either
  removing a launch (fusion, persistent graphs, on-device sampling) or removing a
  pass over memory (no per-run up-casts, O(K) conv). The kernel shapes matter less
  than how many times the GPU is told to start.
- **Cooperative kernels under contention need care.** Occupancy that benchmarks
  well in isolation (4 blocks/CU) timed out on every request once another process
  touched the GPU; 1 block/CU had zero timeouts. A one-shot "fallback forever"
  latch turned a transient stall into a permanent 60% regression - fallbacks must
  re-arm.
- **A fallback path that isn't bit-exact is a correctness bug, not a perf one.**
  The runaway-audio incidents (a 2-char request producing 48 s of babble, 10 s GPU
  pegs) traced to stale hidden state when the fused path fell back mid-frame -
  not to the model. navi-tts has no fallback path; this is one reason why.
- **Bound every request.** Per-request `max_audio_tokens` and chunked vocoder
  decode (64 frames) exist because a single runaway request took the live server
  down with it.

## Commits

Oldest first. Each entry is the commit's own message and file summary.

### `fdafdec` 2026-06-19 - fix(libav): support FFmpeg < 7.1 and fix write callback signature

- Guard avcodec_get_supported_config with #if defined so it works on
  FFmpeg 6.x (Ubuntu 24.04), falling back to AV_SAMPLE_FMT_FLTP
- Fix libav_write_cb callback signature from const uint8_t* to uint8_t*
  to match the FFmpeg API declaration

```
 src/qwen3_tts.cpp | 26 +++++++++++++++++---------
 1 file changed, 17 insertions(+), 9 deletions(-)
```

### `86d0fb7` 2026-06-19 - fix(cmake): set Vulkan RPATH and disable optional httplib dependencies

- Set BUILD_RPATH and INSTALL_RPATH on the ggml target when GGML_VULKAN
  is ON, so executables can find libggml-vulkan.so at runtime
- Force HTTPLIB_REQUIRE_ZSTD and HTTPLIB_REQUIRE_OPENSSL OFF to avoid
  linking failures (USE_*_IF_AVAILABLE defaults remain ON)

```
 CMakeLists.txt | 13 +++++++++++++
 1 file changed, 13 insertions(+)
```

### `dc9fa65` 2026-06-19 - Merge branch 'fix/vulkan-rpath'

```
 CMakeLists.txt | 13 +++++++++++++
 1 file changed, 13 insertions(+)
```

### `d960101` 2026-06-19 - docs: add Vulkan build script and custom voice setup guide

- Add build-vulkan.sh for Vulkan SDK setup and CMake build
- Add docs/custom_voice_setup.md with step-by-step guide for
  creating and registering custom voices with the HTTP server

```
 build-vulkan.sh            |  51 ++++++++++++++
 docs/custom_voice_setup.md | 236 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 287 insertions(+)
```

### `5345881` 2026-06-19 - docs: add detailed Vulkan timing baseline for 0.6B and 1.7B models

```
 docs/performance_plan.md | 121 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 121 insertions(+)
```

### `2eb0cf6` 2026-09-10 - build: disable httplib's opportunistic zstd/openssl, enable timing in build-vulkan.sh

HTTPLIB_USE_*_IF_AVAILABLE still picked up host libs and broke linking
even with the REQUIRE_* flags off.

```
 CMakeLists.txt  | 2 ++
 build-vulkan.sh | 4 +++-
 2 files changed, 5 insertions(+), 1 deletion(-)
```

### `b11fd22` 2026-09-10 - fix: vocoder weights fell back to CPU on discrete GPUs

audio_tokenizer_decoder loads its tensors with GGML_BACKEND_DEVICE_TYPE_IGPU.
On a discrete card that init fails and load_tensor_data_from_file went
straight to CPU, so the scheduler ran the entire vocoder graph on the CPU
even though the compute backend logged as Vulkan0. Try the other GPU
device type before falling back to CPU.

RX 7900 XTX, 0.6B F16: vocoder decode 2883 ms -> 510 ms for 5.2 s of
audio; end-to-end RTF 0.89 -> 0.32.

```
 src/gguf_loader.cpp | 10 ++++++++++
 1 file changed, 10 insertions(+)
```

### `cebfbd3` 2026-09-10 - perf: keep vocoder transposed convs on the GPU by casting F16 weights to F32

ggml-vulkan's CONV_TRANSPOSE_1D kernel only accepts F32 weights; the F16
weights in the tokenizer GGUF made the scheduler bounce all six transposed
convs (2 upsample + 4 decoder blocks) to the CPU, shipping activations of
up to ~45 MB across the bus each way. Cast the small weight tensor in-graph
so the whole vocoder graph stays on Vulkan (13 splits -> 2, 0 on CPU).

RX 7900 XTX, 0.6B F16, 5.2 s clip: vocoder decode 510 ms -> 232 ms.

```
 src/audio_tokenizer_decoder.cpp | 14 ++++++++++++--
 1 file changed, 12 insertions(+), 2 deletions(-)
```

### `500bb6a` 2026-09-10 - docs: code predictor speed-up plan and dGPU profiling notes

Profile of one code-predictor step on RX 7900 XTX (128 dispatches, ~760 us,
~180 us bandwidth floor), the three-phase plan (HIP dispatch-floor test,
ggml graph caching, fused cooperative HIP kernel), profiling recipes, and
what has already been ruled out.

```
 docs/code_predictor_plan.md | 192 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 192 insertions(+)
```

### `e8076e8` 2026-09-10 - docs: upstream status and how to submit the dGPU fixes later

```
 docs/code_predictor_plan.md | 53 +++++++++++++++++++++++++++++++++++++++++++++++++----
 1 file changed, 49 insertions(+), 4 deletions(-)
```

### `2917e8f` 2026-09-10 - docs: Phase 0 results — HIP build, dispatch cost, grid.sync microbench

Phase 0 of the code predictor plan: build with GGML_HIP=ON in the
navi31-llama toolbox (ROCm 10.1 nightly) and compare per-step cost
against Vulkan/RADV on the RX 7900 XTX.

- ROCm per-dispatch cost is ~5.1 us vs ~3.7 us on RADV; the plain HIP
  build is a ~10% loss on the frame (Steps 12.2 vs 10.7 ms/frame).
- GGML_HIP_GRAPHS (compile-time) never engages: the step graphs are
  rebuilt every step with a new n_past, so ggml's warmup never completes.
  Re-test after Phase 1 makes the graphs persistent.
- The HIP vocoder is ~100x slower than Vulkan; it must stay on Vulkan.
- grid.sync() costs 0.6-1.1 us on gfx1100 (scripts/bench_gridsync.hip),
  5-8x cheaper than a dispatch, so the Phase 2 estimate tightens to
  ~4 ms/frame.
- The ROCm nightly's clang ships no host compiler-rt builtins; the plan
  documents the -resource-dir overlay workaround.

Also adds the benchmark harness under scripts/bench/ and ignores all
build*/ directories.

```
 .gitignore                    |  2 +-
 docs/code_predictor_plan.md   | 73 +++++++++++++++++++++++++++++++++++++++++++++++++++++
 scripts/bench/bench_hip.sh    | 25 ++++++++++++++++++
 scripts/bench/bench_vulkan.sh | 32 +++++++++++++++++++++++
 scripts/bench_gridsync.hip    | 87 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 218 insertions(+), 1 deletion(-)
```

### `9e3950c` 2026-09-10 - perf: persistent code-predictor graphs, drop the per-run ffn_down up-cast

Phase 1 of docs/code_predictor_plan.md. 17.4 -> 15.1 ms/frame on the
RX 7900 XTX (Vulkan), RTF 0.27 -> 0.24; HIP 18.9 -> 15.2.

- Build the code predictor's prefill graph and 14 step graphs once, on
  the first frame, each with its own meta context and ggml_gallocr, and
  run them with ggml_backend_graph_compute on the device backend. All
  inputs live in one dedicated buffer so the constant ones (positions,
  per-step causal mask) are set once at init. Per step the host does one
  4-byte tensor_set, one compute, one 8 KB tensor_get. Falls back to the
  scheduler path if the backend can't run an op, or when
  QWEN3_TTS_CODE_PRED_SCHED=1 (A/B knob). Greedy output is bit-identical
  between the two paths on Vulkan and HIP.
- ffn_down is F16 in the GGUF but every graph up-cast it to F32 with
  ggml_cast (a 12 MB copy + dispatch per layer, 5 per predictor step and
  28 per talker frame). Replace with ggml_mul_mat_set_prec(GGML_PREC_F32)
  on the matmul: batched (prefill) GEMMs keep F32 accumulation — without
  it the HIP build's F16-accumulate GEMM derails generation — and the
  decode matvecs are untouched.
- Server gains --max-tokens (default 2048, same as the CLI).
- Bench scripts: VRAM guard, before/after report; the HIP one caps
  generation, kills the server on a runaway or timeout, and never runs
  the pathologically slow HIP vocoder on the long text.

Changes arithmetic order, so sampled codes for a given seed re-roll; the
live service binary in build/ is deliberately not rebuilt yet.

```
 docs/code_predictor_plan.md   |  66 +++++++++++++
 scripts/bench/bench_hip.sh    |  40 +++++++-
 scripts/bench/bench_vulkan.sh |  10 ++
 src/server.cpp                |   6 ++
 src/tts_transformer.cpp       | 332 ++++++++++++++++++++++++++++++++++++++++++++++++++++----------
 src/tts_transformer.h         |  45 ++++++++-
 6 files changed, 440 insertions(+), 59 deletions(-)
```

### `7bc73ca` 2026-09-10 - fix: bound vocoder memory by decoding one-shot requests in 64-frame chunks

ggml's scheduler keeps its compute buffers at the high-water mark of the
largest graph ever run, and a one-shot vocoder decode needs ~4.7 MB of
scratch per frame, so a single runaway 2048-frame request left the live
server holding 16.8 GB of GPU memory until restart. decode() now runs
the streaming decoder over equal chunks of at most 64 frames (utterances
up to 64 frames are unchanged); QWEN3_TTS_DECODE_CHUNK overrides the
chunk size, 0 = single graph. A 496-frame request ends at 3.6 GB instead
of 5.9 and the vocoder share no longer depends on length.

Output is ~51 dB SNR from the one-shot path on the RX 7900 XTX, not
bit-exact: test_streaming_parity already fails its tolerance here with
the untouched one-shot decode because Vulkan's coopmat matmul output
depends on batch size (the streaming design was validated on Strix
Halo). Chunked decode() vs stream_decode() at the same chunk is exact.

```
 docs/code_predictor_plan.md     | 17 +++++++++++++++++
 src/audio_tokenizer_decoder.cpp | 32 +++++++++++++++++++++++++++++++-
 src/audio_tokenizer_decoder.h   | 12 ++++++++++++
 3 files changed, 60 insertions(+), 1 deletion(-)
```

### `e2597f1` 2026-09-10 - perf(ggml-cuda): O(K) instead of O(L_in) per output in conv_transpose_1d

The vendored kernel looped over every input position for every output
element and skipped the ones outside the kernel window, so the cost per
output scaled with the sequence length. On the RX 7900 XTX (HIP) the
six transposed convs of the vocoder took 3.6 s for 1.1 s of audio,
95.7% of the decode's GPU time. Compute the contributing input range
directly; the summation order is unchanged, so the output is
bit-identical. That decode now takes 19 ms, and a full request on HIP
runs at RTF 0.205 vs 0.239 on Vulkan.

```
 docs/code_predictor_plan.md | 19 +++++++++++++++++++
 1 file changed, 19 insertions(+)
```

### `4950330` 2026-09-12 - docs: add dGPU fixes and RX 7900 XTX performance to README

```
 README.md | 12 +++++++++++-
 1 file changed, 11 insertions(+), 1 deletion(-)
```

### `593eb3e` 2026-09-12 - docs: add mermaid pipeline diagram with fork-attribution legend

```
 README.md | 33 +++++++++++++++++++++++++++++++++
 1 file changed, 33 insertions(+)
```

### `5800f57` 2026-09-12 - feat(hip): fused cooperative code predictor with on-device sampling

Single cooperative kernel runs a whole code-predictor frame (16 positions,
5 layers, 15 output heads) with grid-wide barriers instead of 128 dispatches
per step. Sampling runs on device: greedy argmax, or temperature + top-k
(bitonic sort in LDS) + PCG32 multinomial seeded per (frame, position).

Anti-hang: the cg::grid.sync() residency-contention deadlock (occupancy API
only sees the calling process; with other GPU tenants the cooperative grid
can't fully launch and the barrier spins with the display blocked) is replaced
by a deadline-aware sense-reversing barrier (50M-spin cap,
QWEN3_TTS_HIP_BARRIER_SPINS) that unwinds the grid cleanly; cooperativeLaunch
is checked at init and the grid is capped by QWEN3_TTS_HIP_BLOCKS_PER_CU.

Integrated behind QWEN3_TTS_USE_HIP_CODE_PRED=1 via the CoreML seam (host
hidden + cb0 in, 15 codes out, no logits readback); gated to the 0.6B shape,
falls back to the ggml path otherwise. Code-predictor stage 10.1 -> 5.9
ms/frame sampled; greedy reproduces the reference codes 15/15 and sampled
fused == per-phase 15/15. test_hip_code_pred validates against a
QWEN3_TTS_DUMP_CODE_PRED reference frame. Seed re-audition: seed 3 keeper.

```
 CMakeLists.txt               |  25 +++
 docs/code_predictor_plan.md  | 111 +++++++++++
 src/hip/code_pred_hip.h      |  88 +++++++++
 src/hip/code_pred_hip.hip    | 662 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/tts_transformer.cpp      | 152 ++++++++++++++-
 src/tts_transformer.h        |  17 ++
 tests/test_hip_code_pred.cpp | 235 +++++++++++++++++++++++
 7 files changed, 1289 insertions(+), 1 deletion(-)
```

### `22f25a0` 2026-09-12 - chore(ggml): point submodule at shawnshekari/ggml fork with the O(K) conv_transpose_1d fix

The all-HIP vocoder needs the conv_transpose_1d fix (Phase 2 step 0), which
can't be pushed to ggml-org. Repointed the submodule at the fork's
fix/conv-transpose-1d-ok branch (741d8b41) so a GGML_HIP=ON build from a
fresh clone gets the fast vocoder. Vulkan builds are unaffected.

```
 .gitmodules | 3 ++-
 ggml        | 2 +-
 2 files changed, 3 insertions(+), 2 deletions(-)
```

### `d411e57` 2026-09-12 - feat(hip): fused cooperative talker with on-device cb0 sampling

Phase 3 of the dGPU latency plan (docs/code_predictor_plan.md). Mirrors
the fused code predictor: one cooperative kernel runs a whole talker
decode step — 28 Qwen2 layers with attention over the growing KV cache,
the output norm, the codec head — and samples the codebook-0 token on
device, replacing forward_step + the host sampling block.

Talker-specific handling vs the code predictor:
- External KV cache: the kernel reads/writes the same F16 k_cache/v_cache
  the ggml prefill fills, via the tensors' device pointers; position
  stride head_dim*n_kv_head matches the contiguous [head_dim, n_kv_head,
  n_ctx] layout. The prefill stays on the ggml backend.
- Weights read in place through the ggml tensors' device pointers. The
  first cut copied 1.2 GB to a private buffer; that one-time ~209 ms
  upload erased the per-frame win on a ~190-frame request. Reading in
  place drops unaccounted overhead to ~22 ms.
- On-device cb0 sampling mirrors generate(): suppress [vocab-1024,
  vocab) except EOS, HuggingFace repetition penalty over previously
  emitted cb0 (device-side seen[] byte set), then greedy or
  temperature+top-k (bitonic). Frame 0's cb0 still comes from the
  prefill logits on the host and seeds the set via mark_seen.

Integrated behind QWEN3_TTS_USE_HIP_TALKER=1 (off by default); falls
back to the ggml path on init failure. Not yet in production — see
docs/talker_fusion_handoff.md for the three ship gates (mid-generation
fallback, seed re-audition, build path).

Parity vs the QWEN3_TTS_DUMP_TALKER reference (tests/test_hip_talker):
hidden max rel err 0.018%, logits 0.05%, cb0 argmax match; per-phase
and fused agree. RX 7900 XTX, 0.6B F16, greedy: talker 3.9 -> 2.4
ms/frame in-context (1.95 ms unit, 4 blocks/CU); ~19% faster end-to-end
on a ~190-frame request (1962 vs 2428 ms total generate).

Adds a [loop wall / host tail] diagnostic under QWEN3_TTS_TIMING.


```
 CMakeLists.txt                |  11 +
 docs/code_predictor_plan.md   |  62 ++++++
 docs/talker_fusion_handoff.md | 148 +++++++++++++
 src/hip/talker_hip.h          | 119 +++++++++++
 src/hip/talker_hip.hip        | 697 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/tts_transformer.cpp       | 272 ++++++++++++++++++------
 src/tts_transformer.h         |   9 +
 tests/test_hip_talker.cpp     | 257 +++++++++++++++++++++++
 8 files changed, 1515 insertions(+), 60 deletions(-)
```

### `123078d` 2026-09-12 - test(hip): Gate-1 fused-talker fallback test + GPU hog

Adds the mid-generation fallback test (forks a full-grid spin hog, runs
fused-talker requests with the barrier cap tightened so residency failure
is deterministic) and the gpu_saturate helper it uses. Left untracked in
the prior session; required for the test suite to build.

```
 tests/gpu_saturate.h               |  16 ++++++
 tests/gpu_saturate.hip             |  41 ++++++++++++++
 tests/test_hip_talker_fallback.cpp | 171 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 228 insertions(+)
```

### `c1b977a` 2026-09-12 - feat(hip): single fused talker+cp frame kernel (one coop launch/frame)

Replaces the per-frame three-launch chained path (talker.run_device +
cp.run_device + assemble_step_embd) with ONE cooperative kernel that
does the whole frame on-device: talker 28-layer step (-> hidden + cb0 +
KV row), the cp 16-position loop (-> 15 codes), then the next-frame
step_embd assembly. The host reads back only the 16 codes.

- hip_dev.h: the deadline-aware grid_barrier + matvec/norm/argmax
  helpers lifted verbatim out of both shipped kernels into one shared
  header, so the fused kernel shares ONE barrier buffer + spin cap
  across the talker's ~170 and cp's 16x~8 barriers.
- Residency measured: k_frame_fused = 4 blocks/CU, same as the
  standalone kernels -> same 192-block grid, no scheduling regression.
- Bit-identical parity vs the chained path (cb0 + 15/15 codes + next
  step_embd memcmp equal), greedy and sampled; e2e wav byte-identical
  over 120 frames (seed order keeps the rng stream aligned).
- Barrier timeout falls back to the three-launch CHAINED path (not
  ggml): latches hip_frame_failed_, re-syncs the device seen[] set
  (Gate-1 mirror), chained tail redoes the frame.
- Gated QWEN3_TTS_USE_HIP_FRAME_FUSION=1 (requires fused talker+cp);
  default off. NOT deployed to tts-engine pending a live parity audition.

Tests: test_hip_frame_fused (parity), test_hip_frame_fallback
(deterministic mid-generation fallback via the run_frame test hook).

```
 CMakeLists.txt                    |  29 +++
 docs/talker_fusion_handoff.md     | 305 ++++++++++++++++++++--
 src/hip/code_pred_hip.h           |  22 ++
 src/hip/code_pred_hip.hip         | 275 ++++++++------------
 src/hip/frame_fused.h             |  93 +++++++
 src/hip/frame_fused.hip           | 823 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/hip/hip_dev.h                 | 195 ++++++++++++++
 src/hip/talker_hip.h              |  21 ++
 src/hip/talker_hip.hip            | 203 +++------------
 src/tts_transformer.cpp           | 302 ++++++++++++++++++---
 src/tts_transformer.h             |  37 ++-
 tests/test_hip_frame_fallback.cpp | 132 ++++++++++
 tests/test_hip_frame_fused.cpp    | 308 ++++++++++++++++++++++
 13 files changed, 2342 insertions(+), 403 deletions(-)
```

### `dd39236` 2026-09-12 - docs: record permanent-latch RTF regression + deferred self-healing options

Production RTF climbed 0.13 -> 0.21: the fused talker and fused cp each
hit a grid-barrier timeout under GPU contention (llama-server x2 +
embedding-server at 100% util) and latched to GGML for the whole process
(one-way latches). Fused-frame kernel does not help (same coop-residency
requirement). Immediate remedy: restart on a quiet GPU. Self-healing
policy (auto-unlatch/cooldown, per-frame retry, adaptive spin deadline,
contention-aware scheduling) deferred for later discussion.

```
 docs/talker_fusion_handoff.md | 59 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 59 insertions(+)
```

### `5dcca64` 2026-09-12 - feat(server): accept per-request max_audio_tokens

The JSON API had no way to bound a single request's length; the only cap
was the server-wide --max-tokens (600 = 48 s). A runaway generation (the
model never emits EOS - seen today on a 79-char fragment) therefore ran
to 600 frames, 9 s of GPU for 5 s of speech, and the client had to
truncate the babble afterwards.

Clients that know how long their text should take can now send
max_audio_tokens; it is clamped to [1, --max-tokens] so a request can
never raise the server cap. The queue sends 12.5 * (2.5 + 0.12 * chars).

```
 src/server.cpp | 9 ++++++++-
 1 file changed, 8 insertions(+), 1 deletion(-)
```

### `09580fd` 2026-09-12 - feat(hip): self-healing re-armable latch for fused cooperative components

A grid-barrier timeout under GPU contention latched the fused talker/cp/
frame off for the whole process, pinning RTF to the GGML path (~0.21 vs
~0.13) until a manual restart (docs/talker_fusion_handoff.md KNOWN
ISSUE). Replace the one-way latch with a HipHeal state machine (one per
component) driven at the request boundary in generate():

- Cooldown-gated probe + exponential backoff: while latched, re-attempt
  the fused path every `backoff` requests; a failed probe doubles the
  backoff (cap 64) so a persistently-busy GPU stops paying probe stalls,
  the first success resets it to 1 and the fused path returns.
- Tight probe deadline: set_probe_mode() drops the barrier spin cap to
  1/10 (~50ms vs ~500ms) for a probe request so a still-busy GPU fails
  fast; restored on heal. Added to HipTalker/HipCodePredictor/HipFrameFusion.
- Not a special code path: the probe just clears the existing *_failed_
  gate and runs the normal fused path through the existing fallback
  machinery, so the Gate-1 seen[] re-sync invariant holds unchanged.
- Init failures stay permanent (begin_request never probes !ready).
- Journal observability: probing / probe failed+backoff / recovered lines.

Tests: new test_hip_heal (pure-logic policy: cadence, exponential
growth, cap, init-failure permanence, full latch->quiet->recover cycle).
test_hip_talker_fallback updated to the new contract (2nd request
re-probes + re-latches under the hog). test_hip_frame_fallback given a
fixed seed (was nondeterministically hitting early EOS at temp 0.9).
All HIP tests pass.

```
 CMakeLists.txt                     |   7 +++
 docs/talker_fusion_handoff.md      |  75 ++++++++++++++++++++++----------
 src/hip/code_pred_hip.h            |   4 ++
 src/hip/code_pred_hip.hip          |  14 +++++-
 src/hip/frame_fused.h              |   4 ++
 src/hip/frame_fused.hip            |  14 +++++-
 src/hip/talker_hip.h               |   6 +++
 src/hip/talker_hip.hip             |  16 ++++++-
 src/tts_transformer.cpp            |  90 ++++++++++++++++++++++++++++++++++++++
 src/tts_transformer.h              |  66 ++++++++++++++++++++++++++++
 tests/test_hip_frame_fallback.cpp  |   4 ++
 tests/test_hip_heal.cpp            | 133 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/test_hip_talker_fallback.cpp |  22 +++++++---
 13 files changed, 421 insertions(+), 34 deletions(-)
```

### `1bf3904` 2026-09-12 - fix(hip): stale hidden on fused-cp -> ggml fallback (runaway root cause); 1 block/CU; timing

Live investigation of "10 s GPU pegs + runaway audio" under contention.

1. Stale hidden (correctness). The fused talker leaves its hidden on the
   device and skips the host copy; when the fused cp failed or was latched
   mid-request the ggml cp was given last_hidden_ - stale by many frames.
   Codes 1-15 were computed for the wrong frame, the step embedding was
   garbage, the talker drifted and never emitted EOS. Reproduced on a
   quiet GPU: 2048 frames / 164 s without the fix, 107 frames / 8.5 s
   with it. Fix: HipTalker::fetch_hidden() D2H before the ggml cp runs
   whenever it was handed a device hidden.

2. QWEN3_TTS_HIP_CP_FAIL_AT=<n> debug env forces the n-th fused cp call
   to fail so the handoff is testable without contention.

3. Init/KV/embed sub-timers (kv init / clear / hidden fetch / embed /
   graphs). They showed the post-fallback 70-80 ms/frame was contention
   (ggml talker was equally slow), not one op. The per-frame
   clear_code_pred_kv_cache() (2 x n_layers synchronous memsets) was
   unnecessary regardless - every slot is written before read and
   inp_mask hides the rest - so the cache is zeroed once at allocation
   (needs_clear) and only n_used is reset per frame. Bit-identical
   output on the handoff and pure-ggml paths.

4. Backend: label reports HIP fused / HIP fused -> GGML (fallback
   mid-request) / GGML truthfully; it printed GGML for fused runs.

Measured with llama-server + embedding-server at 99% GPU: 4 blocks/CU
timed out on every request (~220 ms/frame); 1 block/CU had zero
timeouts, 14.7 ms/frame fused. The service now runs 1/CU (the queue's old
tts-engine.service). docs/talker_fusion_handoff.md has the write-up.

```
 docs/talker_fusion_handoff.md | 53 +++++++++++++++++++++++++++++++++++++++++++
 src/hip/talker_hip.h          |  5 +++++
 src/hip/talker_hip.hip        |  6 +++++
 src/tts_transformer.cpp       | 77 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 src/tts_transformer.h         |  8 +++++++
 5 files changed, 147 insertions(+), 2 deletions(-)
```

### `b1a1d5f` 2026-09-12 - docs: performance plan Phase 4 - proposed next steps, all marked TO BE DISCUSSED

Results log brought up to date (1.85x -> 0.12x); goal marked met. Eight
proposals with expected gain and gate for each, plus what is explicitly
not proposed. Nothing started.

```
 docs/performance_plan.md      | 83 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 docs/talker_fusion_handoff.md |  5 ++++
 2 files changed, 86 insertions(+), 2 deletions(-)
```

