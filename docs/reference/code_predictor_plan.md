# Code Predictor Speed-Up Plan

Working notes for cutting per-frame latency on a discrete AMD GPU. Written
2026-09-10 after a profiling session on an RX 7900 XTX (RADV, Vulkan backend,
0.6B F16). Companion to `performance_plan.md`, which holds the Strix Halo
baseline.

## Where we are

After the two dGPU vocoder fixes (`b11fd22`, `cebfbd3`) the engine runs at
**~17 ms/frame, RTF 0.27** on the XTX (8.4 s of audio in 2.25 s one-shot,
2.42 s streaming with `stream_batch_size: 8`).

| stage | per frame | notes |
|---|---|---|
| talker (28 layers) | 4.5 ms | bandwidth floor ~1.3 ms (1.2 GB F16 @ ~960 GB/s) |
| code predictor (14 steps) | **12 ms** | 48% of the frame — the target |
| vocoder | ~2 ms | all on Vulkan now |
| graph build + `sched_alloc` | ~2 ms | the 14 step graphs are shape-identical every frame |

The same stages measured on Strix Halo in June (`performance_plan.md`) give
the *same* talker/code-predictor times despite 4x less memory bandwidth.
This model is launch-latency-bound, not bandwidth-bound: lower-precision
weights (Q8/FP4) buy almost nothing on either machine.

## Anatomy of one code-predictor step

Measured with `GGML_VK_PERF_LOGGER=1` (GPU timestamps):

- **128 kernel launches** across 21 distinct op shapes
- **~760 us of GPU time** — and ~760 us of wall time. The GPU is "busy"
  for the whole step; the CPU round-trip (8 KB logits readback, top-k
  sampling, four tiny uploads) is essentially free.
- ~170 MB of weights streamed per step. At ~960 GB/s that is a **~180 us
  floor**. The remaining ~580 us is 128 x ~4.5 us of fixed per-dispatch
  cost (barrier, cache flush, dispatch latency).

So the expensive thing is not "trips to the CPU"; it is 128 separate
dispatches with a pipeline drain between each. `hipGraph`/`GGML_HIP_GRAPHS`
only removes CPU-side submission cost and would help maybe 5-10%. The lever
is **fewer dispatches** — one per step, or one per frame.

Two free findings from the same profile:

- The code predictor's down-projection is stored **F32** in the GGUF
  (`MUL_MAT_VEC f32 m=1024 k=3072`): 63 of the 170 MB per step read at
  double width. Converting that tensor to F16 is worth ~5%.
- The mask is rebuilt on the host and uploaded every step; it only depends
  on `n_past`.

The loop is `TTSTransformer::predict_codes_autoregressive` in
`src/tts_transformer.cpp`: a 2-token prefill graph, then 14x
{`build_code_pred_step_graph`, `ggml_backend_sched_alloc_graph`, 4x
`ggml_backend_tensor_set`, `ggml_backend_sched_graph_compute`,
`ggml_backend_tensor_get(logits)`, CPU top-k sample, `sched_reset`}.

The CoreML path (`predict_codes_autoregressive_coreml`, gated by
`QWEN3_TTS_USE_COREML`) is the existing seam for "alternate backend for the
code predictor stage only". A HIP implementation slots in the same way,
with the talker and vocoder staying on Vulkan; the 4 KB hidden state and
the 15 output codes cross via host memory.

## Phases

### Phase 0 — HIP build, measure the dispatch floor (hours)

Build in the `navi31-llama` toolbox (ROCm 10.1, hipcc; llama.cpp there was
built with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, see
`~/src/fedora_toolbox/Dockerfile.fedora44-therock-navi31`):

```bash
toolbox run --container navi31-llama bash -c '
  cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
        -DQWEN3_TTS_TIMING=ON -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build-hip -j'
```

Run on a spare port with `-V`, then again with `GGML_HIP_GRAPHS=ON`, and
compare `Steps (14)` ms/frame against Vulkan. The question it answers: is
ROCm's per-dispatch fixed cost lower than RADV's ~4.5 us? If it is ~2 us the
plain HIP build is already a ~1.5x win on this stage, and it confirms HIP as
the home for Phase 2.

Risk: the vendored ggml (`d4fcfe88`, April 2026) may not compile against a
ROCm 10.1 nightly.

#### Phase 0 results (2026-09-10, ROCm 10.1.0a20260822 in `navi31-llama`)

Same text (131 chars), same voice, seed 42, second request of each server
(warm), `llama-server` resident but the test instance fully in VRAM.
Harness: `scripts/bench/bench_vulkan.sh <label> <binary>` and
`scripts/bench/bench_hip.sh <label> <binary>` (the latter runs the binary
inside `navi31-llama` and kills the server before the vocoder runs);
`tts-engine` must be stopped first. Timing blocks and WAVs land in
`$BENCH_OUT` (default `/tmp/qwen3-tts-bench`).

| build | frame | talker | code pred | **Steps (14)** | steps compute | steps alloc |
|---|---|---|---|---|---|---|
| Vulkan (RADV) | 17.4 ms | 4.5 ms | 12.0 ms | **10.7 ms** | 9.1 ms | 1.3 ms |
| HIP | 18.9 ms | 4.8 ms | 13.4 ms | **12.2 ms** | 11.7 ms | 0.4 ms |
| HIP + `GGML_HIP_GRAPHS` | 19.2 ms | 4.8 ms | 13.4 ms | **12.2 ms** | 11.7 ms | 0.4 ms |

- **ROCm's per-dispatch cost is higher than RADV's, not lower.** Per step:
  836 us on HIP vs 650 us on Vulkan. Against the ~180 us bandwidth floor
  and 128 dispatches that is ~5.1 us vs ~3.7 us per dispatch. The plain HIP
  build is a ~10% loss on the frame, so HIP is not a drop-in win for this
  stage; the only reason to go there remains Phase 2 (grid-wide sync).
  `sched_alloc` is 3x cheaper on HIP (0.4 vs 1.3 ms/frame) but that is
  Phase 1 territory on either backend.
- **HIP graphs never engage on the step graphs.** `GGML_HIP_GRAPHS` is a
  *compile-time* ggml option (separate build, `build-hip-graphs/`), and
  ggml-cuda keys a captured graph on `cgraph->nodes[0]` and only captures
  after two consecutive computes with identical node properties. The loop
  rebuilds each step graph with a different `n_past`, so warmup resets every
  step (`GGML_LOG_DEBUG` shows 4 "warmup complete" in the whole run, none
  in the predictor). Re-test after Phase 1 step 1 makes the graphs
  persistent; until then the graphs number is meaningless.
- The vendored ggml compiles unchanged against the ROCm 10.1 nightly.
  The toolchain itself needs a workaround: the nightly's clang 23 ships no
  host `libclang_rt.builtins.a` and `hip-lang-config.cmake` forces
  `--rtlib=compiler-rt`, so every HIP link fails. Build with
  `-DCMAKE_HIP_FLAGS=-resource-dir=$HOME/.local/share/rocm-clangrt-overlay`
  (a mirror of the ROCm clang resource dir plus Fedora's compiler-rt 22
  builtins; see the README in that dir). Durable fix for the Dockerfile's
  final stage: `dnf install compiler-rt` and symlink its builtins into
  `/opt/rocm/lib/llvm/lib/clang/23/lib/x86_64-unknown-linux-gnu/`.
- **The vocoder must not run on HIP.** All its ops land on `ROCm0` (2
  splits) but decode takes ~430 ms per frame — 9.8 s for 1.8 s of audio,
  vs ~4 ms/frame on Vulkan — and a 158-frame decode hogged the GPU hard
  enough to stall the desktop. Not investigated (the plan keeps the
  vocoder on Vulkan); likely the naive `conv_transpose_1d` kernel in
  ggml-cuda. The HIP binaries are only usable for measuring the talker and
  code predictor.
- The HIP build binaries link against the container's glibc 2.43 and only
  run inside `toolbox run --container navi31-llama`.
- Sampled codes differ from Vulkan for the same seed (147 vs 158 frames),
  as expected from the arithmetic-order caveat under Constraints.

**grid.sync() microbench** (`scripts/bench_gridsync.hip`, same toolbox),
which is the number Phase 2's estimate hinges on:

| grid (256 threads/block) | cooperative launch | per `grid.sync()` |
|---|---|---|
| 48 blocks (1/CU) | 21 us | 0.78 us |
| 96 blocks (2/CU) | 15 us | 0.56 us |
| 192 blocks (4/CU) | 13 us | 0.61 us |
| 384 blocks (8/CU) | 23 us | 1.06 us |

Back-to-back empty kernel launches on one stream cost 4.7 us each, which
independently confirms the ~5 us/dispatch derived from the ggml numbers
above. So a grid-wide barrier is **5-8x cheaper than a dispatch** on this
card, and the plan's "~5 us per sync" assumption is conservative by that
factor: ~30 syncs/step is ~20-30 us, not ~150 us. The Phase 2 step is then
bounded by weight streaming (~180-250 us), giving **~250-300 us/step,
~4 ms/frame** for the code predictor rather than the 5-6 ms estimated
above. Caveat: the bench syncs with no real work between barriers; with
matvec phases the barrier also waits for the slowest block, so the
per-sync number is a floor, not a typical.

### Phase 1 — ggml-level cleanups (about a day, ~15%, backend-agnostic)

1. Build and allocate the prefill graph and the 14 step graphs once, keep
   them across frames, and only `tensor_set` inputs per step.
2. Precompute the 16 masks (one per `n_past`) at init.
3. Convert the F32 down-projection to F16 in the GGUF (or cast at load).

#### Phase 1 results (2026-09-10) — done, 17.4 -> 15.1 ms/frame on Vulkan

| build | frame | talker | code pred | **Steps (14)** | steps compute | build+alloc |
|---|---|---|---|---|---|---|
| Vulkan before | 17.4 ms | 4.5 ms | 12.0 ms | 10.7 ms | 9.1 ms | 1.5 ms |
| **Vulkan after** | **15.1 ms** | 4.1 ms | 10.1 ms | **8.9 ms** | 8.9 ms | 0 |
| HIP before | 18.9 ms | 4.8 ms | 13.4 ms | 12.2 ms | 11.7 ms | 0.6 ms |
| HIP after | 15.2 ms | 3.8 ms | 10.8 ms | 9.7 ms | 9.4 ms | 0 |
| HIP + graphs after | 15.8 ms | 3.8 ms | 10.9 ms | 9.8 ms | 9.5 ms | 0 |

RTF 0.27 -> 0.24 on Vulkan. What was actually done, and what it taught:

- **Persistent graphs** (`init_code_pred_graphs`, `code_pred_graph` in
  `tts_transformer.h`): the prefill graph and the 14 step graphs are built
  once on the first frame, each with its own meta context and
  `ggml_gallocr`, and run with `ggml_backend_graph_compute` on the device
  backend directly — no scheduler. All input tensors of all 15 graphs live
  in one dedicated buffer (like the KV cache), so the constant ones
  (positions, and the mask for `n_past = step + 1`) are filled once at
  init; that is item 2 for free, and it avoids `ggml_gallocr` reusing an
  input's memory for an intermediate. Per step the host now does one
  4-byte `tensor_set`, one compute, one 8 KB `tensor_get`. Every node is
  checked with `ggml_backend_supports_op` at init; if anything is
  unsupported (or `QWEN3_TTS_CODE_PRED_SCHED=1` is set, for A/B testing)
  the old scheduler path is used. The graphs are dropped whenever the KV
  cache is re-initialised.
- **Verification:** with greedy decoding the persistent and scheduler paths
  are bit-identical (`cmp` on the WAV) on Vulkan over 2048 frames, on HIP
  over 40 frames (`QWEN3_TTS_DUMP_CODES`), and with the card deliberately
  filled to 24.4 GB by a dummy allocator. One earlier pair, run while the
  9B llama-server was also resident, did *not* match and could not be
  reproduced afterwards; the likely mechanism is an allocation failure
  under true exhaustion knocking some stage to CPU, not the graphs.
- **Item 3 was misdiagnosed.** `ffn_down` is F16 in the GGUF; the graphs
  *up-cast* it to F32 with `ggml_cast` on every run (5 per predictor step,
  28 per talker frame — a 12 MB write + read per layer plus a dispatch).
  Upstream added that cast to "mirror Python exactly" and it turns out to
  matter: without it the HIP build runs away (every text generates to
  `max_tokens`), because ggml-cuda's batched F16 GEMM accumulates in F16
  on RDNA3 (`CUBLAS_COMPUTE_16F` unless CDNA/RDNA4), and `silu(gate)*up`
  is where the big values are. The fix is `ggml_mul_mat_set_prec(...,
  GGML_PREC_F32)` on the `ffn_down` matmuls: both backends then take the
  F32-accumulate GEMM for batched (prefill) cases, while the decode
  matvecs — already F32-accumulate — are untouched. Same guarantee, no
  copy. Vulkan was fine either way (its F16xF32 matvec reads F32
  directly and the 2-token prefill goes through the matvec path).
- **HIP graphs are closed as a lever.** With persistent graphs the capture
  finally engages (20 captures per run) and changes nothing: 9.8 vs 9.7
  ms/frame. The per-dispatch cost is on the GPU side, as the Phase 0
  analysis said; only fewer dispatches (Phase 2) moves it.
- Step compute barely moved on Vulkan (9.1 -> 8.9 ms): dropping 5 casts
  per step is offset by the F16 matvec path being no faster than the F32
  one. The gain is all build+alloc. The remaining 0.9 ms/frame of "Data
  I/O" is 14 synchronous 8 KB logits readbacks (~65 us each on Vulkan) —
  that is what on-GPU sampling in Phase 2 removes.
- **The HIP vocoder stalls the desktop.** A runaway 2048-frame request on
  HIP held the GPU for minutes and Wayland got no time. Safeguards added:
  the server takes `--max-tokens` (default 2048, like the CLI), and
  `scripts/bench/bench_hip.sh` runs with `--max-tokens 400`, kills the
  server if a request passes 300 frames or 120 s, and never lets the HIP
  vocoder run on the long text. Both bench scripts refuse to start with
  less than 5 GB of VRAM free and print VRAM before/after.
- Both `build-hip*/` variants and `build-dev/` (Vulkan) are current; the
  live service binary in `build/` has not been rebuilt — that is the
  seed re-audition gate.

### Phase 2 — fused cooperative HIP kernel (1-2 weeks)

#### Step 0 result (2026-09-10): the process goes all-HIP, no mixed backend

The "HIP vocoder is 100x slower" finding from Phase 0 was one kernel:
`rocprofv3 --kernel-trace --stats` on a 13-frame decode put 95.7% of
GPU time in `conv_transpose_1d_kernel` (6 calls, up to 2.5 s each). The
vendored ggml-cuda kernel iterated over *every* input position per
output element and `continue`d outside the kernel window — O(L_in)
instead of O(K) per output. Fixed in `ggml/src/ggml-cuda/conv-transpose-1d.cu`
by computing the valid input range; same summation order, so the
output is bit-identical. Decode 3578 -> 19 ms on that test. (Worth
sending upstream to ggml.)

With that, a full request on HIP: 139 frames at 15.1 ms/frame, vocoder
167 ms (1.2 ms/frame vs 3.9 on Vulkan), **RTF 0.205 vs 0.239 on Vulkan**.
So Phase 2 targets an all-HIP process: talker, code predictor and
vocoder on ROCm0, the fused kernel slotting in behind the CoreML seam,
and the service running inside the toolbox the way `llama-server.service`
already does. No Vulkan+HIP plumbing.

One cooperative-groups kernel that runs an entire step with `grid.sync()`
between phases instead of 128 dispatches, and loops all 14 steps inside a
single launch:

- shapes: 5 layers, hidden 1024, 16 query heads / 8 KV heads x 128,
  intermediate 3072, KV cache <= 16 positions, 15 output heads x vocab 2048
- per step: rmsnorm -> qkv matvec -> rope -> attention over <= 16 keys ->
  o-proj + residual -> rmsnorm -> gate/up matvec -> silu*mul -> down +
  residual, x5; head matvec; temperature/top-k/softmax sampling on-GPU
  with a seeded counter-based RNG
- matvecs partitioned across workgroups by output rows; the small norms
  and softmax done redundantly per block

Estimate: ~30 grid syncs per step at ~5 us plus ~250 us of weight
streaming -> **~400 us/step, 12 ms -> ~5-6 ms per frame**, and the 2 ms of
graph overhead disappears. Frame total roughly 17 -> 9 ms. The same
treatment for the talker (28 layers, 1.3 ms floor vs 4.5 ms today) is the
follow-on.

This needs HIP specifically: Vulkan compute has no grid-wide sync, so a
GLSL version would still be ~10 dispatches per step.

#### Step 1 result (2026-09-12): fused kernel runs, correct, and cannot hang the box

Implemented in `src/hip/code_pred_hip.{h,hip}` (CMake-gated on
`GGML_HIP`), tested by `tests/test_hip_code_pred` against a reference
frame dumped from the ggml engine
(`QWEN3_TTS_DUMP_CODE_PRED=<path> ./qwen3-tts-cli ...` writes the first
frame's hidden state, cb0 token, all 15 heads' logits and sampled codes).

**The hang was residency contention, not a sync bug.** The barrier
structure is uniform (every block hits every `grid.sync()` the same
number of times), but `hipOccupancyMaxActiveBlocksPerMultiprocessor`
only accounts for the *calling* process. With the compositor and
llama-server resident, the cooperative grid cannot fully launch; the
scheduled blocks spin in `grid.sync()` waiting for blocks that never get
a CU, with the display blocked behind them.

Safeguards now in the kernel:

- **Deadline-aware grid barrier** replaces `cg::grid.sync()`:
  sense-reversing with a 50M-spin cap (~0.5 s,
  `QWEN3_TTS_HIP_BARRIER_SPINS`); on timeout a flag is set, every later
  barrier short-circuits, the grid unwinds, and the host reports
  "grid barrier timed out" instead of the workstation dying.
- `prop.cooperativeLaunch` checked at init; grid capped by
  `QWEN3_TTS_HIP_BLOCKS_PER_CU` (default 4).

Measured (GPU otherwise idle, greedy reference frame, 0.6B F16):

| driver | frame | notes |
|---|---|---|
| per-phase launches (16 pos x 8 phases) | 12.8 ms | correctness harness |
| **fused, 1 block/CU (48 blocks)** | **5.1 ms best / 5.5 mean** | sweet spot |
| fused, 2/CU (96 blocks) | 5.35 ms | barrier overhead cancels the gain |
| fused, 4/CU (192 blocks) | 5.5 ms | idem |

15/15 argmax agreement with the ggml reference on both paths, worst
relative logit error 5.2% (F16 numerics). Ran the fused kernel at 4/CU
*concurrently with a Vulkan generation* — completed clean, no abort: the
scenario that previously hung the workstation now survives, and if it
ever stalls >0.5 s per barrier it aborts instead.

Against the live numbers: the ggml code predictor is 10.1 ms/frame, so
the fused kernel is ~2x on this stage; integrated, the frame projects to
**~10 ms (RTF ~0.16)** with the ROCm vocoder at 1.2 ms/frame.

**Gap before integration:** the kernel currently takes all 15 codes as
input (the test feeds the reference codes). Real generation needs the
code sampled at position p to feed position p+1's embedding — so the
on-GPU sampling (greedy argmax first, then temperature/top-k with a
seeded RNG) from the plan is a prerequisite for wiring this into
`predict_codes_autoregressive` via the CoreML seam. Also: the 123 KB
logits readback in `run()` goes to pageable host memory and is likely
inflating the 5.1 ms — pin it.

#### Step 2 result (2026-09-12): on-device sampling + pipeline integration

Sampling is now in the kernel (`PH_SAMPLE`): greedy block-argmax, or
temperature + top-k (iterative max-removal) + softmax + PCG32 multinomial
seeded per `(frame_seed, pos)`. Greedy reproduces the reference codes
15/15; sampled codes are valid and fused == per-phase 15/15. Sampled
frame costs ~7.3 ms vs 5.2 greedy — the 50-pass top-k is the delta and
wants a better selection (bitonic or warp-tournament) as a follow-up.

Integrated via the CoreML seam: `QWEN3_TTS_USE_HIP_CODE_PRED=1` routes
`predict_codes_autoregressive` through `HipCodePredictor` (weights
copied to device once at init from the loaded ggml tensors; 0.6B shape
gated, falls back to the ggml path with a warning otherwise). E2E on the
XTX (0.6B F16, all-HIP process incl. the fixed vocoder):

| path | generate | total | audio | RTF |
|---|---|---|---|---|
| ggml code predictor (greedy) | 1965 ms | 2093 ms | 8.94 s | 0.234 |
| **fused (greedy)** | **1499 ms** | **1626 ms** | 8.78 s | **0.185** |
| **fused (temp 0.9, top-k 50)** | 2834 ms | 3081 ms | 15.98 s | **0.193** |

Greedy HIP vs greedy ggml WAVs are *not* bit-identical — expected: the
talker now runs on the HIP backend too, so hidden states differ from the
Vulkan-dumped reference and greedy codes diverge at some frame. Both
outputs are valid speech of matching duration. The unit-level 15/15
greedy agreement against the dumped frame remains the parity anchor.

**Top-k optimization (2026-09-12):** replaced the 50-pass iterative
top-k with an in-place bitonic sort of the vocab in `xs` (no extra static
shared — a separate sort buffer dropped co-resident blocks/CU and made the
fused matvec phases slower). Sampled fused frame **7.3 -> 5.9 ms**
(unit, 1/CU); fused == per-phase agreement still 15/15. The logits
readback was also made optional (pipeline passes nullptr) but measured
neutral.

**Measurement caveat — e2e RTF is not a clean stage comparison.** The
code predictor's output changes the sampled codes, which changes clip
length, which changes RTF (fixed prefill/load amortize over more or
fewer frames). Random-seed e2e runs spread 0.23-0.25; the early
0.193 was a long-clip outlier and does not reproduce. The trustworthy
signal is the per-frame kernel time:

| code predictor | per-frame |
|---|---|
| ggml (Phase 1) | ~10.1 ms |
| fused, iterative top-k | ~7.3 ms |
| **fused, bitonic top-k** | **~5.9 ms** |

Remaining: the one seed re-audition before any prod move; the talker
fusion is the next big lever (4.1 ms vs ~1.3 ms floor).

**Seed re-audition (2026-09-12): done.** Four ICL-clone candidates
through the fused sampled path across two reference voices (the repo
sample and the user's `nyx_reference.wav`); **seed 3** is the keeper on
both (seed 4 also stood out). The nyx voice audited cleaner with less
seed-to-seed variance. Pin seed 3 for the fused path's reference voice.

### Phase 3 — talker fusion (2026-09-12)

The talker is the same phase machinery as the code predictor with three
differences: 28 layers instead of 5, a growing KV cache instead of a
16-position one, and a single `codec_head` (vocab 3072) whose cb0 is
sampled with the `generate()` rules (suppress `[vocab-1024, vocab)`
except EOS, HuggingFace repetition penalty over previously emitted cb0,
then greedy / temperature+top-k). Implemented in
`src/hip/talker_hip.{h,hip}` (CMake-gated on `GGML_HIP`), tested by
`tests/test_hip_talker` against a reference step dumped with
`QWEN3_TTS_DUMP_TALKER=<path> ./qwen3-tts-cli ...` (step embedding,
hidden, logits, and the full F16 KV cache up to the step position).

Key design points:

- **External KV cache.** The fused kernel reads and writes the *same*
  F16 `k_cache`/`v_cache` the ggml prefill filled, via the tensors'
  device pointers (`tensor->data`); position stride `head_dim*n_kv_head`
  matches the contiguous `[head_dim, n_kv_head, n_ctx]` layout. The
  prefill stays on the ggml backend; only the decode steps are fused.
- **Weights read in place.** The talker weights are already resident on
  the compute backend, so `init_hip_talker` passes the ggml tensors'
  device pointers straight to the kernel instead of copying. The first
  integration copied 1.2 GB to a private allocation and that one-time
  ~209 ms upload ate the whole per-frame win on a 190-frame request;
  reading in place drops the unaccounted overhead to ~22 ms.
- **On-device cb0 sampling** mirrors `generate()` exactly. The
  repetition-penalty set is a device-side byte array the kernel updates
  as it samples (EOS not marked); frame 0's cb0 still comes from the
  prefill logits on the host and seeds the set via `mark_seen`.
- Integrated behind `QWEN3_TTS_USE_HIP_TALKER=1`: the `generate()` loop's
  `forward_step` + host sampling block is replaced by one fused
  `HipTalker::run` that returns the next frame's cb0 in `pending_cb0`.
  Falls back to the ggml path on init failure.

Measured (RX 7900 XTX, 0.6B F16, greedy, GPU otherwise idle):

| path | talker per-frame | notes |
|---|---|---|
| ggml `forward_step` | 3.9 ms | compute 3.6 + build/alloc/IO |
| **fused talker** | **2.4 ms** | in-context (incl. H2D step_embd + D2H hidden + cb0) |
| fused, unit (n_past=10) | 1.95 ms best / 1.96 mean | 4 blocks/CU sweet spot |

Parity vs the dumped reference: hidden max rel err 0.018%, logits max rel
err 0.05%, cb0 argmax matches; per-phase and fused agree. Greedy E2E
produces audio identical in length and RMS to the ggml path.

End-to-end on a ~190-frame request (long text, greedy), fused talker +
fused code predictor vs ggml talker + fused code predictor:
**1962 ms vs 2428 ms total generate (~19% faster)**, 10.5 vs 12.7
ms/frame. The per-frame host tail (code-predictor→talker: embed lookups
+ frame assembly) is 0.47–0.59 ms and is *not* the bottleneck; the
earlier apparent regression was entirely the one-time weight upload, now
gone.

Remaining levers: the talker is at 2.4 ms against a ~1.3 ms bandwidth
floor — the gap is the ~170 grid barriers/step and the attention phase
using only 16 of the grid's blocks. Device-resident chaining (feed the
fused code predictor's hidden from the talker's device buffer to skip the
4 KB round-trip, and ultimately one fused talker+cp kernel with a single
sync per frame) is the next step.

## Constraints

- The engine stays on this PC and keeps serving the `:8080` OpenAI-style
  API; other clients (Open-LLM-Vtuber) hit it directly.
- Any change to arithmetic order — Phases 1 and 2 included — changes the
  sampled codes for a given seed, which re-rolls the cloned voice's
  character. Plan for **one** re-audition of seeds afterwards and pin new
  ones; do not chase bit-exact parity.
- Benchmarks on a spare port are only valid with the live `tts-engine`
  stopped: with llama-server resident the card is otherwise full, the test
  instance spills into GTT, and everything measures 2x slow. Check with
  `grep drm-resident /proc/<pid>/fdinfo/*`.

## Profiling recipes (what actually worked)

- **Where does each op run?** `GGML_SCHED_DEBUG=1` prints one `## SPLIT #n:
  <backend>` line per split; `=2` adds every node. This is how the vocoder
  was caught running on CPU while the load log said "backend: Vulkan0" —
  the load message reports the *compute* backend, not where the weights
  landed. A split labelled `CPU # 0 inputs` covering a whole graph means
  the weights are in a CPU buffer.
- **Per-op GPU time:** `GGML_VK_PERF_LOGGER=1` prints a per-graph block
  with `<op> <shape>: N x t us = total`. It flushes a block when the *next*
  graph runs, so always make two requests, and the last graph of a run is
  never printed. Kernel count per graph = the sum of the `N x` multipliers.
- **Stage wall times:** start the server with `-V`; each request then ends
  with `Code generation / Vocoder decode / Total` in ms plus RTF. The
  `QWEN3_TTS_TIMING=ON` build (default in `build-vulkan.sh`) adds the
  per-frame talker / code-predictor / prefill breakdown.
- **Is it really in VRAM?** `cat /proc/<pid>/fdinfo/* | grep drm-resident`
  sums to VRAM vs GTT residency for the process. The healthy load on the
  XTX is ~1.7-2.4 GB VRAM and ~0.8-1.0 GB GTT (staging); if GTT climbs past
  that after a request, compute buffers spilled and timings are 2x off.
- **Output identity:** same text + same seed is bit-exact run to run on a
  given build, so `cmp` on the WAVs is a valid regression test for "did
  the codes change". When they differ only by vocoder numerics, SNR is
  ~40 dB (GPU vs CPU vocoder measured 41 dB); when the sampled codes
  differ the length changes.
- `pkill -f 'qwen3-tts-server.*-p 8081'` also matches the shell that
  launched it; use `pkill -f '^build/qwen3-tts-server'`.

## Other observations

- Both dGPU bugs came from the fork being developed on Strix Halo (an
  iGPU): `load_tensor_data_from_file(..., GGML_BACKEND_DEVICE_TYPE_IGPU)`
  fails on a discrete card and silently fell to CPU (`b11fd22`), and
  ggml-vulkan's `CONV_TRANSPOSE_1D` only accepts F32 weights so the F16
  ones in the tokenizer GGUF bounced six convs to CPU (`cebfbd3`). When
  adding ops, check `ggml_backend_vk_supports_op` for type restrictions —
  several Vulkan kernels are F32-only.
- Streaming (`stream_format: "audio"`, `stream_batch_size: 8`) runs the
  vocoder once per batch, so vocoder cost is amplified in streaming mode:
  before the conv fix streaming took 4.1 s vs 2.6 s one-shot; after, 2.42
  vs 2.25 s. First audio arrives in under a second.
- `-j` (compute threads) made no measurable difference once nothing ran
  on the CPU.
- Voices registered via `POST /v1/audio/voices` without `ref_text` use the
  speaker embedding only (`[icl] ref_frames=0`). Supplying `ref_text`
  enables ICL cloning, which is a quality lever untested so far — and
  would re-roll the voice like everything else.

- **Vocoder scratch grows with the longest utterance ever decoded and is
  never released.** ggml's scheduler keeps its compute buffers at the
  high-water mark; a one-shot decode costs ~4.7 MB of scratch per frame,
  so one runaway 2048-frame request (seen on the live server 2026-09-10,
  temp 0.9, random seed) left `tts-engine` holding 16.8 GB (4.8 VRAM +
  12 GTT) until restart. Fixed by routing `decode()` through the
  streaming decoder in 64-frame chunks (`decode_chunk_frames_`,
  `QWEN3_TTS_DECODE_CHUNK=0` restores one-shot): a 496-frame request now
  ends at 3.6 GB instead of 5.9, and the vocoder share no longer scales
  with length. The service unit also carries `--max-tokens 600`.
  Chunked vs one-shot is ~51 dB SNR on speech on the XTX, not bit-exact:
  `tests/test_streaming_parity` already fails its 1e-4 tolerance here
  with the untouched one-shot path (max 0.10 / rms 2.8e-3 on random
  codes) because Vulkan's coopmat matmul output depends on batch size;
  the bit-exact claim in `streaming_design.md` was established on Strix
  Halo. Chunked `decode()` vs `stream_decode` at the same chunk is exact.

## Things already ruled out

- **Bumping ggml to v0.23.0** (branch `ggml-bump`): vocoder 232 -> 141 ms,
  but talker and code predictor +20-25% per frame (net +15% per second of
  audio), and it re-rolls the voice. Re-test against a future release; the
  gate is per-second-of-audio cost *and* unchanged output for seed 42.
- **Code predictor on CPU** (5800X3D, 8 threads): 3.5 ms/step, ~4x slower.
- **Q8/FP4 weights or FP4 training**: bandwidth is ~15% of the frame;
  Strix Halo numbers with 4x less bandwidth are identical. Not the
  bottleneck on either machine.
- **Moving upstream**: see "Upstream status" below. Neither upstream is a
  better base than this fork.

## Upstream status and how to submit later

Lineage: `predict-woo/qwen3-tts.cpp` (original ggml port, remote `upstream`)
-> `khimaros/qwen3-tts.cpp` (server, streaming, flash-attn, 1.7B; remote
`origin`) -> `shawnshekari/qwen3-tts.cpp` (this fork, remote `shawnshekari`).

As of 2026-09-10:

- **khimaros** last pushed 2026-06-16 (`0c8b2ba`). Our PRs #3 (libav
  compat) and #4 (Vulkan RPATH / httplib flags) have been open since
  2026-06-19 with no response; #2 (CORS, another contributor) likewise.
  Treat as dormant. Everything after `0c8b2ba` in this repo's history is
  ours, including the June Strix Halo baseline docs.
- **predict-woo** last pushed 2026-07-18; merges small build PRs but closed
  the one performance PR (#28). No server, no streaming (open issues #23,
  #25). Issue **#20** — "Vocoder decode runs on CPU despite Vulkan backend:
  backend/buffer mismatch in load_tensor_data_from_file" — is exactly
  `b11fd22`, open since 2026-03-27. Their `gguf_loader.cpp` moved the
  logic into `init_tensor_loader_backend()` (same bug at the
  `preferred -> CPU` fallback) and their decoder gained CUDA chunked
  decode, so our commits do not cherry-pick cleanly onto their tree.

Not submitted anywhere yet: `b11fd22` (IGPU -> GPU fallback) and
`cebfbd3` (F32 cast for transposed convs).

If submitting later, the target with a live maintainer and a matching open
issue is predict-woo. This is a cherry-pick of two small fixes onto their
tree for the PR only — the working base stays here, do not rebase this
fork onto predict-woo (their decoder rewrite conflicts with the server and
streaming work).

```bash
git fetch upstream
git checkout -b pr/dgpu-vocoder upstream/main
git cherry-pick cebfbd3          # applies with line-offset only
git cherry-pick b11fd22          # will conflict: re-apply the IGPU->GPU
                                 # retry inside init_tensor_loader_backend()
cmake -S . -B build-pr -DGGML_VULKAN=ON -DQWEN3_TTS_TIMING=ON && cmake --build build-pr -j
```

Then re-measure on *their* build (`-V`, vocoder decode ms before/after,
`GGML_SCHED_DEBUG=1` split count) so the PR's numbers are for their code.
In the PR body: reference #20, quote the before/after, and state in one
line that the fix was found with AI-assisted profiling. Keep it to those
two commits. If it sits unmerged, it costs nothing; the fork stays the
working copy either way.
