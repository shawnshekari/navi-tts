# navi-tts design

Status: draft, 2026-09-17. Decisions marked **DECIDED** are settled; **OPEN** ones
need a call before the code that depends on them is written.

## 1. What this is

A text-to-speech inference engine for two AMD machines, written from scratch:

| | Workstation | Mini PC |
|---|---|---|
| GPU | Radeon RX 7900 XTX | Strix Halo iGPU |
| Arch | gfx1100 (RDNA3), 96 CU, wave32 | gfx1151 (RDNA3.5), 40 CU, wave32 |
| Memory | 24 GB GDDR6, ~960 GB/s | unified LPDDR5X, ~256 GB/s |

**DECIDED.** HIP only. No CPU path, no Vulkan, no CUDA, no ggml or other tensor
framework. One binary, one systemd unit, one model file. Anything not on the
list above is out of scope by design, and the binary refuses to start on it.

**DECIDED.** The RX 7900 XTX is built and cut over to production first (M0-M2);
Strix Halo starts only after that is up (M3). Until M3 the build targets gfx1100
alone - nothing in M0-M2 is designed *for* gfx1151, only *not against* it
(runtime-derived grid/occupancy parameters, no compile-time CU counts).

**DECIDED.** Qwen3-TTS architecture (12 Hz, 0.6B first). Fine-tunes of the same
architecture are just different weights and need no engine change; a different
model family is a second `model/` module (section 5), not a rewrite of the runtime.

### Non-goals

- Portability. The point is to be fast on this hardware, not to run elsewhere.
- Bit-exact parity with the old fork or with PyTorch. Same text + same seed must
  be bit-exact **run to run on a given build** (that is the regression test);
  across arithmetic changes, plan one seed re-audition and pin new seeds.
- Batching across requests. Batch one, low latency, conversational.

## 2. What we already know

`HISTORY.md` has the full record; the facts that shape this design:

- **The frame is launch-bound, not bandwidth-bound.** A 12 Hz frame is one talker
  step (28 layers) plus 15 sequential code-predictor steps (5 layers each) plus a
  slice of vocoder. Weight streaming is ~15% of that on the XTX; the rest was
  dispatch latency (~5 us per launch on ROCm, 128 launches per cp step under
  ggml) and host round-trips. Quantized weights were measured and ruled out as a
  lever *for this model size*. Corollary for Strix Halo: its 4x lower bandwidth
  costs far less than 4x; CU count and clocks matter more. Measure before assuming.
- **A grid-wide barrier costs 0.6-0.8 us; a launch costs ~5 us.** One persistent
  cooperative kernel per frame with `grid.sync()` between phases is the design
  center. It took the fork from ~19 ms/frame to 8.5 ms/frame (RTF 0.12).
- **Occupancy that benchmarks well fails under contention.** 4 blocks/CU timed out
  on every request once llama-server touched the card; 1 block/CU had zero
  timeouts and was faster in production. Residency, not throughput, decides.
- **A fallback path is a correctness hazard.** The runaway-audio incidents traced
  to stale hidden state when the fused path fell back to ggml mid-frame. There is
  no fallback here: a barrier timeout fails the *request*, cleanly, and the next
  request starts fresh. The kernel must never be able to hang the box.
- **Bound every request.** `max_audio_tokens` per request (default 600 = 48 s),
  vocoder decoded in fixed-size chunks so scratch never scales with length.
- **Vocoder convs need a real kernel.** The naive transposed conv ran at 430 ms
  per frame on HIP under ggml; the O(K) rewrite made it ~1.2 ms. Write it right
  the first time.
- **First request after start is slow** (RTF ~0.4) until kernels and buffers are
  initialised. Warm up at start, not on the first real utterance.

## 3. Contract

Everything above the engine keeps working unchanged when the binary is swapped:
TTS-Player's queue, the Claude Code / opencode cues, SkyrimNet, and any direct
client (SillyTavern, Open-LLM-VTuber).

### 3.1 HTTP, `127.0.0.1:8080`

**OpenAI dialect** (what the queue and direct clients speak today):

| Endpoint | |
|---|---|
| `GET /health` | engine up, model loaded, device name |
| `GET /v1/models` | the loaded model |
| `GET /v1/audio/languages` | supported language ids |
| `GET /v1/audio/voices` | built-in and cloned voices |
| `POST /v1/audio/voices` | clone from reference audio (multipart: `audio`, `name`, optional `ref_text`) |
| `DELETE /v1/audio/voices/{id}` | |
| `POST /v1/audio/speech` | `{input, voice, seed?, max_audio_tokens?, language?, stream_format?, stream_batch_size?}` → `audio/wav` (24 kHz mono s16) or a streamed PCM/SSE body |

**XTTS dialect** (what SkyrimNet's XTTSInterface speaks; replaces the
`skyrimnet-xtts-shim` process):

| Endpoint | |
|---|---|
| `POST /create_and_store_latents` | multipart `wav_file` + `speaker_name` → clone, stored under the *name* |
| `POST /tts_to_audio/` | `{text, speaker_wav (= name), language}` → `audio/wav` |
| `GET /speakers`, `GET /languages` | whatever else SkyrimNet probes; taken from the shim's `do_GET` |

Both dialects resolve to the same voice store and the same synth path; the XTTS
handlers are thin adapters. Reference WAVs with streaming headers (`0xFFFFFFFF`
RIFF/data sizes) are accepted.

**Operations:** `GET /metrics` in Prometheus text format (llama.cpp naming so the
existing Grafana panels carry over: requests, processing, prompt/predicted
tokens and seconds, audio seconds, stage seconds for tokenize/prefill/generate/
vocode, frame-cap hits, errors by stage). The engine stays loopback-only; the
node_exporter textfile collector copies `/metrics` for Prometheus on the mini PC.

### 3.2 Voice store — DECIDED

Cloned voices persist: name, speaker embedding, optional ref text and the sample,
in a directory the service owns. Reloaded at start. This retires
`tts-register-voices.service` (a one-time import instead of a boot dependency)
and the shim's cache/re-clone machinery. `voice_1`/`voice_2` and the ~50 Skyrim
voice types live in the same map.

### 3.3 Process

One static-ish binary (`navi-tts`), flags for model path, voices dir, port,
max tokens, warm-up on/off. `navi-tts.service` replaces `tts-engine.service`
(same `Before=llama-server` VRAM-ordering trick). `-V` per-request timing on
stderr as before.

## 4. Model

Qwen3-TTS-12Hz-0.6B. The parameter tables, tensor names, special codec tokens
and language ids get their own `docs/model.md`, written at M0 straight from the
HF `config.json` / safetensors index (the fork's tensor-mapping doc was
upstream's and is not carried over; the facts are re-derived, not copied).

| Component | Shape | Role | Runs |
|---|---|---|---|
| Text tokenizer | Qwen3 BPE | text → tokens | CPU, own implementation |
| Speaker encoder | ECAPA-TDNN → 1024-d | reference audio → speaker embedding | GPU, once per clone |
| Talker | 28-layer Qwen3 transformer, 1024 hidden, 16 heads / 8 KV, M-RoPE | one step per frame → hidden + codebook-0 logits | GPU, in the frame kernel |
| Code predictor | 5-layer transformer, 16 codebooks × 2048, delay pattern | 15 sequential steps per frame → codebooks 1-15 | GPU, in the frame kernel |
| Tokenizer decoder (vocoder) | 8-layer transformer + upsampling convnet (8·5·4·3) | codes → 24 kHz PCM | GPU, per batch of frames |
| Tokenizer encoder | Conv1D + 8-layer transformer + RVQ | audio → codes, for ICL cloning | **OPEN** (section 9) |

Prefill (text tokens + speaker embedding through the talker with a KV cache) is
~12 ms and off the per-frame path; it can use a plain batched GEMM.

## 5. Architecture

```
navi-tts/
├── runtime/            survives a model swap
│   ├── device          HIP init, arch detect (gfx1100 | gfx1151 or refuse), CU count → grid
│   ├── weights         .navi blob: mmap, table of tensors, upload to device
│   ├── server          HTTP (both dialects), streaming, /metrics
│   ├── voices          persistent voice store
│   ├── sampling        seeded RNG, temperature / top-k, on-device
│   ├── audio           WAV in (header repair) / out, PCM streaming
│   └── bench           the harness: fixed text + seed → ms/frame, TTFA, RTF, cmp
├── model/qwen3tts/     per architecture
│   ├── tokenizer       BPE
│   ├── speaker         ECAPA-TDNN
│   ├── prefill         batched talker pass
│   ├── frame.hip       the persistent cooperative frame kernel (talker step → cb0 sample → 15 cp steps)
│   ├── vocoder         transformer + convs, chunked, state threaded for streaming
│   └── graph           orchestrates one request: prefill → frames → vocoder batches
├── tools/convert.py    HF safetensors → .navi (offline; the only Python)
└── docs/
```

### 5.1 The frame kernel — DECIDED

Starts from `reference/hip/` - the fork-era `talker_hip.hip`, `code_pred_hip.hip`
and `frame_fused.hip` are the author's own, have no ggml dependency in the
kernel code, and already reached 8.5 ms/frame. What changes is the host side:
their `.h` interfaces assume the fork's types and the ggml fallback handoff,
both of which go away.

One cooperative launch per frame, persistent for the request. Phases separated
by `grid.sync()`: talker layer × 28 → norm → cb0 logits → sample on device →
step embedding → cp layer × 5, × 15 steps, sampling each → write 16 codes.
Hidden state and step embeddings never leave the device between phases.

- **1 block per CU**, grid = CU count from `hipDeviceProp` (96 / 40), not a
  compile-time constant. Block size and per-phase work split derive from it.
- Barrier spin cap set from measured p99, not a guess (the fork ran 0.5 s; data
  suggested ~50 ms). A timed-out barrier sets an abort flag; every block exits;
  the request fails with a clear error; nothing is latched for the next request.
- Weights resident in device memory for the process lifetime. KV caches
  allocated once at the max frame budget and reused (zeroed once, `n_used`
  reset per request — the per-frame memset was found unnecessary).
- Bit-exact run to run for a given seed. Sampling is on-device with a
  counter-based RNG so this holds.

### 5.2 Vocoder

Runs once per batch of `stream_batch_size` frames (default 8) with decoder state
threaded across batches; one-shot requests use the same path with a fixed chunk
so scratch is bounded. Transposed convs with the O(K)-per-output formulation.
Gate: chunked output equals one-shot output within 1e-5 PCM float, no audible
seam. (Upstream's streaming design established this invariant; the fork found
it holds exactly for chunked-vs-streamed at the same chunk size and to ~51 dB
SNR against a one-shot decode whose matmul output depends on batch size.)

### 5.3 Weight format — DECIDED

`.navi`: magic, version, architecture id, then a table of `{name, dtype, shape,
byte offset}` and page-aligned tensor data. `mmap` at start, one upload per
tensor. F16 first. The converter is where a fine-tune or a quantised variant
plugs in later; the engine reads whatever dtype the table says.

### 5.4 Streaming

`stream_format: "audio"` returns PCM as it is produced; TTFA target under 300 ms
for a typical sentence (prefill + first batch + one vocoder batch). The queue in
TTS-Player renders whole chunks today; it can move to streaming later without an
engine change.

## 6. Correctness strategy

1. **Reference dumps** from the PyTorch model for a fixed text/voice/seed: token
   ids, speaker embedding, first-step hidden and logits, first frame's 16 codes
   (greedy), the full KV after prefill, and the vocoder's PCM for a fixed code
   sequence. Stored under `tests/reference/`, produced by our own
   `tools/dump_reference.py` against the HF model (upstream's reference scripts
   are not carried over).
2. **Parity gates**, as the fork used: hidden within 0.02%, logits within 0.05%,
   greedy codes match 15/15 on the first frames, vocoder PCM within 1e-4.
   Greedy is a code-for-code gate only: argmax decoding of this model collapses
   into silence codes after ~10 frames and never emits EOS, so a greedy run
   that sounds like nothing is correct, not a bug. Listen to the seeded sampled
   reference (`tests/reference/sampled.wav`) instead.
3. **Regression**: `cmp` on the WAV for the bench text at a pinned seed after
   every change; a difference is either an intended arithmetic change (then one
   seed re-audition, documented) or a bug.
4. **Contention test**: a GPU hog process alongside; zero timeouts over a day of
   journal is the bar for any occupancy change.
5. **Runaway test**: temperature 0.9, random seed, long text, `max_audio_tokens`
   cap — must end at the cap, must not grow memory.

## 7. Performance targets

| | gfx1100 | gfx1151 |
|---|---|---|
| Frame (talker + cp) | ≤ 8.5 ms at M1 (parity with the fork); goal ~7.5 | measure at M3, then set |
| Vocoder | ~1.2 ms/frame | measure |
| TTFA, streaming | < 300 ms | < 500 ms |
| RTF, queue median under llama-server contention | ≤ 0.12 | ≤ 0.5 is the usefulness bar |
| First request after start | no slower than steady state (warm-up) | same |

Numbers go on the README front page from the bench harness, per release.

**Measurement, two tools:** `GET /metrics` (Prometheus text format, llama.cpp
naming, scraped via the workstation's node_exporter textfile collector) is what
production experiences over time - live RTF, stage split, TTFA histogram,
barrier timeouts, frame-cap hits. `navi-tts bench` is the controlled A/B: same
code path as serving, no HTTP, fixed text/voice/seed, emits JSON (git hash, gfx
target, ROCm version, ms/frame by stage, TTFA, RTF, WAV sha256) that is appended
to `bench/results.jsonl`; the README table is generated from that file.

## 8. Strix Halo

- Starts at M3, after the XTX is in production. Until then the build is
  gfx1100 only; M3 adds gfx1151 to `CMAKE_HIP_ARCHITECTURES` (fat binary).
  Nothing arch-specific at compile time beyond that; grid, block and spin
  parameters come from the device at runtime.
- Unified memory: still allocate device memory and upload; host-coherent
  mappings are an experiment for M3, not a design assumption.
- Expect launch and sync latency to look similar (same wave32 RDNA lineage) and
  bandwidth-bound phases (vocoder convs, prefill) to be the ones that slow down.
  If the frame kernel holds ~10-12 ms on 40 CUs, RTF lands ~0.15-0.2, which is
  comfortably useful for the mini PC.

## 8b. Host and toolchain — DECIDED

- **Compiler:** the ROCm clang (AMD clang 23, ROCm 10.1.0 therock) for host and
  device alike, invoked directly with `-x hip --offload-arch=gfx1100`; no
  `hipcc` wrapper, no system g++ for host objects (the mixed-toolchain link
  failure in `docs/reference/toolchain.md` is why). C++20, `-O3 -march=znver3
  -flto`, `-ffast-math` off until the parity gates pass without it. CMake +
  Ninja with presets (`xtx`, later `strix`).
- **CPU is off the fast path.** Ryzen 7 5800X3D; the host does HTTP, BPE, one
  launch per frame, one wait, WAV out. What it owes the design is latency
  hygiene, not throughput: spin-wait on completion (`hipDeviceScheduleSpin` or a
  mapped flag) rather than yield; pinned host buffers for codes and PCM; one
  pinned serving thread for the single in-flight request, HTTP accept on
  another; `mmap` → upload → `munmap` for weights, no host copy retained.
- **Python** only offline, in `tools/` (converter, reference dumps), via `uv`.

## 9. Open questions

- **DECIDED: HTTP library - cpp-httplib** (single header, MIT, vendored in
  `third_party/`), with its opportunistic zstd/openssl probes disabled at
  configure time. It covers routing, multipart uploads and chunked streaming
  responses, which is the whole contract. Batch-one, loopback-only traffic
  makes an async framework (Drogon, Oat++) pure weight here; if the engine ever
  becomes a shared multi-client service, the HTTP layer is the cheapest part to
  swap. Streaming is a bounded queue between the serving thread (pushes each
  vocoder batch's PCM) and the response's content-provider callback (writes
  chunks as they arrive); a client that stops reading fails the request rather
  than stalling the GPU worker.
- **OPEN: ICL cloning (tokenizer encoder).** Voices registered with `ref_text`
  use in-context cloning and need the audio *encoder*; without it, cloning is
  speaker-embedding only (what runs today). The encoder is a fourth model
  component (Conv1D + 8-layer transformer + RVQ). Recommendation: out of M1/M2;
  revisit as a quality lever in M4 with a listen test first.
- **OPEN: 1.7B.** Same architecture scaled; RTF 3.0 on the fork's Vulkan path in
  June, never tried on HIP. Support via config is nearly free; tuning is not.
  Recommendation: load it as a stretch test in M3, don't target it.
- **DECIDED: ROCm floor, not a pin - 10.1 or newer.** Enforced at configure
  time (`NAVI_ROCM_MIN_VERSION`, from the install's `.info/version`) and at
  runtime (the loaded HIP runtime must be at least the one the binary was
  compiled against). The fork's compiler-rt overlay workaround
  (`docs/reference/toolchain.md`) is not needed when the ROCm clang is called
  directly. The version actually built with goes on the README front page.

## 10. Milestones

M0-M2 are the XTX. Nothing runs on the mini PC before M3.

- **M0 — skeleton.** Repo layout, CMake + ROCm toolchain on the workstation,
  `runtime/device` with arch detection and refusal, `tools/convert.py` →
  `.navi`, loader + upload, bench harness scaffold, reference dumps generated.
- **M1 — parity.** Tokenizer, prefill, frame kernel, vocoder; `POST
  /v1/audio/speech` returns correct audio for the bench text; greedy codes match
  the reference; gfx1100 frame ≤ 8.5 ms; run-to-run bit-exact. Old engine still
  in production.
- **M2 — cutover.** Voice store with persistence, XTTS dialect, streaming,
  `/metrics`, warm-up, runaway/contention tests green for a day. `navi-tts.service`
  replaces `tts-engine.service`; `skyrimnet-xtts-shim.service` and
  `tts-register-voices.service` retired; TTS-Player docs updated.
- **M3 — Strix Halo.** Fat binary, build and bench on the mini PC, tune grid/
  spin from data, front-page numbers for both targets.
- **M4 — levers.** Frame-kernel fusion of the host tail, spin cap from p99, ICL
  cloning, quantised GEMV (worth re-measuring on gfx1151 even though the XTX
  ruled it out), 1.7B, fine-tuned weights through the converter.

## 11. Carried over from the fork

Only the author's own work - files *added* by the fork's commits, verified with
`git diff --name-status origin/main..HEAD`. Nothing inherited from either
upstream, source or docs. It lives under `reference/` (unbuilt starting
material) and `docs/reference/` (the record):

- `reference/hip/` - fused talker, code predictor and frame kernels, `hip_dev.h`
  (2,571 lines; the seed of `model/qwen3tts/frame.hip`)
- `reference/tests/` - HIP parity / fallback / self-heal tests, the GPU hog
- `reference/bench/` - the bench harness scripts, the `grid.sync()` microbench
- `docs/reference/code_predictor_plan.md` - dispatch-floor and barrier
  measurements, profiling recipes, things ruled out
- `docs/reference/talker_fusion_handoff.md` - fused-kernel gates, invariants,
  the contention investigation
- `docs/reference/performance_plan.md` - results log and Phase 4 proposals
  (4.1-4.9, including the `/metrics` design)
- `docs/reference/custom_voice_setup.md`, `docs/reference/toolchain.md`

Not carried over, deliberately: upstream's `tensor_mapping.md`,
`streaming_design.md`, `optimization.md`, the GGUF converters and reference
scripts, and all of `src/` outside `src/hip/`. Their facts are re-derived in
`docs/model.md` and `tools/` at M0.
