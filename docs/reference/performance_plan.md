# Performance Optimization Plan

## Goals
- Target: RTF < 0.5x (2x faster than real-time) for 0.6B model — **met 2026-09-10; now 0.12x** (see Results Log and Phase 4)
- Hardware: RX 7900 XTX (HIP/ROCm since Sep 10; Vulkan before), 16-core CPU, 24 GB RAM
- Use case: SillyTavern / Open-LLM-VTuber (low latency, conversational)

## Current Baseline (Jun 19, 2026)

| Model | Audio | Time | RTF | Notes |
|-------|-------|------|-----|-------|
| 0.6B F16 | 3.9s | 3222ms | 0.83x | Voice cloning (nyx_voice) |
| 0.6B F16 | 1.9s | 1557ms | 0.84x | Voice cloning (amy_voice) |
| 1.7B Q8_0 | 3.7s | 11033ms | 3.00x | Voice cloning (nyx_voice) |
| 1.7B Q8_0 | 2.4s | 7126ms | 3.00x | Voice cloning (amy_voice) |

## Phase 1: Baseline & Quick Wins

### Task 1.1: Capture Vulkan baseline timings ✅
- [x] Build with `-DQWEN3_TTS_TIMING=ON`
- [x] Run 0.6B F16 with voice cloning (nyx_voice, amy_voice)
- [x] Record: prefill, talker, code predictor, vocoder, total
- [x] Save results to this file

**Detailed breakdown (0.6B F16, nyx_voice, 3.66s audio):**
| Stage | Time | % |
|-------|------|---|
| Tokenization | 1ms | 0% |
| Speaker encode | 0ms | 0% (cached) |
| Code generation | 922ms | 30% |
| - Prefill | 40ms | 1% |
| - Talker | 211ms | 7% |
| - Code predictor | 601ms | 20% |
| Vocoder decode | 2149ms | 70% |
| **Total** | **3072ms** | **RTF 0.84x** |

**Detailed breakdown (1.7B Q8_0, nyx_voice, 3.42s audio):**
| Stage | Time | % |
|-------|------|---|
| Tokenization | 1ms | 0% |
| Speaker encode | 0ms | 0% (cached) |
| Code generation | 8304ms | 80% |
| - Prefill | 194ms | 2% |
| - Talker | 4682ms | 45% |
| - Code predictor | 3274ms | 32% |
| Vocoder decode | 2056ms | 20% |
| **Total** | **10362ms** | **RTF 3.03x** |

**Key insight:** For 0.6B, **vocoder is 70%** of time. For 1.7B, **talker is 45%** and **code predictor is 32%**.

### Task 1.2: Convert 0.6B to Q8_0
- [ ] Use khimaros' quantization script from source checkpoint
- [ ] Convert `models/Qwen3-TTS-12Hz-0.6B-Base` → `qwen3-tts-0.6b-q8_0.gguf`
- [ ] Test: same curl requests, compare speed vs F16
- [ ] Generate side-by-side WAVs (F16 vs Q8_0) for quality comparison
- [ ] **Quality test needed:** listen to output WAVs, compare F16 vs Q8_0

### Task 1.3: Speaker embedding cache (disk-based)
- [ ] Add `--voice-cache-dir` CLI flag (default: `~/.qwen3-tts-cache/`)
- [ ] Server loads cached embeddings on startup from disk
- [ ] Pre-computed embeddings stored as `.spk` files (~4 KB each)
- [ ] New voices get encoded and cached automatically
- [ ] Persists across server restarts
- [ ] Update `docs/custom_voice_setup.md` with new workflow
- [ ] Test: restart server, verify cached voices work without re-encoding

## Phase 2: Streaming Vocoder

### Task 2.1: Transformer frame callback
- [ ] Add `frame_emit_fn` to `TTSTransformer::generate()`
- [ ] Fire callback after each frame's 16 codebooks are generated
- [ ] Unit test: callback fires once per frame with 16 codes

### Task 2.2: Decoder stream state
- [ ] Implement `AudioTokenizerDecoderStream` class in `src/audio_tokenizer_decoder.{h,cpp}`
- [ ] KV cache for pre-tfm layers (8 layers, ~32 MiB)
- [ ] Conv ring buffers (15 rings, ~265 KiB)
- [ ] Unit test: chunked decode matches one-shot decode (bit-exact or < 1e-5 max abs diff)

### Task 2.3: Server SSE integration
- [ ] Wire streaming to `stream_format=="sse"` path in `src/server.cpp`
- [ ] Flush PCM chunks as they're decoded via HTTP data sink
- [ ] Add `--streaming-batch-size` CLI flag (default 0 = off)
- [ ] Test: curl with streaming, measure time-to-first-audio

### Task 2.4: Parity test
- [ ] Implement `tests/test_streaming_parity.cpp`
- [ ] Compare streaming vs one-shot at batch sizes 1, 4, 8, 16, 32
- [ ] Pass criterion: max abs diff < 1e-5 PCM float

## Phase 3: Micro-optimizations

### Task 3.1: Snake activation broadcast
- [ ] Profile vocoder with timing instrumentation
- [ ] Replace `ggml_reshape_3d` + `ggml_repeat` with pre-broadcast alpha/beta tensors
- [ ] Profile before/after vocoder time

### Task 3.2: Batch processing
- [ ] Process multiple texts with shared speaker embedding
- [ ] Useful for SillyTavern multi-utterance scenarios

## Results Log

| Date | Model | Audio | Time | RTF | Notes |
|------|-------|-------|------|-----|-------|
| Jun 19 | 0.6B F16 | 3.66s | 3072ms | 0.84x | Baseline (nyx_voice) |
| Jun 19 | 0.6B F16 | 1.98s | 1637ms | 0.83x | Baseline (amy_voice) |
| Jun 19 | 1.7B Q8_0 | 3.42s | 10362ms | 3.03x | Baseline (nyx_voice) |
| Jun 19 | 1.7B Q8_0 | 1.98s | 5967ms | 3.02x | Baseline (amy_voice) |

| Sep 10 | 0.6B F16 | 8.4s | 15.5s | 1.85x | Vulkan; engine pushed into GTT by llama-server VRAM |
| Sep 10 | 0.6B F16 | 8.4s | 2.25s | 0.27x | VRAM ordering + vocoder dGPU placement fixes |
| Sep 10 | 0.6B F16 | 8.4s | 1.7s | 0.21x | HIP build, persistent code-predictor graphs |
| Sep 12 | 0.6B F16 | — | 9 ms/frame | 0.14x | fused cooperative talker + cp, on-device sampling/chaining |
| Sep 12 | 0.6B F16 | — | 8.5 ms/frame | **0.12x** | 1 cooperative block per CU; queue median over live traffic (p90 0.19) |

## Phase 4: Next steps — ALL TO BE DISCUSSED (proposed 2026-09-12, nothing started)

Status: **proposals only.** None of these is agreed, scheduled or in progress.
Decide each one before writing code; the correctness invariants in
`talker_fusion_handoff.md` (Gate 1 `seen[]` re-sync, seed pin coupling) apply
to anything that touches the fused/ggml boundary.

Context: at 8.5 ms/frame the generation loop is ~12x real time. What the queue
measures (RTF 0.12) is now dominated by per-request overhead rather than the
loop, which changes what is worth doing. For a typical 4 s chunk: loop ~425 ms
(~75%), vocoder ~60 ms (~10%), prefill ~12 ms, HTTP + WAV + queue ~50-80 ms
(~10%), fused-kernel lazy init ~1 s on the first request after a restart only.

### Task 4.1: Enable frame fusion at 1 block/CU — TO BE DISCUSSED
- [ ] `QWEN3_TTS_USE_HIP_FRAME_FUSION=1` merges talker + cp into one launch per
  frame and removes the host tail. Built and parity-tested; left off because it
  shares the cooperative residency requirement — an objection 1/CU has removed.
- [ ] Gate: CLI run `cmp`-identical to current output at seed 2; one listen;
  a day of journal with zero `timed out` lines.
- Expected: ~8.5 -> ~7.5 ms/frame (~10% of the loop).

### Task 4.2: Warm-up request at service start — TO BE DISCUSSED
- [ ] After `tts-register-voices` registers, send one short synthesis per voice
  so the fused kernels, persistent cp graphs and vocoder buffers initialise
  before the first real utterance (today's first request measures RTF ~0.41).
- Expected: removes the one remaining visibly slow moment after a restart.

### Task 4.3: Lower the normal barrier spin cap from data — TO BE DISCUSSED
- [ ] Instrument: `atomicMax` of observed spins per launch, log p99 for a day.
- [ ] Then set the cap (currently 50M spins ~= 0.5 s) to ~10x the p99 — likely
  ~50 ms. Legit stragglers are <1 ms; a rare timeout would then cost ~50 ms
  instead of 0.5 s per component.

### Task 4.4: Streaming for long chunks (queue-side) — TO BE DISCUSSED
- [ ] The server supports `stream_format` + `stream_batch_size`; the queue
  renders whole chunks before playing. Per-sentence chunks are fine; the Claude
  Code hook's long paragraphs (8-10 s of audio) would start in ~0.3 s with
  batch 8. Requires the queue to play from a pipe rather than a file.
- Latency win only; throughput unchanged.

### Task 4.5: Keep-alive connection from the queue — TO BE DISCUSSED
- [ ] `urllib` opens a new TCP connection per chunk; reuse an `http.client`
  connection. ~1 ms/chunk. Trivial, low value; listed for completeness.

### Task 4.6: Seed audition on the ggml path — TO BE DISCUSSED
- [ ] Seed 2 was picked by ear on the fused arithmetic. If a seed sounds right
  on both paths, a fallback becomes inaudible rather than a voice change.
  4-5 seeds, HIP env vars unset, same texts, one CLI session.
- Robustness, not speed; low priority now that timeouts are rare.

### Task 4.7: Re-evaluate the ggml bump — TO BE DISCUSSED
- [ ] Sep 10 rejected v0.23.0 because it changed talker numerics (voice
  re-roll). Talker and cp are now HIP kernels, so a bump only touches vocoder
  (232 -> 141 ms per 8 s) and prefill — but prefill still feeds frame 0, so the
  roll can still shift. Try `ggml-bump`, `cmp` the WAV; not bit-identical means
  a re-audition for ~1.5% end to end. Probably not worth it.

### Task 4.8: Upstream the general fixes — TO BE DISCUSSED
- [ ] `e2597f1` ggml `conv_transpose_1d` O(K) -> PR to ggml.
- [ ] `b11fd22` / `cebfbd3` dGPU vocoder placement, `5dcca64` per-request
  `max_audio_tokens` -> PR to khimaros.
- Not perf; hygiene. All model-agnostic.

### Task 4.9: Native `/metrics` (Prometheus) on the engine — TO BE DISCUSSED (added 2026-09-14)
- Not perf; observability. Context: the household now has Prometheus + Grafana
  on the mini PC (the halogen monitoring stack), already scraping every
  llama-server and halogen. The TTS queue will export its own view
  (`TTS-Player/README.md`, "Metrics"), but SillyTavern, OpenVtuber and other
  clients hit this server **directly**, so the queue's numbers are a subset —
  the engine has to count its own.
- [ ] `GET /metrics` in `server.cpp`, llama.cpp's shape and naming so the
  existing Grafana panels can reuse expressions: `tts:requests_total`,
  `tts:requests_processing`, `tts:prompt_tokens_total` /
  `tts:prompt_seconds_total`, `tts:predicted_tokens_total` /
  `tts:predicted_seconds_total` (= talker tokens and the autoregressive loop —
  the same `timings` already emitted on every response, just accumulated),
  `tts:audio_seconds_total` (so `rate(predicted_seconds)/rate(audio_seconds)`
  is the engine-side RTF), `tts:tokenize_seconds_total` / `encode` /
  `generate` / `vocode` (the same split as `timings.*_ms`, for the
  70/20/7% breakdown above as a live graph instead of a one-off measurement),
  `tts:frame_cap_hits_total` (no-EOS generations — currently only visible as
  the queue's warning), and `tts:errors_total{stage}`.
- Implementation: a handful of `std::atomic<uint64_t>`/`double` counters
  bumped where `timings` is assembled (one place, ~line 212), one httplib
  `Get` handler printing them in the text exposition format. No dependency;
  ~40 lines. Keep `--host 127.0.0.1` — the local clients are the point — and
  let the workstation's node_exporter textfile writer copy
  `http://127.0.0.1:8080/metrics` verbatim into `tts_engine.prom` every 5 s
  (the format passes straight through the textfile collector), so Prometheus
  on the mini PC sees it without the engine ever being LAN-exposed. Same pattern as
  the amdgpu and llama-server `/slots` collectors already running there.
- Why it earns a place in a *performance* plan: Task 4.1/4.2/4.3 all change
  latency in ways that are currently checked with one-off runs. A permanent
  RTF and stage-breakdown graph — on the same axis as GPU busy/VRAM and
  Skyrim's share of the card — is the regression check for every later
  change here, and the only way to see what direct clients experience.

### Explicitly not proposed
- cp kernel micro-work: 5.1 ms for 15 sequential steps x 5 layers is near the
  launch-latency floor for a cooperative grid on RDNA3 (see the grid.sync
  microbench in `code_predictor_plan.md`).
- CPU offload of anything (measured slower, Sep 10).
- Vocoder beyond 4.7 (1.2 ms/frame is ~1.5% of real time).

If only one thing is agreed: 4.1 then 4.2 — about an hour, no new kernels,
median toward ~0.10 and the slow first utterance gone.

## Optimization Priority (based on timing data)

For **0.6B F16** (primary use case):
1. **Vocoder** (70%) - streaming decode, snake activation
2. **Code predictor** (20%) - MTP optimization, quantization
3. **Talker** (7%) - lower priority

For **1.7B Q8_0** (quality use case):
1. **Talker** (45%) - flash attention, kernel fusion
2. **Code predictor** (32%) - same as above
3. **Vocoder** (20%) - streaming decode
