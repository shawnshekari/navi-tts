# Talker Fusion — Session Handoff

Start here if you're picking up the fused HIP talker. Written 2026-09-12.
See `code_predictor_plan.md` Phases 2-3 for the kernel design and parity
numbers. This doc covers the **ship state** and the **next step**, not
the kernel internals.

## Current state (what's live in production)

**All three gates passed and shipped.** `tts-engine.service` now runs the
**host** binary `build-hip-host/qwen3-tts-server` with BOTH
`QWEN3_TTS_USE_HIP_TALKER=1` and `QWEN3_TTS_USE_HIP_CODE_PRED=1`
(fused talker + fused code predictor + vocoder, all on ROCm0), client
seed pinned to 2. Live steady-state under llama-server contention:
fused talker 2.2 ms/frame + fused cp 6.6 ms/frame + host tail 0.03
ms/frame, RTF ~0.14, zero fallbacks.

- Fused talker kernel: `src/hip/talker_hip.{h,hip}`, parity-verified
  (hidden 0.018%, logits 0.05%, cb0 argmax match vs the
  `QWEN3_TTS_DUMP_TALKER` reference).
- Fused cp kernel: `src/hip/code_pred_hip.{h,hip}`, 15/15 argmax parity.
- Gate 1 (mid-gen fallback), Gate 2 (seed 2 keeper), Gate 3 (host-binary
  cutover) — all DONE, details in the gate sections below.
- Device-resident chaining (hidden + step_embd) DONE & DEPLOYED — see
  "Next technical lever" for what it set up.
- The toolbox `build-hip/` is no longer used by the service (kept for
  reference). The host `build-hip-host/` is the production build.

> ⚠️ **LIVE REGRESSION (2026-09-12):** the running process latched both
> fused kernels to GGML after barrier timeouts under contention — RTF
> ~0.21 vs the ~0.13/0.14 fused steady state. **Self-healing is now
> implemented** (see KNOWN ISSUE below): the engine re-probes the fused
> path on its own and recovers once the GPU quiets, so no manual restart
> is needed — just keep serving requests.

## Next session: single fused talker+cp kernel

The remaining big lever is **one cooperative launch per frame** that
produces cb0 + all 15 codes + the next step_embd on-device, removing
the last host round-trips and the cp's separate launch. All the
prerequisite plumbing is already in place and deployed (see the chaining
notes at the bottom of this doc). Full design notes and the kickoff
prompt are in **"Next technical lever"** near the end of this file.

## Do NOT just flip the env var in the service. Three gates first.

### Gate 1 — mid-generation fallback (correctness on a shared GPU)

The fused talker is a cooperative kernel needing full grid residency.
The service shares the GPU with `llama-server` and `embedding-server`.
If the grid can't launch, the deadline barrier times out and
`HipTalker::run` returns false — and the current integration does
`return false` from `generate()` (tts_transformer.cpp ~3651), failing
the whole request.

The fused code predictor already handles this: on failure
`predict_codes_autoregressive` logs and falls back to the ggml path.
The talker needs the same, but mid-generation is trickier because the
fused path owns `pending_cb0` and the device-side `seen[]` set.

Required fix: on `run()` failure, fall back to `forward_step` + the
host `sample_cb0` lambda for that frame and keep going. The host must
then re-sync the device `seen[]` set with `generated_cb0_tokens` (call
`reset_repetition()` then `mark_seen()` for each token) so the penalty
stays consistent across the fused/ggml boundary. Decide: latch
`hip_talker_failed_` on first failure (simplest, one fallback frame)
or retry per frame. Recommend latch-and-fallback.

Add a test: run the fused talker while a second process saturates the
GPU and confirm the request completes via fallback rather than erroring.

**DONE (2026-09-12).** Latch-and-fallback implemented in `generate()`:
on `run()` failure the frame is redone with `forward_step` (which also
overwrites any partial K/V the aborted kernel wrote at `n_past`),
`hip_talker_failed_` latches for the process, and the device `seen[]`
set is re-synced from `generated_cb0_tokens`. Test:
`test_hip_talker_fallback` (forks a full-grid spin hog from
`tests/gpu_saturate.hip`, runs two requests with stderr captured).
Note: the GPU time-slices between processes, so with the default ~0.5 s
barrier cap the 2.4 ms cooperative step still completes under a hog —
the test sets `QWEN3_TTS_HIP_BARRIER_SPINS=1` to make the residency
failure deterministic. Pass criteria: both requests complete, exactly
one fallback line (latch held), fused init line present. Verified on
the live box with llama-server + embedding-server resident; fused
success path (CLI, default cap) and `test_hip_talker` parity unchanged.

### Gate 2 — seed re-audition (voice identity)

The fused talker changes the arithmetic order, so sampled codes diverge
from the ggml-talker path. Seed 3 was auditioned for **ggml talker +
fused cp**. With the fused talker the voice is a new roll.

Before pinning: run 3-4 seeds through the **fused talker + fused cp**
path on the reference voice(s) (`~/tools/reference_voices/nyx_reference.wav`),
listen, pick the keeper, pin it. Same procedure as the Phase 2 re-audition.
Do this AFTER gate 1 so you audition the shippable code.

**DONE (2026-09-12). Keeper: seed 2.** Four seeds (1-4) through the
Gate-1 build, fused talker + fused cp, temp 0.9 / top-k 50, both voices
(nyx + amy mirrors); candidates in `/tmp/gate2_audition/`. User picked
**seed 2** on nyx.

**Pin coupling — apply at the Gate 3 cutover, not before.** The seed is
path-dependent: seed 2 is only auditioned against the fused-talker
arithmetic. The live client pin is `~/src/TTS-Player/tts_config.yaml`
(`voices.seeds`, hot-reloaded; voice_1=nyx, voice_2=amy, currently 42
= the ggml-talker roll). Changing it now would re-roll the live voice
onto an unauditioned seed on the *old* path. At cutover, change both
entries to `2` in the same window as adding
`QWEN3_TTS_USE_HIP_TALKER=1` to the service.

### Gate 3 — build path

The service ExecStart points at `build-hip/` inside the `navi31-llama`
toolbox. Two options:

- **Rebuild `build-hip` in the toolbox** with the talker code (keeps the
  current infra). Use the Phase 0 toolbox recipe
  (`~/src/fedora_toolbox/Dockerfile.fedora44-therock-navi31`) plus the
  compiler-rt overlay workaround noted in `code_predictor_plan.md`.
- **Move the service to a host binary** (`build-hip-host/`). The host
  now has full ROCm and the E2E (incl. vocoder) runs natively — no
  toolbox glibc dance. Cleaner long-term, but changes the service's
  container/kill-mode assumptions (the current unit shares the toolbox
  with llama-server via `KillMode=process`). Decide deliberately.

**DONE (2026-09-12). Host binary cutover, live in production.**
Decision drivers: the host build is a strict superset (links libav →
mp3/opus works, which the old toolbox binary never had; current code
with Gate 1), and a direct A/B of the fused cp on the same reference
frame showed toolchain parity (host 5.63/5.70 ms vs toolbox 5.67/5.70
ms best/mean, 15/15 argmax). Unit rewritten: `ExecStart` →
`build-hip-host/qwen3-tts-server` directly, toolbox/podman/`KillMode=
process`/pkill lines dropped, `Before=llama-server` VRAM ordering kept,
fused flags via `Environment=`. **Gotcha hit during cutover:** systemd
doesn't source `~/.bashrc`, so the therock libs weren't on the path —
the unit now sets `LD_LIBRARY_PATH=/home/sreed/tools/therock-tarball/
install/lib:.../llvm/lib` explicitly (the ld cache only has the old
system `libamdhip64.so.5`). Client seed pins flipped 42→2 with the
restart (Gate 2 keeper).

**Voice trio self-heals:** the engine does NOT persist ICL voices across
restarts — `tts-queue`'s `ensure_voices` re-registers voice_1=nyx /
voice_2=amy on its next request cycle. A speech request that races the
fresh engine gets a 400 `unknown voice`; restart order that avoids the
race is engine → wait for queue to re-register → traffic. (The
`tts-register-voices` oneshot adds duplicate ids voice_3/voice_4 —
harmless.)

**Steady state under live llama-server load:** fused talker 2.2 ms/frame
+ fused cp 6.6 ms/frame, zero fallbacks. Caveat worth watching: the
very first request after the cutover *did* hit a fused-cp barrier
timeout (my gate tests + register-voices speaker encoding were
saturating the GPU in that window) and latched the cp to the GGML path
at ~128 ms/frame under contention — the fallback keeps requests correct
but latency collapses if it latches. If production latency ever
degrades, check the journal for `HIP fused code predictor failed,
falling back` and restart the engine on a quieter GPU. Rollback: remove
the two `Environment=` fused flags (or restore
`/tmp/tts-engine.service.bak`) + daemon-reload + restart, and restore
seed pins to 42.

## RESOLVED 2026-09-12 (evening) — contention timeouts, the runaway, and the 10 s GPU pegs

Three findings from a live investigation, all fixed and measured:

**1. `QWEN3_TTS_HIP_BLOCKS_PER_CU=1` eliminates the barrier timeouts.**
With llama-server + embedding-server driven to 99% GPU by a load
generator, the default 4/CU (192-block) grids hit the barrier deadline on
*every* request — talker AND cp -> GGML, ~220 ms/frame under time-slicing,
23-28 s per ~110-frame request. At 1/CU (48 blocks): **zero timeouts in
every run**, fused path intact at 14.7 ms/frame under full load (RTF ~0.18),
12.8 ms/frame through the live queue. On a quiet GPU 1/CU is also slightly
faster (9.0 vs 9.5 ms/frame) with bit-identical output. The service unit
now sets it (TTS-Player `tts-engine.service`). The self-heal latch remains
as the safety net; it should rarely fire now.

**2. Mid-request fused-cp -> GGML fallback used a stale hidden (runaway root cause).**
The fused talker leaves its post-norm hidden on the device and skips the
host copy (chaining); if the fused cp then failed or was latched, the ggml
cp was handed `last_hidden_` — whatever `forward_step` last wrote, many
frames earlier. Codes 1-15 were computed for the wrong frame from then on,
the next step embedding was garbage, the talker drifted and never emitted
EOS. Reproduced deterministically on a quiet GPU with the new
`QWEN3_TTS_HIP_CP_FAIL_AT=<n>` debug env: 2048 frames (CLI cap) / 164 s of
audio without the fix, 107 frames / 8.5 s with it. Fix:
`HipTalker::fetch_hidden()` D2H before the ggml cp runs whenever it was
given a device hidden. Every "runaway" observed today was on such a
request.

**3. The 70-80 ms/frame `Init/KV/embed` after a fallback was contention, not a bug in the bucket.**
Sub-timers were added (kv init / clear / hidden fetch / embed / graphs) —
on a quiet GPU all are ~0. Under external load every synchronous device
op waits out a scheduler timeslice, and the ggml talker was equally slow
(79 ms/frame). Still, the per-frame `clear_code_pred_kv_cache()` (2 x
n_layers synchronous memsets) was unnecessary: every slot is written before
it is read within a frame and `inp_mask` hides the rest, so the cache is now
zeroed once at allocation (`needs_clear`) and only `n_used` is reset per
frame. Output bit-identical on the fused->ggml handoff and the pure ggml
path. With (1) in place the ggml path is rarely exercised anyway.

Also: the timing block's `Backend:` line now reports `HIP fused`,
`HIP fused -> GGML (fallback mid-request)` or `GGML` truthfully — it used
to print GGML even when the fused cp ran, which misled the diagnosis.

Useful commands:
```
# reproduce a mid-request handoff on a quiet GPU
QWEN3_TTS_USE_HIP_TALKER=1 QWEN3_TTS_USE_HIP_CODE_PRED=1 QWEN3_TTS_HIP_CP_FAIL_AT=5 \
  ./build-hip-host/qwen3-tts-cli -m models/qwen3-tts-0.6b-f16.gguf --vocoder models/qwen3-tts-tokenizer-f16.gguf \
  -r ~/tools/reference_voices/nyx_reference.wav -t "..." --seed 2 -o /tmp/x.wav
# per-request frame budget from a client (clamped to --max-tokens)
curl :8080/v1/audio/speech -d '{"input":"...","voice":"voice_1","max_audio_tokens":150}'
```

**Proposed next steps (all TO BE DISCUSSED, none started):** see
`performance_plan.md` Phase 4 — frame fusion at 1/CU, warm-up request,
data-driven spin cap, queue-side streaming, ggml-path seed audition, ggml
bump re-evaluation, upstream PRs.

## KNOWN ISSUE — permanent latch under contention degrades RTF (self-healing deferred)

**Symptom (observed 2026-09-12):** client RTF climbed from the ~0.13
fused steady state to **0.21** and stayed there. Per-request timing in
the journal showed `Code predictor Backend: GGML` (not fused), talker
3.8 ms/frame + cp 10.8 ms/frame ≈ 15.8 ms/frame, vs the fused
2.2 + 6.6 ≈ 9 ms/frame.

**Root cause:** the fused talker and fused cp each hit a `grid barrier
timed out` under GPU contention and latched to GGML for the **whole
process lifetime** (`hip_talker_failed_` / `hip_code_pred_failed_` are
one-way latches). Trigger was `llama-server` ×2 + `embedding-server`
resident at 100% GPU util / 88% VRAM — the cooperative grids couldn't
get a full-residency slice inside the spin deadline, so they aborted and
latched. This is the Gate-3 caveat above, now actually observed in
production.

**Why it's sticky:** the latch is intentionally one-way (correctness:
once a barrier timeout is seen, the process never trusts the cooperative
grid again). That makes a *single transient* contention spike permanently
degrade the process until restart — the fused path never comes back on
its own.

**Why the fused-frame kernel does NOT help here:** `k_frame_fused` has
the *same* cooperative full-residency requirement, so under this
contention it would time out and fall back too. Frame fusion reduces
launch/sync overhead on a *schedulable* GPU; it does not make the grid
easier to schedule. (Measured occupancy is unchanged at 4 blocks/CU.)

**Immediate remedy (operational, not code):** restart `tts-engine` when
the GPU is quiet (llama-server/embedding idle) so the fused path
re-engages. Restarting *during* saturation just re-times-out and
re-latches (and risks the voice-registration race — see Gate 3).

**Self-healing — IMPLEMENTED (2026-09-12).** The one-way latch is now a
re-armable `HipHeal` state machine (`src/tts_transformer.h`), one per
fused component (talker / cp / fused-frame), driven at the request
boundary in `generate()`:

- *Cooldown-gated probe + exponential backoff:* while latched, every
  `backoff` requests we clear the component's gate for one request and
  re-attempt the fused path. A failed probe doubles `backoff` (capped at
  64) so a persistently-busy GPU stops paying the probe's spin-deadline
  stall; the first successful probe resets it to 1 and the fused path is
  back. The first latch probes on the very next request (backoff 1) for
  fast recovery.
- *Tight probe deadline:* `set_probe_mode(true)` (on `HipTalker` /
  `HipCodePredictor` / `HipFrameFusion`) drops the barrier spin cap to
  1/10 of normal (~50 ms vs ~500 ms) for the probe request, so a probe
  into a still-busy GPU gives up fast instead of stalling the full
  deadline. Restored to normal on heal.
- *Not a special code path:* the probe just clears the existing
  `*_failed_` gate and runs the normal fused path through the existing
  fallback machinery, so the Gate-1 `seen[]` re-sync invariant holds
  unchanged if the probe fails mid-request.
- *Init failures stay permanent:* `begin_request(ready)` never probes a
  component whose init failed (`*_ready_` false), so a config error isn't
  retried forever.
- *Observability:* the journal logs `HIP self-heal: probing fused
  talker/cp/frame`, `... probe failed, backoff N`, and `... recovered`.

To recover the live regression now: just keep serving requests — the
engine re-probes on its own and flips back to the fused path once the
GPU quiets (no restart needed). Restarting on a quiet GPU still works
and is instant, but is no longer required.

**Tests:** `test_hip_heal` (pure-logic policy: cadence, exponential
growth, cap, init-failure permanence, full latch→quiet→recover cycle).
`test_hip_talker_fallback` updated to the new contract (second request
re-probes + re-latches under the hog instead of asserting a permanent
latch). `test_hip_frame_fallback` given a fixed seed (it samples at
temp 0.9 and was nondeterministically hitting early EOS). Manual GPU
test: saturate the GPU to trip the latch, then idle it and watch the
journal for the `recovered` line.

**Rejected alternatives** (kept for context): per-frame retry (worst —
every frame pays the spin-deadline under sustained contention);
adaptive-widen deadline (just converts fast-fallback into slow-fallback);
GPU-util polling (coarse/laggy metric + polling overhead — the probe is
a cleaner form of sensing).

## Enabling, once gates pass

Add `QWEN3_TTS_USE_HIP_TALKER=1` to the service ExecStart env (next to
`QWEN3_TTS_USE_HIP_CODE_PRED=1`), flip the client seed pin
(`tts_config.yaml` `voices.seeds`: voice_1/voice_2 `42` -> `2`, the
Gate-2 fused-path keeper — see gate 2 note on why these move together),
`systemctl --user daemon-reload && restart tts-engine`, and watch the
journal for the `HIP fused talker: N blocks` line and any
`grid barrier timed out` aborts under real load. Roll back by removing
the env var and restoring the seed pins to 42.

## Next technical lever (after shipping, or instead of shipping now)

The fused talker sits at 2.4 ms against a ~1.3 ms bandwidth floor. The
gap is ~170 grid barriers/step and the attention phase using only 16 of
the grid's blocks. Two directions, in rough value order:

1. **Device-resident chaining.** Today each frame does a 4 KB hidden
   D2H (talker) + 4 KB H2D (cp) round-trip. `HipTalker::device_hidden()`
   already exposes the device buffer. Add a cp entry point that reads
   hidden from a device pointer (skip the host copy), with a stream
   event between the talker stream and the cp stream. Removes ~8 KB/frame
   of PCIe and one sync from the critical path.
2. **Single fused talker+cp kernel.** One cooperative launch per frame,
   one sync, cb0 and the 15 codes all produced on device. Biggest win,
   biggest change: the cp's 16-position loop and the talker's 28-layer
   step share one barrier schedule. Revisit the residency math — a larger
   cooperative grid is harder to schedule on a busy GPU, which makes
   gate 1's fallback even more important.

Cheaper follow-ups: parallelize the talker attention across more blocks
(split-K over positions); tune blocks/CU per n_past.

**Device-resident chaining — DONE (2026-09-12), perf-neutral.**
`HipCodePredictor::run_device()` takes a device pointer for the talker
hidden (skips the 4 KB H2D); `generate()` skips the talker's 4 KB
hidden D2H when the fused talker produced it (kept only under
`QWEN3_TTS_DUMP_LOGITS` for the diagnostic) and threads the pointer
via a `prev_hidden_device` flag (frame 0's hidden is the ggml prefill
on the host; a Gate-1 fallback flips it back to host). Verified
**byte-identical** output vs the pre-chaining path (same seed/text, 3
runs) — the device path reads the same source buffer, so it's
equivalence-by-construction. Gate-1 fallback + talker parity tests pass.
**Measured win: none** — talker 2.2 ms + cp ~8.6 ms before and after;
the 8 KB PCIe round-trip is ~1-2 us/frame, lost in contention noise.
Kept as fusion plumbing, not deployed for perf (production binary is
unchanged until the next meaningful restart).

**Revised lever ranking after measuring:** the per-frame budget under
live contention is talker 2.2 + cp 8.6 + host tail 0.47 ms. The cp
stage dominates and is contention-sensitive (5.9 ms idle -> 8.6 ms with
llama-server resident). The two remaining real levers, in value order:
1. **Single fused talker+cp kernel** — one cooperative launch produces
   cb0 + all 15 codes + the next step_embd on device, removing the
   cb0 round-trip, the cp's separate launch, and the 0.47 ms host
   embed tail from the critical path (~4% + a sync). Biggest win,
   biggest change, and a larger cooperative grid is harder to schedule
   on the busy GPU (Gate 1 fallback becomes load-bearing).
2. **Device-side embed lookup** — fold the step_embd assembly
   (codec_embd(cb0) + code_pred_embd sum + trailing row) into the cp
   kernel output so the host never builds it; removes the 0.47 ms
   serial tail (~4%) without the full fusion. Cheaper than (1),
   same tail win.

**Device-side embed lookup — DONE & DEPLOYED (2026-09-12).**
`HipCodePredictor::assemble_step_embd(trailing_row)` runs a small
non-cooperative kernel (`k_assemble_step_embd`) that folds the 16
embedding rows (codec_embd[cb0] + the 15 code_pred_embd rows for the
codes just sampled, read from the cp's device `d_codes`) + the trailing
text row into a device `d_step_embd`, in the host's exact summation
order so the F32 result is **bit-identical** to the host path.
`HipTalker::run_device()` (D2D copy into the residual buffer) consumes
it. `generate()` gates this on `chain_embed = fused_talker &&
cp_last_frame_fused_` — if the fused cp falls back to GGML the flag is
false and the host assembly takes over (verified: a contention-induced
cp fallback correctly reverts and still completes). Measured: host tail
**0.47 -> 0.03 ms/frame** (embed lookups bucket 0.47 -> ~0.0); output
byte-identical to the pre-chaining baseline across runs where the fused
path holds. Gate-1 fallback + talker + cp parity tests pass. Deployed
to `tts-engine` (restart picked up the new binary; production journal
shows host tail 0.03 ms/frame, no fallbacks).
The talker's 2.2-vs-1.3 ms floor is barrier count (~170 grid
barriers/step); reducing phases is a kernel redesign with less
end-to-end value than (1)/(2) right now.

### THE NEXT TASK — single fused talker+cp kernel

Both chaining steps above are the groundwork for this. The goal: replace
the per-frame `HipTalker::run_device()` + `HipCodePredictor::run_device()`
+ `assemble_step_embd()` (three launches, three syncs) with **one
cooperative kernel** that does the whole frame on-device:

1. talker 28-layer step (reads the previous frame's `d_step_embd`,
   writes cb0 + hidden + the F16 KV cache row at `n_past`),
2. cp 16-position loop (position 0 = talker hidden, position 1 =
   codec_embd(cb0), positions 2..15 = code_pred_embd rows of the
   sampled codes) producing the 15 codes,
3. the step_embd assembly for the NEXT frame (the `k_assemble_step_embd`
   math) — so the host only reads back the 16 codes + cb0 and never
   touches hidden or step_embd.

Design considerations:
- **Shared barrier schedule.** The talker's ~170 barriers/step and the
  cp's 16-position × ~8-phase barriers must interleave in one grid.
  Reuse the deadline-aware `grid_barrier` from both kernels (same
  sense-reversing pattern); the fused kernel needs one `bar` buffer and
  one `spin_cap`.
- **Residency math is the risk.** The fused grid must be co-resident in
  one cooperative launch. The talker alone uses 192 blocks (4/CU × 61
  CU on the 7900 XTX); the cp uses the same 192. A fused kernel with
  the union of their LDS/register pressure may drop blocks/CU and shrink
  the grid — measure `hipOccupancyMaxActiveBlocksPerMultiprocessor`
  for the fused kernel first. A larger cooperative grid is harder to
  schedule on the busy GPU, which makes the Gate-1 fallback load-bearing
  (the fused kernel must return false on barrier timeout exactly like
  the talker does, and `generate()` must fall back to the current
  three-launch chained path, NOT to ggml, so the win degrades
  gracefully).
- **Keep the existing three-launch path as the fallback.** Don't remove
  `run_device`/`assemble_step_embd` — they become the fallback when the
  fused kernel can't launch. The `chain_embed`/`prev_hidden_device`
  flags already model "device-resident vs host" and can extend to
  "fused vs chained".
- **Parity anchor.** Bit-identical output vs the current chained path
  (same summation order, same sampling RNG seeds per position) is the
  correctness bar — same as the chaining. Use the existing
  `test_hip_talker` / `test_hip_code_pred` reference dumps as anchors
  and add a fused-frame test comparing the 16 codes + next step_embd.
- **Don't touch production until it's parity-verified and the fallback
  is tested** (same discipline as Gates 1-3).

Expected win: removes one full sync + the cp's separate launch + the
cb0/hidden round-trips from the critical path. The talker+cp are
already device-resident after the chaining, so the fused kernel mostly
saves launch/sync overhead (~0.5-1 ms/frame) rather than PCIe.

**Single fused talker+cp frame kernel — DONE & PARITY-VERIFIED
(2026-09-12). NOT yet deployed to `tts-engine` (awaiting a parity
audition on the live box before cutover, same discipline as Gates 1-3).**

`src/hip/frame_fused.{h,hip}` — `HipFrameFusion::run_frame()` does the
whole frame in ONE cooperative launch: copy step_embd→x, talker 28-layer
step (→ hidden + cb0 into `codes[0]` + the F16 KV row at `n_past`), the
cp 16-position loop (→ `codes[1..15]`), then the next-frame step_embd
assembly. The host reads back only the 16 codes (`get_codes`). The
talker's ~170 barriers and the cp's 16×~8 barriers share ONE
deadline-aware `grid_barrier` buffer + ONE spin cap — the shared device
helpers (incl. `grid_barrier`) were lifted out of the two shipped kernels
into `src/hip/hip_dev.h` (verbatim move; the shipped kernels now include
it, so there is a single source of truth for the barrier).

- **Residency (the risk): MEASURED, no regression.**
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` for `k_frame_fused` =
  **4 blocks/CU**, identical to the standalone talker and cp kernels →
  same 192-block cooperative grid. The fused kernel's union of register/
  LDS pressure does NOT drop blocks/CU, so the grid is no harder to
  schedule than the existing ones. Logged at init: `HIP fused frame: N
  blocks (X/CU)`.
- **Parity: BIT-IDENTICAL to the three-launch chained path.**
  `test_hip_frame_fused` runs both paths on the same reference step with
  the same seeds and the seen set reset between: cb0 match, 15/15 codes,
  next step_embd `memcmp` equal (max|diff| 0.0). Greedy AND sampled
  (temp 0.9 / top-k 50). E2E CLI: fused-frame wav is byte-identical to
  the chained wav across a 120-frame generation, greedy and sampled — the
  per-frame seed order (talker seed then cp seed drawn at end of frame N)
  keeps the whole rng stream aligned with the chained path.
- **Fallback: chained path, NOT ggml.** On `run_frame` barrier timeout,
  `generate()` latches `hip_frame_failed_`, re-syncs the device `seen[]`
  set from `generated_cb0_tokens` (mirror of Gate 1), and the existing
  three-launch chained tail redoes the frame; subsequent frames run the
  chained path. `run_device`/`assemble_step_embd` are untouched.
  `test_hip_frame_fallback` drives this deterministically via the
  `QWEN3_TTS_TEST_FRAME_FAIL_AFTER=<n>` hook in `run_frame` (fails the
  Nth call, simulating a mid-generation busy-GPU timeout) — asserts the
  request completes, the fallback fires exactly once (latch holds), and
  all frames are produced. (A raw hog does NOT trip the fused frame the
  way it trips the talker: the fused frame is one long cooperative kernel
  that reliably gets a full-residency slice, whereas the bootstrap cp
  fails first and skips the fused launch — hence the deterministic hook.)
- **Enable:** `QWEN3_TTS_USE_HIP_FRAME_FUSION=1` alongside
  `QWEN3_TTS_USE_HIP_TALKER=1` + `QWEN3_TTS_USE_HIP_CODE_PRED=1`
  (frame fusion requires both; it force-inits the cp before the loop).
  Default off.
- **Measured (idle-ish GPU, 120 frames):** fused frame 7.9 ms/frame
  device time replaces the chained talker 2.3 + cp 6.7 + host-tail
  0.03; loop wall 1073→964 ms, throughput 11.8→11.3 ms/frame. The win
  is the removed launch/sync overhead, as predicted. Re-measure under
  llama-server contention before trusting the number.
- **Tests:** `test_hip_frame_fused` (parity), `test_hip_frame_fallback`
  (fallback). Existing `test_hip_talker` / `test_hip_code_pred` /
  `test_hip_talker_fallback` all still pass after the `hip_dev.h`
  refactor.

## Build / test / bench (host, this machine)

```bash
# configure (host therock ROCm; use clang++ directly, NOT the hipcc wrapper)
cmake -S . -B build-hip-host -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
      -DQWEN3_TTS_TIMING=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/home/sreed/tools/therock-tarball/install/lib/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/home/sreed/tools/therock-tarball/install
cmake --build build-hip-host -j8

# reference dump (first decode step: step_embd, hidden, logits, full KV cache)
QWEN3_TTS_DUMP_TALKER=/tmp/talker_ref_step.bin \
  ./build-hip-host/qwen3-tts-cli -m models/qwen3-tts-0.6b-f16.gguf \
  --vocoder models/qwen3-tts-tokenizer-f16.gguf \
  -r ~/tools/reference_voices/nyx_reference.wav -t "Hello, this is a test." \
  --temperature 0 --max-tokens 8 -o /tmp/talker_ref.wav

# parity test (greedy default; QWEN3_TTS_TEST_TEMP/TOPK for sampled smoke)
./build-hip-host/test_hip_talker --model models/qwen3-tts-0.6b-f16.gguf \
  --ref /tmp/talker_ref_step.bin --iters 30

# fused vs ggml talker, long text (amortize init; compare total generate)
LONG="...a few sentences..."
QWEN3_TTS_USE_HIP_CODE_PRED=1 ./build-hip-host/qwen3-tts-cli ... -t "$LONG" --temperature 0 --max-tokens 400
QWEN3_TTS_USE_HIP_TALKER=1 QWEN3_TTS_USE_HIP_CODE_PRED=1 ./build-hip-host/qwen3-tts-cli ... -t "$LONG" --temperature 0 --max-tokens 400

# fused-frame parity (bit-identical vs chained) + fallback
./build-hip-host/test_hip_frame_fused --model models/qwen3-tts-0.6b-f16.gguf \
  --ref /tmp/talker_ref_step.bin --iters 10
QWEN3_TTS_TEST_TEMP=0.9 QWEN3_TTS_TEST_TOPK=50 \
  ./build-hip-host/test_hip_frame_fused --model models/qwen3-tts-0.6b-f16.gguf \
  --ref /tmp/talker_ref_step.bin --iters 10
./build-hip-host/test_hip_frame_fallback --model models/qwen3-tts-0.6b-f16.gguf \
  --frames 24 --fail-after 2

# e2e byte-identical check: fused-frame vs chained
QWEN3_TTS_USE_HIP_TALKER=1 QWEN3_TTS_USE_HIP_CODE_PRED=1 ./build-hip-host/qwen3-tts-cli \
  ... --seed 2 -o /tmp/chained.wav
QWEN3_TTS_USE_HIP_TALKER=1 QWEN3_TTS_USE_HIP_CODE_PRED=1 QWEN3_TTS_USE_HIP_FRAME_FUSION=1 \
  ./build-hip-host/qwen3-tts-cli ... --seed 2 -o /tmp/fused.wav
cmp /tmp/chained.wav /tmp/fused.wav
```

## Gotchas

- **Frame-count confound.** Greedy output length depends on where EOS
  lands, which changes with the arithmetic. End-to-end ms across paths
  is only comparable at similar frame counts; trust the per-frame
  talker bucket and the unit-test kernel time, not raw totals.
- **GPU must be otherwise idle** for clean timings (see the
  `drm-resident` recipe in `code_predictor_plan.md`). With
  llama-server resident the cooperative grid may not launch at all —
  which is exactly the gate-1 scenario.
- **`tensor->data` is the device pointer** for backend-allocated ggml
  tensors; that's how the fused talker shares the prefill KV cache and
  reads weights in place. Don't re-copy them.
- The `[loop wall / host tail]` line in the timing dump is diagnostic
  added this session (under `QWEN3_TTS_TIMING`); host tail is ~0.5
  ms/frame and is not the bottleneck.
