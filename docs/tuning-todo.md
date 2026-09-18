# Tuning TODO

The next steps after M2's first pass (`docs/tuning.md` is the record of what
already moved and what did not). Ordered by payoff per hour, bit-exact work
first. Numbers are the XTX at `5aa11e7`: frame 6.13 ms, vocoder 0.62 ms/frame
streaming, prefill 14.2 ms, TTFA 41 ms, RTF 0.088, WAV sha `4cb7232a…`.

Per-frame budget for reference: 83.3 ms of audio at 12 Hz costs ~7.3 ms wall —
frame 6.13, vocoder 0.62, prefill amortised 0.2, host tail ~0.1. The frame
kernel is now 84 % of it.

## Before any of it

- [ ] **M2's day-long soak** - deferred 2026-09-18 (the service will not see
      real use until later; the tuning work does not wait on it). When it
      runs: `navi_tts_barrier_timeouts` and the frame-cap counter in
      `/metrics`, not just "it sounded fine".
- [x] **Re-confirm the bench protocol.** Done 2026-09-18: `navi-tts` and
      `tts-queue` stopped, `llama-server` + `embedding-server` left resident
      and idle (0 % busy, as for every earlier row), three runs, bench process
      peaked at 2741 MiB VRAM / 8 MiB GTT - no spill. Frame 6.14-6.17 ms,
      TTFA 59, RTF 0.099, sha unchanged: `cf18dfd` reproduces. Note the `git`
      field is compiled in at build time (`cmake/build_info.cmake`), so a row
      tags the commit the binary was *built* at, and the dirty flag is
      build-time too - rebuild after committing, then bench.

Each item below keeps the bench WAV's sha256 unless it says otherwise. A step
that changes the sha needs a listen test and a parity number before it lands.

## 1. Vocoder k=7 convs — done 2026-09-18, `5aa11e7`

Predicted ≈0.004 RTF from register blocking; got 0.011 RTF from something
else. The premise ("LDS-bound, 6 loads per 8 FMAs") was wrong: the weight
loads were 16-way bank-conflicted, the full unroll cost 154 VGPRs, and the
staging loop out-instructed the FMAs. Fixed all three in the shared kernel
(padded weight rows, no unroll for K > 1, 8-byte/float4 staging, dilation as
a template argument), tile unchanged. k=7 group 9.4 → 4.3 ms per 16-frame
chunk; vocoder 1.30 → 0.62 ms/frame; and because the kernel also runs the
prefill GEMMs, prefill 26.6 → 14.2 ms and TTFA 59 → 41. 12/12 WAVs
identical, sha unchanged. Record: `docs/tuning.md`, `bench/micro/conv7.hip`.

- [x] Microbench at the real shapes, variants checked bit-exact first.
- [x] Production kernel, five gates, 12 seed × text `cmp`, bench rows.
- [ ] *(follow-up, low value)* a second tile (64×128, 8×4) for dilation 9 at
      96–192 channels: ~0.03 ms/frame. Only if the vocoder is ever the item.

## 2. Prefill — TTFA 41 → ~30 ms, bit-exact

Item 1 took prefill from 26.6 to 14.2 ms on its own (the K=1 GEMMs share the
kernel), past this item's original ~46 ms TTFA target. What is left: a
prompt of a few dozen tokens is one 32-row tile, so each GEMM is a single
pass over its weights at the staging loop's pace, through LDS it does not
need.

- [ ] Measure first: add the K=1 shapes at T = 16/32/64 to
      `bench/micro/conv7.hip` (or a sibling) and get the GB/s. If the GEMMs
      already stream near the matvec's ~700 GB/s, this item is closed.
- [ ] If not: a T ≤ 32 path that reads the weight row straight from global
      memory into registers (the frame's matvec shape, `e8872cd`) with the
      activations in LDS — same per-output order, bit-exact.
- **Gate:** `tests/test_prefill.cpp` green, WAV sha unchanged.
- **Expected:** TTFA ~30 ms if prefill halves again. RTF moves ~0.002.

## 3. Contention RTF — investigation, no expected gain yet

The production card is shared. Quiet RTF is 0.099; under a 745 GB/s hog it is
0.30, 3× the quiet number and 2.5× the ≤0.12 target. The target was written
for the quiet case and the hog is deliberately brutal, but nothing has
measured the *real* mix — `llama-server` + `embedding-server` at their normal
load, not `tests/gpu_hog.hip` at 100 %.

- [ ] Measure RTF against the actual resident neighbours (scrape `/metrics`
      over a normal day rather than running a synthetic hog).
- [ ] If the real number is well under 0.12, say so in `docs/DESIGN.md` §7 and
      close this out — the hog row stays as the worst case, not the target.
- [ ] If it is not, the levers are blocks/CU (1/CU today, from the fork's
      contention finding, `docs/DESIGN.md` §2) and the barrier spin cap.
- **Gate:** zero barrier timeouts throughout; a day of journal (§6.4).

## 4. Barrier spin cap from p99 — robustness, not speed

The cap is a constant today (`frame.hip:70`, `B.spin_cap = opt.spin_cap`).

- [ ] Take the barrier-wait p99 from `/metrics` over the soak, set the cap from
      it with headroom, and record the number and its source.
- **Gate:** no timeout on a quiet card, still bounded under the hog; the
  runaway test (`tests/runaway.sh`) unchanged.

## M4 — not next, but the shape is known

- **int8 code-predictor weights.** The only real lever left on the frame
  itself: 2.4 of the 3.3 GB streamed per frame is the code predictor, streamed
  15×. Halving it puts the frame near ~4 ms and RTF near ~0.073. **Not
  bit-exact** — a model change, needs the listen test, a ≥60 dB SNR parity
  number (§6.3) and a new reference sha. Re-measure on gfx1151 too: the XTX
  ruled quantised GEMV out for the talker, Strix may not.
- **On-device frame loop.** Host tail ~0.1 ms/frame plus TTFA jitter; worth
  ~0.001 RTF, mostly a jitter and a tidiness win.
- **ICL cloning (tokenizer encoder).** A quality lever, not a perf one
  (§9, OPEN). Listen test first.
- **1.7B.** Stretch test at M3, not a target.

## Decisions this list is waiting on

- **Text-derived frame cap.** ~15 % of seeds on one- or two-token prompts run
  to the 600-frame cap because the model never emits EOS. Bounded today, not
  fixed. A cap derived from the text length is a §2 decision and needs the
  user's call before code depends on it.

## Explicitly not on this list

Ruled out with measurements in `docs/tuning.md`: balanced contiguous row
partition, fused q/k/v matvec, norm/rope folded into attend (also not
bit-exact), R8 and R1-U8 matvec shapes. A matvec phase ends when the memory
system finishes, not when the last wave does — row assignment is not the lever.
