# Tuning TODO

The next steps after M2's first pass (`docs/tuning.md` is the record of what
already moved and what did not). Ordered by payoff per hour, bit-exact work
first. Numbers are the XTX at `cf18dfd`: frame 6.24 ms, vocoder 1.28 ms/frame
streaming, prefill 25.7 ms, TTFA 58 ms, RTF 0.099, WAV sha `4cb7232a…`.

Per-frame budget for reference: 83.3 ms of audio at 12 Hz costs ~8.25 ms wall —
frame 6.24, vocoder 1.28, prefill amortised 0.37, host tail ~0.1.

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

## 1. Vocoder k=7 convs — bit-exact, ≈0.004 RTF

`k_conv<K=7>` is 40 of the vocoder's 90 ms per 70 frames. The kernel
(`model/qwen3tts/kernels.h:54`) is LDS-bound: a 4×2 thread tile does 6 LDS
loads per 8 FMAs.

- [ ] Register-block the inner loop to 8×4 (then try 8×8), keeping the
      per-output summation order — `acc[r][c] += w * xv` over `ci`, then `j` —
      so the result is bit-identical.
- [ ] Watch LDS per block: `xs[WIN][CC]` grows with `(K-1)*MAX_DIL`; an 8×8
      tile at CT=64 may cost occupancy. Measure, don't assume.
- **Gate:** `tests/test_vocoder.cpp` green, bench WAV sha unchanged.
- **Expected:** k=7 group 40 → ~20 ms per 70 frames, RTF 0.099 → ~0.095.

## 2. Prefill GEMMs — bit-exact, TTFA 58 → ~46 ms

Prefill is 25.7 ms, 44 % of TTFA, and its GEMMs run through the same conv
kernel at K=1 (`model/qwen3tts/talker.hip:161`) at ~60 GB/s.

- [ ] Route the K=1 prompt GEMMs through a matvec-style skinny GEMM — the
      2 rows × 4 loads in flight per wave shape that took the frame's matvec to
      707 GB/s (`e8872cd`), widened to the prompt's token count.
- [ ] `bench/micro/matvec.hip` first, at the prefill's real shapes, from a
      buffer larger than the 96 MB Infinity Cache.
- **Gate:** `tests/test_prefill.cpp` green, WAV sha unchanged.
- **Expected:** TTFA ~46 ms. RTF moves ~0.003 (prefill is amortised over the
  request); this one is for the queue's felt latency, not throughput.

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
