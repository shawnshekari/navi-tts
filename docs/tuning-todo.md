# Tuning TODO

The next steps after M2's first pass (`docs/tuning.md` is the record of what
already moved and what did not). Ordered by payoff per hour, bit-exact work
first. Numbers are the XTX at `c01b791`: frame 6.13 ms, vocoder 0.62 ms/frame
streaming, prefill 7.0 ms, TTFA 34 ms, RTF 0.086, WAV sha `4cb7232a…`.

Per-frame budget for reference: 83.3 ms of audio at 12 Hz costs ~7.3 ms wall —
frame 6.13, vocoder 0.61, prefill amortised 0.1, host tail ~0.1. The frame
kernel is now 88 % of it.

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

## 2. Prefill — done 2026-09-18, `e64cc29`

Predicted TTFA ~30 from a matvec-style GEMM; got 35. The measure-first step
found the prefill GEMMs at 60-130 GB/s and a 10-row prompt (the text rides
in as trailing rows). `k_gemm_skinny` (lane per prompt row, f32 weight
tile, gate as its own pass) took the GEMMs 11.7 → 5.5 ms, prefill 14.2 →
7.7, TTFA 41 → 35, bit-exact (12/12 WAVs, sha). It does not reach the
matvec's bandwidth and the record says why: the bit-exact rule leaves a
10-row GEMM with 1-2 waves per SIMD of dependent chains stalling on LDS;
four mappings and every latency-hiding trick land at 150-240 GB/s
(`docs/tuning.md`, `bench/micro/gemm_skinny.hip`).

- [x] Measure first (T = 10, shapes, GB/s), variants memcmp'd against `k_conv`.
- [x] Production kernel, gates, 12 seed × text `cmp`, bench rows.
- [x] Fuse q/k/v and gate/up — `c01b791`: `DeviceWeights::upload` places
      named groups adjacently in the arena, the talker checks and falls back
      per layer. Prefill 7.7 → 7.0, TTFA 35 → 34. Fallback path exercised
      (groups disabled: identical bytes).
- [ ] *(follow-up, small)* `EPI_SCALE_RESIDUAL` / `EPI_GELU` on
      `k_gemm_skinny` so the vocoder transformer's T = 4/16 GEMMs use it.

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

- **int8 weights.** Measured 2026-09-18 (`bench/micro/matvec.hip`, frame-8
  profile): the frame is 6.0 ms of which matvec phases are 4.7 (code
  predictor 3.5, talker 1.2). int8 with block-32 scales runs the matvec at
  0.55× f16's time, not 0.5× — the convert per byte makes it VALU-bound
  unless activations go int8 too (v_dot4). Projection: **cp only → frame
  ~4.5 ms, RTF ~0.066 (−23 %); cp + talker → ~3.9 / ~0.059 (−30 %)**;
  ceiling ~0.056. TTFA barely moves. VRAM 1.96 → ~1.2 GB. Cost: converter
  quantiser + `.navi` dtype, int8 matvec in the frame kernel (every phase),
  a quality gate — §6.3's ≥60 dB SNR is an f16 gate that int8 will not
  meet, so a listen test and a new number — and a new reference sha.
  Worth more on Strix (bandwidth-bound harder) than on the XTX; on the XTX
  it is headroom under contention, not felt latency. Decide after item 3.
- **On-device frame loop.** Host tail ~0.1 ms/frame plus TTFA jitter; worth
  ~0.001 RTF, mostly a jitter and a tidiness win.
- **ICL cloning (tokenizer encoder).** A quality lever, not a perf one
  (§9, OPEN). Listen test first.
- **1.7B.** Stretch test at M3, not a target.

## Strix Halo (M3) — first numbers 2026-09-18, `403af27`

Frame 20.1 ms, prefill 16.5, vocoder 1.27 ms/frame, TTFA 98 ms, RTF 0.271,
sha `4cb7232a…` (= the XTX). Everything is 2.5-3.3× the XTX, the bandwidth
ratio, so the levers are the same ones in the same order — and int8 (M4)
is worth proportionally more here. Not yet done: fat binary, a service
unit, the grid/spin parameters from data (the frame kernel runs 20 blocks
at occupancy 3; the spin cap is the XTX's), `libstdc++-devel` on the host
(`docs/reference/toolchain.md`).

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
