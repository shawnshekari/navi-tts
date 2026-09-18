# Tuning record

What moved the numbers after M2's cutover, what did not, and why. Every step
kept the bench WAV's sha256 (`4cb7232a…`): all of this is arithmetic-order-
preserving. Numbers are `navi-tts bench` on the XTX, streaming configuration,
`bench/results.jsonl` has the rows.

| step | commit | frame ms | TTFA ms | RTF |
|---|---|---|---|---|
| M2 baseline | `2d2286a` | 8.33 | 102 | 0.129 |
| vocoder batching (4 first, then 16; max for whole-body) | `faabcaa` | 8.31 | 66 | 0.125 |
| sampler: radix select + draw over candidates | `0b25a25` | 7.56 | 63 | 0.116 |
| matvec 2 rows × 4 loads in flight per wave | `e8872cd` | 6.46 | 59 | 0.102 |
| barrier on one monotonic counter | `cf18dfd` | 6.24 | 58 | 0.099 |

Whole-body requests (the queue) sit ~0.004 below the streaming RTF.

## How the frame was measured

- `rocprofv3 --kernel-trace --stats` on a `synth` run: the frame kernel is 81 %
  of GPU time, the vocoder's `k_conv` 17 %, everything else 1.4 %.
- `-DNAVI_FRAME_PROFILE` (a separate build dir): block 0 timestamps every grid
  barrier; `Frame` prints a per-phase table for frame 8. Per frame, before the
  barrier change: matvec phases 4.67 ms for ~3.3 GB of weights (707 GB/s),
  barrier waits 0.95 ms over 700 barriers, attention 0.35, sampling 0.22,
  norm/rope/embed 0.11.
- `bench/micro/matvec.hip`: the frame's matvec at the real grid from a 500 MB
  rotation (the 96 MB Infinity Cache flatters anything smaller). `bench/micro/
  sampling.hip`: the sort/draw pieces.

A frame streams ~3.3 GB (talker 0.89, code predictor 15 × 157 MB): at 6.24 ms
that is ~530 GB/s end to end, ~707 inside the matvec phases. The card sustains
~745–880 with a plain streaming kernel.

## What did not help (measured, reverted)

- **Balanced contiguous row partition** and a **fused q/k/v matvec**: block 0's
  phases shortened, its barrier waits lengthened by the same amount. A matvec
  phase ends when the memory system finishes, not when the last wave does;
  row assignment is not the lever.
- **norm/rope folded into attend** (one barrier per layer fewer): slower - the
  q-norm, k-norm, v and score latency chains serialised inside 16 blocks cost
  more than the barrier that let 32 blocks run them in parallel - and not
  bit-exact.
- **More rows per wave (R8)** and **one row with 8 loads (R1 U8)**: both slower
  than R2 U4 in the microbench.
- On RDNA3 `clock64()` does not count shader cycles; spin loops use
  `wall_clock64()` (the hog learned this the hard way).

## What is left

- **Vocoder k=7 convs** (`k_conv<K=7>`, 40 of the vocoder's 90 ms per 70
  frames): the kernel is LDS-bound - a 4×2 thread tile does 6 LDS loads per
  8 FMAs. Register blocking (8×4 or 8×8) with the same per-output summation
  order is bit-exact; ~5 % of RTF.
- **Prefill** (25.7 ms, 44 % of TTFA): its GEMMs run through the same conv
  kernel at ~60 GB/s. A matvec-style skinny GEMM would take TTFA to ~46 ms.
  RTF unaffected.
- **int8 code-predictor weights**: the only lever left on the frame itself
  (2.4 of the 3.3 GB per frame is the code predictor, streamed 15×). A model
  change - not bit-exact, needs the listen test and a parity number. M4.
- **On-device frame loop** (host tail ~0.1 ms/frame, TTFA jitter): M4.
- The model's own no-EOS behaviour on one-token prompts (~15 % of seeds run
  to the 600-frame cap) is bounded, not fixed; a text-derived cap is a
  DESIGN §2 decision.
