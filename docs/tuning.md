# Tuning record

What moved the numbers after M2's cutover, what did not, and why. Every step
kept the bench WAV's sha256 (`4cb7232a…`): all of this is arithmetic-order-
preserving. Numbers are `navi-tts bench` on the XTX, streaming configuration,
`bench/results.jsonl` has the rows.

| step | commit | frame ms | TTFA ms | RTF |
|---|---|---|---|---|
| M2 baseline | `1bb04b9` | 8.33 | 102 | 0.129 |
| vocoder batching (4 first, then 16; max for whole-body) | `7e9364b` | 8.31 | 66 | 0.125 |
| sampler: radix select + draw over candidates | `c3dac3d` | 7.56 | 63 | 0.116 |
| matvec 2 rows × 4 loads in flight per wave | `c8fe709` | 6.46 | 59 | 0.102 |
| barrier on one monotonic counter | `016acb3` | 6.24 | 58 | 0.099 |
| conv kernel: conflict-free weight tile, vector staging, static dilation | `90c88b8` | 6.13 | 41 | 0.088 |
| prefill: skinny GEMM, lane per prompt row | `52fcd7d` | 6.13 | 35 | 0.086 |
| prefill: q/k/v and gate/up as one GEMM each | `86f6372` | 6.13 | 34 | 0.086 |

Whole-body requests (the queue) sit ~0.004 below the streaming RTF.

The conv change moved everything but the frame: vocoder 1.30 -> 0.62 ms/frame,
prefill 25.7 -> 14.2 ms. One kernel serves the vocoder's convs and GEMMs, the
talker prefill and the speaker encoder, so a fix aimed at the k=7 convs
landed on all of them.

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
- **Register-blocking the conv kernel (8×4, 8×8, 16×4 thread tiles)**: the
  premise was wrong. The kernel was not LDS-bandwidth-bound; its weight loads
  were 16-way bank-conflicted (56-dword lane stride), its fully unrolled 16×7
  inner loop cost 154 VGPRs for 8 accumulators, and its staging spent more
  instructions than its FMAs. Every wider tile spilled and ran slower. With
  those three fixed the 4×2 tile is within noise of the wider ones; a
  64×128 8×4 tile wins only on dilation 9 at 96-192 channels (~0.03 ms/frame,
  not worth a second instantiation). `bench/micro/conv7.hip`.
- **Unrolling the conv's channel-chunk loop** at all: u1 < u2 < u4 < u16, every
  step. The unrolled loads hoist into registers and occupancy pays.
- **Streaming the prefill GEMMs at the matvec's ~700 GB/s.** Not reachable
  under the bit-exact rule at T = 10: one lane must own each (t, co) dot
  product end to end, so a GEMM has Cout/R waves of dependent chains - 1-2
  per SIMD - and they stall on LDS reads (273 `s_waitcnt` per 2952 ISA
  instructions). Four mappings (lane per row with f16 or f32 tiles, lane per
  channel, quad-coalesced with DPP exchange), deeper prefetch, register
  double-buffering and occupancy hints all sit at 150-240 GB/s where pure
  streams do 450-730. Fusing q/k/v and gate/up measured ~15 % on top.
  `bench/micro/gemm_skinny.hip`.

## What is left

Ordered, with gates and expected numbers: `docs/tuning-todo.md`.

- **Vocoder**, now 0.62 ms/frame: the k=7 group is ~19 of the ~43 ms per 70
  frames and runs at 13-19 TFLOP/s on the wide layers, roughly half of the
  card's single-issue f32 rate. What is left there is VALU issue (the f16
  unpack per tap, ~25 % of the loop) and the two `__syncthreads` per 16-
  channel chunk. ~2 % of RTF at most; not next.
- **Prefill** (7.0 ms, 21 % of TTFA): the GEMMs are ~4.8 of it at 150-240
  GB/s, the ceiling for a bit-exact 10-row GEMM (above); q/k/v and gate/up
  are fused (the arena places named groups adjacently). Left: the non-GEMM
  ~2 ms (norms, rope, attention, 115 launches). TTFA levers only; RTF
  unaffected.
- **Vocoder transformer K=1 GEMMs** at T = 4 and 16 still go through
  `k_conv`; `k_gemm_skinny` needs `EPI_SCALE_RESIDUAL` and `EPI_GELU` to take
  them. Small, but on the TTFA path (the first 4-frame batch).
- **int8 code-predictor weights**: the only lever left on the frame itself
  (2.4 of the 3.3 GB per frame is the code predictor, streamed 15×). A model
  change - not bit-exact, needs the listen test and a parity number. M4.
- **On-device frame loop** (host tail ~0.1 ms/frame, TTFA jitter): M4.
- The model's own no-EOS behaviour on one-token prompts (~15 % of seeds run
  to the 600-frame cap) is bounded, not fixed; a text-derived cap is a
  DESIGN §2 decision.
