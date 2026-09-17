# reference/

Verbatim copies of the author's own work from the frozen fork at
`~/tools/qwen3-tts.cpp` (see `HISTORY.md`). Nothing here is built; it is the
starting material for `model/qwen3tts/` and `runtime/bench/`.

- `hip/` - the fused cooperative talker, code predictor and frame kernels
  (`talker_hip.hip`, `code_pred_hip.hip`, `frame_fused.hip`) and `hip_dev.h`.
  These are the hot loop that reached 8.5 ms/frame; their `.hip` files have no
  ggml dependency. The `.h` interfaces do assume the fork's host-side types and
  will be reshaped, not reused.
- `tests/` - the HIP parity, fallback and self-healing tests, and the GPU hog
  (`gpu_saturate.hip`) used for contention testing.
- `bench/` - the benchmark harness scripts and the `grid.sync()` microbench that
  the design's launch-vs-barrier numbers come from.

Nothing inherited from either upstream is copied here; those repos are MIT and
this project shares no code with them.
