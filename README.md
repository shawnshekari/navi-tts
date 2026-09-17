# navi-tts

**AMD-first text-to-speech inference. RDNA3 (gfx1100) and Strix Halo (gfx1151), HIP only, tuned and measured. Other hardware is out of scope by design.**

A small, focused engine: one binary, no CPU or Vulkan fallback, no framework
underneath. The hot loop is hand-written HIP; the model runs at batch one and is
memory-bandwidth-bound, so the kernels are built around that.

| Target | GPU | ROCm | RTF | Time to first audio |
|---|---|---|---|---|
| Workstation | Radeon RX 7900 XTX (gfx1100, 96 CU) | — | — | — |
| Mini PC | Strix Halo (gfx1151, 40 CU) | — | — | — |

*(Numbers land here from the benchmark harness as milestones are reached.)*

## Scope

- **Model:** Qwen3-TTS architecture (0.6B), own weight format converted offline.
  Fine-tunes of the same architecture are just different weights.
- **API:** OpenAI-compatible `/v1/audio/speech` and `/v1/audio/voices` on `:8080`,
  plus the XTTS dialect SkyrimNet speaks. Cloned voices persist across restarts.
- **Serves:** [TTS-Player](../TTS-Player) (queue, Claude Code / opencode cues) and SkyrimNet.

## Lineage

Written from scratch. The Qwen3-TTS model was first brought to GGML by
[predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp), extended by
[khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp); this project shares
no code with either but owes them the map of the model. `HISTORY.md` records the
discrete-GPU/HIP work that preceded this rewrite.
