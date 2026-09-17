#!/bin/bash
# usage: bench-hip.sh <label> <binary> [ENV=VAL ...]   (runs inside navi31-llama)
# Warm-up with a 1-word request (tiny vocoder decode), then the full text; the
# server is killed as soon as the 2nd timing block prints, so the pathologically
# slow HIP vocoder never runs on the long text.
set -u
LABEL=$1; BIN=$2; shift 2
S=${BENCH_OUT:-/tmp/qwen3-tts-bench}; mkdir -p "$S"; LOG=$S/$LABEL.log
MODEL=/home/sreed/tools/qwen3-tts.cpp/models/qwen3-tts-0.6b-f16.gguf
TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you, the whole thing fell apart."

# VRAM guard: a test instance that spills into GTT measures 2x slow, so
# refuse to run when the card is nearly full (BENCH_FORCE=1 overrides).
vram() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1e9}'; }
VRAM_TOTAL=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1e9}')
echo "VRAM before: $(vram) / $VRAM_TOTAL GB"
if [ -z "${BENCH_FORCE:-}" ] && awk -v u="$(vram)" -v t="$VRAM_TOTAL" 'BEGIN{exit !(t - u < 5)}'; then
  echo "less than 5 GB of VRAM free; stop the other GPU services first (or BENCH_FORCE=1)" >&2; exit 1
fi
pkill -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server' 2>/dev/null; sleep 1
# --max-tokens bounds a runaway generation; the watchdog below is the backstop.
toolbox run --container navi31-llama env "$@" "$BIN" -m "$MODEL" -p 8081 -V --seed 42 --max-tokens 400 >"$LOG" 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8081/v1/audio/voices >/dev/null 2>&1 && break; sleep 0.5; done
VID=$(curl -s -X POST localhost:8081/v1/audio/voices -F name=nyx -F audio_sample=@/home/sreed/tools/reference_voices/nyx_reference.wav | jq -r .id)
req() { curl -s -m 300 -X POST localhost:8081/v1/audio/speech -H 'Content-Type: application/json' \
    -d "{\"input\": $(jq -Rn --arg t "$1" '$t'), \"voice\": \"$VID\", \"seed\": 42}" --output "$S/$LABEL.$2.wav"; }
kill_server() { pkill -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server'; }
# Warm-up on a 1-word text, then the real text. Watchdog: the HIP vocoder is
# ~100x slower than Vulkan and hogs the GPU hard enough to stall the desktop,
# so the server is killed the moment the 2nd request's codes are out, and
# earlier if a request runs past 300 frames or the whole thing exceeds 120 s.
wait_codes() { # <n>: wait until n requests have produced codes; 1 = abort
  local deadline=$((SECONDS + 120))
  while [ $SECONDS -lt $deadline ]; do
    [ "$(grep -c 'Speech codes generated' "$LOG")" -ge "$1" ] && return 0
    if grep -q 'decode: frame 300/' "$LOG"; then echo "runaway generation (>300 frames), killing server" >&2; return 1; fi
    sleep 0.2
  done
  echo "timeout, killing server" >&2; return 1
}
req "Hi." 1 &
req_pid=$!
if wait_codes 1; then
  wait $req_pid  # let the ~10-frame warm-up decode finish (~4 s on HIP)
  req "$TEXT" 2 &
  wait_codes 2
fi
for p in $(pgrep -f '^/home/sreed/tools/qwen3-tts.cpp/build[^ ]*/qwen3-tts-server'); do
  cat /proc/$p/fdinfo/* 2>/dev/null | awk -v p=$p '/drm-resident-vram/{v+=$2} /drm-resident-gtt/{g+=$2} END{if(v||g) printf "residency(pid %d): VRAM %.2f GB  GTT %.2f GB\n", p, v/1e6, g/1e6}'
done
kill_server; wait 2>/dev/null
echo "== $LABEL =="
grep -E "Speech codes generated" "$LOG"
awk '/Talker forward_step/{n++} n==2' "$LOG" | grep -E "Total:|Steps|Compute|Data I/O|Graph alloc|Graph build|Embed|Throughput|Speech codes" | head -16
echo "VRAM after: $(vram) / $VRAM_TOTAL GB"
