#!/usr/bin/env bash
# Contention test (DESIGN 2, CLAUDE.md): requests while tests/gpu_hog.hip
# fills every CU. A request must either complete (slower) or fail with a
# clean engine_error inside the bound; the server must never hang and must
# be back at normal RTF once the hog exits. Runs briefly, on purpose, against
# a side-port navi-tts, never production.
#   usage: tests/contention.sh [port=8090] [hog_seconds=20] [blocks_per_cu=4] [spin|stream=stream]
set -u
PORT=${1:-8090}
HOG_S=${2:-20}
PER_CU=${3:-4}
MODE=${4:-stream}
URL="http://127.0.0.1:$PORT"
HERE="$(dirname "$(readlink -f "$0")")"
HOG="$HERE/../build/xtx/gpu_hog"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
say() { printf '%s\n' "$*"; }
bad() { say "FAIL: $*"; fail=$((fail + 1)); }
[ -x "$HOG" ] || { say "build gpu_hog first (ninja gpu_hog)"; exit 2; }
curl -sf "$URL/health" >/dev/null || { say "no server on $URL"; exit 2; }
VOICE=$(curl -sf "$URL/v1/audio/voices" | python3 -c 'import json,sys; print([v for v in list(json.load(sys.stdin).values())[0] if v != "default"][0])')
TEXT="The quick brown fox jumps over the lazy dog, and the bench text stays fixed."
BOUND=30   # seconds; a frame kernel has a 0.5 s spin cap, so a 600-frame request cannot legitimately take this long

# SSE, so the done event carries rtf / frames / eos per request
one() {  # seed out -> "status rtf frames eos seconds"
    local t0 t1 code
    t0=$(date +%s.%N)
    code=$(curl -s -N --max-time $BOUND -o "$2" -w '%{http_code}' -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
        -d "{\"input\":\"$TEXT\",\"voice\":\"$VOICE\",\"seed\":$1,\"stream_format\":\"sse\"}")
    t1=$(date +%s.%N)
    python3 - "$2" "$code" "$t0" "$t1" <<'PY'
import json, sys
path, code, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rtf = frames = eos = "-"; err = ""
try:
    for ln in open(path):
        if ln.startswith("data:"):
            ev = json.loads(ln[5:])
            if ev.get("type") == "speech.audio.done":
                n = ev["navi"]; rtf, frames, eos = f'{n["rtf"]:.3f}', n["frames"], n["eos"]
            elif ev.get("type") == "error":
                err = ev.get("error", "")
    if code != "200" and not err:
        err = json.load(open(path)).get("error", {}).get("message", "")
except Exception:
    pass
print(code, rtf, frames, eos, f"{t1 - t0:.2f}", err)
PY
}

say "-- baseline (quiet card)"
read -r code rtf frames eos secs err < <(one 1 "$TMP/base.sse")
say "   HTTP $code rtf $rtf frames $frames eos $eos in ${secs}s"
[ "$code" = 200 ] || bad "baseline failed: $err"
base_rtf=$rtf

say "-- hog ($MODE): $PER_CU blocks/CU for ${HOG_S}s; requests back to back meanwhile"
"$HOG" "$HOG_S" "$PER_CU" "$MODE" 2> "$TMP/hog.log" &
hogpid=$!
sleep 1
ok=0; failed=0; hung=0; n=0; worst=0
while kill -0 $hogpid 2>/dev/null; do
  n=$((n + 1))
  read -r code rtf frames eos secs err < <(one $((10 + n)) "$TMP/h$n.sse")
  say "   #$n HTTP $code rtf $rtf frames $frames eos $eos in ${secs}s ${err:+- $err}"
  if [ "$code" = 200 ]; then ok=$((ok + 1)); worst=$(python3 -c "print(max($worst, $rtf))")
  elif [ "$code" = 000 ]; then hung=$((hung + 1))
  else failed=$((failed + 1)); fi
done
wait $hogpid
cat "$TMP/hog.log" | sed 's/^/   /'
say "   under the hog: $ok ok, $failed failed cleanly, $hung exceeded ${BOUND}s; worst RTF $worst"
[ $hung -eq 0 ] || bad "$hung request(s) did not return within ${BOUND}s"
[ $n -gt 0 ] || bad "no requests ran under the hog"

say "-- after the hog"
sleep 1
read -r code rtf frames eos secs err < <(one 1 "$TMP/after.sse")
say "   HTTP $code rtf $rtf frames $frames eos $eos in ${secs}s"
[ "$code" = 200 ] || bad "post-hog request failed: $err"
cmp -s <(grep delta "$TMP/base.sse") <(grep delta "$TMP/after.sse") || bad "post-hog audio differs from the baseline (same seed)"
python3 -c "import sys; sys.exit(0 if float('$rtf') < float('$base_rtf') * 1.5 else 1)" || bad "post-hog RTF $rtf vs baseline $base_rtf"
curl -sf "$URL/health" >/dev/null || bad "server unhealthy at the end"
say "-- $([ $fail -eq 0 ] && echo PASS || echo "FAIL ($fail)")"
exit $((fail > 0))
