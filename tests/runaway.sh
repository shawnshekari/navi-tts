#!/usr/bin/env bash
# Runaway test (DESIGN 2, 10 M2): against a running navi-tts, never production.
# Prose must end in EOS (the model's no-EOS rate on real sentences is ~0; more
# than 10% is a regression), degenerate inputs ("a", "...", "Hello.") are
# allowed to run to the cap - the model never emits EOS once it drifts into
# silence codes, ~15% of seeds on a two-token prompt - and are reported as a
# rate; the cap must bound them and be honoured exactly when set; a repeated
# request must be bit-exact; parallel clients must all be served; a 4096-char
# input must complete. Failures must be clean HTTP statuses; the server must
# be healthy at the end.
#   usage: tests/runaway.sh [port=8090] [rounds=6]
set -u
PORT=${1:-8090}
ROUNDS=${2:-6}
URL="http://127.0.0.1:$PORT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
say() { printf '%s\n' "$*"; }
bad() { say "FAIL: $*"; fail=$((fail + 1)); }

metric() { curl -sf "$URL/metrics" | awk -v m="$1" '$1 == m { print $2 }'; }
speak() {  # text voice seed extra_json out
    curl -s -o "$5" -w '%{http_code}' -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' \
        -d "{\"input\":$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"voice\":\"$2\",\"seed\":$3$4}"
}
frames_of() { local b; b=$(stat -c %s "$1"); echo $(( (b - 44) / 2 / 1920 )); }

curl -sf "$URL/health" >/dev/null || { say "no server on $URL"; exit 2; }
mapfile -t VOICES < <(curl -sf "$URL/v1/audio/voices" | python3 -c 'import json,sys; [print(v) for v in list(json.load(sys.stdin).values())[0] if v != "default"]')
[ ${#VOICES[@]} -gt 0 ] || { say "no voices registered"; exit 2; }

PROSE=(
  "The quick brown fox jumps over the lazy dog, and the bench text stays fixed."
  "Wait... what? No! I said, \"turn left at the second junction,\" not the first; honestly."
  "In 1847, 3,200 settlers crossed 1,100 miles in 142 days, averaging 7.7 miles per day at 4 mph."
  "Skyrim belongs to the Nords! I have been asleep for a very long time. Who are you, and why have you come here?"
  "This sentence is here to be long enough to cross several vocoder batches and it keeps going with more clauses, more commas, and more words until it finally, eventually, comes to an end."
)
DEGENERATE=("Hello." "a" "..." "Yes.")
CAP=600

err0=$(metric 'tts:errors_total{stage="generate"}')
n=0; seed=100; prose_cap=0; degen_cap=0; degen_n=0
say "-- $ROUNDS rounds x ${#PROSE[@]} prose + ${#DEGENERATE[@]} degenerate texts over ${#VOICES[@]} voice(s)"
for r in $(seq 1 "$ROUNDS"); do
  for t in "${PROSE[@]}" "${DEGENERATE[@]}"; do
    v=${VOICES[$((n % ${#VOICES[@]}))]}
    seed=$((seed + 1)); n=$((n + 1))
    code=$(speak "$t" "$v" "$seed" "" "$TMP/o.wav")
    if [ "$code" != 200 ]; then bad "HTTP $code for seed $seed voice $v text '$t'"; continue; fi
    f=$(frames_of "$TMP/o.wav")
    chars=${#t}
    [ "$f" -ge 1 ] || bad "empty audio for '$t' (seed $seed)"
    [ "$f" -le $CAP ] || bad "$f frames exceeds the cap $CAP"
    if [ "$chars" -lt 8 ]; then
      degen_n=$((degen_n + 1)); [ "$f" -lt $CAP ] || degen_cap=$((degen_cap + 1))
    else
      # 12.5 frames/s; prose runs ~2-3 frames per word, numbers read out run longer
      if [ "$f" -ge $CAP ]; then prose_cap=$((prose_cap + 1)); bad "no EOS: '$t' (seed $seed, voice $v)"
      elif [ "$f" -gt $((chars * 3 + 24)) ]; then bad "runaway? $f frames for $chars chars (seed $seed, voice $v): '$t'"; fi
    fi
  done
done
err1=$(metric 'tts:errors_total{stage="generate"}')
prose_n=$((n - degen_n))
say "   $n requests; prose no-EOS $prose_cap/$prose_n, degenerate no-EOS $degen_cap/$degen_n (bounded by the cap), generate errors $((err1 - err0))"
[ $((prose_cap * 10)) -le $prose_n ] || bad "prose no-EOS rate over 10%"
[ $((err1 - err0)) -eq 0 ] || bad "$((err1 - err0)) request(s) failed in generate"
cap1=$(metric 'tts:frame_cap_hits_total')

say "-- cap honoured exactly (max_audio_tokens 8)"
code=$(speak "${PROSE[4]}" "${VOICES[0]}" 7 ',"max_audio_tokens":8' "$TMP/cap.wav")
[ "$code" = 200 ] || bad "cap request HTTP $code"
[ "$(frames_of "$TMP/cap.wav")" -eq 8 ] || bad "cap: $(frames_of "$TMP/cap.wav") frames, expected 8"
cap2=$(metric 'tts:frame_cap_hits_total'); [ $((cap2 - cap1)) -eq 1 ] || bad "cap hit not counted"

say "-- bit-exact repeat"
speak "${PROSE[0]}" "${VOICES[0]}" 2 "" "$TMP/a.wav" >/dev/null
speak "${PROSE[0]}" "${VOICES[0]}" 2 "" "$TMP/b.wav" >/dev/null
cmp -s "$TMP/a.wav" "$TMP/b.wav" || bad "same text + seed differs run to run"

say "-- 4 parallel clients"
for i in 1 2 3 4; do speak "${PROSE[3]}" "${VOICES[$(( (i - 1) % ${#VOICES[@]} ))]}" $((900 + i)) "" "$TMP/p$i.wav" > "$TMP/p$i.code" & done; wait
for i in 1 2 3 4; do [ "$(cat "$TMP/p$i.code")" = 200 ] || bad "parallel client $i: HTTP $(cat "$TMP/p$i.code")"; done

say "-- 4096-char input, cap 600"
long=$(python3 -c 'print(("The quick brown fox jumps over the lazy dog. " * 100)[:4096])')
t0=$(date +%s.%N)
code=$(speak "$long" "${VOICES[0]}" 3 "" "$TMP/long.wav")
t1=$(date +%s.%N)
if [ "$code" = 200 ]; then say "   $(frames_of "$TMP/long.wav") frames in $(python3 -c "print(round($t1-$t0,1))") s"; else bad "4096-char input: HTTP $code"; fi
code=$(speak "$long." "${VOICES[0]}" 3 "" "$TMP/toolong.wav"); [ "$code" = 400 ] || bad "4097 chars accepted (HTTP $code)"

say "-- bad requests are statuses, not hangs"
for body in '{}' '{"input":""}' '{"input":"x","voice":"nobody"}' '{"input":"x","max_audio_tokens":0}' 'not json'; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST "$URL/v1/audio/speech" -H 'Content-Type: application/json' -d "$body")
  [ "$code" = 400 ] || bad "'$body' -> HTTP $code"
done

curl -sf "$URL/health" >/dev/null || bad "server unhealthy at the end"
say "-- $([ $fail -eq 0 ] && echo PASS || echo "FAIL ($fail)")"
exit $((fail > 0))
