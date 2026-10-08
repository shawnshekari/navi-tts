#!/usr/bin/env bash
# One-time import into the persistent voice store (DESIGN 3.2): the queue's
# two hand-made voices under the ids the queue uses, and every Skyrim voice
# type the retired shim had cached. Offline - the server need not run; a
# re-run with unchanged samples is a no-op.
#   usage: import_voices.sh [voices-dir]
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
NAVI="$HERE/../build/xtx/navi-tts"
MODEL="$HERE/../models/qwen3-tts-0.6b-f16.navi"
VOICES="${1:-$HOME/.local/share/navi-tts/voices}"
export LD_LIBRARY_PATH="$HOME/tools/therock-tarball/install/lib:$HOME/tools/therock-tarball/install/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

add() { "$NAVI" voices --voices "$VOICES" add --model "$MODEL" --name "$1" --voice "$2"; }

TP="${TTS_QUEUE_VOICES:-}"
if [ -n "$TP" ] && [ -f "$TP/nyx_reference.wav" ]; then
  add voice_1 "$TP/nyx_reference.wav"
else
  add voice_1 "$HERE/../voices/voice_1.wav"   # bundled reference clip
fi
if [ -n "$TP" ] && [ -f "$TP/amy_reference.wav" ]; then
  add voice_2 "$TP/amy_reference.wav"
fi

SHIM="$HOME/.cache/skyrimnet-xtts-shim/samples"
if [ -d "$SHIM" ]; then
  for f in "$SHIM"/*.wav; do
    [ -e "$f" ] || continue
    add "$(basename "$f" .wav)" "$f"
  done
fi
"$NAVI" voices --voices "$VOICES" list
