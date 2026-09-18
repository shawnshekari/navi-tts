#!/usr/bin/env bash
# Copy navi-tts's /metrics into the node_exporter textfile dir every 5 s. The
# engine stays loopback-only; Prometheus on the mini PC sees it through
# node_exporter (DESIGN 3.1). Same pattern as llamacpp_slots_textfile.sh.
#   usage: navi_tts_metrics_textfile.sh <textfile-dir> [<port>]
set -u
DIR=${1:?textfile dir}
PORT=${2:-8080}
mkdir -p "$DIR"
while :; do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/metrics" > "$DIR/navi_tts.prom.$$"; then
    mv -f "$DIR/navi_tts.prom.$$" "$DIR/navi_tts.prom"
  else
    rm -f "$DIR/navi_tts.prom.$$" "$DIR/navi_tts.prom"   # engine down: no stale series
  fi
  sleep 5
done
