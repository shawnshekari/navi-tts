# deploy/

Units and helpers for running navi-tts as a **systemd user service**, the way
it runs in production. Paths in the units assume the repo at `~/src/navi-tts`
and the model blob under `models/` (`%h`-relative); adjust `ExecStart` if yours
differ.

## navi-tts.service (the engine)

    cp deploy/navi-tts.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now navi-tts.service

Then: `curl -s localhost:8080/v1/audio/voices` should list `default` plus
whatever the voices dir holds; `journalctl --user -u navi-tts` shows the
start line with the model id and voice count.

Things in the unit that are load-bearing - read before trimming:

- **`Before=llama-server.service embedding-server.service`** - VRAM ordering.
  The engine needs its ~2.7 GiB claimed before an LLM fills the card, or it
  gets pushed into GTT and runs at half speed. Ordering only, not a
  `Requires=`: navi-tts works with or without those services. On a host with
  different GPU services, give them the same treatment or start navi-tts
  first.
- **`Environment=LD_LIBRARY_PATH=...`** - points at the TheRock install's
  libs; systemd does not source your shell profile, so without it the binary
  picks up whatever `libamdhip64` the ld cache knows.
- **`Environment=GPU_MAX_HW_QUEUES=1`** - matches the queue cap the other GPU
  services use; without it, concurrent load oversubscribed the card's HW
  queues and faulted the driver under the old engine (2026-09-12; the
  unit file's comment tells the short version).
- **`--xtts-port 8020`, `--max-frames 600`** - SkyrimNet's expected port and
  the per-request frame cap. `--host 0.0.0.0` exposes the (unauthenticated)
  API to the LAN; drop it to stay loopback-only.

For a headless box, keep the unit alive across logout: `loginctl
enable-linger $USER`.

A fresh install can populate the store with the bundled clip:
`deploy/import_voices.sh` registers `voice_1` from `voices/voice_1.wav`
plus any Skyrim samples cached under `~/.cache/skyrimnet-xtts-shim/samples`.
Migrating from the old fork-era engine instead, point it at the queue repo's
clips first: `TTS_QUEUE_VOICES=<queue voices dir> deploy/import_voices.sh`.

## navi-tts-metrics-textfile.service (optional, needs Prometheus)

Copies `GET /metrics` into the node_exporter textfile dir every 5 s, so the
engine can stay loopback-only while Prometheus scrapes it via node_exporter.

    cp deploy/navi-tts-metrics-textfile.service ~/.config/systemd/user/
    systemctl --user daemon-reload && systemctl --user enable --now navi-tts-metrics-textfile.service

The unit writes to `~/.local/share/node_exporter/textfile`; if your
node_exporter runs as a system service, make sure its textfile collector
directory is the same path and is writable by your user session, or point
`ExecStart` at a directory that is. Disable the unit - or just don't install
it - if you are not running Prometheus.
