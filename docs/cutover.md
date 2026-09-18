# Cutover: tts-engine -> navi-tts (M2)

What changes on the workstation when `navi-tts.service` replaces the fork.
Every client keeps its configuration: the queue still talks to
`localhost:8080`, SkyrimNet still talks to `localhost:8020`.

## Units

| Before | After |
|---|---|
| `tts-engine.service` (the fork on :8080) | `navi-tts.service` (:8080 OpenAI + XTTS, :8020 XTTS) |
| `skyrimnet-xtts-shim.service` (:8020 -> :8080) | retired: the XTTS dialect is native |
| `tts-register-voices.service` (one-shot after the engine) | retired: voices persist in `~/.local/share/navi-tts/voices` |
| `tts-queue.service` `Requires=tts-engine tts-register-voices` | `Requires=navi-tts.service`, `After=navi-tts.service` |
| - | `navi-tts-metrics-textfile.service`: `/metrics` -> node_exporter |

`navi-tts.service` keeps `Before=llama-server.service embedding-server.service`
(VRAM first) and `GPU_MAX_HW_QUEUES=1`, and adds `Conflicts=` on the three
retired units so two engines can never hold :8080 at once (this happened
during the stand-in test: starting `tts-queue` pulled `tts-engine` back in
through its `Requires=`).

## Sequence

1. `deploy/import_voices.sh` - fills the store offline: `voice_1`, `voice_2`
   from TTS-Player, the Skyrim voice types from the shim's cache. Safe to run
   while the old engine serves; re-runs are no-ops.
2. `cp deploy/navi-tts.service deploy/navi-tts-metrics-textfile.service
   ~/.config/systemd/user/ && systemctl --user daemon-reload`
3. TTS-Player: `tts-queue.service` depends on `navi-tts.service`; the Makefile's
   `start`/`stop`/`status`/`install` name `navi-tts tts-queue`; `tts-toggle`
   follows the Makefile. Install the unit, `daemon-reload`.
4. `systemctl --user stop tts-queue skyrimnet-xtts-shim tts-register-voices tts-engine`
   `systemctl --user disable tts-engine skyrimnet-xtts-shim tts-register-voices`
   `systemctl --user enable --now navi-tts navi-tts-metrics-textfile`
   `systemctl --user start tts-queue`
5. Verify: `curl :8080/v1/audio/voices` lists `voice_1`, `voice_2` and the
   Skyrim names; one queue request (`tts-speak`); `curl :8020/speakers`;
   `~/.local/share/node_exporter/textfile/navi_tts.prom` exists.

Rollback: `systemctl --user start tts-engine` (its `Conflicts=` stops
navi-tts), then `start tts-register-voices tts-queue` with the old unit files.

## Voices

The id is the registered name. The queue's `~/.local/share/tts/voice_id_map.json`
already maps `voice_1 -> voice_1`, `voice_2 -> voice_2`, and its
`ensure_voices()` only registers what `/v1/audio/voices` lacks, so a boot needs
no registration step. SkyrimNet re-uploads a sample on first use of a voice
type per session; with the same bytes that is a no-op ("exists").
