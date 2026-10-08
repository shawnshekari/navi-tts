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
(VRAM first) and `GPU_MAX_HW_QUEUES=1`. Through the cutover it also carried
`Conflicts=` on the three retired units, so two engines could never hold :8080
at once (that happened during the stand-in test: starting `tts-queue` pulled
`tts-engine` back in through its `Requires=`). The `Conflicts=` line was dropped
on 2026-09-19 with the cleanup below - there are no units left to conflict with.

## Sequence

1. `deploy/import_voices.sh` - fills the store offline: `voice_1`, `voice_2`
   from the queue, the Skyrim voice types from the shim's cache. Safe to run
   while the old engine serves; re-runs are no-ops.
2. `cp deploy/navi-tts.service deploy/navi-tts-metrics-textfile.service
   ~/.config/systemd/user/ && systemctl --user daemon-reload`
3. The queue repo: `tts-queue.service` depends on `navi-tts.service`; the Makefile's
   `start`/`stop`/`status`/`install` name `navi-tts tts-queue`; `tts-toggle`
   follows the Makefile. Install the unit, `daemon-reload`.
4. `systemctl --user stop tts-queue skyrimnet-xtts-shim tts-register-voices tts-engine`
   `systemctl --user disable tts-engine skyrimnet-xtts-shim tts-register-voices`
   `systemctl --user enable --now navi-tts navi-tts-metrics-textfile`
   `systemctl --user start tts-queue`
5. Verify: `curl :8080/v1/audio/voices` lists `voice_1`, `voice_2` and the
   Skyrim names; one queue request (`tts-speak`); `curl :8020/speakers`;
   `~/.local/share/node_exporter/textfile/navi_tts.prom` exists.

## Cleanup (2026-09-19)

The three retired units were removed from `~/.config/systemd/user` once navi-tts
had been live for two days. All three were already `disabled` and `inactive`;
nothing outside prose referenced them (`tts-toggle` -> the queue's Makefile ->
`navi-tts tts-queue`, and `tts-queue.service` already `Requires=navi-tts.service`).
Each was archived beside its source first:

| Unit | Archived to |
|---|---|
| `tts-engine.service` | archived in the upstream fork's repo |
| `skyrimnet-xtts-shim.service` | archived in the queue's repo (integrations/skyrimnet/) |
| `tts-register-voices.service` | archived in the queue's repo |

Each archived copy carries a `# RETIRED` header saying where it ran and why it
stopped. Six stale `*.service.*.bak` / `.pre-metrics` files (embedding-server,
llama-server) went at the same time; systemd ignored them, but they were the
clutter.

Rollback: `systemctl --user stop navi-tts`, `cp` the archived
`tts-engine.service` (and `tts-register-voices.service` if the queue needs it)
back to `~/.config/systemd/user/`, `daemon-reload`, `start tts-engine
tts-register-voices tts-queue`. Stopping navi-tts is now a manual step - the
`Conflicts=` that used to do it automatically is gone.

## Voices

The id is the registered name. The queue's `~/.local/share/tts/voice_id_map.json`
already maps `voice_1 -> voice_1`, `voice_2 -> voice_2`, and its
`ensure_voices()` only registers what `/v1/audio/voices` lacks, so a boot needs
no registration step. SkyrimNet re-uploads a sample on first use of a voice
type per session; with the same bytes that is a no-op ("exists").
