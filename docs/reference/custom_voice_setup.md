# Custom voice setup

One clip of speech in, a named voice out - registered with the engine, kept
across restarts, usable by any client.

## 1. Prepare the reference audio

- **5-15 seconds** of clear speech, ideally one speaker, no music, heavy
  reverb or overlaid noise: the speaker encoder embeds the clip, and anything
  that isn't the voice dilutes the embedding.
- **Any sample rate** - the engine resamples to the encoder's 24 kHz
  (windowed-sinc, `runtime/audio/wav.h`).
- A natural, representative sentence works better than a read script; the
  clone carries the clip's prosody.

Cut a clip out of anything with `ffmpeg` (`sudo apt install ffmpeg` /
`sudo dnf install ffmpeg`):

```bash
# 12 seconds starting at 8 s into a podcast mp3, as mono WAV at any rate
ffmpeg -i source.mp3 -ss 8 -t 12 -ac 1 -y /tmp/myref.wav
```

## 2. Register the voice

Against a **running server** (multipart upload; re-registering the same name
with the same sample is a no-op, a new sample replaces):

```bash
curl -s -F "name=myvoice" -F "audio_sample=@/tmp/myref.wav" http://localhost:8080/v1/audio/voices
```

**Offline**, no server needed (`navi-tts voices ...` also takes
`--voices DIR`; the default is `$XDG_DATA_HOME/navi-tts/voices`):

```bash
./build/xtx/navi-tts voices add --model models/qwen3-tts-0.6b-f16.navi --name myvoice --voice /tmp/myref.wav
./build/xtx/navi-tts voices list
```

`--ref-text "the words spoken in the clip"` is stored for the planned ICL
cloning mode (DESIGN 9, OPEN); it does not affect today's embeddings.

## 3. Use and verify

```bash
curl -s http://localhost:8080/v1/audio/voices
curl -s -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-0.6b-f16","input":"Testing my new voice.","voice":"myvoice","seed":2}' \
  --output /tmp/out.wav
```

Or without the server at all: `navi-tts synth ... --voice /tmp/myref.wav`.
Pin a `seed` (fixed seed per voice is what keeps a cloned character
sounding like itself - the engine otherwise seeds randomly per request).

## 4. Where voices live, and moving them

`~/.local/share/navi-tts/voices/<name>/` holds the reference sample and a
`voice.json` (sample sha256, duration, model, date). The directory is the
whole story: copy it to another host running the same `.navi` model and the
voices load at start. `navi-tts voices rm <name>` deletes one.

To expose the voice through the household TTS queue, add its name to
`voices.available` and pin a seed under `voices.seeds` in the queue config.
