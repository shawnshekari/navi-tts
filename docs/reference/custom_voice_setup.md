# Custom Voice Setup for Qwen3-TTS

This guide explains how to create and register a custom voice for your Open-LLM-VTuber using Qwen3-TTS.

## Prerequisites

- Qwen3-TTS server running:

```shell
./build/qwen3-tts-server -m models/qwen3-tts-0.6b-f16.gguf
```

- FFmpeg installed (`sudo apt install ffmpeg` on Ubuntu/Debian)

---

## Step 1: Extract Reference Audio Clip

### From an MP3 file:

```bash
# Extract 15 seconds starting at 8 seconds (adjust -ss and -t as needed)
ffmpeg -i "/path/to/source.mp3" -ss 8 -t 15 -ac 1 -ar 24000 -y /tmp/nyx_reference.wav
```

### Parameters explained:
- `-ss 8` - Start at 8 seconds into the file
- `-t 15` - Extract 15 seconds of audio
- `-ac 1` - Convert to mono (1 channel)
- `-ar 24000` - Sample rate 24kHz (required by Qwen3-TTS)
- `-y` - Overwrite output file if it exists

### Alternative extraction options:

```bash
# Extract from the beginning (first 10 seconds)
ffmpeg -i "source.mp3" -t 10 -ac 1 -ar 24000 -y reference.wav

# Extract from 30 seconds, for 20 seconds
ffmpeg -i "source.mp3" -ss 30 -t 20 -ac 1 -ar 24000 -y reference.wav

# Pitch shift the audio (useful for voice modification)
ffmpeg -i "source.mp3" -ss 8 -t 15 -ac 1 -ar 24000 -af "atempo=1.0,asetrate=24000*1.1,pitch=1.2" -y reference.wav
```

### Recommended clip length:
- **Minimum:** 5 seconds
- **Ideal:** 10-15 seconds  
- **Maximum:** 30 seconds (longer doesn't improve quality)

---

## Step 2: Register Voice with TTS Server

```bash
# Upload the reference audio as a custom voice
curl -X POST http://localhost:8080/v1/audio/voices \
  -F "name=nyx_voice" \
  -F "audio_sample=@/tmp/nyx_reference.wav"
```

### Response:
```json
{"id":"voice_3","name":"nyx_voice"}
```

**Important:** The `id` field (e.g., `voice_3`) is what you'll use in your config.

---

## Step 3: Verify Voice Registration

```bash
# List all available voices
curl -s http://localhost:8080/v1/audio/voices | python3 -m json.tool
```

### Example output:
```json
{
    "qwen3-tts-0.6b-f16": [
        "default",
        "voice_1",
        "voice_2",
        "voice_3"
    ]
}
```

---

## Step 4: Test the Custom Voice

```bash
# Test with the voice ID from Step 2
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-0.6b-f16",
    "input": "Hello, this is a test of the custom voice.",
    "voice": "voice_3"
  }' \
  --output test_voice.wav

# Play the result
ffplay test_voice.wav
```

---

## Step 5: Configure Open-LLM-VTuber

Edit your `conf.yaml` file:

```yaml
tts_config:
  tts_model: 'openai_tts'
  
  openai_tts:
    model: 'qwen3-tts-0.6b-f16.gguf'
    voice: 'voice_3'              # Use the ID from Step 2
    api_key: 'not-needed'
    base_url: 'http://localhost:8080/v1'
    file_extension: 'wav'
    seed: 42                      # Fixed seed for consistent voice
    temperature: 0.3              # Lower = more consistent
    top_k: 0                      # Disable top-k filtering
```

**Restart Open-LLM-VTuber** after making changes.

---

## Important Notes

### Voice ID Persistence
- Voice IDs (`voice_1`, `voice_2`, etc.) are **stored in memory only**
- If you restart the TTS server, all registered voices are lost
- Next registration after restart will be `voice_1` again

### Making Voices Permanent
To avoid re-registering voices after server restarts:

**Option 1: Create a setup script**
```bash
#!/bin/bash
# register_voices.sh

# Wait for server to be ready
sleep 2

# Register your voices
curl -X POST http://localhost:8080/v1/audio/voices \
  -F "name=nyx_voice" \
  -F "audio_sample=@/path/to/nyx_reference.wav"

curl -X POST http://localhost:8080/v1/audio/voices \
  -F "name=another_voice" \
  -F "audio_sample=@/path/to/another_reference.wav"

echo "Voices registered!"
```

**Option 2: Save your reference audio files**
```bash
# Create a directory for voice references
mkdir -p ~/voices

# Save your reference files
cp /tmp/nyx_reference.wav ~/voices/nyx_original.wav

# When you need to re-register after server restart:
ffmpeg -i "source.mp3" -ss 8 -t 15 -ac 1 -ar 24000 -y ~/voices/nyx_reference.wav
curl -X POST http://localhost:8080/v1/audio/voices \
  -F "name=nyx_voice" \
  -F "audio_sample=@~/voices/nyx_reference.wav"
```

---

## Troubleshooting

### Error: "unknown voice 'voice_3'"
- The TTS server was restarted and voices were lost
- Re-register the voice using Step 2
- Check available voices with `curl http://localhost:8080/v1/audio/voices`

### Error: "'audio_sample' file is required"
- Make sure you're using `-F` (form data), not `-d` (JSON)
- Check the file path is correct: `@/path/to/file.wav`

### Error: "failed to extract speaker embedding"
- Audio file may be corrupted or in wrong format
- Re-extract with: `ffmpeg -i input.mp3 -ac 1 -ar 24000 -y output.wav`

### Voice sounds different each time
- Make sure you're using a voice ID (e.g., `voice_3`), NOT `default`
- Check that `seed` and `temperature` are set in your config

### No audio output
- Test with `default` voice first to verify server is working
- Check server logs for errors
- Verify the text input is not empty

---

## Quick Reference Commands

```bash
# Extract audio from MP3
ffmpeg -i "source.mp3" -ss 8 -t 15 -ac 1 -ar 24000 -y reference.wav

# Register voice
curl -X POST http://localhost:8080/v1/audio/voices \
  -F "name=my_voice" \
  -F "audio_sample=@reference.wav"

# List voices
curl -s http://localhost:8080/v1/audio/voices

# Test voice
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-tts-0.6b-f16", "input": "Test text", "voice": "voice_3"}' \
  --output test.wav
```

---

## Tips for Better Voice Quality

1. **Choose clean audio:** Use reference clips without background music or noise
2. **Clear speech:** 10-15 seconds of clear speech works best
3. **Consistent tone:** Avoid clips with big volume variations
4. **Sample rate:** Always convert to 24kHz (the model's native rate)
5. **Mono audio:** Stereo doesn't improve quality and wastes space
