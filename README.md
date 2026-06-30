# Voice Assistant - Ubuntu Edition

An offline, on-device conversational AI that listens, understands, and replies - running natively on Ubuntu Linux. You speak into a microphone, the system transcribes your speech, a local LLM generates a reply, and a local TTS engine speaks it back, **fully offline with no cloud or internet at runtime**.

This is a parallel track to the [Raspberry Pi project](https://github.com/siddoboi/voice-assistant), built to develop and validate the full conversational pipeline on real audio hardware without waiting on Pi/GSM hardware. The Ubuntu machine is both development and deployment target.

> A software-based prototype is ready and complete: the full streaming pipeline, push-to-talk mode with barge-in, software audio normalization, and bad-transcript rejection all work end-to-end in a quiet room, backed by a passing test suite.

---

## Architecture

```
Your voice (mic) → VAD (Silero) → ASR (faster-whisper small.en)
   → LLM (Llama 3.2 3B via Ollama) → TTS (Piper) → earphones
```

The pipeline streams sentence-by-sentence with a bounded `asyncio.Queue` for back-pressure, so the first words play before the full reply finishes generating.

**Two input modes:**
- **VAD mode** - continuous listening; Silero detects speech onset/offset automatically
- **Push-to-talk (`--ptt`)** - records only between keypresses; structurally eliminates the mic re-hearing the assistant's own TTS reply

---

## What Makes the Ubuntu Track Different

Building on real laptop audio surfaced problems a mocked test suite never could. The fixes here are production-minded:

- **Software RMS normalization + noise gate** - input gain no longer needs hand-tuning; ASR becomes independent of input level.
- **Bad-transcript rejection** - empty or hallucinated input (Whisper emits "thank you" on silence) is caught and answered with a spoken "Sorry, I didn't catch that" instead of a wrong reply.
- **Push-to-talk mode** - eliminates TTS-into-mic echo entirely.
- **Push-to-talk with barge-in** - press Enter to interrupt the assistant mid-reply; playback stops instantly via continuous interruptible audio, and it immediately starts listening for your next question.
- **Indian-English tuning** - small.en ASR with an Indian-English prompt; 3B LLM for reliable one-sentence answers.
- **Per-session logging** - JSON + text transcript + combined per-turn WAVs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama - Llama 3.2 3B Instruct (Q4_K_M) primary, TinyLlama 1.1B fallback |
| ASR | faster-whisper (small.en, int8, CPU) with Indian-English prompt |
| TTS | Piper TTS (en_US-amy-medium) |
| VAD | Silero VAD v4 via ONNX Runtime |
| Audio I/O | sounddevice (PortAudio) + software RMS normalization |
| Concurrency | asyncio (bounded-queue streaming, continuous interruptible playback) |
| Persistence | SQLite |
| Testing | pytest |
| OS | Ubuntu (tested on 26.04, Python 3.14) |

---

## Setup

> **Note:** This repo was developed on Ubuntu 26.04 with Python 3.14.4. `setup_pi.sh` is the legacy script for the Raspberry Pi target (hardcodes `python3.13`) - do not use it here. For Ubuntu, use `setup_ubuntu.sh` below, which handles Python 3.14 and the correct package pins automatically. `onnxruntime` must be ≥ 1.27.0 for Python 3.14 wheels.

```bash
git clone https://github.com/siddoboi/voice-assistant-ubuntu.git
cd voice-assistant-ubuntu
bash setup_ubuntu.sh
```

`setup_ubuntu.sh` installs system packages, builds the venv, installs all pinned Python packages, installs Ollama and pulls both models, downloads the Piper/Silero models, pre-fetches the faster-whisper weights, and runs the test suite. It cannot fully automate audio device selection or mic gain, since those are physically different per machine - it prints your detected devices and sets a starting gain, but you must verify the config manually (next step).

Find your audio device indices and set up the config:

```bash
aplay -l && arecord -l          # note your mic + earphone card indices
# edit configs/ubuntu_config.yaml - set integer input_device / output_device
export VOICE_ASSISTANT_CONFIG=configs/ubuntu_config.yaml
```

---

## Usage

```bash
# Push-to-talk mode (recommended - no echo issues)
python src/main.py --ptt

# VAD-driven continuous mode
python src/main.py

# Run the pipeline on a pre-recorded file
python -m src.pipeline --input recordings/sample1.wav
```

While the assistant is replying in push-to-talk mode, press Enter again to interrupt and speak immediately - no waiting for the reply to finish.

---

## Testing

```bash
unset VOICE_ASSISTANT_CONFIG    # IMPORTANT - see note below
pytest tests/
```

> **Caveat:** `vad.py` reads `VOICE_ASSISTANT_CONFIG` at import time. Leaving it exported makes VAD/audio tests read the wrong threshold and fail. Always `unset` it before `pytest`, re-export before running the assistant.

The suite covers every module with fast mocked unit tests by default; real-model integration tests are opt-in via `--run-integration`.

---

## Project Structure

```
voice-assistant-ubuntu/
├── src/
│   ├── main.py            # VAD state machine + push-to-talk loop + barge-in + normalization + logging
│   ├── pipeline.py        # streaming pipeline + interruptible playback + precomputed_transcript bypass + markdown stripping
│   ├── asr.py             # faster-whisper (config model_size + Indian-English prompt)
│   ├── vad.py              # Silero VAD v4 (config-driven silence_threshold)
│   └── tts.py / llm_client.py / audio_io.py / conversation.py
├── configs/
│   ├── ubuntu_config.yaml # live config
│   └── dev_config.yaml / models.yaml
├── scripts/                # benchmark + VAD tuning
├── tests/                  # pytest suite
├── setup_ubuntu.sh         # automated Ubuntu/Phase 0a setup
├── setup_pi.sh             # legacy Raspberry Pi setup (not for this repo's target)
└── recordings/sessions/    # per-session logs (local only)
```

---

## Roadmap

- **Core pipeline (done):** Full conversational pipeline on laptop audio, live-voice fixes, push-to-talk, barge-in, normalization, bad-transcript gate
- **Later:** WER evaluation, broader hardening, expanded testing

A GPU-accelerated multilingual variant is planned in [voice-assistant-multilingual](https://github.com/siddoboi/voice-assistant-multilingual). Real-call telephony integration is being pursued separately in the [Raspberry Pi project](https://github.com/siddoboi/voice-assistant).

---

## License

MIT