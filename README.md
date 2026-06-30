# Voice Assistant - Ubuntu Edition

An offline, on-device conversational AI that answers phone calls - running natively on Ubuntu Linux with the GSM module connected over USB. The caller speaks, the system transcribes their speech, a local LLM generates a reply, and a local TTS engine speaks it back, **fully offline with no cloud or internet at runtime**.

This is a parallel track to the [Raspberry Pi project](https://github.com/siddoboi/voice-assistant). It exists so the full pipeline can be built and tested on real audio hardware immediately, rather than waiting for the Pi. The Ubuntu machine is both development and deployment target.

> A software-based prototype is ready: Phase 0a (laptop mic and earphones, no GSM) is fully implemented - the streaming pipeline, push-to-talk mode, software audio normalization, and bad-transcript rejection all work end-to-end in a quiet room, backed by a passing test suite. Wiring in the A7672S GSM module over USB (Phase 0b) is the next step.

---

## The Two Sub-Phases

| Phase | What it is | Hardware |
|---|---|---|
| **0a** | Full AI pipeline on laptop mic → earphones. No GSM. Proves the pipeline on real audio. | Laptop + TRRS earphones |
| **0b** | Wire in the A7672S GSM module via USB. Replace mic/earphones with a real cellular call. | A7672S module + 12 V supply + nano SIM |

Phase 0a is built first so that by the time the GSM module is added, every line of AI code is already tested with real voice - GSM integration becomes a one-day audio-source swap rather than debugging everything at once.

---

## Architecture

```
Your voice (mic / GSM call) → VAD (Silero) → ASR (faster-whisper small.en)
   → LLM (Llama 3.2 3B via Ollama) → TTS (Piper) → earphones / GSM call
```

The pipeline streams sentence-by-sentence with a bounded `asyncio.Queue` for back-pressure, so the first words play before the full reply finishes generating.

**Two input modes:**
- **VAD mode** - continuous listening; Silero detects speech onset/offset automatically
- **Push-to-talk (`--ptt`)** - records only between keypresses; structurally eliminates the mic re-hearing the assistant's own TTS reply

---

## What Makes the Ubuntu Track Different

Building on real laptop audio surfaced problems the mocked test suite never could. The fixes here are production-minded and feed back to the Pi track:

- **Software RMS normalization + noise gate** - input gain no longer needs hand-tuning; ASR becomes independent of input level. Critical for an unpredictable GSM line.
- **Bad-transcript rejection** - empty or hallucinated input (Whisper emits "thank you" on silence) is caught and answered with a spoken "Sorry, I didn't catch that" instead of a wrong reply.
- **Push-to-talk mode** - eliminates TTS-into-mic echo entirely.
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
| Telephony (Phase 0b) | A7672S 4G LTE module over USB, pyserial AT commands |
| Audio I/O | sounddevice (PortAudio) + software RMS normalization |
| Concurrency | asyncio (bounded-queue streaming) |
| Persistence | SQLite |
| Testing | pytest |
| OS | Ubuntu (tested on 26.04, Python 3.14) |

---

## Setup

> **Note:** This repo was developed on Ubuntu 26.04 with Python 3.14.4. `setup.sh` hardcodes `python3.13`; on newer Ubuntu, run the steps individually with your Python version. `onnxruntime` must be ≥ 1.27.0 for Python 3.14 wheels.

```bash
git clone https://github.com/siddoboi/voice-assistant-ubuntu.git
cd voice-assistant-ubuntu
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install faster-whisper piper-tts onnxruntime sounddevice pyyaml \
            noisereduce pyserial ollama numpy pytest
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull tinyllama:1.1b
# download Piper voice + Silero VAD v4 into models/ (see setup.sh)
```

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
│   ├── main.py            # VAD state machine + push-to-talk loop + normalization + logging
│   ├── pipeline.py        # streaming pipeline + precomputed_transcript bypass + markdown stripping
│   ├── asr.py             # faster-whisper (config model_size + Indian-English prompt)
│   ├── vad.py             # Silero VAD v4 (config-driven silence_threshold)
│   ├── tts.py / llm_client.py / audio_io.py / conversation.py
│   └── telephony/gsm_adapter.py   # A7672S AT-command control (Phase 0b)
├── configs/
│   ├── ubuntu_config.yaml # Phase 0a live config
│   ├── dev_config.yaml / pi_config.yaml / models.yaml
├── scripts/               # benchmark + VAD tuning
├── tests/                 # pytest suite (303 passing)
└── recordings/sessions/   # per-session logs (local only)
```

---

## Roadmap

- **Phase 0a (done):** Full pipeline on laptop audio, live-voice fixes, push-to-talk, normalization, bad-transcript gate
- **Phase 0b:** Solder A7672S headers, wire audio, GSM call lifecycle in `main.py`, first real call
- **Later:** WER evaluation, systemd service, error hardening

A GPU-accelerated multilingual variant is being developed in [voice-assistant-multilingual](https://github.com/siddoboi/voice-assistant-multilingual).

---

## License

MIT