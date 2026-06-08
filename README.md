# Voice Assistant — Offline Phone Call Responder

A fully offline, on-device conversational AI assistant for the Raspberry Pi 5.
It answers incoming cellular phone calls over a SIM7600EI 4G LTE HAT,
transcribes the caller's speech, generates a reply with a local LLM, and
speaks the response back through synthesized voice — with no cloud services
and no internet required at runtime. All speech recognition, language
modelling, and speech synthesis run locally on the Pi.

## Hardware Requirements

- Raspberry Pi 5 (4 GB RAM) with the official active cooler and 27 W USB-C PD power supply
- 64 GB A2-rated microSD card (e.g. SanDisk Extreme Pro / Samsung Pro Plus)
- SIM7600EI 4G LTE GSM HAT with antenna (Indian LTE bands; AT-command interface over serial)
- Prepaid voice SIM card (any Indian carrier)
- USB audio adapter plus TRRS earphones with inline mic (for local pipeline testing; live call audio routes through the HAT's 3.5 mm jack)

Development is done on WSL2 (Debian Trixie); only live audio, GSM calls, and
latency profiling require the Pi.

## Setup

```bash
git clone https://github.com/siddoboi/voice-assistant.git
cd voice-assistant
bash setup.sh
source venv/bin/activate
ollama serve &
```

`setup.sh` installs system packages, creates the Python 3.13 virtual
environment, installs the Python dependencies, pulls the Ollama models, and
downloads the Piper TTS voice and Silero VAD v4 model. It runs identically on
WSL2 Debian Trixie and Raspberry Pi OS 64-bit.

On the Pi, activate the Pi configuration before running:

```bash
export VOICE_ASSISTANT_CONFIG=configs/pi_config.yaml
```

## Usage

Run the pipeline against a pre-recorded WAV file:

```bash
python -m src.pipeline --input recordings/sample1.wav
```

Useful flags: `--no-play` (skip audio playback, e.g. for headless
benchmarking), `--output <path>` (save the synthesized reply WAV),
`--model <name>` (override the LLM), and `--session-id <id>` (resume a stored
conversation).

Helper scripts:

```bash
python scripts/tune_vad_threshold.py recordings/sample1.wav   # VAD threshold sweep
python scripts/benchmark_pi.py --input recordings/sample1.wav # full benchmark suite
```

## Project Structure

```
voice-assistant/
├── src/
│   ├── audio_io.py        # record/play/resample/WAV I/O + noise reduction
│   ├── asr.py             # faster-whisper transcription
│   ├── llm_client.py      # Ollama LLM client + sentence streaming
│   ├── tts.py             # Piper TTS batch + streaming synthesis
│   ├── vad.py             # Silero VAD v4 voice activity detection
│   ├── conversation.py    # multi-turn history + SQLite session persistence
│   ├── pipeline.py        # streaming record→ASR→LLM→TTS→play orchestration
│   ├── main.py            # VAD-driven call loop (in progress; needs Pi)
│   └── telephony/
│       └── gsm_adapter.py # SIM7600EI AT-command call control
├── configs/
│   ├── dev_config.yaml    # WSL2 development settings
│   ├── pi_config.yaml     # Raspberry Pi deployment settings
│   └── models.yaml        # model paths and benchmark metrics
├── scripts/
│   ├── tune_vad_threshold.py
│   └── benchmark_pi.py
├── tests/                 # pytest unit + opt-in integration suite
├── recordings/            # WAV samples + SQLite DB (gitignored)
└── setup.sh
```

## Architecture

The core is a streaming pipeline that overlaps language-model generation with
speech synthesis and playback to minimise perceived latency. Transcribed
caller text is sent to the local LLM, whose token stream is buffered into
complete sentences as they form. Each finished sentence is synthesized to
audio immediately and pushed onto a bounded `asyncio.Queue`, while a separate
consumer task pulls audio off the queue and plays it. The queue's maximum size
applies back-pressure — synthesis pauses when playback falls behind — which
caps memory use on the 4 GB Pi and lets the first words of a reply play before
the full response has finished generating. Perceived latency is measured as
ASR time plus time-to-first-audio, reflecting the gap between the caller
finishing speaking and hearing the first word back.

## Current Status

**Phase 1 in progress.** Weeks 1–3 are complete: core modules, the streaming
pipeline, multi-turn conversation management with SQLite persistence, GSM
call-control signalling, and config-toggleable noise reduction, backed by a
passing unit and integration test suite. Week 4 (live GSM call audio routing,
full call lifecycle, on-Pi VAD/mic tuning) begins when the Pi 5 hardware
arrives. Weeks 5–6 cover WER and latency evaluation, stability testing, and
packaging as a systemd service.

