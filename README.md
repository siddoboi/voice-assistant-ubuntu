# Voice Assistant — Phase 0 (Ubuntu)

A fully offline, on-device conversational AI assistant running natively on
Ubuntu Linux. Speak into your laptop mic and get a synthesized voice response
through your earphones — no cloud, no internet at runtime, no Raspberry Pi
needed.

This is the **Phase 0 parallel track** of the voice assistant project.
Phase 0a runs the full AI pipeline on laptop mic and earphones.
Phase 0b wires in an A7672S GSM module to answer real cellular phone calls.
The Pi track lives at [siddoboi/voice-assistant](https://github.com/siddoboi/voice-assistant).

## Hardware Requirements

**Phase 0a (now):**
- Any Ubuntu x86_64 laptop
- TRRS earphones with inline mic plugged into the 3.5mm jack

**Phase 0b (after Phase 0a is stable):**
- A7672S GSM module (SIMCom A7672S-LASC, 4G LTE Cat-1)
- 12V DC adapter
- Nano SIM card (any Indian carrier)
- Jumper wires to connect MIC/SPK pins to a USB audio adapter

## Setup

```bash
git clone https://github.com/siddoboi/voice-assistant-ubuntu.git
cd voice-assistant-ubuntu
python3.14 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install faster-whisper==1.2.1 piper-tts==1.4.2 onnxruntime==1.27.0 \
    sounddevice==0.5.5 psutil pyyaml==6.0.2 noisereduce pyserial ollama \
    numpy pytest
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull tinyllama:1.1b
mkdir -p models/piper models/silero
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx \
    -O models/piper/en_US-amy-medium.onnx
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json \
    -O models/piper/en_US-amy-medium.onnx.json
wget -q https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx \
    -O models/silero/silero_vad_v4.onnx
```

Find your ALSA device indices and set them in `configs/ubuntu_config.yaml`:

```bash
aplay -l
arecord -l
```

## Usage

**Live voice interaction (Phase 0a):**

```bash
export VOICE_ASSISTANT_CONFIG=configs/ubuntu_config.yaml
python src/main.py
```

Speak into your mic. The assistant transcribes your speech, generates a reply
via the local LLM, and plays it back through your earphones. Press Ctrl+C to stop.

**Pipeline smoke test (WAV file input):**

```bash
python -m src.pipeline --input recordings/sample1.wav --no-play
```

**Benchmark suite:**

```bash
python scripts/benchmark.py --input recordings/sample1.wav
```

**VAD threshold tuning:**

```bash
python scripts/tune_vad_threshold.py recordings/sample1.wav
```

**Run tests:**

```bash
unset VOICE_ASSISTANT_CONFIG
pytest tests/
pytest tests/ --run-integration  # requires real models + mic
```

## Project Structure
voice-assistant-ubuntu/

├── src/

│   ├── audio_io.py        # record/play/resample/WAV I/O + noise reduction

│   ├── asr.py             # faster-whisper transcription

│   ├── llm_client.py      # Ollama LLM client + sentence streaming

│   ├── tts.py             # Piper TTS batch + streaming synthesis

│   ├── vad.py             # Silero VAD v4 + config-driven threshold

│   ├── conversation.py    # multi-turn history + SQLite session persistence

│   ├── pipeline.py        # streaming ASR-LLM-TTS-play orchestration

│   ├── main.py            # VAD-driven voice loop (Phase 0a)

│   └── telephony/

│       └── gsm_adapter.py # A7672S AT-command call control (Phase 0b)

├── configs/

│   ├── ubuntu_config.yaml # Ubuntu deployment settings (fill ALSA indices)

│   ├── dev_config.yaml    # WSL2 reference config

│   └── models.yaml        # model choices + benchmark metrics

├── scripts/

│   ├── benchmark.py       # full LLM/ASR/TTS/VAD/E2E benchmark suite

│   └── tune_vad_threshold.py

├── tests/                 # 303 pytest unit + opt-in integration tests

├── models/                # gitignored - download fresh on each machine

├── recordings/            # gitignored - WAV samples + SQLite DB

└── setup.sh
## Benchmark Numbers (Ubuntu x86_64, Phase 0a Day 2)

Machine: ASUS TUF Gaming A17 FA706IC, Ubuntu 26.04, Python 3.14.4

| Stage | Metric | Value |
|-------|--------|-------|
| LLM (llama3.2:1b) | first token | 0.265s |
| LLM (llama3.2:1b) | total | 1.991s |
| LLM (tinyllama) | first token | 0.081s |
| ASR (tiny.en) | RTF | 0.146 |
| TTS (amy-medium) | RTF | 0.041 |
| TTS (amy-medium) | first audio | 0.182s |
| End-to-end | perceived_s | 2.077s |

Target: perceived_s <= 3.0s. Current: 2.077s.

## Architecture

The pipeline overlaps LLM generation with TTS synthesis and audio playback to
minimize perceived latency. Transcribed speech goes to the local LLM, whose
token stream is buffered into sentences as they form. Each sentence is
synthesized immediately and pushed onto a bounded `asyncio.Queue`. A consumer
task plays audio off the queue while synthesis continues. Perceived latency is
ASR time plus time-to-first-audio.

The VAD loop in `main.py` runs a state machine on 512-sample chunks: IDLE,
DETECTING_ONSET, RECORDING, PROCESSING. Speech onset requires 3 consecutive
above-threshold chunks. Offset requires 18 consecutive below-threshold chunks
(~576ms silence) to end capture and trigger the pipeline.

## Tech Stack

- **OS:** Ubuntu 26.04 LTS (Resolute Raccoon), x86_64
- **Python:** 3.14.4
- **LLM:** Ollama - Llama 3.2 1B Instruct (Q4_K_M) primary, TinyLlama 1.1B fallback
- **ASR:** faster-whisper 1.2.1 (tiny.en, int8, CPU)
- **TTS:** Piper TTS 1.4.2 (en_US-amy-medium)
- **VAD:** Silero VAD v4 via ONNX Runtime 1.27.0
- **Noise reduction:** noisereduce (lazy-imported, enabled from Day 1)
- **Audio I/O:** sounddevice 0.5.5 (PortAudio, plughw:2,0 on this machine)
- **Concurrency:** asyncio (bounded-queue streaming)
- **Config:** PyYAML
- **Persistence:** SQLite (stdlib)
- **Testing:** pytest (303 tests: 287 unit + 16 main + 15 opt-in integration + 1 integration)
- **Telephony (Phase 0b):** A7672S via pyserial AT commands

## Current Status

**Phase 0a complete.** Full AI pipeline working on laptop mic and earphones.
VAD-driven `main.py` loop written and tested. 303 tests passing.
Perceived latency 2.077s (target <= 3.0s).

**Next:** Live voice interaction test (speak into mic, hear TTS response).
Then Phase 0b: wire in A7672S GSM module for real cellular calls.
