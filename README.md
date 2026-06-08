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

