#!/bin/bash
# setup.sh — Full environment setup for voice-assistant
# Run this on any new Debian Trixie WSL2 machine after cloning the repo
# Usage: bash setup.sh

set -e  # Exit on any error

echo "================================================"
echo " Voice Assistant — Environment Setup"
echo "================================================"
echo ""

# ── Step 1: System dependencies ──────────────────────
echo "[1/6] Installing system dependencies..."
sudo apt update -q
sudo apt install -y \
    python3.13 python3.13-venv python3.13-dev python3-pip \
    git curl wget build-essential pkg-config \
    libssl-dev libffi-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libportaudio2 portaudio19-dev \
    ffmpeg sox alsa-utils zstd
echo "✓ System dependencies installed"
echo ""

# ── Step 2: Python virtual environment ───────────────
echo "[2/6] Creating Python virtual environment..."
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
echo "✓ Virtual environment created"
echo ""

# ── Step 3: Python packages ───────────────────────────
echo "[3/6] Installing Python packages..."
pip install -q \
    ollama \
    faster-whisper \
    piper-tts \
    onnxruntime \
    sounddevice \
    psutil \
    pyyaml
echo "✓ Python packages installed"
echo ""

# ── Step 4: Ollama ────────────────────────────────────
echo "[4/6] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "✓ Ollama installed"
echo ""

echo "  Pulling Llama 3.2 1B (primary model ~807MB)..."
ollama pull llama3.2:1b-instruct-q4_K_M
echo "  Pulling TinyLlama 1.1B (fallback ~637MB)..."
ollama pull tinyllama:1.1b
echo "✓ LLM models pulled"
echo ""

# ── Step 5: Piper TTS voice model ─────────────────────
echo "[5/6] Downloading Piper TTS voice model (~61MB)..."
mkdir -p models/piper
wget -q --show-progress \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx \
    -O models/piper/en_US-amy-medium.onnx
wget -q --show-progress \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json \
    -O models/piper/en_US-amy-medium.onnx.json
echo "✓ Piper TTS model downloaded"
echo ""

# ── Step 6: Silero VAD model ──────────────────────────
echo "[6/6] Downloading Silero VAD v4 model (~1.8MB)..."
mkdir -p models/silero
wget -q --show-progress \
    https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx \
    -O models/silero/silero_vad_v4.onnx
echo "✓ Silero VAD model downloaded"
echo ""

# ── Done ──────────────────────────────────────────────
echo "================================================"
echo " Setup complete! Everything is ready."
echo "================================================"
echo ""
echo "To start working:"
echo "  source venv/bin/activate"
echo "  ollama serve &"
echo ""
echo "To verify everything works:"
echo "  python src/llm_client.py"
echo "  python src/asr.py"
echo "  python src/tts.py"
echo "  python src/vad.py"
echo ""
