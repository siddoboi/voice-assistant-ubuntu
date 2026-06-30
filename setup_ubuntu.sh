#!/usr/bin/env bash
# =============================================================================
# setup_voice_assistant_ubuntu.sh
# Full Phase 0a environment setup for voice-assistant-ubuntu
# Target: Ubuntu 26.04 LTS, Python 3.14.x, x86_64
# Run from the directory where you want the repo cloned (e.g. ~/projects)
# =============================================================================
set -e

echo "============================================================"
echo " Phase 0a Setup - voice-assistant-ubuntu"
echo "============================================================"

# -----------------------------------------------------------------------
# 1. System packages (python excluded - handled separately below)
# -----------------------------------------------------------------------
echo ""
echo "[1/9] Installing system packages..."
sudo apt update
sudo apt install -y \
    git curl wget build-essential pkg-config \
    libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libportaudio2 portaudio19-dev \
    ffmpeg sox alsa-utils zstd \
    python3.14 python3.14-venv python3.14-dev python3-pip

# -----------------------------------------------------------------------
# 2. Clone repo (skip if already present)
# -----------------------------------------------------------------------
echo ""
echo "[2/9] Cloning repository..."
REPO_DIR="voice-assistant-ubuntu"
if [ -d "$REPO_DIR" ]; then
    echo "  $REPO_DIR already exists, skipping clone."
else
    git clone https://github.com/siddoboi/voice-assistant-ubuntu.git
fi
cd "$REPO_DIR"

# -----------------------------------------------------------------------
# 3. Python venv + pip packages
# -----------------------------------------------------------------------
echo ""
echo "[3/9] Creating venv and installing Python packages..."
python3.14 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install \
    faster-whisper==1.2.1 \
    piper-tts==1.4.2 \
    onnxruntime==1.27.0 \
    sounddevice==0.5.5 \
    psutil \
    pyyaml==6.0.2 \
    noisereduce \
    pyserial \
    ollama \
    numpy \
    pytest

# -----------------------------------------------------------------------
# 4. Ollama + models
# -----------------------------------------------------------------------
echo ""
echo "[4/9] Installing Ollama and pulling models..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
(ollama serve &>/dev/null &)
sleep 3
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull tinyllama:1.1b

# -----------------------------------------------------------------------
# 5. Model files (Piper TTS + Silero VAD)
# -----------------------------------------------------------------------
echo ""
echo "[5/9] Downloading TTS/VAD model files..."
mkdir -p models/piper models/silero

if [ ! -f models/silero/silero_vad_v4.onnx ]; then
    wget -q --show-progress \
        https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx \
        -O models/silero/silero_vad_v4.onnx
fi

if [ ! -f models/piper/en_US-amy-medium.onnx ]; then
    wget -q --show-progress \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx \
        -O models/piper/en_US-amy-medium.onnx
fi

if [ ! -f models/piper/en_US-amy-medium.onnx.json ]; then
    wget -q --show-progress \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json \
        -O models/piper/en_US-amy-medium.onnx.json
fi

# -----------------------------------------------------------------------
# 6. faster-whisper small.en weights (pre-download so first run isn't slow)
# -----------------------------------------------------------------------
echo ""
echo "[6/9] Pre-downloading faster-whisper small.en weights..."
python -c "
from faster_whisper import WhisperModel
print('Downloading small.en...')
WhisperModel('small.en', device='cpu', compute_type='int8')
print('Done.')
"

# -----------------------------------------------------------------------
# 7. Audio device detection (informational - manual config step required)
# -----------------------------------------------------------------------
echo ""
echo "[7/9] Audio devices on this machine:"
echo "--- aplay -l ---"
aplay -l || true
echo "--- arecord -l ---"
arecord -l || true
echo ""
echo "  NOTE: configs/ubuntu_config.yaml must have input_device/output_device"
echo "  set to the correct PortAudio integer indices for THIS machine."
echo "  Run: python -c \"import sounddevice as sd; print(sd.query_devices())\""
echo "  to find them, then edit configs/ubuntu_config.yaml manually."

# -----------------------------------------------------------------------
# 8. Mic gain (prevents saturation - tune per machine, this is a starting point)
# -----------------------------------------------------------------------
echo ""
echo "[8/9] Setting a conservative starting mic gain (28%)..."
amixer -c 2 set Capture 28% 2>/dev/null || echo "  (skip - adjust card index if this machine differs)"
sudo alsactl store 2>/dev/null || true

# -----------------------------------------------------------------------
# 9. Verify
# -----------------------------------------------------------------------
echo ""
echo "[9/9] Running test suite..."
unset VOICE_ASSISTANT_CONFIG
pytest tests/ -q

echo ""
echo "============================================================"
echo " Setup complete."
echo "============================================================"
echo ""
echo " IMPORTANT MANUAL STEPS STILL REQUIRED:"
echo "  1. Edit configs/ubuntu_config.yaml: set input_device / output_device"
echo "     to YOUR machine's PortAudio indices (see step 7 output above)."
echo "  2. Tune mic gain for your hardware (target RMS ~3000-4000, no clipping):"
echo "       amixer -c <card> set Capture <N>%"
echo "       sudo alsactl store"
echo "  3. Run live:"
echo "       export VOICE_ASSISTANT_CONFIG=configs/ubuntu_config.yaml"
echo "       python src/main.py --ptt          # push-to-talk (recommended)"
echo "       python src/main.py                # VAD mode (needs quiet room)"
echo "============================================================"
