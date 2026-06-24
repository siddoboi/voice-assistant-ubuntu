"""
main.py - VAD-driven voice assistant loop for Phase 0a (laptop mic/earphones).

State machine: IDLE -> DETECTING_ONSET -> RECORDING -> PROCESSING -> IDLE

Device records at native rate (48kHz stereo on this machine). Each chunk is
downmixed to mono and resampled to 16kHz/512 samples for VAD. Native-rate
chunks are accumulated separately and used for ASR (better quality).

No ring detection in Phase 0a. Press Ctrl+C to stop.
Phase 0b will extend this with gsm_adapter.wait_for_ring() and answer_call().

Usage:
    export VOICE_ASSISTANT_CONFIG=configs/ubuntu_config.yaml
    python src/main.py
    python src/main.py --onset-chunks 3 --offset-chunks 18
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src import asr, audio_io, tts, vad
from src.conversation import ConversationManager
from src import pipeline

# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------

class _State(Enum):
    IDLE             = auto()
    DETECTING_ONSET  = auto()
    RECORDING        = auto()
    PROCESSING       = auto()


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

VAD_CHUNK_SIZE  = vad.CHUNK_SIZE    # 512 samples at 16kHz
SAMPLE_RATE     = vad.SAMPLE_RATE   # 16000 Hz (VAD requirement)

DEFAULT_ONSET_CHUNKS  = 3    # consecutive above-threshold chunks to start recording
DEFAULT_OFFSET_CHUNKS = 18   # consecutive below-threshold chunks to end recording (~576ms)

# Keep CHUNK_SIZE alias so existing tests that import it still work
CHUNK_SIZE = VAD_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _record_chunk() -> tuple[np.ndarray, np.ndarray]:
    """Record one VAD-sized window.

    Returns (chunk_16k_mono_int16, chunk_native) where:
    - chunk_16k_mono_int16: 512 samples at 16kHz mono int16 for VAD
    - chunk_native: raw recording at device native rate/channels for ASR
    """
    chunk_native = audio_io.record(
        duration_sec=VAD_CHUNK_SIZE / SAMPLE_RATE,
    )
    # Squeeze mono channel axis if present
    if chunk_native.ndim == 2 and chunk_native.shape[1] == 1:
        chunk_native = chunk_native[:, 0]
    # Resample 44100 -> 16kHz for VAD
    indices = np.linspace(0, len(chunk_native) - 1, VAD_CHUNK_SIZE)
    chunk_16k = np.interp(
        indices,
        np.arange(len(chunk_native)),
        chunk_native.astype(np.float32)
    ).astype(np.int16)

    return chunk_16k, chunk_native


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_loop(
    onset_chunks: int  = DEFAULT_ONSET_CHUNKS,
    offset_chunks: int = DEFAULT_OFFSET_CHUNKS,
) -> None:
    """Main VAD-driven loop. Blocks until Ctrl+C."""

    print("Loading models...")
    vad.load_model()
    asr.load_model()
    tts.load_voice()
    print("All models loaded. Speak to interact. Ctrl+C to quit.\n")

    conversation = ConversationManager()

    state        = _State.IDLE
    onset_count  = 0
    offset_count = 0
    vad_chunks:    list[np.ndarray] = []   # 16kHz mono for VAD decisions
    native_chunks: list[np.ndarray] = []   # native rate for ASR quality

    try:
        while True:
            chunk_16k, chunk_native = _record_chunk()

            # VAD on 16kHz mono float32
            chunk_f32 = chunk_16k.astype(np.float32) / 32768.0
            prob = vad.get_speech_prob(chunk_f32)
            is_speech = prob >= vad.SILENCE_THRESHOLD

            # ---- state transitions ----
            if state == _State.IDLE:
                if is_speech:
                    state = _State.DETECTING_ONSET
                    onset_count = 1
                    vad_chunks    = [chunk_16k]
                    native_chunks = [chunk_native]

            elif state == _State.DETECTING_ONSET:
                vad_chunks.append(chunk_16k)
                native_chunks.append(chunk_native)
                if is_speech:
                    onset_count += 1
                    if onset_count >= onset_chunks:
                        state = _State.RECORDING
                        offset_count = 0
                        print("[VAD] Speech onset detected - recording...")
                else:
                    state = _State.IDLE
                    onset_count = 0
                    vad_chunks    = []
                    native_chunks = []

            elif state == _State.RECORDING:
                vad_chunks.append(chunk_16k)
                native_chunks.append(chunk_native)
                if not is_speech:
                    offset_count += 1
                    if offset_count >= offset_chunks:
                        state = _State.PROCESSING
                else:
                    offset_count = 0

            # ---- processing ----
            if state == _State.PROCESSING:
                print("[VAD] Silence detected - processing...")
                # Use native-rate mono audio for ASR
                audio_native = np.concatenate(native_chunks)
                if audio_native.ndim == 2 and audio_native.shape[1] == 1:
                    audio_native = audio_native[:, 0]
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir="recordings"
                ) as f:
                    tmp_path = f.name
                audio_io.save_wav(audio_native, tmp_path, sample_rate=44100)

                try:
                    import subprocess
                    subprocess.run(["amixer", "-c", "2", "set", "Capture", "nocap"], capture_output=True)
                    result = pipeline.run(
                        input_wav=tmp_path,
                        conversation=conversation,
                        skip_play=False,
                    )
                    perceived = result.get("latencies", {}).get("perceived_s")
                    print(f"[LATENCY] perceived_s={perceived:.3f}s")
                    time.sleep(0.5)
                    vad.reset_state()
                finally:
                    import subprocess as _sp
                    _sp.run(["amixer", "-c", "2", "set", "Capture", "cap"], capture_output=True)
                    Path(tmp_path).unlink(missing_ok=True)

                state         = _State.IDLE
                onset_count   = 0
                offset_count  = 0
                vad_chunks    = []
                native_chunks = []

    except KeyboardInterrupt:
        print("\nStopping. Ending conversation session.")
        conversation.end_session()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 0a VAD-driven voice assistant loop."
    )
    p.add_argument(
        "--onset-chunks", type=int, default=DEFAULT_ONSET_CHUNKS,
        help=f"Consecutive above-threshold chunks to start recording (default: {DEFAULT_ONSET_CHUNKS}).",
    )
    p.add_argument(
        "--offset-chunks", type=int, default=DEFAULT_OFFSET_CHUNKS,
        help=f"Consecutive below-threshold chunks to end recording (default: {DEFAULT_OFFSET_CHUNKS}).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_loop(
        onset_chunks=args.onset_chunks,
        offset_chunks=args.offset_chunks,
    )
