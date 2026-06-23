"""
main.py - VAD-driven voice assistant loop for Phase 0a (laptop mic/earphones).

State machine: IDLE -> DETECTING_ONSET -> RECORDING -> PROCESSING -> IDLE

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
from src import llm_client
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

CHUNK_SIZE   = vad.CHUNK_SIZE    # 512 samples at 16kHz
SAMPLE_RATE  = vad.SAMPLE_RATE   # 16000

DEFAULT_ONSET_CHUNKS  = 3    # consecutive above-threshold chunks to start recording
DEFAULT_OFFSET_CHUNKS = 18   # consecutive below-threshold chunks to end recording (~576ms)


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
    recorded_chunks: list[np.ndarray] = []

    try:
        while True:
            # Record one chunk from mic
            chunk = audio_io.record(
                duration_sec=CHUNK_SIZE / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
            )

            # Convert int16 -> float32 for VAD
            chunk_f32 = chunk.astype(np.float32) / 32768.0

            prob = vad.get_speech_prob(chunk_f32)
            is_speech = prob >= vad.SILENCE_THRESHOLD

            # ---- state transitions ----
            if state == _State.IDLE:
                if is_speech:
                    state = _State.DETECTING_ONSET
                    onset_count = 1
                    recorded_chunks = [chunk]

            elif state == _State.DETECTING_ONSET:
                recorded_chunks.append(chunk)
                if is_speech:
                    onset_count += 1
                    if onset_count >= onset_chunks:
                        state = _State.RECORDING
                        offset_count = 0
                        print("[VAD] Speech onset detected - recording...")
                else:
                    # False start - reset
                    state = _State.IDLE
                    onset_count = 0
                    recorded_chunks = []

            elif state == _State.RECORDING:
                recorded_chunks.append(chunk)
                if not is_speech:
                    offset_count += 1
                    if offset_count >= offset_chunks:
                        state = _State.PROCESSING
                else:
                    offset_count = 0

            # ---- processing ----
            if state == _State.PROCESSING:
                print("[VAD] Silence detected - processing...")
                audio = np.concatenate(recorded_chunks)

                # Save to temp WAV for ASR
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir="recordings"
                ) as f:
                    tmp_path = f.name
                audio_io.save_wav(audio, tmp_path, sample_rate=SAMPLE_RATE)

                try:
                    result = pipeline.run(
                        input_wav=tmp_path,
                        conversation=conversation,
                        skip_play=False,
                    )
                    perceived = result.get("latencies", {}).get("perceived_s")
                    print(f"[LATENCY] perceived_s={perceived:.3f}s\n")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

                # Reset for next turn
                state        = _State.IDLE
                onset_count  = 0
                offset_count = 0
                recorded_chunks = []

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
