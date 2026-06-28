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
import json
import shutil
import datetime
import wave

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

def _is_bad_transcript(text: str) -> bool:
    """True if the transcript is empty, too short, or a known Whisper
    silence-hallucination (e.g. 'Thank you', 'Thanks for watching')."""
    if not text:
        return True
    cleaned = text.strip().lower().rstrip(".!?, ")
    if len(cleaned) < 3:
        return True
    hallucinations = {
        "thank you", "thanks", "thank you very much",
        "thanks for watching", "thank you for watching",
        "you", "bye", "bye bye", ".", "thank you.",
        "please subscribe", "see you next time",
    }
    return cleaned in hallucinations


class _SkipTurn(Exception):
    """Raised to skip processing a turn with bad/empty transcript."""
    pass


_DIDNT_CATCH_MSG = "Sorry, I didn't catch that. Could you please say it again?"


def _speak_didnt_catch() -> None:
    """Synthesize and play a fixed 'didn't catch that' message via Piper."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="recordings")
        tmp.close()
        tts.synthesize(_DIDNT_CATCH_MSG, tmp.name)
        with wave.open(tmp.name, "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio_io.play(data, sample_rate=sr)
        Path(tmp.name).unlink(missing_ok=True)
    except Exception as e:
        print(f"[WARN] didnt_catch playback failed: {e}")
    print(f"Bot:  {_DIDNT_CATCH_MSG}\n")


def _normalize_audio(audio: np.ndarray, target_rms: float = 3000.0,
                     noise_gate: float = 150.0) -> np.ndarray:
    """Scale audio to a consistent target RMS, immune to input gain drift.

    - If the signal RMS is below noise_gate, it is treated as silence and
      returned near-zero (prevents amplifying background hiss into fake speech).
    - Otherwise scaled so RMS == target_rms, then clipped to int16 range.

    This makes ASR independent of mic/line input level - the core fix for
    production where input gain cannot be tuned per call.
    """
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < noise_gate:
        return (audio * 0.1).astype(np.int16)  # treat as silence
    gain = target_rms / rms
    # Cap gain so we never amplify by more than 20x (avoids blowing up pure noise)
    gain = min(gain, 20.0)
    out = audio * gain
    out = np.clip(out, -32768, 32767)
    return out.astype(np.int16)


def _save_combined_wav(input_wav: str, reply_wav: str, out_path: str) -> None:
    """Concatenate input + 0.5s silence + reply into one WAV (16kHz mono)."""
    def _read(path):
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            ch = w.getnchannels()
        if ch == 2:
            frames = frames.reshape(-1, 2).mean(axis=1).astype(np.int16)
        # Resample to 16kHz
        if rate != 16000:
            n_out = int(len(frames) * 16000 / rate)
            idx = np.linspace(0, len(frames) - 1, n_out)
            frames = np.interp(idx, np.arange(len(frames)), frames.astype(np.float32)).astype(np.int16)
        return frames

    parts = []
    if input_wav and Path(input_wav).exists():
        parts.append(_read(input_wav))
    parts.append(np.zeros(8000, dtype=np.int16))  # 0.5s silence at 16kHz
    if reply_wav and Path(reply_wav).exists():
        parts.append(_read(reply_wav))

    combined = np.concatenate(parts)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(combined.tobytes())


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

    # --- Session logging setup ---
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("recordings/sessions") / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_json = session_dir / f"session_{session_id}.json"
    session_txt = session_dir / f"session_{session_id}.txt"
    session_turns = []
    turn_number = 0
    print(f"Session log: {session_dir}\n")

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
                audio_native = _normalize_audio(audio_native)
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir="recordings"
                ) as f:
                    tmp_path = f.name
                audio_io.save_wav(audio_native, tmp_path, sample_rate=44100)

                try:
                    pre_tx = asr.transcribe(tmp_path).get("text", "")
                    if _is_bad_transcript(pre_tx):
                        print(f"[SKIP] bad transcript: {pre_tx!r}")
                        _speak_didnt_catch()
                        raise _SkipTurn()
                    result = pipeline.run(
                        input_wav=tmp_path,
                        conversation=conversation,
                        skip_play=False,
                        precomputed_transcript=pre_tx,
                    )
                    perceived = result.get("latencies", {}).get("perceived_s")
                    print(f"[LATENCY] perceived_s={perceived:.3f}s")

                    # --- Save this turn ---
                    turn_number += 1
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    transcript_text = result.get("transcript", "")
                    reply_text = result.get("reply_text", "")
                    reply_wav = result.get("reply_wav", "")

                    # Combined WAV: input + 0.5s silence + reply
                    combined_path = session_dir / f"turn_{turn_number:02d}.wav"
                    try:
                        _save_combined_wav(tmp_path, reply_wav, str(combined_path))
                    except Exception as e:
                        print(f"[WARN] combined wav failed: {e}")

                    turn_record = {
                        "turn": turn_number,
                        "timestamp": ts,
                        "transcript": transcript_text,
                        "reply": reply_text,
                        "perceived_s": perceived,
                        "wav": str(combined_path),
                    }
                    session_turns.append(turn_record)

                    # Write JSON
                    with open(session_json, "w") as jf:
                        json.dump({"session_id": session_id, "turns": session_turns}, jf, indent=2)

                    # Write TXT
                    with open(session_txt, "w") as tf:
                        tf.write(f"Session {session_id}\n{'='*50}\n\n")
                        for t in session_turns:
                            tf.write(f"[Turn {t['turn']} @ {t['timestamp']}]\n")
                            tf.write(f"You:  {t['transcript']}\n")
                            tf.write(f"Bot:  {t['reply']}\n")
                            tf.write(f"Latency: {t['perceived_s']:.2f}s\n\n")

                    time.sleep(1.5)
                    vad.reset_state()
                except _SkipTurn:
                    pass
                finally:
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

def run_loop_ptt() -> None:
    """Push-to-talk loop. Press Enter to start recording, Enter again to stop.

    Eliminates the TTS-echo problem: the mic only records between the two
    Enter presses, so it never hears the assistant's own reply.
    """
    import threading
    sd = audio_io.sd

    print("Loading models...")
    asr.load_model()
    tts.load_voice()
    print("All models loaded.\n")
    print("=== PUSH TO TALK ===")
    print("Press Enter to START recording, press Enter again to STOP.\n")

    conversation = ConversationManager()

    # Session logging
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("recordings/sessions") / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_json = session_dir / f"session_{session_id}.json"
    session_txt = session_dir / f"session_{session_id}.txt"
    session_turns = []
    turn_number = 0
    print(f"Session log: {session_dir}\n")

    import sounddevice as sd
    cfg_rate = 44100
    cfg_device = audio_io._audio_cfg("input_device", 5) if hasattr(audio_io, "_audio_cfg") else 5

    try:
        while True:
            input("[Press Enter to speak] ")
            print("Recording... press Enter to stop.")

            frames = []
            stop_flag = threading.Event()

            def _record():
                with sd.InputStream(samplerate=cfg_rate, channels=1,
                                    device=cfg_device, dtype="int16") as stream:
                    while not stop_flag.is_set():
                        data, _ = stream.read(1024)
                        frames.append(data.copy())

            rec_thread = threading.Thread(target=_record, daemon=True)
            rec_thread.start()
            input()  # wait for second Enter
            stop_flag.set()
            rec_thread.join(timeout=2)

            if not frames:
                print("No audio captured.\n")
                continue

            audio = np.concatenate(frames).flatten()
            audio = _normalize_audio(audio)
            print(f"Captured {len(audio)/cfg_rate:.1f}s. Processing...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="recordings") as f:
                tmp_path = f.name
            audio_io.save_wav(audio, tmp_path, sample_rate=cfg_rate)

            try:
                pre_tx = asr.transcribe(tmp_path).get("text", "")
                if _is_bad_transcript(pre_tx):
                    print(f"[SKIP] bad transcript: {pre_tx!r}")
                    _speak_didnt_catch()
                    Path(tmp_path).unlink(missing_ok=True)
                    continue
                result = pipeline.run(input_wav=tmp_path, conversation=conversation, skip_play=False, precomputed_transcript=pre_tx)
                perceived = result.get("latencies", {}).get("perceived_s", 0.0)
                print(f"[LATENCY] perceived_s={perceived:.3f}s")

                turn_number += 1
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                combined_path = session_dir / f"turn_{turn_number:02d}.wav"
                try:
                    _save_combined_wav(tmp_path, result.get("reply_wav", ""), str(combined_path))
                except Exception as e:
                    print(f"[WARN] combined wav failed: {e}")

                session_turns.append({
                    "turn": turn_number, "timestamp": ts,
                    "transcript": result.get("transcript", ""),
                    "reply": result.get("reply_text", ""),
                    "perceived_s": perceived, "wav": str(combined_path),
                })
                with open(session_json, "w") as jf:
                    json.dump({"session_id": session_id, "turns": session_turns}, jf, indent=2)
                with open(session_txt, "w") as tf:
                    tf.write(f"Session {session_id}\n{'='*50}\n\n")
                    for t in session_turns:
                        tf.write(f"[Turn {t['turn']} @ {t['timestamp']}]\n")
                        tf.write(f"You:  {t['transcript']}\n")
                        tf.write(f"Bot:  {t['reply']}\n")
                        tf.write(f"Latency: {t['perceived_s']:.2f}s\n\n")
                print()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    except KeyboardInterrupt:
        print("\nStopping. Ending conversation session.")
        conversation.end_session()


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
    p.add_argument(
        "--ptt", action="store_true",
        help="Push-to-talk mode: press Enter to start/stop recording (no VAD, no echo).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.ptt:
        run_loop_ptt()
        sys.exit(0)
    run_loop(
        onset_chunks=args.onset_chunks,
        offset_chunks=args.offset_chunks,
    )
