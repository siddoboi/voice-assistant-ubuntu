"""
pipeline.py — Hardcoded voice assistant chain (Phase 1, Week 2 Day 3).

The minimum viable end-to-end loop. Records for a fixed duration, transcribes,
queries the LLM, synthesises the reply, plays it back. No VAD, no streaming,
no barge-in — those land later in Week 2 Day 5 and Phase 2.

Chain:
    record(5s)  ->  asr.transcribe  ->  llm_client.generate  ->  tts.synthesize  ->  audio_io.play

Inputs:
    --duration  : seconds to record (default 5, from dev_config.yaml)
    --input     : optional .wav file path; when provided, skips recording
                  entirely and runs the chain from that file. Useful on
                  WSL2 where there's no live microphone.
    --output    : where to write the TTS reply .wav (default: a timestamped
                  file under recordings/).
    --no-play   : skip playback; just write the reply file. WSL2 default.

This script is structured so each stage logs its latency. The cumulative
budget for Phase 1 end-to-end on Pi is ~2-3s per turn; numbers here on
WSL2 are reference, not representative of Pi.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make `from src import ...` work whether pipeline is run as a module
# (`python -m src.pipeline`) or as a script (`python src/pipeline.py`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import asr, audio_io, llm_client, tts


# ---------------------------------------------------------------------------
# Stage wrappers — each returns (result, latency_s) and prints a one-liner.
# ---------------------------------------------------------------------------


def _stage_record(duration_s: float, input_wav: Optional[str]) -> tuple[str, float]:
    """Either record live or use a pre-recorded .wav.

    Returns the path to the captured audio and the wall-clock seconds spent
    on this stage. When `input_wav` is supplied, we skip recording entirely
    and the reported latency is ~0.
    """
    if input_wav is not None:
        in_path = Path(input_wav)
        if not in_path.exists():
            raise FileNotFoundError(f"Input WAV not found: {in_path}")
        print(f"[1/5] Using input file: {in_path}  (skipping live record)")
        return str(in_path), 0.0

    # Live recording path — used on the Pi once it arrives. On WSL2 this
    # will most likely raise sounddevice.PortAudioError because there's no
    # audio passthrough; that's expected and is why --input exists.
    print(f"[1/5] Recording {duration_s:.1f}s of audio ...")
    t0 = time.perf_counter()
    audio_array = audio_io.record(duration_sec=duration_s)
    latency = time.perf_counter() - t0

    # Stamp the file so multiple turns in one debug session don't clobber.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "recordings" / f"pipeline_input_{ts}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_io.save_wav(audio_array, str(out_path))
    print(f"      recorded -> {out_path}  ({latency:.3f}s)")
    return str(out_path), latency


def _stage_asr(audio_path: str) -> tuple[str, float]:
    print(f"[2/5] Transcribing {audio_path} ...")
    t0 = time.perf_counter()
    result = asr.transcribe(audio_path)
    latency = time.perf_counter() - t0
    text = result.get("text", "").strip()
    print(f"      text: {text!r}  ({latency:.3f}s)")
    if not text:
        # Empty transcripts are common when --input is a sine wave; we still
        # need *some* prompt to feed the LLM, otherwise it hallucinates.
        text = "(no speech detected)"
    return text, latency


def _stage_llm(prompt: str, model: Optional[str]) -> tuple[str, float]:
    print(f"[3/5] LLM generating reply ...")
    t0 = time.perf_counter()
    # The master context locks llm_client.generate(prompt, model) -> {text, latency_s}.
    # If a project-specific signature differs, this is the only call site
    # in the pipeline that needs adjustment.
    if model is None:
        result = llm_client.generate(prompt)
    else:
        result = llm_client.generate(prompt, model=model)
    latency = time.perf_counter() - t0
    reply = (result.get("text") or "").strip()
    print(f"      reply: {reply!r}  ({latency:.3f}s)")
    if not reply:
        reply = "I'm sorry, I didn't quite catch that."
    return reply, latency


def _stage_tts(text: str, output_wav: str) -> tuple[str, float]:
    print(f"[4/5] Synthesising reply ...")
    t0 = time.perf_counter()
    result = tts.synthesize(text, output_wav)
    latency = time.perf_counter() - t0
    print(
        f"      wrote {result['output_path']}  "
        f"(audio={result['duration_s']:.2f}s, tts={latency:.3f}s, rtf={result['rtf']:.3f})"
    )
    return result["output_path"], latency


def _stage_play(wav_path: str, skip: bool) -> float:
    if skip:
        print(f"[5/5] --no-play set; skipping playback")
        return 0.0
    print(f"[5/5] Playing {wav_path} ...")
    t0 = time.perf_counter()
    audio_array, sample_rate = audio_io.load_wav(wav_path)
    audio_io.play(audio_array, sample_rate=sample_rate)
    latency = time.perf_counter() - t0
    print(f"      done  ({latency:.3f}s)")
    return latency


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(
    duration_s: float = 5.0,
    input_wav: Optional[str] = None,
    output_wav: Optional[str] = None,
    skip_play: bool = False,
    llm_model: Optional[str] = None,
) -> dict:
    """Run one full turn through the hardcoded chain.

    Args:
        duration_s: live record duration in seconds (ignored if input_wav given)
        input_wav: pre-recorded WAV path; bypasses live recording
        output_wav: where to write the TTS reply; auto-generated if None
        skip_play: don't play the reply through speakers; just write the file
        llm_model: optional Ollama model override (default = llm_client default)

    Returns:
        Dict with per-stage latencies and final paths/text — handy for tests
        and for the Day 5 carry-over once VAD wraps the same chain.
    """
    print("=" * 60)
    print("Pipeline (Day 3 hardcoded chain)")
    print("=" * 60)

    if output_wav is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "recordings" / f"pipeline_reply_{ts}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_wav = str(out_path)

    # ---- 1. record (or load) ----------------------------------------
    input_path, rec_lat = _stage_record(duration_s, input_wav)

    # ---- 2. ASR -----------------------------------------------------
    transcript, asr_lat = _stage_asr(input_path)

    # ---- 3. LLM -----------------------------------------------------
    reply_text, llm_lat = _stage_llm(transcript, llm_model)

    # ---- 4. TTS -----------------------------------------------------
    reply_wav, tts_lat = _stage_tts(reply_text, output_wav)

    # ---- 5. play ----------------------------------------------------
    play_lat = _stage_play(reply_wav, skip_play)

    total = rec_lat + asr_lat + llm_lat + tts_lat + play_lat
    print("-" * 60)
    print(f"Latencies: rec={rec_lat:.3f}s  asr={asr_lat:.3f}s  "
          f"llm={llm_lat:.3f}s  tts={tts_lat:.3f}s  play={play_lat:.3f}s")
    print(f"Total wall-clock: {total:.3f}s")
    print("=" * 60)

    return {
        "input_path": input_path,
        "transcript": transcript,
        "reply_text": reply_text,
        "reply_wav": reply_wav,
        "latencies": {
            "record_s": rec_lat,
            "asr_s": asr_lat,
            "llm_s": llm_lat,
            "tts_s": tts_lat,
            "play_s": play_lat,
            "total_s": total,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hardcoded voice assistant pipeline (Day 3): "
                    "record/load -> ASR -> LLM -> TTS -> play."
    )
    p.add_argument("--duration", type=float, default=5.0,
                   help="Seconds to record (ignored if --input given). Default: 5.")
    p.add_argument("--input", type=str, default=None,
                   help="Pre-recorded .wav file to use instead of live mic.")
    p.add_argument("--output", type=str, default=None,
                   help="Where to write the TTS reply .wav. Default: recordings/pipeline_reply_TS.wav")
    p.add_argument("--no-play", action="store_true",
                   help="Skip speaker playback. Default on WSL2 where audio passthrough is absent.")
    p.add_argument("--model", type=str, default=None,
                   help="LLM model override (e.g. 'tinyllama:1.1b'). "
                        "Default uses llm_client's primary model.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        run(
            duration_s=args.duration,
            input_wav=args.input,
            output_wav=args.output,
            skip_play=args.no_play,
            llm_model=args.model,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        # Don't swallow tracebacks during development — re-raise after a
        # human-readable line so the cause is obvious in the log.
        print(f"\nERROR ({type(e).__name__}): {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())