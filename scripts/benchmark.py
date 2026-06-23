#!/usr/bin/env python3
"""Run the full performance benchmark suite and emit numbers for models.yaml.

On Ubuntu Day 2 this is run once to populate every Pi-specific latency/RTF field
in configs/models.yaml:

    LLM  : first-token + total latency for primary and fallback models
    ASR  : real-time factor (RTF) on a sample WAV
    TTS  : RTF over 5 standard sentences
    VAD  : speech_ratio + processing latency on a sample WAV
    E2E  : pipeline.run() perceived_s on a sample WAV

Results are printed as a formatted table and saved to
recordings/ubuntu_benchmarks.json. The same script runs unchanged on WSL2 for a
dev baseline.

RTF (real-time factor) = processing_time / audio_duration. Lower is better;
< 1.0 means faster than real time, which is required for live calls.

No new dependencies: drives the existing src modules (asr, tts, vad,
llm_client, pipeline) through their documented entry points only.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --input recordings/sample2.wav
    python scripts/benchmark.py --output recordings/ubuntu_benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- Make the project importable and fix CWD ------------------------------
# Same rationale as tune_vad_threshold.py: src modules use relative model
# paths, so CWD must be the project root. Resolve from __file__.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src import asr, llm_client, pipeline, tts, vad  # noqa: E402

DEFAULT_INPUT = Path("recordings") / "sample1.wav"
DEFAULT_OUTPUT = Path("recordings") / "ubuntu_benchmarks.json"

# Five standard sentences for the TTS RTF measurement. Mix of lengths so the
# RTF reflects realistic phone-reply phrasing rather than one outlier.
TTS_SENTENCES: tuple[str, ...] = (
    "Hello, thanks for calling. How can I help you today?",
    "I'm sorry, I didn't quite catch that. Could you say it again?",
    "Let me check that for you.",
    "Your appointment is confirmed for Tuesday at three in the afternoon.",
    "Is there anything else I can help you with before you go?",
)


# --------------------------------------------------------------------------
# Individual benchmarks
# --------------------------------------------------------------------------
def benchmark_llm() -> dict:
    """Measure first-token and total latency for primary and fallback models.

    Returns:
        Dict keyed "primary"/"fallback", each a llm_client.measure_latency()
        result ({model, first_token_latency_s, total_latency_s, response}).
    """
    print("  [LLM] primary  …", flush=True)
    primary = llm_client.measure_latency(llm_client.PRIMARY_MODEL)
    print("  [LLM] fallback …", flush=True)
    fallback = llm_client.measure_latency(llm_client.FALLBACK_MODEL)
    return {"primary": primary, "fallback": fallback}


def benchmark_asr(wav_path: Path) -> dict:
    """Measure ASR latency and RTF on a WAV file.

    asr.transcribe() returns {text, duration_s, latency_s}. RTF is
    latency_s / duration_s.

    Returns:
        Dict with model, duration_s, latency_s, rtf, and a transcript preview.
    """
    print("  [ASR] transcribing …", flush=True)
    result = asr.transcribe(str(wav_path))
    duration = float(result.get("duration_s") or 0.0)
    latency = float(result.get("latency_s") or 0.0)
    rtf = round(latency / duration, 4) if duration > 0 else None
    return {
        "model": "tiny.en",
        "duration_s": round(duration, 3),
        "latency_s": round(latency, 3),
        "rtf": rtf,
        "transcript_preview": (result.get("text") or "")[:80],
    }


def benchmark_tts() -> dict:
    """Measure TTS RTF over the 5 standard sentences.

    Drives tts.synthesize_stream() (the documented streaming entry point),
    summing produced audio samples to get total audio duration and using
    wall-clock to get synthesis time. RTF = synth_time / audio_duration.

    Returns:
        Dict with sample_rate, num_sentences, audio_duration_s,
        synth_time_s, time_to_first_audio_s, and rtf.
    """
    print("  [TTS] synthesising 5 sentences …", flush=True)
    sample_rate = int(tts.output_sample_rate())

    total_samples = 0
    stats: dict = {}
    t0 = time.perf_counter()
    gen = tts.synthesize_stream(TTS_SENTENCES, sample_rate=sample_rate)
    try:
        while True:
            chunk = next(gen)
            total_samples += int(np.asarray(chunk).size)
    except StopIteration as stop:
        stats = stop.value or {}
    synth_time = time.perf_counter() - t0

    audio_duration = total_samples / sample_rate if sample_rate else 0.0
    rtf = round(synth_time / audio_duration, 4) if audio_duration > 0 else None
    return {
        "model": "en_US-amy-medium",
        "sample_rate": sample_rate,
        "num_sentences": stats.get("num_sentences", len(TTS_SENTENCES)),
        "audio_duration_s": round(audio_duration, 3),
        "synth_time_s": round(synth_time, 3),
        "time_to_first_audio_s": stats.get("time_to_first_audio_s"),
        "rtf": rtf,
    }


def benchmark_vad(wav_path: Path) -> dict:
    """Measure VAD speech_ratio and processing latency on a WAV file.

    Uses vad.test_on_file(), which returns speech_ratio and latency_s among
    other fields.

    Returns:
        Dict with model, speech_ratio, latency_s, total_chunks, and the
        threshold the run used (the module default).
    """
    print("  [VAD] scanning …", flush=True)
    result = vad.test_on_file(str(wav_path))
    return {
        "model": "silero_vad_v4",
        "threshold_used": vad.SILENCE_THRESHOLD,
        "speech_ratio": result.get("speech_ratio"),
        "latency_s": result.get("latency_s"),
        "total_chunks": result.get("total_chunks"),
    }


def benchmark_e2e(wav_path: Path) -> dict:
    """Measure end-to-end pipeline perceived latency on a WAV file.

    Runs pipeline.run() with the WAV as input and playback disabled. The
    returned latencies dict carries perceived_s (= asr_s + first_audio_s).

    Returns:
        Dict with perceived_s plus the component latencies and counts.
    """
    print("  [E2E] running full pipeline (no playback) …", flush=True)
    result = pipeline.run(input_wav=str(wav_path), skip_play=True)
    latencies = result.get("latencies", {})
    return {
        "perceived_s": latencies.get("perceived_s"),
        "asr_s": latencies.get("asr_s"),
        "first_audio_s": latencies.get("first_audio_s"),
        "stream_s": latencies.get("stream_s"),
        "total_s": latencies.get("total_s"),
        "num_sentences": result.get("num_sentences"),
        "num_audio_chunks": result.get("num_audio_chunks"),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt(value, suffix: str = "") -> str:
    """Format a possibly-None numeric value for the table."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def print_table(report: dict) -> None:
    """Print the benchmark report as an aligned text table."""
    llm = report["llm"]
    asr_r = report["asr"]
    tts_r = report["tts"]
    vad_r = report["vad"]
    e2e = report["e2e"]

    print("\n" + "=" * 64)
    print(f"  Ubuntu x86_64 Benchmark Suite — input: {report['input']}")
    print("=" * 64)

    print("\n  LLM latency")
    print(f"  {'model':<34}{'first token':>14}{'total':>14}")
    print("  " + "-" * 60)
    for key in ("primary", "fallback"):
        m = llm[key]
        print(
            f"  {m['model']:<34}"
            f"{_fmt(m.get('first_token_latency_s'), 's'):>14}"
            f"{_fmt(m.get('total_latency_s'), 's'):>14}"
        )

    print("\n  ASR / TTS / VAD")
    print(f"  {'stage':<14}{'metric':<22}{'value':>16}")
    print("  " + "-" * 52)
    print(f"  {'ASR':<14}{'RTF':<22}{_fmt(asr_r.get('rtf')):>16}")
    print(f"  {'ASR':<14}{'latency':<22}{_fmt(asr_r.get('latency_s'), 's'):>16}")
    print(f"  {'TTS':<14}{'RTF':<22}{_fmt(tts_r.get('rtf')):>16}")
    print(
        f"  {'TTS':<14}{'first audio':<22}"
        f"{_fmt(tts_r.get('time_to_first_audio_s'), 's'):>16}"
    )
    print(f"  {'VAD':<14}{'speech_ratio':<22}{_fmt(vad_r.get('speech_ratio')):>16}")
    print(f"  {'VAD':<14}{'latency':<22}{_fmt(vad_r.get('latency_s'), 's'):>16}")

    print("\n  End-to-end (pipeline.run)")
    print("  " + "-" * 52)
    print(f"  {'perceived_s':<22}{_fmt(e2e.get('perceived_s'), 's'):>30}")
    print(f"  {'asr_s':<22}{_fmt(e2e.get('asr_s'), 's'):>30}")
    print(f"  {'first_audio_s':<22}{_fmt(e2e.get('first_audio_s'), 's'):>30}")
    print(f"  {'total_s':<22}{_fmt(e2e.get('total_s'), 's'):>30}")
    print("=" * 64 + "\n")


def run_all(wav_path: Path, output_path: Path) -> dict:
    """Run every benchmark, print the table, and save JSON.

    Args:
        wav_path: Sample WAV used for ASR, VAD, and end-to-end stages.
        output_path: Where to write the JSON results.

    Returns:
        The full results dict.
    """
    print(f"Running benchmark suite (input: {wav_path}) …")
    report = {
        "input": str(wav_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm": benchmark_llm(),
        "asr": benchmark_asr(wav_path),
        "tts": benchmark_tts(),
        "vad": benchmark_vad(wav_path),
        "e2e": benchmark_e2e(wav_path),
    }

    print_table(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {output_path}\n")

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full LLM/ASR/TTS/VAD/end-to-end benchmark suite and "
            "save numbers for configs/models.yaml."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Sample WAV for ASR/VAD/E2E stages (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write JSON results (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    wav_path = args.input
    if not wav_path.is_absolute() and not wav_path.exists():
        candidate = _PROJECT_ROOT / wav_path
        if candidate.exists():
            wav_path = candidate

    if not wav_path.exists():
        print(f"error: input WAV not found: {args.input}", file=sys.stderr)
        return 2

    run_all(wav_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())