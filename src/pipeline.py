"""
pipeline.py — Streaming voice assistant chain (Week 3 Day 4).

Replaces the Day 3 hardcoded single-shot chain with a streaming one:

    record/load -> asr.transcribe
                -> ConversationManager.add_user_turn / build_messages
                -> llm_client.stream_sentences_from_messages   (sentence stream)
                -> tts.synthesize_stream                       (PCM per sentence)
                -> bounded asyncio.Queue (back-pressure)       -> audio_io.play

Key properties:
    - Multi-turn memory + system prompt via ConversationManager.
    - First audio plays as soon as the first sentence is synthesised, while
      later sentences are still being generated — the streaming latency win.
    - A bounded asyncio.Queue(maxsize=tts_buffer_max_chunks) sits between TTS
      synthesis and playback so synthesis cannot run unboundedly ahead and
      exhaust memory on the Pi (back-pressure).
    - The full reply text is reassembled from the sentence stream and stored
      with ConversationManager.add_assistant_turn.

run() handles its own ConversationManager session by default (create + end),
or accepts a caller-supplied `conversation` for multi-turn call loops, in
which case the caller owns end_session() (called on hangup).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import asr, audio_io, llm_client, tts
from src.conversation import ConversationManager

# Fallbacks (kept from the Day 3 pipeline).
FALLBACK_TRANSCRIPT = "(no speech detected)"
FALLBACK_REPLY = "I'm sorry, I didn't quite catch that."

# Sentinel used to close the producer→consumer queue.
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _max_chunks(config_path: Optional[str]) -> int:
    """Read pipeline.tts_buffer_max_chunks (default 3) from config.

    Reuses audio_io's cached config loader since the pipeline already depends
    on audio_io. The bound caps how far TTS synthesis may run ahead of
    playback before back-pressure pauses it.
    """
    cfg = audio_io._get_config(config_path).get("pipeline", {}) or {}
    value = cfg.get("tts_buffer_max_chunks", 3)
    return int(value) if int(value) > 0 else 3


# ---------------------------------------------------------------------------
# Record / ASR stages (unchanged in spirit from Day 3)
# ---------------------------------------------------------------------------


def _stage_record(duration_s: float, input_wav: Optional[str]) -> tuple[str, float]:
    if input_wav is not None:
        in_path = Path(input_wav)
        if not in_path.exists():
            raise FileNotFoundError(f"Input WAV not found: {in_path}")
        print(f"[1/4] Using input file: {in_path}  (skipping live record)")
        return str(in_path), 0.0

    print(f"[1/4] Recording {duration_s:.1f}s of audio ...")
    t0 = time.perf_counter()
    audio_array = audio_io.record(duration_sec=duration_s)
    latency = time.perf_counter() - t0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "recordings" / f"pipeline_input_{ts}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_io.save_wav(audio_array, str(out_path))
    print(f"      recorded -> {out_path}  ({latency:.3f}s)")
    return str(out_path), latency


def _stage_asr(audio_path: str) -> tuple[str, float]:
    print(f"[2/4] Transcribing {audio_path} ...")
    t0 = time.perf_counter()
    result = asr.transcribe(audio_path)
    latency = time.perf_counter() - t0
    text = (result.get("text") or "").strip()
    print(f"      text: {text!r}  ({latency:.3f}s)")
    if not text:
        text = FALLBACK_TRANSCRIPT
    return text, latency


# ---------------------------------------------------------------------------
# Streaming LLM -> TTS -> playback with a bounded queue (back-pressure)
# ---------------------------------------------------------------------------


def _recording_iter(sentence_iter, collected: list[str]):
    """Pass sentences through to TTS while recording each for the reply text."""
    for sentence in sentence_iter:
        collected.append(sentence)
        yield sentence


def _safe_next(gen, stats_holder: dict):
    """next(gen) that captures the generator's StopIteration.value as stats."""
    try:
        return next(gen)
    except StopIteration as stop:
        stats_holder["stats"] = stop.value or {}
        return _SENTINEL


def _play_chunk(chunk, sample_rate: int) -> None:
    audio_io.play(chunk, sample_rate=sample_rate)


async def _produce(queue, sentence_iter, sample_rate, tts_stats) -> None:
    """Synthesise PCM per sentence (off-thread) and feed the bounded queue.

    `await queue.put()` blocks when the queue is full → synthesis pauses →
    back-pressure. Each synth runs in the default executor so it overlaps
    with playback of the previous chunk.
    """
    loop = asyncio.get_event_loop()
    gen = tts.synthesize_stream(sentence_iter, sample_rate=sample_rate)
    try:
        while True:
            chunk = await loop.run_in_executor(None, _safe_next, gen, tts_stats)
            if chunk is _SENTINEL:
                break
            await queue.put(chunk)
    finally:
        # Always unblock the consumer, even if synthesis/LLM raised — otherwise
        # the consumer waits on an empty queue forever (deadlock).
        await queue.put(_SENTINEL)


async def _consume(queue, sample_rate, skip_play, t0, result) -> None:
    """Drain the queue, playing each chunk (off-thread) and timing first audio."""
    loop = asyncio.get_event_loop()
    chunks: list = []
    while True:
        chunk = await queue.get()
        if chunk is _SENTINEL:
            queue.task_done()
            break
        if result["first_audio_s"] is None:
            result["first_audio_s"] = time.perf_counter() - t0
        chunks.append(chunk)
        if not skip_play:
            await loop.run_in_executor(None, _play_chunk, chunk, sample_rate)
        queue.task_done()
    result["chunks"] = chunks


async def _stream_and_play_async(sentence_iter, sample_rate, max_chunks, skip_play, t0):
    queue: asyncio.Queue = asyncio.Queue(maxsize=max_chunks)
    tts_stats: dict = {}
    result = {"first_audio_s": None, "chunks": []}
    producer = asyncio.create_task(
        _produce(queue, sentence_iter, sample_rate, tts_stats)
    )
    await _consume(queue, sample_rate, skip_play, t0, result)
    await producer
    return result["chunks"], result["first_audio_s"], tts_stats.get("stats", {})


def _stream_and_play(sentence_iter, sample_rate, max_chunks, skip_play):
    """Sync wrapper: run the producer/consumer event loop to completion.

    Returns (chunks, first_audio_s, tts_stats).
    """
    t0 = time.perf_counter()
    return asyncio.run(
        _stream_and_play_async(sentence_iter, sample_rate, max_chunks, skip_play, t0)
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(
    duration_s: float = 5.0,
    input_wav: Optional[str] = None,
    output_wav: Optional[str] = None,
    skip_play: bool = False,
    llm_model: Optional[str] = None,
    conversation: Optional[ConversationManager] = None,
    session_id: Optional[str] = None,
    config_path: Optional[str] = None,
) -> dict:
    """Run one streaming turn through the chain.

    Args:
        duration_s: live record duration (ignored if input_wav given).
        input_wav: pre-recorded WAV path; bypasses live recording.
        output_wav: where to write the concatenated TTS reply; auto if None.
        skip_play: don't play through speakers (still writes output_wav).
        llm_model: optional Ollama model override.
        conversation: optional caller-owned ConversationManager for multi-turn
            loops. If None, run() creates and ends its own session.
        session_id: resume an existing session (only when conversation is None).
        config_path: optional explicit config path.

    Returns:
        Dict with transcript, reply_text, session_id, reply_wav, counts, and a
        per-stage latency dict.
    """
    print("=" * 60)
    print("Pipeline (Day 4 streaming chain)")
    print("=" * 60)

    if output_wav is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "recordings" / f"pipeline_reply_{ts}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_wav = str(out_path)

    # ---- 1. record / load ----
    input_path, rec_lat = _stage_record(duration_s, input_wav)

    # ---- 2. ASR ----
    transcript, asr_lat = _stage_asr(input_path)

    sample_rate = tts.output_sample_rate()
    max_chunks = _max_chunks(config_path)

    own_conversation = conversation is None
    if own_conversation:
        conversation = ConversationManager(config_path=config_path, session_id=session_id)

    t_stream0 = time.perf_counter()
    try:
        # ---- 3. conversation ----
        conversation.add_user_turn(transcript)
        messages = conversation.build_messages()

        # ---- 4. stream LLM -> TTS -> play ----
        print(f"[3/4] Streaming reply (buffer max {max_chunks} chunks) ...")
        collected: list[str] = []
        if llm_model is None:
            sentence_iter = llm_client.stream_sentences_from_messages(messages)
        else:
            sentence_iter = llm_client.stream_sentences_from_messages(messages, model=llm_model)
        chunks, first_audio_s, _ = _stream_and_play(
            _recording_iter(sentence_iter, collected), sample_rate, max_chunks, skip_play
        )
        reply_text = " ".join(collected).strip()

        # Empty reply → speak a fallback so the caller always hears something.
        if not reply_text:
            reply_text = FALLBACK_REPLY
            fb_chunks, fb_first, _ = _stream_and_play(
                iter([reply_text]), sample_rate, max_chunks, skip_play
            )
            chunks = fb_chunks
            if first_audio_s is None:
                first_audio_s = fb_first

        conversation.add_assistant_turn(reply_text)
        session_id_out = conversation.session_id
    finally:
        if own_conversation:
            conversation.end_session()

    stream_lat = time.perf_counter() - t_stream0
    if first_audio_s is None:
        first_audio_s = 0.0

    # ---- 5. write concatenated reply wav (debug/inspection) ----
    print(f"[4/4] Writing reply -> {output_wav}")
    if chunks:
        full = np.concatenate(chunks)
        audio_io.save_wav(full, output_wav, sample_rate=sample_rate)

    total = rec_lat + asr_lat + stream_lat
    perceived = asr_lat + first_audio_s

    print("-" * 60)
    print(f"transcript : {transcript!r}")
    print(f"reply      : {reply_text!r}")
    print(f"Latencies  : rec={rec_lat:.3f}s asr={asr_lat:.3f}s "
          f"first_audio={first_audio_s:.3f}s stream={stream_lat:.3f}s")
    print(f"Perceived (asr+first_audio): {perceived:.3f}s   Total: {total:.3f}s")
    print("=" * 60)

    return {
        "input_path": input_path,
        "transcript": transcript,
        "reply_text": reply_text,
        "reply_wav": output_wav,
        "session_id": session_id_out,
        "num_sentences": len(collected),
        "num_audio_chunks": len(chunks),
        "latencies": {
            "record_s": rec_lat,
            "asr_s": asr_lat,
            "first_audio_s": first_audio_s,
            "stream_s": stream_lat,
            "perceived_s": perceived,
            "total_s": total,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Streaming voice assistant pipeline (Day 4): "
                    "record/load -> ASR -> ConversationManager -> streaming "
                    "LLM -> streaming TTS -> play."
    )
    p.add_argument("--duration", type=float, default=5.0,
                   help="Seconds to record (ignored if --input given). Default: 5.")
    p.add_argument("--input", type=str, default=None,
                   help="Pre-recorded .wav file to use instead of live mic.")
    p.add_argument("--output", type=str, default=None,
                   help="Where to write the TTS reply .wav.")
    p.add_argument("--no-play", action="store_true",
                   help="Skip speaker playback (still writes the reply file).")
    p.add_argument("--model", type=str, default=None,
                   help="LLM model override (e.g. 'tinyllama:1.1b').")
    p.add_argument("--session-id", type=str, default=None,
                   help="Resume an existing ConversationManager session.")
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
            session_id=args.session_id,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\nERROR ({type(e).__name__}): {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())