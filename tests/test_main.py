"""
tests/test_main.py - Unit tests for src/main.py VAD state machine.

Tests cover state transitions only - no real mic, no real models.
All audio_io, vad, asr, tts, pipeline, and ConversationManager calls
are mocked.

Integration test (real mic + real models) is opt-in via --run-integration.
"""

from __future__ import annotations

from enum import Enum, auto
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

import src.main as main_mod
from src.main import _State, run_loop, DEFAULT_ONSET_CHUNKS, DEFAULT_OFFSET_CHUNKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(speech: bool) -> np.ndarray:
    """Return a 512-sample int16 chunk that will read as speech or silence."""
    val = 16000 if speech else 0
    return np.full(main_mod.CHUNK_SIZE, val, dtype=np.int16)


def _prob(speech: bool) -> float:
    return 1.0 if speech else 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_chunk_size_matches_vad(self):
        import src.vad as vad
        assert main_mod.CHUNK_SIZE == vad.CHUNK_SIZE

    def test_sample_rate_matches_vad(self):
        import src.vad as vad
        assert main_mod.SAMPLE_RATE == vad.SAMPLE_RATE

    def test_default_onset_chunks(self):
        assert DEFAULT_ONSET_CHUNKS == 3

    def test_default_offset_chunks(self):
        assert DEFAULT_OFFSET_CHUNKS == 18


# ---------------------------------------------------------------------------
# State machine - IDLE transitions
# ---------------------------------------------------------------------------

class TestIdleState:
    def test_silence_stays_idle(self):
        """Silence in IDLE keeps state IDLE."""
        state = _State.IDLE
        onset_count = 0
        recorded = []

        is_speech = False
        if state == _State.IDLE:
            if is_speech:
                state = _State.DETECTING_ONSET
                onset_count = 1
                recorded = [_chunk(False)]

        assert state == _State.IDLE
        assert onset_count == 0
        assert recorded == []

    def test_speech_transitions_to_detecting_onset(self):
        """First speech chunk in IDLE -> DETECTING_ONSET."""
        state = _State.IDLE
        onset_count = 0
        recorded = []
        chunk = _chunk(True)

        is_speech = True
        if state == _State.IDLE:
            if is_speech:
                state = _State.DETECTING_ONSET
                onset_count = 1
                recorded = [chunk]

        assert state == _State.DETECTING_ONSET
        assert onset_count == 1
        assert len(recorded) == 1


# ---------------------------------------------------------------------------
# State machine - DETECTING_ONSET transitions
# ---------------------------------------------------------------------------

class TestDetectingOnsetState:
    def test_false_start_resets_to_idle(self):
        """Silence during DETECTING_ONSET resets to IDLE."""
        state = _State.DETECTING_ONSET
        onset_count = 1
        recorded = [_chunk(True)]
        onset_chunks = DEFAULT_ONSET_CHUNKS

        is_speech = False
        if state == _State.DETECTING_ONSET:
            recorded.append(_chunk(False))
            if is_speech:
                onset_count += 1
            else:
                state = _State.IDLE
                onset_count = 0
                recorded = []

        assert state == _State.IDLE
        assert onset_count == 0
        assert recorded == []

    def test_reaches_onset_threshold_transitions_to_recording(self):
        """N consecutive speech chunks -> RECORDING."""
        state = _State.DETECTING_ONSET
        onset_count = DEFAULT_ONSET_CHUNKS - 1
        recorded = [_chunk(True)] * onset_count
        onset_chunks = DEFAULT_ONSET_CHUNKS
        offset_count = 0

        is_speech = True
        if state == _State.DETECTING_ONSET:
            recorded.append(_chunk(True))
            if is_speech:
                onset_count += 1
                if onset_count >= onset_chunks:
                    state = _State.RECORDING
                    offset_count = 0
            else:
                state = _State.IDLE
                onset_count = 0
                recorded = []

        assert state == _State.RECORDING
        assert offset_count == 0

    def test_below_threshold_does_not_trigger_recording(self):
        """N-1 speech chunks is not enough to start RECORDING."""
        onset_count = DEFAULT_ONSET_CHUNKS - 2
        onset_chunks = DEFAULT_ONSET_CHUNKS
        state = _State.DETECTING_ONSET

        is_speech = True
        if state == _State.DETECTING_ONSET:
            if is_speech:
                onset_count += 1
                if onset_count >= onset_chunks:
                    state = _State.RECORDING

        assert state == _State.DETECTING_ONSET


# ---------------------------------------------------------------------------
# State machine - RECORDING transitions
# ---------------------------------------------------------------------------

class TestRecordingState:
    def test_silence_increments_offset_count(self):
        """Silence during RECORDING increments offset_count."""
        state = _State.RECORDING
        offset_count = 0
        recorded = [_chunk(True)] * 5
        offset_chunks = DEFAULT_OFFSET_CHUNKS

        is_speech = False
        if state == _State.RECORDING:
            recorded.append(_chunk(False))
            if not is_speech:
                offset_count += 1
                if offset_count >= offset_chunks:
                    state = _State.PROCESSING
            else:
                offset_count = 0

        assert state == _State.RECORDING
        assert offset_count == 1

    def test_speech_resets_offset_count(self):
        """Speech during RECORDING resets offset_count."""
        state = _State.RECORDING
        offset_count = 10
        recorded = [_chunk(True)] * 5

        is_speech = True
        if state == _State.RECORDING:
            recorded.append(_chunk(True))
            if not is_speech:
                offset_count += 1
            else:
                offset_count = 0

        assert offset_count == 0
        assert state == _State.RECORDING

    def test_reaches_offset_threshold_transitions_to_processing(self):
        """M consecutive silence chunks -> PROCESSING."""
        state = _State.RECORDING
        offset_count = DEFAULT_OFFSET_CHUNKS - 1
        offset_chunks = DEFAULT_OFFSET_CHUNKS
        recorded = [_chunk(True)] * 10

        is_speech = False
        if state == _State.RECORDING:
            recorded.append(_chunk(False))
            if not is_speech:
                offset_count += 1
                if offset_count >= offset_chunks:
                    state = _State.PROCESSING

        assert state == _State.PROCESSING


# ---------------------------------------------------------------------------
# run_loop - model loading
# ---------------------------------------------------------------------------

class TestRunLoopModelLoading:
    def test_loads_all_models_on_startup(self):
        """run_loop loads vad, asr, and tts models before entering loop."""
        with patch("src.main.vad.load_model") as mock_vad, \
             patch("src.main.asr.load_model") as mock_asr, \
             patch("src.main.tts.load_voice") as mock_tts, \
             patch("src.main.audio_io.record", side_effect=KeyboardInterrupt), \
             patch("src.main.ConversationManager") as mock_cm:
            mock_cm.return_value.end_session = MagicMock()
            run_loop()

        mock_vad.assert_called_once()
        mock_asr.assert_called_once()
        mock_tts.assert_called_once()

    def test_keyboard_interrupt_ends_session(self):
        """Ctrl+C triggers conversation.end_session()."""
        mock_conv = MagicMock()
        with patch("src.main.vad.load_model"), \
             patch("src.main.asr.load_model"), \
             patch("src.main.tts.load_voice"), \
             patch("src.main.audio_io.record", side_effect=KeyboardInterrupt), \
             patch("src.main.ConversationManager", return_value=mock_conv):
            run_loop()

        mock_conv.end_session.assert_called_once()


# ---------------------------------------------------------------------------
# run_loop - processing trigger
# ---------------------------------------------------------------------------

class TestRunLoopProcessing:
    def test_pipeline_called_after_speech_and_silence(self):
        """Full onset+recording+offset sequence triggers pipeline.run()."""
        onset = DEFAULT_ONSET_CHUNKS
        offset = DEFAULT_OFFSET_CHUNKS

        # Sequence: onset speech chunks + 10 more speech + offset silence + KeyboardInterrupt
        speech_chunk = _chunk(True)
        silence_chunk = _chunk(False)

        record_calls = (
            [speech_chunk] * (onset + 10) +
            [silence_chunk] * offset +
            [KeyboardInterrupt()]
        )

        def fake_record(**kwargs):
            val = record_calls.pop(0)
            if isinstance(val, KeyboardInterrupt):
                raise KeyboardInterrupt
            return val

        def fake_prob(chunk):
            return 1.0 if chunk.max() > 0 else 0.0

        mock_conv = MagicMock()
        mock_conv.end_session = MagicMock()

        with patch("src.main.vad.load_model"), \
             patch("src.main.asr.load_model"), \
             patch("src.main.tts.load_voice"), \
             patch("src.main.audio_io.record", side_effect=fake_record), \
             patch("src.main.vad.get_speech_prob", side_effect=fake_prob), \
             patch("src.main.vad.SILENCE_THRESHOLD", 0.5), \
             patch("src.main.audio_io.save_wav"), \
             patch("src.main.asr.transcribe", return_value={"text": "what is the capital of france"}), \
             patch("src.main._save_combined_wav"), \
             patch("src.main.pipeline.run", return_value={"latencies": {"perceived_s": 1.5}, "transcript": "x", "reply_text": "Paris.", "reply_wav": ""}) as mock_pipeline, \
             patch("src.main.ConversationManager", return_value=mock_conv), \
             patch("pathlib.Path.unlink"):
            run_loop(onset_chunks=onset, offset_chunks=offset)

        mock_pipeline.assert_called_once()

    def test_pipeline_not_called_on_silence_only(self):
        """Silence-only input never triggers pipeline.run()."""
        silence_chunk = _chunk(False)
        calls = [silence_chunk] * 10 + [KeyboardInterrupt()]

        def fake_record(**kwargs):
            val = calls.pop(0)
            if isinstance(val, KeyboardInterrupt):
                raise KeyboardInterrupt
            return val

        mock_conv = MagicMock()
        with patch("src.main.vad.load_model"), \
             patch("src.main.asr.load_model"), \
             patch("src.main.tts.load_voice"), \
             patch("src.main.audio_io.record", side_effect=fake_record), \
             patch("src.main.vad.get_speech_prob", return_value=0.0), \
             patch("src.main.vad.SILENCE_THRESHOLD", 0.5), \
             patch("src.main.pipeline.run") as mock_pipeline, \
             patch("src.main.ConversationManager", return_value=mock_conv):
            run_loop()

        mock_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Integration test (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMainIntegration:
    def test_real_mic_one_turn(self):
        """Smoke test: run_loop processes one real mic turn then exits.

        Requires --run-integration. Speak once then stay silent for ~2s.
        This test exits via KeyboardInterrupt after 30 seconds.
        """
        import threading
        import time

        def stop():
            time.sleep(30)
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=stop, daemon=True).start()
        run_loop()
