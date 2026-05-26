"""
test_asr.py — Unit tests for src/asr.py.

Two layers:
  - Unit tests (default): mock faster_whisper.WhisperModel so we can
    verify load caching, transcribe() shape, beam_size/language params,
    error propagation, and timing — without downloading or running the
    real model.
  - Integration tests (--run-integration): run real faster-whisper
    tiny.en on a generated .wav and on the project's recordings/sample
    files if present. Validates the real model still returns the
    documented contract.

Run:
    pytest tests/test_asr.py -v
    pytest tests/test_asr.py -v --run-integration
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import asr


# ---------------------------------------------------------------------------
# Reset module state between tests — _model is cached at module level.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_asr_model_cache():
    asr._model = None
    yield
    asr._model = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(text: str) -> SimpleNamespace:
    """faster-whisper segments expose a .text attribute."""
    return SimpleNamespace(text=text)


def _make_info(duration: float) -> SimpleNamespace:
    return SimpleNamespace(duration=duration)


def _fake_model_returning(text: str, duration: float = 1.0) -> MagicMock:
    """Build a MagicMock that behaves like a faster-whisper WhisperModel."""
    model = MagicMock()
    # transcribe() returns (iterable_of_segments, info)
    model.transcribe.return_value = ([_make_segment(text)], _make_info(duration))
    return model


def _patch_whisper_model():
    """Patch WhisperModel regardless of how src.asr imported it.

    Handles all import styles:
      - `from faster_whisper import WhisperModel` at top  -> src.asr.WhisperModel
      - `import faster_whisper as fw` at top              -> src.asr.fw.WhisperModel
      - `import faster_whisper`                           -> src.asr.faster_whisper.WhisperModel
      - Lazy import inside load_model()                   -> patch faster_whisper.WhisperModel
        on the module object itself via the canonical import
    """
    if hasattr(asr, "WhisperModel"):
        return patch.object(asr, "WhisperModel")
    if hasattr(asr, "fw") and hasattr(asr.fw, "WhisperModel"):
        return patch.object(asr.fw, "WhisperModel")
    if hasattr(asr, "faster_whisper") and hasattr(asr.faster_whisper, "WhisperModel"):
        return patch.object(asr.faster_whisper, "WhisperModel")
    # Lazy-import fallback: patch the canonical module attribute, which
    # all `import faster_whisper` / `from faster_whisper import X` calls
    # resolve through.
    import faster_whisper
    return patch.object(faster_whisper, "WhisperModel")


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_model_size_is_tiny_en(self):
        assert asr.MODEL_SIZE == "tiny.en"

    def test_model_cache_starts_empty(self):
        assert asr._model is None


# ---------------------------------------------------------------------------
# load_model — caching behaviour
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_loads_with_cpu_int8(self):
        with _patch_whisper_model() as MockModel:
            MockModel.return_value = MagicMock(name="WhisperInstance")
            asr.load_model()
            MockModel.assert_called_once_with("tiny.en", device="cpu", compute_type="int8")

    def test_caches_model_across_calls(self):
        with _patch_whisper_model() as MockModel:
            MockModel.return_value = MagicMock(name="WhisperInstance")
            first = asr.load_model()
            second = asr.load_model()
            third = asr.load_model()
            assert first is second is third
            assert MockModel.call_count == 1

    def test_returns_cached_instance(self):
        sentinel = MagicMock(name="WhisperInstance")
        asr._model = sentinel
        with _patch_whisper_model() as MockModel:
            result = asr.load_model()
            assert result is sentinel
            MockModel.assert_not_called()


# ---------------------------------------------------------------------------
# transcribe — contract and parameter passing
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_returns_expected_keys(self, short_wav: Path):
        asr._model = _fake_model_returning("hello world", duration=2.5)
        result = asr.transcribe(str(short_wav))
        assert set(result.keys()) == {"text", "duration_s", "latency_s"}

    def test_text_concatenates_segments(self):
        model = MagicMock()
        model.transcribe.return_value = (
            [_make_segment(" hello"), _make_segment(" world ")],
            _make_info(1.0),
        )
        asr._model = model
        result = asr.transcribe("/dev/null")
        # Segments stripped and joined with space, outer string stripped too.
        assert result["text"] == "hello world"

    def test_duration_from_info(self):
        asr._model = _fake_model_returning("x", duration=12.34)
        result = asr.transcribe("/dev/null")
        assert result["duration_s"] == 12.34

    def test_passes_beam_size_and_language(self):
        model = _fake_model_returning("x")
        asr._model = model
        asr.transcribe("recordings/sample1.wav")
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["beam_size"] == 5
        assert kwargs["language"] == "en"

    def test_latency_is_positive(self):
        """transcribe must report latency from perf_counter, not raw 0.0."""
        model = MagicMock()
        # Make transcribe sleep a bit so latency is measurable.
        def slow_transcribe(*args, **kwargs):
            time.sleep(0.01)
            return ([_make_segment("ok")], _make_info(1.0))
        model.transcribe.side_effect = slow_transcribe
        asr._model = model
        result = asr.transcribe("/dev/null")
        assert result["latency_s"] >= 0.01
        assert result["latency_s"] < 1.0  # but not absurd

    def test_propagates_model_errors(self):
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("decoder boom")
        asr._model = model
        with pytest.raises(RuntimeError, match="decoder boom"):
            asr.transcribe("/dev/null")

    def test_empty_transcript_handled(self):
        """A file with no recognisable speech yields empty text — not a crash."""
        asr._model = _fake_model_returning("", duration=0.5)
        result = asr.transcribe("/dev/null")
        assert result["text"] == ""
        assert result["duration_s"] == 0.5


# ---------------------------------------------------------------------------
# Integration — real faster-whisper tiny.en
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealModel:
    """Slow tests — opt in with --run-integration."""

    def test_real_model_loads(self):
        model = asr.load_model()
        assert model is not None

    def test_real_transcribe_on_synthetic_wav(self, short_wav: Path):
        """Tiny sine wave produces some result (likely empty text) without crashing."""
        result = asr.transcribe(str(short_wav))
        assert "text" in result
        assert result["duration_s"] > 0
        assert result["latency_s"] > 0

    def test_real_transcribe_on_sample1_if_present(self):
        """If recordings/sample1.wav exists, run the documented benchmark
        and assert RTF stays in a sane range (broader than the 0.064 logged
        on WSL2 — Pi will differ)."""
        path = Path("recordings/sample1.wav")
        if not path.exists():
            pytest.skip("recordings/sample1.wav not present in this checkout")
        result = asr.transcribe(str(path))
        assert result["text"] != ""
        rtf = result["latency_s"] / result["duration_s"]
        # WSL2 logged 0.064; allow up to 1.0 for any reasonable hardware.
        assert rtf < 1.0, f"RTF {rtf:.3f} exceeds real-time budget"