"""
test_tts.py — Unit tests for src/tts.py.

Two layers:
  - Unit tests (default): mock PiperVoice so we can verify load caching,
    synthesize() contract, wave file params (channels=1, sampwidth=2,
    framerate from voice.config.sample_rate), and the RTF calculation,
    without downloading the 61MB voice model.
  - Integration tests (--run-integration): run real Piper TTS on a
    short sentence. Validates the documented contract — synthesize_wav()
    works, .wav is non-empty, RTF < 1.0.

Run:
    pytest tests/test_tts.py -v
    pytest tests/test_tts.py -v --run-integration
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import tts


# ---------------------------------------------------------------------------
# Reset module state between tests — _voice is cached at module level.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tts_voice_cache():
    tts._voice = None
    yield
    tts._voice = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_voice(sample_rate: int = 22050, n_frames: int = 22050):
    """Build a MagicMock with the .config.sample_rate attribute Piper exposes.

    synthesize_wav() takes (text, wave_writer) and writes frames. Our fake
    writes n_frames of silence so .wav inspection works.
    """
    voice = MagicMock()
    voice.config = SimpleNamespace(sample_rate=sample_rate)

    def fake_synth(text, wf):
        # Wave writer already has params set by tts.synthesize() before
        # this is called.
        wf.writeframes(b"\x00\x00" * n_frames)

    voice.synthesize_wav.side_effect = fake_synth
    return voice


def _patch_piper_voice():
    """Patch PiperVoice regardless of how src.tts imported it.

    Handles all import styles:
      - `from piper import PiperVoice` at top   -> src.tts.PiperVoice
      - `import piper`                          -> src.tts.piper.PiperVoice
      - Lazy import inside load_voice()         -> patch piper.PiperVoice via the
        canonical module attribute (all `import piper` calls resolve here)
    """
    if hasattr(tts, "PiperVoice"):
        return patch.object(tts, "PiperVoice")
    if hasattr(tts, "piper") and hasattr(tts.piper, "PiperVoice"):
        return patch.object(tts.piper, "PiperVoice")
    import piper
    return patch.object(piper, "PiperVoice")


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_model_path_is_amy_medium(self):
        assert tts.MODEL_PATH == "models/piper/en_US-amy-medium.onnx"

    def test_config_path_matches_model(self):
        assert tts.CONFIG_PATH == "models/piper/en_US-amy-medium.onnx.json"

    def test_voice_cache_starts_empty(self):
        assert tts._voice is None


# ---------------------------------------------------------------------------
# load_voice — caching
# ---------------------------------------------------------------------------


class TestLoadVoice:
    def test_calls_piper_load_once(self):
        with _patch_piper_voice() as MockVoice:
            MockVoice.load.return_value = MagicMock(name="VoiceInstance")
            tts.load_voice()
            tts.load_voice()
            tts.load_voice()
            assert MockVoice.load.call_count == 1

    def test_load_passes_model_and_config_path(self):
        with _patch_piper_voice() as MockVoice:
            MockVoice.load.return_value = MagicMock(name="VoiceInstance")
            tts.load_voice()
            MockVoice.load.assert_called_once_with(
                "models/piper/en_US-amy-medium.onnx",
                config_path="models/piper/en_US-amy-medium.onnx.json",
            )

    def test_returns_cached_instance(self):
        sentinel = MagicMock(name="VoiceInstance")
        tts._voice = sentinel
        with _patch_piper_voice() as MockVoice:
            assert tts.load_voice() is sentinel
            MockVoice.load.assert_not_called()


# ---------------------------------------------------------------------------
# synthesize — wave params, contract, rtf
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_writes_valid_mono_16bit_wav(self, tmp_path: Path):
        tts._voice = _make_fake_voice(sample_rate=22050, n_frames=11025)
        out = tmp_path / "out.wav"
        tts.synthesize("hello", str(out))

        assert out.exists()
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 11025

    def test_framerate_follows_voice_config(self, tmp_path: Path):
        """Different voice configs must propagate to the wav file."""
        tts._voice = _make_fake_voice(sample_rate=16000, n_frames=16000)
        out = tmp_path / "out.wav"
        tts.synthesize("hello", str(out))
        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == 16000

    def test_returns_expected_keys(self, tmp_path: Path):
        tts._voice = _make_fake_voice()
        result = tts.synthesize("hello", str(tmp_path / "out.wav"))
        assert set(result.keys()) == {"output_path", "duration_s", "latency_s", "rtf"}

    def test_duration_matches_frames_over_rate(self, tmp_path: Path):
        # 22050 frames @ 22050 Hz = exactly 1.0s
        tts._voice = _make_fake_voice(sample_rate=22050, n_frames=22050)
        result = tts.synthesize("hello", str(tmp_path / "out.wav"))
        assert result["duration_s"] == pytest.approx(1.0)

    def test_rtf_is_latency_over_duration(self, tmp_path: Path):
        tts._voice = _make_fake_voice(sample_rate=22050, n_frames=22050)
        result = tts.synthesize("hello", str(tmp_path / "out.wav"))
        assert result["rtf"] == pytest.approx(result["latency_s"] / result["duration_s"])

    def test_output_path_in_result(self, tmp_path: Path):
        tts._voice = _make_fake_voice()
        out = str(tmp_path / "out.wav")
        result = tts.synthesize("hello", out)
        assert result["output_path"] == out

    def test_passes_text_to_synthesize_wav(self, tmp_path: Path):
        voice = _make_fake_voice()
        tts._voice = voice
        tts.synthesize("the rain in spain", str(tmp_path / "out.wav"))
        call_args = voice.synthesize_wav.call_args
        assert call_args.args[0] == "the rain in spain"

    def test_latency_is_positive(self, tmp_path: Path):
        voice = _make_fake_voice()
        def slow_synth(text, wf):
            time.sleep(0.01)
            wf.writeframes(b"\x00\x00" * 22050)
        voice.synthesize_wav.side_effect = slow_synth
        tts._voice = voice
        result = tts.synthesize("hello", str(tmp_path / "out.wav"))
        assert result["latency_s"] >= 0.01

    def test_critical_param_order(self, tmp_path: Path):
        """Regression test for the Day 4 Piper bug: synthesize_wav() is
        called AFTER wave params are set. If the order is reversed,
        wave.Error fires inside writeframes."""
        order: list[str] = []
        voice = MagicMock()
        voice.config = SimpleNamespace(sample_rate=22050)

        def tracking_synth(text, wf):
            # By the time we get here, all params must be set or
            # writeframes will fail.
            order.append("synthesize_wav")
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            wf.writeframes(b"\x00\x00" * 22050)

        voice.synthesize_wav.side_effect = tracking_synth
        tts._voice = voice
        tts.synthesize("x", str(tmp_path / "out.wav"))
        assert order == ["synthesize_wav"]

    def test_propagates_synth_errors(self, tmp_path: Path):
        voice = MagicMock()
        voice.config = SimpleNamespace(sample_rate=22050)
        voice.synthesize_wav.side_effect = RuntimeError("piper boom")
        tts._voice = voice
        with pytest.raises(RuntimeError, match="piper boom"):
            tts.synthesize("hello", str(tmp_path / "out.wav"))


# ---------------------------------------------------------------------------
# Integration — real Piper voice
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealVoice:

    def test_real_voice_loads(self):
        if not Path(tts.MODEL_PATH).exists():
            pytest.skip(f"Voice model not at {tts.MODEL_PATH}")
        voice = tts.load_voice()
        assert voice is not None
        assert hasattr(voice, "config")
        assert hasattr(voice.config, "sample_rate")

    def test_real_synthesize_short_sentence(self, tmp_path: Path):
        if not Path(tts.MODEL_PATH).exists():
            pytest.skip(f"Voice model not at {tts.MODEL_PATH}")
        out = tmp_path / "real.wav"
        result = tts.synthesize("Hello world.", str(out))

        assert out.exists()
        assert out.stat().st_size > 1000  # bytes — non-trivial wav
        assert result["duration_s"] > 0.1
        assert result["latency_s"] > 0
        # WSL2 logged avg 0.067; allow up to 1.0 for any hardware.
        assert result["rtf"] < 1.0, f"RTF {result['rtf']:.3f} exceeds real-time"

        # Verify wav file is structurally what we expect.
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2