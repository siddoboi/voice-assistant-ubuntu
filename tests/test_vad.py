"""
test_vad.py — Unit tests for src/vad.py.

Two layers:
  - Unit tests (default): mock the onnxruntime InferenceSession so we can
    verify state init, reset_state, get_speech_prob shape handling,
    is_speech threshold logic, test_on_file resampling + downmixing,
    without loading the real 1.8MB ONNX model.
  - Integration tests (--run-integration): run the real Silero VAD v4
    ONNX on the project sample wavs if present. Validates the documented
    contract — v4 produces non-zero probs for real speech and works at
    the configured chunk size.

Critical: v4 ONLY. v5 is broken with onnxruntime 1.26.0 (all-zero probs).
Tests don't enforce this directly but document it.

Run:
    pytest tests/test_vad.py -v
    pytest tests/test_vad.py -v --run-integration
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import vad


# ---------------------------------------------------------------------------
# Reset all module state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_vad_state():
    vad._session = None
    vad._h = None
    vad._c = None
    yield
    vad._session = None
    vad._h = None
    vad._c = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_session(prob_sequence: list[float] | None = None):
    """Build a MagicMock onnxruntime session that returns a sequence of
    probabilities. State (h, c) is echoed back unchanged."""
    session = MagicMock()
    probs_iter = iter(prob_sequence or [0.9])

    def fake_run(_outputs, feeds):
        try:
            p = next(probs_iter)
        except StopIteration:
            p = 0.0
        out = np.array([[p]], dtype=np.float32)
        return [out, feeds["h"], feeds["c"]]

    session.run.side_effect = fake_run
    return session


def _patch_inference_session():
    """Patch InferenceSession in src.vad regardless of how it was imported.

    Handles all import styles:
      - `import onnxruntime as ort` at module top      -> src.vad.ort.InferenceSession
      - `import onnxruntime` at module top             -> src.vad.onnxruntime.InferenceSession
      - `from onnxruntime import InferenceSession`     -> src.vad.InferenceSession
      - Lazy `import onnxruntime as ort` inside a func -> patch onnxruntime.InferenceSession
        via sys.modules (the underlying module object — all aliases point here).
    """
    # Module-top: from onnxruntime import InferenceSession
    if hasattr(vad, "InferenceSession"):
        return patch.object(vad, "InferenceSession")
    # Module-top: import onnxruntime as ort
    if hasattr(vad, "ort") and hasattr(vad.ort, "InferenceSession"):
        return patch.object(vad.ort, "InferenceSession")
    # Module-top: import onnxruntime
    if hasattr(vad, "onnxruntime") and hasattr(vad.onnxruntime, "InferenceSession"):
        return patch.object(vad.onnxruntime, "InferenceSession")
    # Lazy import inside a function — the only shared truth is the
    # onnxruntime module object itself in sys.modules. Patching its
    # attribute affects every `import onnxruntime` everywhere.
    import onnxruntime  # ensures sys.modules['onnxruntime'] is populated
    return patch.object(onnxruntime, "InferenceSession")


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_model_path_is_v4(self):
        assert vad.MODEL_PATH == "models/silero/silero_vad_v4.onnx"
        # Hard-document the v4-only requirement.
        assert "v4" in vad.MODEL_PATH

    def test_sample_rate_is_16k(self):
        assert vad.SAMPLE_RATE == 16000

    def test_chunk_size_is_512(self):
        """512 samples @ 16kHz = 32ms — the Silero VAD frame size."""
        assert vad.CHUNK_SIZE == 512

    def test_silence_threshold_default(self):
        assert vad.SILENCE_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# load_model — state initialisation
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_initialises_h_and_c_zeros(self):
        with _patch_inference_session() as MockSession:
            MockSession.return_value = MagicMock(name="OrtSession")
            vad.load_model()
            assert vad._h is not None and vad._c is not None
            assert vad._h.shape == (2, 1, 64)
            assert vad._c.shape == (2, 1, 64)
            assert vad._h.dtype == np.float32
            assert vad._c.dtype == np.float32
            assert np.all(vad._h == 0)
            assert np.all(vad._c == 0)

    def test_caches_session(self):
        with _patch_inference_session() as MockSession:
            MockSession.return_value = MagicMock(name="OrtSession")
            vad.load_model()
            vad.load_model()
            vad.load_model()
            assert MockSession.call_count == 1

    def test_uses_correct_model_path(self):
        """The session must be opened against the v4 model file. We don't
        assert on `providers=` because vad.py relies on the onnxruntime
        default (which is CPU on systems without a GPU build installed)."""
        with _patch_inference_session() as MockSession:
            MockSession.return_value = MagicMock(name="OrtSession")
            vad.load_model()
            args, _kwargs = MockSession.call_args
            assert args[0] == "models/silero/silero_vad_v4.onnx"


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------


class TestResetState:
    def test_zeros_h_and_c(self):
        vad._session = _make_fake_session()
        vad._h = np.ones((2, 1, 64), dtype=np.float32)
        vad._c = np.full((2, 1, 64), 7.0, dtype=np.float32)
        vad.reset_state()
        assert np.all(vad._h == 0)
        assert np.all(vad._c == 0)

    def test_zeros_h_and_c_when_session_not_loaded(self):
        """reset_state() is intentionally session-agnostic — it only zeros
        the state tensors and does not trigger model loading. The lazy-load
        path is exercised by TestGetSpeechProb.test_auto_loads_model_if_unloaded.
        """
        vad._session = None
        vad._h = np.ones((2, 1, 64), dtype=np.float32)
        vad._c = np.full((2, 1, 64), 7.0, dtype=np.float32)
        vad.reset_state()
        assert np.all(vad._h == 0)
        assert np.all(vad._c == 0)
        # Session remains unloaded — that is correct behaviour.
        assert vad._session is None


# ---------------------------------------------------------------------------
# get_speech_prob
# ---------------------------------------------------------------------------


class TestGetSpeechProb:
    def test_returns_float(self, sine_chunk_16k):
        vad._session = _make_fake_session([0.8])
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.get_speech_prob(sine_chunk_16k)
        assert isinstance(result, float)
        assert result == pytest.approx(0.8)

    def test_feeds_correct_input_keys(self, sine_chunk_16k):
        session = _make_fake_session([0.5])
        vad._session = session
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad.get_speech_prob(sine_chunk_16k)

        _, feeds = session.run.call_args.args
        assert set(feeds.keys()) == {"input", "sr", "h", "c"}
        assert feeds["input"].shape == (1, 512)
        assert feeds["input"].dtype == np.float32
        assert feeds["sr"] == 16000

    def test_converts_int16_to_float32(self):
        int_chunk = np.zeros(512, dtype=np.int16)
        session = _make_fake_session([0.1])
        vad._session = session
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad.get_speech_prob(int_chunk)
        _, feeds = session.run.call_args.args
        assert feeds["input"].dtype == np.float32

    def test_updates_h_and_c(self, sine_chunk_16k):
        """Returned hn/cn from the session must overwrite cached state."""
        new_h = np.full((2, 1, 64), 3.14, dtype=np.float32)
        new_c = np.full((2, 1, 64), 2.71, dtype=np.float32)
        session = MagicMock()
        session.run.return_value = [np.array([[0.5]], dtype=np.float32), new_h, new_c]
        vad._session = session
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        vad.get_speech_prob(sine_chunk_16k)
        assert np.allclose(vad._h, 3.14)
        assert np.allclose(vad._c, 2.71)

    def test_auto_loads_model_if_unloaded(self, sine_chunk_16k):
        with _patch_inference_session() as MockSession:
            MockSession.return_value = _make_fake_session([0.5])
            vad.get_speech_prob(sine_chunk_16k)
            MockSession.assert_called_once()


# ---------------------------------------------------------------------------
# is_speech — threshold logic
# ---------------------------------------------------------------------------


class TestIsSpeech:
    def test_above_threshold_returns_true(self, sine_chunk_16k):
        vad._session = _make_fake_session([0.7])
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        assert vad.is_speech(sine_chunk_16k) is True

    def test_below_threshold_returns_false(self, sine_chunk_16k):
        vad._session = _make_fake_session([0.3])
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        assert vad.is_speech(sine_chunk_16k) is False

    def test_at_threshold_returns_true(self, sine_chunk_16k):
        """is_speech uses >=, so exactly 0.5 counts as speech."""
        vad._session = _make_fake_session([0.5])
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        assert vad.is_speech(sine_chunk_16k) is True


# ---------------------------------------------------------------------------
# test_on_file — resampling, downmix, stats
# ---------------------------------------------------------------------------


class TestTestOnFile:
    def test_returns_expected_keys(self, short_wav: Path):
        vad._session = _make_fake_session([0.9] * 1000)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(short_wav))
        expected = {
            "file", "original_rate", "total_chunks", "speech_chunks",
            "silence_chunks", "speech_ratio", "latency_s",
        }
        assert set(result.keys()) == expected

    def test_chunk_counts_consistent(self, short_wav: Path):
        vad._session = _make_fake_session([0.9] * 1000)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(short_wav))
        assert result["total_chunks"] == result["speech_chunks"] + result["silence_chunks"]

    def test_speech_ratio_matches(self, short_wav: Path):
        # Alternating speech/silence
        vad._session = _make_fake_session([0.9, 0.1] * 500)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(short_wav))
        expected_ratio = result["speech_chunks"] / result["total_chunks"]
        # vad.py rounds speech_ratio to 3 decimal places, so we allow
        # a 1e-3 absolute tolerance rather than pytest.approx's default 1e-6.
        assert result["speech_ratio"] == pytest.approx(expected_ratio, abs=1e-3)

    def test_original_rate_recorded(self, short_wav: Path):
        """short_wav is 16kHz — recorded as-is."""
        vad._session = _make_fake_session([0.5] * 1000)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(short_wav))
        assert result["original_rate"] == 16000

    def test_resamples_8k_to_16k(self, tmp_path: Path):
        """8kHz input must be resampled before chunking."""
        path = tmp_path / "8k.wav"
        sr = 8000
        n = sr * 2  # 2s of audio
        samples = (np.random.randn(n).astype(np.float32) * 10000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

        vad._session = _make_fake_session([0.5] * 1000)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(path))

        assert result["original_rate"] == 8000
        # 2s @ 16kHz = 32000 samples / 512 = 62 chunks
        assert result["total_chunks"] == 62

    def test_downmixes_stereo(self, stereo_wav_8k: Path):
        """Stereo input must be mixed to mono — no shape errors."""
        vad._session = _make_fake_session([0.5] * 1000)
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)
        result = vad.test_on_file(str(stereo_wav_8k))
        assert result["original_rate"] == 8000
        assert result["total_chunks"] > 0
        # No exception means downmix happened correctly.

    def test_resets_state_per_file(self, short_wav: Path):
        """test_on_file must call reset_state at start — otherwise the
        state from a previous file leaks into the new one."""
        vad._session = _make_fake_session([0.5] * 1000)
        vad._h = np.full((2, 1, 64), 99.0, dtype=np.float32)  # dirty state
        vad._c = np.full((2, 1, 64), 99.0, dtype=np.float32)
        vad.test_on_file(str(short_wav))
        # After test_on_file, state may have been updated by the session,
        # but reset must have happened first. Confirm via a fresh hook.
        # Easier: spy that h is no longer 99 (it was zeroed then echoed).
        # Fake session echoes whatever feeds["h"] is, so after first call
        # _h equals zeros (because reset_state ran).
        assert not np.allclose(vad._h, 99.0)


# ---------------------------------------------------------------------------
# Integration — real Silero VAD v4
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealVad:

    def test_real_model_loads(self):
        if not Path(vad.MODEL_PATH).exists():
            pytest.skip(f"VAD model not at {vad.MODEL_PATH}")
        vad.load_model()
        assert vad._session is not None
        assert vad._h.shape == (2, 1, 64)

    def test_real_silence_below_threshold(self, silence_chunk_16k):
        if not Path(vad.MODEL_PATH).exists():
            pytest.skip(f"VAD model not at {vad.MODEL_PATH}")
        vad.load_model()
        vad.reset_state()
        prob = vad.get_speech_prob(silence_chunk_16k)
        # Silence should produce a low probability (well under 0.5).
        # If v5 was loaded by mistake this always returns ~0.0005 — we'd
        # still pass this assertion. The non_zero_for_speech test below
        # is what actually catches v5.
        assert prob < 0.5

    def test_real_not_v5_regression(self):
        """v5 returns ~0.0005 for everything. v4 returns varied probs.
        Run many random chunks; if all are < 0.001, we're on v5.
        """
        if not Path(vad.MODEL_PATH).exists():
            pytest.skip(f"VAD model not at {vad.MODEL_PATH}")
        vad.load_model()
        vad.reset_state()
        rng = np.random.default_rng(42)
        probs = []
        for _ in range(20):
            chunk = rng.standard_normal(512).astype(np.float32) * 0.3
            probs.append(vad.get_speech_prob(chunk))
        assert max(probs) > 0.001, (
            f"All probs < 0.001 (max={max(probs):.6f}) — likely loaded v5 by mistake. "
            "Re-download from github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
        )

    def test_real_test_on_file_if_sample_present(self):
        if not Path(vad.MODEL_PATH).exists():
            pytest.skip(f"VAD model not at {vad.MODEL_PATH}")
        sample = Path("recordings/sample1.wav")
        if not sample.exists():
            pytest.skip("recordings/sample1.wav not present")
        result = vad.test_on_file(str(sample))
        # Master context logged 88.1% speech ratio on sample1. Allow wide range.
        assert 0.5 < result["speech_ratio"] < 1.0, (
            f"Speech ratio {result['speech_ratio']:.2%} far from 88% — check VAD."
        )