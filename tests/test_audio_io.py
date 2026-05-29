"""
test_audio_io.py — Unit tests for src/audio_io.py.

All tests run on WSL2 without a live microphone. Tests that would need
real hardware (record() against a physical mic) are intentionally NOT
included here; they live in the live-mic test suite that runs on the Pi.

Coverage:
    - load_wav / save_wav round-trip (int16 and float32 inputs)
    - save_wav handling of mono vs stereo, float clipping, dtype edge cases
    - resample_8_to_16k length, dtype preservation, error paths
    - load_wav error paths (missing file)
    - play() on a null/virtual device — verified by monkeypatching
      sounddevice so we never touch the OS audio stack
    - list_devices() returns a list and does not crash even when PortAudio
      has no devices
    - Config loading: defaults applied when keys missing, env-var override
    - reduce_noise(): validation, config toggle, dtype preservation,
      param forwarding (Week 3 Day 3)

Run with:
    pytest tests/test_audio_io.py -v
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# Make 'src' importable when running pytest from the project root.
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import audio_io  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_audio_io_config_cache():
    """Reset the module-level config cache before every test.

    audio_io caches the parsed config dict; tests that monkeypatch the
    config path or env var need a clean slate.
    """
    audio_io._config = None
    audio_io._config_path_loaded = None
    yield
    audio_io._config = None
    audio_io._config_path_loaded = None


@pytest.fixture
def sine_8k_float() -> np.ndarray:
    """0.5s 440Hz sine wave at 8 kHz, float32 in [-1, 1]."""
    t = np.linspace(0.0, 0.5, 4000, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def sine_16k_int16() -> np.ndarray:
    """0.5s 440Hz sine wave at 16 kHz, int16."""
    t = np.linspace(0.0, 0.5, 8000, endpoint=False, dtype=np.float32)
    f = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    return (f * 32767).astype(np.int16)


@pytest.fixture
def tmp_wav_path(tmp_path: Path) -> Path:
    """Path to a writable .wav file inside pytest's tmp_path."""
    return tmp_path / "test.wav"


def _write_config(
    tmp_path: Path,
    *,
    enabled=True,
    stationary=False,
    prop_decrease=1.0,
    sample_rate=16000,
    include_nr=True,
) -> Path:
    """Write a minimal config (with optional noise_reduction section) to disk."""
    cfg = {"audio": {"sample_rate": sample_rate, "channels": 1}, "paths": {}}
    if include_nr:
        cfg["noise_reduction"] = {
            "enabled": enabled,
            "stationary": stationary,
            "prop_decrease": prop_decrease,
        }
    p = tmp_path / "dev_config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# save_wav / load_wav
# ---------------------------------------------------------------------------


class TestSaveLoadWavRoundTrip:
    """save_wav and load_wav must round-trip int16 and float32 data."""

    def test_int16_round_trip(self, sine_16k_int16: np.ndarray, tmp_wav_path: Path):
        audio_io.save_wav(sine_16k_int16, tmp_wav_path, sample_rate=16000)
        loaded, sr = audio_io.load_wav(tmp_wav_path)

        assert sr == 16000
        assert loaded.dtype == np.int16
        assert loaded.shape == sine_16k_int16.shape
        # Exact equality: int16 -> wav -> int16 has no lossy conversion.
        np.testing.assert_array_equal(loaded, sine_16k_int16)

    def test_float32_round_trip_within_tolerance(
        self, sine_8k_float: np.ndarray, tmp_wav_path: Path
    ):
        audio_io.save_wav(sine_8k_float, tmp_wav_path, sample_rate=8000)
        loaded, sr = audio_io.load_wav(tmp_wav_path)

        assert sr == 8000
        assert loaded.dtype == np.int16
        # float -> int16 scaling: max absolute error is 1 LSB.
        reconstructed = loaded.astype(np.float32) / 32767.0
        np.testing.assert_allclose(reconstructed, sine_8k_float, atol=1.0 / 32767.0 + 1e-6)

    def test_default_sample_rate_from_config(
        self, sine_16k_int16: np.ndarray, tmp_wav_path: Path
    ):
        """save_wav with sample_rate=None uses audio.sample_rate from config."""
        audio_io.save_wav(sine_16k_int16, tmp_wav_path)  # no sample_rate
        with wave.open(str(tmp_wav_path), "rb") as wf:
            assert wf.getframerate() == 16000  # the dev_config default
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_stereo_input_downmixed_to_mono(self, tmp_wav_path: Path):
        """Stereo (n, 2) input is averaged to mono before writing."""
        left = np.full(1000, 10000, dtype=np.int16)
        right = np.full(1000, -10000, dtype=np.int16)
        stereo = np.stack([left, right], axis=1)
        audio_io.save_wav(stereo, tmp_wav_path, sample_rate=16000)

        loaded, _ = audio_io.load_wav(tmp_wav_path)
        # Mean of +10000 and -10000 is 0.
        assert loaded.ndim == 1
        assert np.all(loaded == 0)

    def test_float_clipping(self, tmp_wav_path: Path):
        """Out-of-range floats are clipped, not wrapped."""
        arr = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        audio_io.save_wav(arr, tmp_wav_path, sample_rate=16000)
        loaded, _ = audio_io.load_wav(tmp_wav_path)
        # 2.0 -> clipped to 1.0 -> 32767; -2.0 -> -1.0 -> -32767
        assert loaded[0] == 32767
        assert loaded[1] == -32767


class TestSaveWavErrors:
    def test_rejects_non_ndarray(self, tmp_wav_path: Path):
        with pytest.raises(TypeError):
            audio_io.save_wav([1, 2, 3], tmp_wav_path)  # type: ignore[arg-type]

    def test_rejects_empty(self, tmp_wav_path: Path):
        with pytest.raises(ValueError):
            audio_io.save_wav(np.array([], dtype=np.int16), tmp_wav_path)

    def test_rejects_invalid_sample_rate(self, sine_16k_int16: np.ndarray, tmp_wav_path: Path):
        with pytest.raises(ValueError):
            audio_io.save_wav(sine_16k_int16, tmp_wav_path, sample_rate=0)

    def test_rejects_missing_parent_dir(self, sine_16k_int16: np.ndarray, tmp_path: Path):
        bogus = tmp_path / "no_such_dir" / "out.wav"
        with pytest.raises(FileNotFoundError):
            audio_io.save_wav(sine_16k_int16, bogus, sample_rate=16000)

    def test_rejects_unsupported_dtype(self, tmp_wav_path: Path):
        arr = np.array([1 + 0j, 2 + 0j], dtype=np.complex64)
        with pytest.raises(ValueError):
            audio_io.save_wav(arr, tmp_wav_path, sample_rate=16000)


class TestLoadWavErrors:
    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            audio_io.load_wav(tmp_path / "nope.wav")

    def test_8bit_wav_widened_to_int16(self, tmp_path: Path):
        """8-bit unsigned WAV input is centred and widened to int16."""
        path = tmp_path / "u8.wav"
        # Build a tiny 8-bit unsigned PCM file manually.
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(8000)
            # 8-bit PCM is unsigned, centred at 128.
            wf.writeframes(bytes([128, 0, 255, 128]))
        arr, sr = audio_io.load_wav(path)
        assert sr == 8000
        assert arr.dtype == np.int16
        # 128 -> 0; 0 -> -32768; 255 -> 32512; 128 -> 0.
        assert arr[0] == 0
        assert arr[1] == -32768
        assert arr[3] == 0


# ---------------------------------------------------------------------------
# resample_8_to_16k
# ---------------------------------------------------------------------------


class TestResample8To16k:
    def test_output_length_is_2n_minus_1(self, sine_8k_float: np.ndarray):
        out = audio_io.resample_8_to_16k(sine_8k_float)
        assert out.shape[0] == 2 * sine_8k_float.shape[0] - 1

    def test_float32_input_yields_float32(self, sine_8k_float: np.ndarray):
        out = audio_io.resample_8_to_16k(sine_8k_float)
        assert out.dtype == np.float32

    def test_int16_input_yields_int16(self):
        arr = np.array([0, 32767, -32768, 100, -100], dtype=np.int16)
        out = audio_io.resample_8_to_16k(arr)
        assert out.dtype == np.int16

    def test_endpoints_preserved(self):
        """Linear interpolation must preserve the original sample values."""
        arr = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        out = audio_io.resample_8_to_16k(arr)
        # Even indices in the output land on the original samples.
        np.testing.assert_allclose(out[0::2], arr)
        # Odd indices are midpoints between neighbours.
        expected_midpoints = (arr[:-1] + arr[1:]) / 2.0
        np.testing.assert_allclose(out[1::2], expected_midpoints, rtol=1e-6)

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            audio_io.resample_8_to_16k([1.0, 2.0])  # type: ignore[arg-type]

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            audio_io.resample_8_to_16k(np.zeros((100, 2), dtype=np.float32))

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            audio_io.resample_8_to_16k(np.array([], dtype=np.float32))

    def test_round_trip_via_wav(self, sine_8k_float: np.ndarray, tmp_wav_path: Path):
        """Save 8kHz, load, resample to 16kHz — length and sample rate
        line up the way the pipeline expects."""
        audio_io.save_wav(sine_8k_float, tmp_wav_path, sample_rate=8000)
        loaded, sr = audio_io.load_wav(tmp_wav_path)
        assert sr == 8000
        upsampled = audio_io.resample_8_to_16k(loaded)
        assert upsampled.shape[0] == 2 * loaded.shape[0] - 1


# ---------------------------------------------------------------------------
# play() — mocked, no real audio hardware
# ---------------------------------------------------------------------------


class _FakeSoundDevice:
    """Captures play() and wait() calls so we can assert on them."""

    def __init__(self) -> None:
        self.play_calls: list[dict[str, Any]] = []
        self.wait_called: int = 0

    def play(self, data, samplerate=None, device=None) -> None:
        self.play_calls.append(
            {
                "shape": data.shape,
                "dtype": data.dtype,
                "samplerate": samplerate,
                "device": device,
            }
        )

    def wait(self) -> None:
        self.wait_called += 1


class TestPlay:
    def test_play_int16_passes_through(
        self, sine_16k_int16: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _FakeSoundDevice()
        monkeypatch.setattr(audio_io.sd, "play", fake.play)
        monkeypatch.setattr(audio_io.sd, "wait", fake.wait)

        audio_io.play(sine_16k_int16, sample_rate=16000, device=None)

        assert len(fake.play_calls) == 1
        call = fake.play_calls[0]
        assert call["samplerate"] == 16000
        assert call["dtype"] == np.int16
        assert call["device"] is None
        assert fake.wait_called == 1  # blocking=True is default

    def test_play_float64_downcast_to_float32(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _FakeSoundDevice()
        monkeypatch.setattr(audio_io.sd, "play", fake.play)
        monkeypatch.setattr(audio_io.sd, "wait", fake.wait)

        arr64 = np.linspace(-0.5, 0.5, 100, dtype=np.float64)
        audio_io.play(arr64, sample_rate=16000)
        assert fake.play_calls[0]["dtype"] == np.float32

    def test_play_uses_config_defaults(
        self, sine_16k_int16: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ):
        """sample_rate=None and device=None come from config."""
        fake = _FakeSoundDevice()
        monkeypatch.setattr(audio_io.sd, "play", fake.play)
        monkeypatch.setattr(audio_io.sd, "wait", fake.wait)

        audio_io.play(sine_16k_int16)  # all defaults
        call = fake.play_calls[0]
        assert call["samplerate"] == 16000  # dev_config sample_rate
        assert call["device"] is None       # dev_config input/output default

    def test_play_non_blocking_skips_wait(
        self, sine_16k_int16: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ):
        fake = _FakeSoundDevice()
        monkeypatch.setattr(audio_io.sd, "play", fake.play)
        monkeypatch.setattr(audio_io.sd, "wait", fake.wait)

        audio_io.play(sine_16k_int16, blocking=False)
        assert fake.wait_called == 0

    def test_play_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            audio_io.play([1, 2, 3])  # type: ignore[arg-type]

    def test_play_rejects_empty(self):
        with pytest.raises(ValueError):
            audio_io.play(np.array([], dtype=np.int16))

    def test_play_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError):
            audio_io.play(np.array([1, 2, 3], dtype=np.int8))

    def test_play_rejects_invalid_sample_rate(self, sine_16k_int16, monkeypatch):
        fake = _FakeSoundDevice()
        monkeypatch.setattr(audio_io.sd, "play", fake.play)
        monkeypatch.setattr(audio_io.sd, "wait", fake.wait)
        with pytest.raises(ValueError):
            audio_io.play(sine_16k_int16, sample_rate=0)


# ---------------------------------------------------------------------------
# list_devices
# ---------------------------------------------------------------------------


class TestListDevices:
    def test_returns_list_even_when_portaudio_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """If sounddevice can't enumerate, list_devices returns []
        rather than raising — that's what we want on bare WSL2."""

        def boom(*args, **kwargs):
            raise RuntimeError("No PortAudio host APIs available")

        monkeypatch.setattr(audio_io.sd, "query_devices", boom)
        result = audio_io.list_devices()
        assert result == []

    def test_returns_list_of_dicts_when_devices_present(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        fake_devices = [
            {
                "name": "Fake Mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
            },
            {
                "name": "Fake Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]
        monkeypatch.setattr(audio_io.sd, "query_devices", lambda: fake_devices)

        result = audio_io.list_devices()
        assert len(result) == 2
        assert result[0]["name"] == "Fake Mic"
        assert result[1]["name"] == "Fake Speaker"

        captured = capsys.readouterr()
        assert "Fake Mic" in captured.out
        assert "Fake Speaker" in captured.out


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_loads_default_config(self):
        cfg = audio_io._get_config()
        assert cfg["audio"]["sample_rate"] == 16000
        assert cfg["audio"]["channels"] == 1

    def test_missing_audio_keys_use_function_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If config file has an empty audio section, _audio_cfg falls
        back to the default passed in by the caller."""
        sparse = tmp_path / "sparse.yaml"
        sparse.write_text("audio: {}\npaths: {}\n")
        monkeypatch.setenv("VOICE_ASSISTANT_CONFIG", str(sparse))

        # Cache is reset by the autouse fixture.
        assert audio_io._audio_cfg("sample_rate", 22050) == 22050
        assert audio_io._audio_cfg("channels", 7) == 7

    def test_missing_config_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VOICE_ASSISTANT_CONFIG", str(tmp_path / "nope.yaml"))
        with pytest.raises(FileNotFoundError):
            audio_io._get_config()

    def test_explicit_config_path_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        env_cfg = tmp_path / "env.yaml"
        env_cfg.write_text("audio:\n  sample_rate: 8000\n")
        monkeypatch.setenv("VOICE_ASSISTANT_CONFIG", str(env_cfg))

        explicit_cfg = tmp_path / "explicit.yaml"
        explicit_cfg.write_text("audio:\n  sample_rate: 44100\n")

        cfg = audio_io._get_config(explicit_cfg)
        assert cfg["audio"]["sample_rate"] == 44100


# ---------------------------------------------------------------------------
# reduce_noise — Week 3 Day 3 (noisereduce mocked in unit tests)
# ---------------------------------------------------------------------------


class TestReduceNoiseValidation:
    def test_rejects_non_ndarray(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        with pytest.raises(TypeError, match="must be np.ndarray"):
            audio_io.reduce_noise([1, 2, 3], config_path=cfg)  # type: ignore[arg-type]

    def test_rejects_2d_array(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        arr = np.zeros((10, 2), dtype=np.int16)
        with pytest.raises(ValueError, match="1-D mono"):
            audio_io.reduce_noise(arr, config_path=cfg)

    def test_rejects_empty(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            audio_io.reduce_noise(np.array([], dtype=np.int16), config_path=cfg)

    def test_rejects_non_positive_sample_rate(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        arr = np.zeros(100, dtype=np.int16)
        with pytest.raises(ValueError, match="sample_rate must be > 0"):
            audio_io.reduce_noise(arr, sample_rate=0, config_path=cfg)


class TestReduceNoiseBehaviour:
    def test_disabled_returns_input_unchanged(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, enabled=False)
        arr = np.array([1, 2, 3, 4], dtype=np.int16)
        out = audio_io.reduce_noise(arr, config_path=cfg)
        # Passthrough: same object, no noisereduce import/call needed.
        assert out is arr

    def test_int16_in_int16_out(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        arr = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        # Mock noisereduce to echo its float input back unchanged.
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y):
            out = audio_io.reduce_noise(arr, config_path=cfg)
        assert out.dtype == np.int16
        # Echoed float (arr/32768) scaled back ≈ original (within rounding).
        np.testing.assert_allclose(out, arr, atol=1)

    def test_float32_in_float32_out(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        arr = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y):
            out = audio_io.reduce_noise(arr, config_path=cfg)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_passes_sample_rate_to_noisereduce(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, sample_rate=16000)
        arr = np.zeros(100, dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y) as m:
            audio_io.reduce_noise(arr, config_path=cfg)
        _, kwargs = m.call_args
        assert kwargs["sr"] == 16000

    def test_explicit_sample_rate_overrides_config(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, sample_rate=16000)
        arr = np.zeros(100, dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y) as m:
            audio_io.reduce_noise(arr, sample_rate=8000, config_path=cfg)
        _, kwargs = m.call_args
        assert kwargs["sr"] == 8000

    def test_passes_stationary_and_prop_decrease(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, stationary=True, prop_decrease=0.8)
        arr = np.zeros(100, dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y) as m:
            audio_io.reduce_noise(arr, config_path=cfg)
        _, kwargs = m.call_args
        assert kwargs["stationary"] is True
        assert kwargs["prop_decrease"] == 0.8

    def test_missing_nr_section_defaults_to_enabled(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, include_nr=False)
        arr = np.zeros(100, dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y) as m:
            out = audio_io.reduce_noise(arr, config_path=cfg)
        # Default enabled → noisereduce IS called, output returned.
        assert m.called
        assert out.dtype == np.float32

    def test_output_length_preserved(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        arr = np.zeros(512, dtype=np.float32)
        with patch("noisereduce.reduce_noise", side_effect=lambda y, **k: y):
            out = audio_io.reduce_noise(arr, config_path=cfg)
        assert out.shape == arr.shape


@pytest.mark.integration
class TestReduceNoiseReal:
    def test_real_reduces_white_noise_on_tone(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, sample_rate=16000)
        # 0.5s 440Hz tone + white noise at 16kHz.
        sr = 16000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        rng = np.random.default_rng(0)
        noisy = (tone + 0.1 * rng.standard_normal(tone.shape).astype(np.float32)).astype(np.float32)

        out = audio_io.reduce_noise(noisy, config_path=cfg)

        assert out.dtype == np.float32
        assert out.shape == noisy.shape
        # Residual noise power should drop vs the noisy input.
        noise_before = float(np.mean((noisy - tone) ** 2))
        noise_after = float(np.mean((out - tone) ** 2))
        assert noise_after < noise_before