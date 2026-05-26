"""
tests/conftest.py — shared pytest configuration for the voice-assistant suite.

- Adds the project root to sys.path so `from src import ...` works in every
  test file without each one needing its own boilerplate.
- Registers the `integration` marker. By default these tests are skipped
  unless the user passes --run-integration on the pytest command line.
  Integration tests load real models (Ollama, faster-whisper, Piper, Silero)
  and read .wav files from disk; they are slower and have external deps.
- Provides shared fixtures used by multiple module test files.

Usage:
    pytest tests/                       # unit tests only (mocked, fast)
    pytest tests/ --run-integration     # unit + integration (real models)
    pytest tests/ -m integration --run-integration   # integration only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Integration marker — opt-in via --run-integration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real models (Ollama, Piper, "
             "faster-whisper, Silero VAD) and pre-recorded .wav files.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: test requires real models or external services "
        "(skipped by default; enable with --run-integration)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_chunk_16k() -> np.ndarray:
    """A 512-sample 440Hz sine at 16kHz, float32 in [-1, 1].

    This is exactly the VAD chunk size — used by VAD tests to verify
    correct input shape handling.
    """
    n = 512
    t = np.arange(n, dtype=np.float32) / 16000.0
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def silence_chunk_16k() -> np.ndarray:
    """A 512-sample silent chunk at 16kHz, float32 zeros."""
    return np.zeros(512, dtype=np.float32)


@pytest.fixture
def short_wav(tmp_path: Path) -> Path:
    """A 0.5s sine-wave WAV file at 16kHz, 16-bit mono — for transcribe tests."""
    import wave

    path = tmp_path / "short.wav"
    sr = 16000
    n = sr // 2  # 0.5s
    t = np.arange(n, dtype=np.float32) / sr
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


@pytest.fixture
def stereo_wav_8k(tmp_path: Path) -> Path:
    """A 0.25s stereo WAV at 8kHz — for VAD test_on_file resample/downmix path."""
    import wave

    path = tmp_path / "stereo_8k.wav"
    sr = 8000
    n = sr // 4
    t = np.arange(n, dtype=np.float32) / sr
    left = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    right = (0.3 * np.sin(2 * np.pi * 880.0 * t) * 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())
    return path