"""
test_pipeline.py — Unit tests for src/pipeline.py.

The pipeline is orchestration code. Unit tests mock the four stage modules
(`audio_io`, `asr`, `llm_client`, `tts`) and verify:
  - stages execute in the right order
  - data flows correctly between them (transcript -> LLM prompt, reply -> TTS, etc)
  - the --input path bypasses recording
  - the --no-play flag skips playback
  - empty transcripts and empty LLM replies get sensible fallbacks
  - the per-stage latency dict is well-formed

Integration test (--run-integration) wires the real ASR + LLM + TTS modules
against a small pre-recorded .wav and asserts the chain completes.

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --run-integration
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_wav(tmp_path: Path) -> Path:
    """A 1s 16kHz mono wav — used as the pipeline's --input."""
    path = tmp_path / "input.wav"
    sr = 16000
    samples = np.zeros(sr, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


@pytest.fixture
def reply_wav(tmp_path: Path) -> Path:
    """A 0.5s 22050Hz mono wav — represents the TTS output."""
    path = tmp_path / "reply.wav"
    sr = 22050
    samples = np.zeros(sr // 2, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


@pytest.fixture
def mocked_stages(monkeypatch, reply_wav: Path):
    """Mock every external stage the pipeline depends on.

    Returns a SimpleNamespace exposing each mock so tests can inspect calls.
    """
    asr_mock = MagicMock(return_value={"text": "hello world", "duration_s": 1.0, "latency_s": 0.1})
    llm_mock = MagicMock(return_value={"text": "Hi there!", "latency_s": 0.5})
    tts_mock = MagicMock(return_value={
        "output_path": str(reply_wav),
        "duration_s": 0.5,
        "latency_s": 0.2,
        "rtf": 0.4,
    })
    record_mock = MagicMock(return_value=np.zeros(16000, dtype=np.int16))
    save_wav_mock = MagicMock()
    load_wav_mock = MagicMock(return_value=(np.zeros(11025, dtype=np.int16), 22050))
    play_mock = MagicMock()

    monkeypatch.setattr(pipeline.asr, "transcribe", asr_mock)
    monkeypatch.setattr(pipeline.llm_client, "generate", llm_mock)
    monkeypatch.setattr(pipeline.tts, "synthesize", tts_mock)
    monkeypatch.setattr(pipeline.audio_io, "record", record_mock)
    monkeypatch.setattr(pipeline.audio_io, "save_wav", save_wav_mock)
    monkeypatch.setattr(pipeline.audio_io, "load_wav", load_wav_mock)
    monkeypatch.setattr(pipeline.audio_io, "play", play_mock)

    class Stages:
        pass
    s = Stages()
    s.asr = asr_mock
    s.llm = llm_mock
    s.tts = tts_mock
    s.record = record_mock
    s.save_wav = save_wav_mock
    s.load_wav = load_wav_mock
    s.play = play_mock
    return s


# ---------------------------------------------------------------------------
# Stage wiring — input path, ASR -> LLM -> TTS flow
# ---------------------------------------------------------------------------


class TestWiring:
    def test_input_wav_skips_recording(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        # record() must NOT have been called when --input was given.
        mocked_stages.record.assert_not_called()
        mocked_stages.save_wav.assert_not_called()
        assert result["input_path"] == str(input_wav)

    def test_no_input_triggers_recording(self, mocked_stages, tmp_path: Path):
        result = pipeline.run(
            duration_s=3.0,
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        mocked_stages.record.assert_called_once()
        # duration_sec=3.0 should be passed through.
        call_kwargs = mocked_stages.record.call_args.kwargs
        assert call_kwargs.get("duration_sec") == 3.0
        # save_wav called once to persist the captured audio.
        mocked_stages.save_wav.assert_called_once()
        # Path must point at the recordings dir.
        assert "recordings" in result["input_path"]

    def test_asr_called_with_input_path(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        mocked_stages.asr.assert_called_once_with(str(input_wav))

    def test_llm_receives_transcript(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.return_value = {"text": "what's the weather", "duration_s": 2.0, "latency_s": 0.2}
        pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        mocked_stages.llm.assert_called_once()
        # First positional arg is the prompt.
        assert mocked_stages.llm.call_args.args[0] == "what's the weather"

    def test_tts_receives_llm_reply(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.llm.return_value = {"text": "It's sunny today.", "latency_s": 0.4}
        out = tmp_path / "out.wav"
        pipeline.run(input_wav=str(input_wav), output_wav=str(out), skip_play=True)
        mocked_stages.tts.assert_called_once_with("It's sunny today.", str(out))


# ---------------------------------------------------------------------------
# Playback flag
# ---------------------------------------------------------------------------


class TestPlayback:
    def test_no_play_skips_playback(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        mocked_stages.load_wav.assert_not_called()
        mocked_stages.play.assert_not_called()

    def test_play_loads_then_plays(self, mocked_stages, input_wav: Path, reply_wav: Path, tmp_path: Path):
        pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=False,
        )
        mocked_stages.load_wav.assert_called_once_with(str(reply_wav))
        mocked_stages.play.assert_called_once()
        # play() must be passed the audio array and sample_rate from load_wav.
        play_args = mocked_stages.play.call_args
        # sample_rate could be positional or kwarg — accept either.
        assert play_args.kwargs.get("sample_rate") == 22050


# ---------------------------------------------------------------------------
# LLM model override
# ---------------------------------------------------------------------------


class TestModelOverride:
    def test_no_model_uses_default(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        # When model is None, generate is called with just the prompt — no `model` kwarg.
        assert "model" not in mocked_stages.llm.call_args.kwargs

    def test_explicit_model_forwarded(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
            llm_model="tinyllama:1.1b",
        )
        assert mocked_stages.llm.call_args.kwargs.get("model") == "tinyllama:1.1b"


# ---------------------------------------------------------------------------
# Fallbacks for empty stages
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_empty_transcript_gets_fallback_prompt(self, mocked_stages, input_wav: Path, tmp_path: Path):
        """ASR returning empty text must not produce an empty LLM prompt
        (the LLM hallucinates badly on empty input)."""
        mocked_stages.asr.return_value = {"text": "", "duration_s": 0.0, "latency_s": 0.05}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        prompt = mocked_stages.llm.call_args.args[0]
        assert prompt != ""

    def test_whitespace_transcript_gets_fallback_prompt(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.return_value = {"text": "   \n  ", "duration_s": 0.0, "latency_s": 0.05}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        prompt = mocked_stages.llm.call_args.args[0]
        assert prompt.strip() != ""

    def test_empty_llm_reply_gets_fallback_text(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.llm.return_value = {"text": "", "latency_s": 0.1}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        # TTS must be called with non-empty text.
        spoken = mocked_stages.tts.call_args.args[0]
        assert spoken.strip() != ""


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_returns_expected_keys(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        assert set(result.keys()) == {
            "input_path", "transcript", "reply_text", "reply_wav", "latencies",
        }
        assert set(result["latencies"].keys()) == {
            "record_s", "asr_s", "llm_s", "tts_s", "play_s", "total_s",
        }

    def test_total_latency_equals_sum(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        lat = result["latencies"]
        total = lat["record_s"] + lat["asr_s"] + lat["llm_s"] + lat["tts_s"] + lat["play_s"]
        assert lat["total_s"] == pytest.approx(total)

    def test_skipped_record_has_zero_latency(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        assert result["latencies"]["record_s"] == 0.0

    def test_skipped_play_has_zero_latency(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(
            input_wav=str(input_wav),
            output_wav=str(tmp_path / "out.wav"),
            skip_play=True,
        )
        assert result["latencies"]["play_s"] == 0.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_input_wav_raises(self, mocked_stages, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                input_wav=str(tmp_path / "nope.wav"),
                output_wav=str(tmp_path / "out.wav"),
                skip_play=True,
            )

    def test_asr_error_propagates(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.side_effect = RuntimeError("asr crashed")
        with pytest.raises(RuntimeError, match="asr crashed"):
            pipeline.run(
                input_wav=str(input_wav),
                output_wav=str(tmp_path / "out.wav"),
                skip_play=True,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_returns_2_when_input_missing(self, monkeypatch, capsys, tmp_path: Path):
        # Mock everything else so the only error is the missing file.
        monkeypatch.setattr(pipeline.asr, "transcribe", MagicMock())
        monkeypatch.setattr(pipeline.llm_client, "generate", MagicMock())
        monkeypatch.setattr(pipeline.tts, "synthesize", MagicMock())
        monkeypatch.setattr(pipeline.audio_io, "record", MagicMock())

        rc = pipeline.main([
            "--input", str(tmp_path / "missing.wav"),
            "--output", str(tmp_path / "out.wav"),
            "--no-play",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "ERROR" in err

    def test_main_forwards_model_flag(self, mocked_stages, input_wav: Path, tmp_path: Path):
        rc = pipeline.main([
            "--input", str(input_wav),
            "--output", str(tmp_path / "out.wav"),
            "--no-play",
            "--model", "tinyllama:1.1b",
        ])
        assert rc == 0
        assert mocked_stages.llm.call_args.kwargs.get("model") == "tinyllama:1.1b"


# ---------------------------------------------------------------------------
# Integration — real ASR + LLM + TTS
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealChain:
    def test_full_chain_on_input_wav(self, tmp_path: Path):
        """End-to-end with real models. Uses a pre-recorded sample if
        available so ASR has something to transcribe."""
        sample = Path("recordings/sample1.wav")
        if not sample.exists():
            pytest.skip("recordings/sample1.wav not present")

        result = pipeline.run(
            input_wav=str(sample),
            output_wav=str(tmp_path / "reply.wav"),
            skip_play=True,
        )
        assert Path(result["reply_wav"]).exists()
        assert Path(result["reply_wav"]).stat().st_size > 1000  # non-trivial wav
        assert result["transcript"]  # ASR returned something
        assert result["reply_text"]  # LLM returned something
        # Per-stage latencies all positive (live record skipped).
        lat = result["latencies"]
        assert lat["asr_s"] > 0
        assert lat["llm_s"] > 0
        assert lat["tts_s"] > 0