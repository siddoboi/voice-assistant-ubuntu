"""
test_pipeline.py — Unit tests for src/pipeline.py (Week 3 Day 4 streaming chain).

The pipeline is orchestration code. Unit tests mock every stage module
(`audio_io`, `asr`, `llm_client`, `tts`) and the `ConversationManager` class,
and verify:
  - stages execute and data flows correctly (transcript -> user turn ->
    build_messages -> streaming LLM -> streaming TTS -> playback)
  - the --input path bypasses recording
  - the --no-play flag skips playback but still writes the reply
  - empty transcripts and empty replies get sensible fallbacks
  - the conversation lifecycle is correct (own vs caller-supplied session)
  - the bounded queue plays every chunk, in order
  - the per-stage latency dict is well-formed

Integration test (--run-integration) wires real ASR + LLM + TTS + a real
ConversationManager against recordings/sample1.wav.

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


def _fake_synth_stream_factory(samples_per_sentence: int = 100):
    """Build a fake tts.synthesize_stream that CONSUMES its sentence iterator
    (so the pipeline's reply-text recorder populates) and yields one int16
    PCM chunk per sentence."""

    def fake_synth_stream(sentences, sample_rate=None):
        for _sentence in sentences:
            yield np.full(samples_per_sentence, 1, dtype=np.int16)

    return fake_synth_stream


@pytest.fixture
def mocked_stages(monkeypatch):
    """Mock every external dependency the streaming pipeline touches."""
    asr_mock = MagicMock(return_value={"text": "hello world", "duration_s": 1.0, "latency_s": 0.1})

    # Streaming LLM: yields two sentences.
    def llm_stream(messages, model=None):
        yield "Hi there."
        yield "How can I help?"
    llm_mock = MagicMock(side_effect=llm_stream)

    synth_mock = MagicMock(side_effect=_fake_synth_stream_factory())
    output_rate_mock = MagicMock(return_value=22050)

    record_mock = MagicMock(return_value=np.zeros(16000, dtype=np.int16))
    save_wav_mock = MagicMock()
    load_wav_mock = MagicMock(return_value=(np.zeros(11025, dtype=np.int16), 22050))
    play_mock = MagicMock()

    # ConversationManager: a mock instance with the methods the pipeline calls.
    conv = MagicMock()
    conv.build_messages.return_value = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello world"},
    ]
    conv.session_id = "test-session-id"
    cm_class_mock = MagicMock(return_value=conv)

    monkeypatch.setattr(pipeline.asr, "transcribe", asr_mock)
    monkeypatch.setattr(pipeline.llm_client, "stream_sentences_from_messages", llm_mock)
    monkeypatch.setattr(pipeline.tts, "synthesize_stream", synth_mock)
    monkeypatch.setattr(pipeline.tts, "output_sample_rate", output_rate_mock)
    monkeypatch.setattr(pipeline.audio_io, "record", record_mock)
    monkeypatch.setattr(pipeline.audio_io, "save_wav", save_wav_mock)
    monkeypatch.setattr(pipeline.audio_io, "load_wav", load_wav_mock)
    monkeypatch.setattr(pipeline.audio_io, "play", play_mock)
    monkeypatch.setattr(pipeline, "ConversationManager", cm_class_mock)
    # Pin the buffer size so tests are deterministic regardless of config file.
    monkeypatch.setattr(pipeline, "_max_chunks", lambda config_path=None: 3)

    class Stages:
        pass

    s = Stages()
    s.asr = asr_mock
    s.llm = llm_mock
    s.synth = synth_mock
    s.output_rate = output_rate_mock
    s.record = record_mock
    s.save_wav = save_wav_mock
    s.load_wav = load_wav_mock
    s.play = play_mock
    s.cm_class = cm_class_mock
    s.conv = conv
    return s


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_input_wav_skips_recording(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.record.assert_not_called()
        assert result["input_path"] == str(input_wav)

    def test_no_input_triggers_recording(self, mocked_stages, tmp_path: Path):
        result = pipeline.run(duration_s=3.0, output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.record.assert_called_once()
        assert mocked_stages.record.call_args.kwargs.get("duration_sec") == 3.0
        assert "recordings" in result["input_path"]

    def test_asr_called_with_input_path(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.asr.assert_called_once_with(str(input_wav))

    def test_user_turn_added_with_transcript(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.return_value = {"text": "what's the weather", "duration_s": 2.0, "latency_s": 0.2}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.conv.add_user_turn.assert_called_once_with("what's the weather")

    def test_llm_receives_built_messages(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.conv.build_messages.assert_called_once()
        # First positional arg to the streaming LLM is the messages list.
        assert mocked_stages.llm.call_args.args[0] == mocked_stages.conv.build_messages.return_value

    def test_tts_stream_invoked(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.synth.assert_called_once()
        # synthesize_stream gets the voice sample rate.
        assert mocked_stages.synth.call_args.kwargs.get("sample_rate") == 22050

    def test_assistant_turn_added_with_full_reply(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        # Full reply = both streamed sentences joined.
        mocked_stages.conv.add_assistant_turn.assert_called_once_with("Hi there. How can I help?")

    def test_reply_text_in_result(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert result["reply_text"] == "Hi there. How can I help?"
        assert result["num_sentences"] == 2


# ---------------------------------------------------------------------------
# Playback / bounded queue
# ---------------------------------------------------------------------------


class TestPlayback:
    def test_no_play_skips_playback(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.play.assert_not_called()

    def test_plays_one_chunk_per_sentence(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=False)
        # Two sentences -> two PCM chunks -> two play calls.
        assert mocked_stages.play.call_count == 2

    def test_play_uses_voice_sample_rate(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=False)
        assert mocked_stages.play.call_args.kwargs.get("sample_rate") == 22050

    def test_all_chunks_played_under_small_buffer(self, mocked_stages, input_wav: Path, tmp_path: Path):
        # Many sentences with a buffer of 1 → exercises back-pressure; all must play.
        def many(messages, model=None):
            for i in range(10):
                yield f"Sentence {i}."
        mocked_stages.llm.side_effect = many
        with patch.object(pipeline, "_max_chunks", lambda config_path=None: 1):
            result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=False)
        assert mocked_stages.play.call_count == 10
        assert result["num_audio_chunks"] == 10

    def test_reply_wav_written(self, mocked_stages, input_wav: Path, tmp_path: Path):
        out = tmp_path / "out.wav"
        pipeline.run(input_wav=str(input_wav), output_wav=str(out), skip_play=True)
        # Concatenated PCM saved at the voice rate.
        mocked_stages.save_wav.assert_called()
        assert mocked_stages.save_wav.call_args.kwargs.get("sample_rate") == 22050


# ---------------------------------------------------------------------------
# Model override
# ---------------------------------------------------------------------------


class TestModelOverride:
    def test_no_model_omits_model_kwarg(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert "model" not in mocked_stages.llm.call_args.kwargs

    def test_explicit_model_forwarded(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"),
                     skip_play=True, llm_model="tinyllama:1.1b")
        assert mocked_stages.llm.call_args.kwargs.get("model") == "tinyllama:1.1b"


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_empty_transcript_gets_fallback(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.return_value = {"text": "", "duration_s": 0.0, "latency_s": 0.05}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        added = mocked_stages.conv.add_user_turn.call_args.args[0]
        assert added.strip() != ""
        assert added == pipeline.FALLBACK_TRANSCRIPT

    def test_whitespace_transcript_gets_fallback(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.return_value = {"text": "   \n ", "duration_s": 0.0, "latency_s": 0.05}
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        added = mocked_stages.conv.add_user_turn.call_args.args[0]
        assert added.strip() != ""

    def test_empty_reply_gets_fallback_text(self, mocked_stages, input_wav: Path, tmp_path: Path):
        # LLM yields no sentences → reply falls back, and TTS is invoked twice
        # (once for the empty main stream, once for the fallback).
        mocked_stages.llm.side_effect = lambda messages, model=None: iter([])
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert result["reply_text"] == pipeline.FALLBACK_REPLY
        mocked_stages.conv.add_assistant_turn.assert_called_once_with(pipeline.FALLBACK_REPLY)


# ---------------------------------------------------------------------------
# Conversation lifecycle
# ---------------------------------------------------------------------------


class TestConversationLifecycle:
    def test_creates_and_ends_own_session(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.cm_class.assert_called_once()
        mocked_stages.conv.end_session.assert_called_once()

    def test_caller_supplied_conversation_not_constructed_or_ended(
        self, mocked_stages, input_wav: Path, tmp_path: Path
    ):
        external = MagicMock()
        external.build_messages.return_value = [{"role": "user", "content": "hi"}]
        external.session_id = "external-session"
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"),
                     skip_play=True, conversation=external)
        # The pipeline must NOT build its own manager or end the caller's.
        mocked_stages.cm_class.assert_not_called()
        external.end_session.assert_not_called()
        external.add_user_turn.assert_called_once()
        external.add_assistant_turn.assert_called_once()

    def test_session_id_in_result(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert result["session_id"] == "test-session-id"

    def test_session_id_forwarded_to_manager(self, mocked_stages, input_wav: Path, tmp_path: Path):
        pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"),
                     skip_play=True, session_id="resume-me")
        assert mocked_stages.cm_class.call_args.kwargs.get("session_id") == "resume-me"


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_returns_expected_keys(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert set(result.keys()) == {
            "input_path", "transcript", "reply_text", "reply_wav",
            "session_id", "num_sentences", "num_audio_chunks", "latencies",
        }
        assert set(result["latencies"].keys()) == {
            "record_s", "asr_s", "first_audio_s", "stream_s", "perceived_s", "total_s",
        }

    def test_total_is_record_plus_asr_plus_stream(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        lat = result["latencies"]
        assert lat["total_s"] == pytest.approx(lat["record_s"] + lat["asr_s"] + lat["stream_s"])

    def test_perceived_is_asr_plus_first_audio(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        lat = result["latencies"]
        assert lat["perceived_s"] == pytest.approx(lat["asr_s"] + lat["first_audio_s"])

    def test_skipped_record_zero_latency(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert result["latencies"]["record_s"] == 0.0

    def test_first_audio_non_negative(self, mocked_stages, input_wav: Path, tmp_path: Path):
        result = pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        assert result["latencies"]["first_audio_s"] >= 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestMaxChunksConfig:
    def test_reads_value_from_config(self, monkeypatch):
        monkeypatch.setattr(
            pipeline.audio_io, "_get_config",
            lambda config_path=None: {"pipeline": {"tts_buffer_max_chunks": 5}},
        )
        assert pipeline._max_chunks(None) == 5

    def test_defaults_to_3_when_absent(self, monkeypatch):
        monkeypatch.setattr(pipeline.audio_io, "_get_config", lambda config_path=None: {})
        assert pipeline._max_chunks(None) == 3

    def test_non_positive_falls_back_to_3(self, monkeypatch):
        monkeypatch.setattr(
            pipeline.audio_io, "_get_config",
            lambda config_path=None: {"pipeline": {"tts_buffer_max_chunks": 0}},
        )
        assert pipeline._max_chunks(None) == 3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_input_wav_raises(self, mocked_stages, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            pipeline.run(input_wav=str(tmp_path / "nope.wav"), output_wav=str(tmp_path / "out.wav"), skip_play=True)

    def test_asr_error_propagates(self, mocked_stages, input_wav: Path, tmp_path: Path):
        mocked_stages.asr.side_effect = RuntimeError("asr crashed")
        with pytest.raises(RuntimeError, match="asr crashed"):
            pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)

    def test_own_session_ended_even_on_error(self, mocked_stages, input_wav: Path, tmp_path: Path):
        # If the LLM stream raises mid-turn, the owned session must still end.
        def boom(messages, model=None):
            raise RuntimeError("llm crashed")
            yield  # pragma: no cover
        mocked_stages.llm.side_effect = boom
        with pytest.raises(RuntimeError, match="llm crashed"):
            pipeline.run(input_wav=str(input_wav), output_wav=str(tmp_path / "out.wav"), skip_play=True)
        mocked_stages.conv.end_session.assert_called_once()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_returns_2_when_input_missing(self, mocked_stages, capsys, tmp_path: Path):
        rc = pipeline.main([
            "--input", str(tmp_path / "missing.wav"),
            "--output", str(tmp_path / "out.wav"),
            "--no-play",
        ])
        assert rc == 2
        assert "ERROR" in capsys.readouterr().err

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
# Integration — real ASR + LLM + TTS + ConversationManager
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealChain:
    def test_full_streaming_chain_on_sample(self, tmp_path: Path):
        sample = Path("recordings/sample1.wav")
        if not sample.exists():
            pytest.skip("recordings/sample1.wav not present")

        result = pipeline.run(
            input_wav=str(sample),
            output_wav=str(tmp_path / "reply.wav"),
            skip_play=True,
            config_path=None,
        )
        assert Path(result["reply_wav"]).exists()
        assert Path(result["reply_wav"]).stat().st_size > 1000
        assert result["transcript"]
        assert result["reply_text"]
        assert result["num_audio_chunks"] >= 1
        lat = result["latencies"]
        assert lat["asr_s"] > 0
        assert lat["first_audio_s"] > 0
        assert lat["stream_s"] >= lat["first_audio_s"]
        # The Week 3 goal: perceived latency target (informational, not asserted hard).
        print(f"\nPerceived latency (asr+first_audio): {lat['perceived_s']:.3f}s")