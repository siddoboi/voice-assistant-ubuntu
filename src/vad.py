"""
vad.py — Voice Activity Detection via Silero VAD v4 (ONNX Runtime)
Model: silero_vad_v4.onnx
"""

import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/silero/silero_vad_v4.onnx"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
def _load_silence_threshold() -> float:
    import os, yaml
    config_path = os.environ.get('VOICE_ASSISTANT_CONFIG',
                                 'configs/dev_config.yaml')
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get('vad', {}).get('silence_threshold', 0.5))
    except Exception:
        return 0.5

SILENCE_THRESHOLD = _load_silence_threshold()

_session = None
_h = None
_c = None


def load_model():
    """Load Silero VAD v4 ONNX model (cached after first call)."""
    global _session, _h, _c
    if _session is None:
        print(f"Loading VAD model: {MODEL_PATH}")
        _session = ort.InferenceSession(MODEL_PATH)
        _h = np.zeros((2, 1, 64), dtype=np.float32)
        _c = np.zeros((2, 1, 64), dtype=np.float32)
        print("VAD model loaded.")
    return _session


def reset_state():
    """Reset VAD state — call at the start of each new utterance."""
    global _h, _c
    _h = np.zeros((2, 1, 64), dtype=np.float32)
    _c = np.zeros((2, 1, 64), dtype=np.float32)


def get_speech_prob(audio_chunk: np.ndarray) -> float:
    """
    Get speech probability for a single audio chunk.
    Args:
        audio_chunk: numpy array of float32 audio, length CHUNK_SIZE
    Returns:
        float probability of speech (0.0 to 1.0)
    """
    global _h, _c
    session = load_model()

    chunk = audio_chunk.reshape(1, -1).astype(np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)

    out, _h, _c = session.run(
        None,
        {"input": chunk, "sr": sr, "h": _h, "c": _c}
    )
    return float(out[0][0])


def is_speech(audio_chunk: np.ndarray) -> bool:
    """Return True if chunk contains speech."""
    return get_speech_prob(audio_chunk) >= SILENCE_THRESHOLD


def test_on_file(wav_path: str) -> dict:
    """
    Test VAD on a .wav file — auto resamples to 16kHz mono.
    Args:
        wav_path: path to .wav file
    Returns:
        dict with speech_ratio, total_chunks, speech_chunks, latency_s
    """
    import wave

    reset_state()

    with wave.open(wav_path, "r") as wf:
        original_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # Mix down to mono if stereo
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    # Resample to 16kHz if needed
    if original_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / original_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    start = time.time()
    total_chunks = 0
    speech_chunks = 0

    for i in range(0, len(audio) - CHUNK_SIZE, CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        if len(chunk) == CHUNK_SIZE:
            if is_speech(chunk):
                speech_chunks += 1
            total_chunks += 1

    latency = round(time.time() - start, 3)
    speech_ratio = round(speech_chunks / total_chunks, 3) if total_chunks > 0 else 0

    return {
        "file": wav_path,
        "original_rate": original_rate,
        "total_chunks": total_chunks,
        "speech_chunks": speech_chunks,
        "silence_chunks": total_chunks - speech_chunks,
        "speech_ratio": speech_ratio,
        "latency_s": latency
    }


if __name__ == "__main__":
    test_files = [
        "recordings/sample1.wav",
        "recordings/sample2.wav",
        "recordings/tts_test_1.wav",
    ]

    print("=== VAD Benchmark ===\n")
    for wav_path in test_files:
        print(f"File: {wav_path}")
        result = test_on_file(wav_path)
        print(f"Original rate : {result['original_rate']}Hz -> resampled to 16000Hz")
        print(f"Total chunks  : {result['total_chunks']}")
        print(f"Speech chunks : {result['speech_chunks']}")
        print(f"Silence chunks: {result['silence_chunks']}")
        print(f"Speech ratio  : {result['speech_ratio']*100:.1f}%")
        print(f"VAD latency   : {result['latency_s']}s")
        print()