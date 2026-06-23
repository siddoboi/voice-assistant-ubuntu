"""
asr.py — Automatic Speech Recognition via faster-whisper
Model: tiny.en (Phase 1)
"""

import time
from faster_whisper import WhisperModel

def _load_model_size() -> str:
    import os, yaml
    config_path = os.environ.get('VOICE_ASSISTANT_CONFIG', 'configs/dev_config.yaml')
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get('asr', {}).get('model_size', 'tiny.en')
    except Exception:
        return 'tiny.en'

MODEL_SIZE = _load_model_size()
_model = None


def load_model():
    """Load the Whisper model (cached after first call)."""
    global _model
    if _model is None:
        print(f"Loading ASR model: {MODEL_SIZE}")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("ASR model loaded.")
    return _model


def transcribe(audio_path: str) -> dict:
    """
    Transcribe a .wav file and return text + timing.
    Args:
        audio_path: path to .wav file
    Returns:
        dict with keys: text, duration_s, latency_s
    """
    model = load_model()
    start = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5, language="en", initial_prompt="Indian English speaker.")
    text = " ".join([seg.text.strip() for seg in segments])
    latency = round(time.time() - start, 3)
    return {
        "text": text,
        "duration_s": round(info.duration, 2),
        "latency_s": latency
    }


if __name__ == "__main__":
    samples = [
        "recordings/sample1.wav",
        "recordings/sample2.wav",
    ]

    print("=== ASR Benchmark ===\n")
    for path in samples:
        print(f"File: {path}")
        result = transcribe(path)
        print(f"Transcription : {result['text']}")
        print(f"Audio duration: {result['duration_s']}s")
        print(f"ASR latency   : {result['latency_s']}s")
        rtf = round(result['latency_s'] / result['duration_s'], 3)
        print(f"Real-time factor (RTF): {rtf}")
        print()