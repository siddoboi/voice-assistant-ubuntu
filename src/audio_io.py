"""
audio_io.py — Audio capture, playback, and file I/O for the voice assistant.

Provides config-driven device selection and the small set of audio utilities
used by the rest of the pipeline:

    - list_devices()          : enumerate input/output devices
    - record()                : capture from mic to a numpy array
    - play()                  : play a numpy array to the speaker
    - resample_8_to_16k()     : telephony 8kHz -> ASR 16kHz
    - save_wav() / load_wav() : .wav file persistence

All defaults (sample rate, channels, dtype, device indices) are loaded
from configs/dev_config.yaml (or configs/pi_config.yaml on the Pi). No
hardcoded magic values inside the functions themselves.

Audio convention across the project:
    - In-memory numpy arrays are float32 in the range [-1.0, 1.0] for
      processing (VAD, ASR) and int16 for capture/playback hardware paths.
    - .wav files on disk are int16 PCM, mono, at the configured sample rate.
    - resample_8_to_16k uses np.interp — the same approach already proven
      in vad.py.test_on_file (see SECTION 6.4 of the master context).

This module is imported by pipeline.py, vad.py tests, and tests/test_audio_io.py.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sounddevice as sd
import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Default config path is resolved relative to the project root (one level
# above this file's parent — src/audio_io.py -> project_root/configs/...).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "dev_config.yaml"

# Module-level config cache. Populated on first call to _get_config().
_config: Optional[dict] = None
_config_path_loaded: Optional[Path] = None


def _get_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load and cache the audio config.

    Args:
        config_path: Optional explicit path. If None, falls back to the
            VOICE_ASSISTANT_CONFIG env var, then to the default
            configs/dev_config.yaml.

    Returns:
        Parsed config dict. The 'audio' and 'paths' subsections are
        guaranteed to exist (empty dicts if not present in the file).

    Raises:
        FileNotFoundError: if the resolved config path does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    global _config, _config_path_loaded

    if config_path is None:
        env_path = os.environ.get("VOICE_ASSISTANT_CONFIG")
        resolved = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    else:
        resolved = Path(config_path)

    # Re-load if the requested path differs from the cached one.
    if _config is not None and _config_path_loaded == resolved:
        return _config

    if not resolved.exists():
        raise FileNotFoundError(
            f"Audio config not found at {resolved}. "
            "Set VOICE_ASSISTANT_CONFIG or pass config_path explicitly."
        )

    with resolved.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    # Normalise: ensure expected subsections exist.
    loaded.setdefault("audio", {})
    loaded.setdefault("paths", {})

    _config = loaded
    _config_path_loaded = resolved
    return _config


def _audio_cfg(key: str, default: Any, config_path: Optional[Union[str, Path]] = None) -> Any:
    """Fetch a single key from the audio: subsection with a default fallback."""
    return _get_config(config_path).get("audio", {}).get(key, default)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def list_devices() -> list[dict]:
    """List all available audio devices.

    Prints a formatted table of (index, name, max_input_channels,
    max_output_channels, default_samplerate) to stdout and also returns
    the underlying list of device dicts from sounddevice.

    Returns:
        List of device-info dicts as produced by sounddevice.query_devices().
        Empty list if PortAudio cannot enumerate any devices (common on
        bare WSL2 without WSLg audio).
    """
    try:
        devices = sd.query_devices()
    except Exception as e:  # PortAudio errors surface as generic Exception
        print(f"[audio_io] Could not query devices: {e}")
        return []

    # sounddevice returns a DeviceList; coerce to plain list of dicts.
    device_list = [dict(d) for d in devices]

    header = (
        f"{'IDX':>3}  {'IN':>3}  {'OUT':>3}  {'RATE':>7}  NAME"
    )
    print(header)
    print("-" * len(header))
    for idx, d in enumerate(device_list):
        print(
            f"{idx:>3}  "
            f"{d.get('max_input_channels', 0):>3}  "
            f"{d.get('max_output_channels', 0):>3}  "
            f"{int(d.get('default_samplerate', 0)):>7}  "
            f"{d.get('name', '?')}"
        )
    return device_list


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record(
    duration_sec: Optional[float] = None,
    sample_rate: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    channels: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Record audio from a microphone into a numpy array.

    Args:
        duration_sec: Seconds to record. If None, uses
            audio.default_record_duration_sec from config.
        sample_rate: Capture rate in Hz. If None, uses audio.sample_rate.
        device: Input device index or name. If None, uses
            audio.input_device from config (which may itself be null,
            in which case sounddevice's system default is used).
        channels: Channel count. If None, uses audio.channels.
        config_path: Optional explicit config file path.

    Returns:
        np.ndarray of shape (n_samples,) for mono or (n_samples, channels)
        for multi-channel, dtype int16.

    Raises:
        ValueError: if duration_sec <= 0 or sample_rate <= 0.
        sounddevice.PortAudioError: if the device cannot be opened
            (e.g. WSL2 with no audio passthrough).
    """
    cfg_sample_rate = _audio_cfg("sample_rate", 16000, config_path)
    cfg_channels = _audio_cfg("channels", 1, config_path)
    cfg_device = _audio_cfg("input_device", None, config_path)
    cfg_duration = _audio_cfg("default_record_duration_sec", 5.0, config_path)

    sr = int(sample_rate if sample_rate is not None else cfg_sample_rate)
    ch = int(channels if channels is not None else cfg_channels)
    dev = device if device is not None else cfg_device
    dur = float(duration_sec if duration_sec is not None else cfg_duration)

    if dur <= 0:
        raise ValueError(f"duration_sec must be > 0, got {dur}")
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sr}")
    if ch <= 0:
        raise ValueError(f"channels must be > 0, got {ch}")

    n_samples = int(round(dur * sr))
    audio = sd.rec(
        n_samples,
        samplerate=sr,
        channels=ch,
        device=dev,
        dtype="int16",
    )
    sd.wait()  # block until capture completes

    # For mono, squeeze the trailing singleton axis to give a 1-D array.
    if ch == 1 and audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]

    return audio


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def play(
    audio_array: np.ndarray,
    sample_rate: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    config_path: Optional[Union[str, Path]] = None,
    blocking: bool = True,
) -> None:
    """Play an audio array through the configured output device.

    Args:
        audio_array: 1-D mono or 2-D (n_samples, channels) array.
            Accepts int16, float32, or float64. float arrays are assumed
            to be in [-1.0, 1.0].
        sample_rate: Playback rate in Hz. If None, uses audio.sample_rate.
        device: Output device index or name. If None, uses
            audio.output_device from config.
        config_path: Optional explicit config file path.
        blocking: If True (default), wait for playback to finish before
            returning. If False, return immediately after starting.

    Raises:
        TypeError: if audio_array is not a numpy ndarray.
        ValueError: if audio_array is empty or has an unsupported dtype.
    """
    if not isinstance(audio_array, np.ndarray):
        raise TypeError(
            f"audio_array must be np.ndarray, got {type(audio_array).__name__}"
        )
    if audio_array.size == 0:
        raise ValueError("audio_array is empty; nothing to play")

    cfg_sample_rate = _audio_cfg("sample_rate", 16000, config_path)
    cfg_device = _audio_cfg("output_device", None, config_path)

    sr = int(sample_rate if sample_rate is not None else cfg_sample_rate)
    dev = device if device is not None else cfg_device

    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sr}")

    # sounddevice accepts int16 and float32 directly. Cast float64 down to
    # float32 to avoid a sounddevice copy + warning.
    if audio_array.dtype == np.float64:
        audio_array = audio_array.astype(np.float32)
    elif audio_array.dtype not in (np.int16, np.float32):
        raise ValueError(
            f"Unsupported dtype {audio_array.dtype}; expected int16, float32, or float64"
        )

    sd.play(audio_array, samplerate=sr, device=dev)
    if blocking:
        sd.wait()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_8_to_16k(audio_array: np.ndarray) -> np.ndarray:
    """Resample an 8 kHz audio array to 16 kHz via linear interpolation.

    This mirrors the np.interp-based resampling already used in
    vad.py.test_on_file() for consistency across the pipeline. It is
    fast enough for the per-turn 30 ms resample budget on Pi and produces
    audio that faster-whisper transcribes correctly.

    Args:
        audio_array: 1-D numpy array of 8 kHz samples (any numeric dtype).

    Returns:
        1-D numpy array at 16 kHz. Output length = 2 * input length - 1
        (np.interp does not extrapolate past the final sample). Dtype is
        preserved for int16 input; otherwise float32.

    Raises:
        TypeError: if audio_array is not a numpy ndarray.
        ValueError: if audio_array is not 1-D or is empty.
    """
    if not isinstance(audio_array, np.ndarray):
        raise TypeError(
            f"audio_array must be np.ndarray, got {type(audio_array).__name__}"
        )
    if audio_array.ndim != 1:
        raise ValueError(
            f"resample_8_to_16k expects a 1-D mono array, got shape {audio_array.shape}"
        )
    if audio_array.size == 0:
        raise ValueError("audio_array is empty")

    original_dtype = audio_array.dtype
    src = audio_array.astype(np.float32, copy=False)

    n_in = src.shape[0]
    # Source sample positions: 0, 1, ..., n_in-1
    # Target sample positions at 2x rate: 0.0, 0.5, 1.0, 1.5, ..., n_in-1
    # That gives 2*n_in - 1 output samples.
    n_out = 2 * n_in - 1
    src_x = np.arange(n_in, dtype=np.float32)
    tgt_x = np.linspace(0.0, n_in - 1, n_out, dtype=np.float32)

    resampled = np.interp(tgt_x, src_x, src).astype(np.float32)

    # Preserve int16 input dtype for callers passing raw PCM. Float inputs
    # stay float32 — that's what VAD and ASR want anyway.
    if np.issubdtype(original_dtype, np.integer):
        # Clip to int16 range before casting, just in case interpolation
        # produced anything fractionally outside.
        info = np.iinfo(np.int16)
        resampled = np.clip(resampled, info.min, info.max).astype(np.int16)

    return resampled


def reduce_noise(audio_array, sample_rate=None, config_path=None):
    """Spectral noise suppression via noisereduce, applied before VAD/ASR."""
    if not isinstance(audio_array, np.ndarray):
        raise TypeError(f"audio_array must be np.ndarray, got {type(audio_array).__name__}")
    if audio_array.ndim != 1:
        raise ValueError(f"reduce_noise expects a 1-D mono array, got shape {audio_array.shape}")
    if audio_array.size == 0:
        raise ValueError("audio_array is empty")

    nr_cfg = _get_config(config_path).get("noise_reduction", {}) or {}
    if not nr_cfg.get("enabled", True):
        return audio_array

    sr = int(sample_rate if sample_rate is not None else _audio_cfg("sample_rate", 16000, config_path))
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sr}")

    stationary = bool(nr_cfg.get("stationary", False))
    prop_decrease = float(nr_cfg.get("prop_decrease", 1.0))

    import noisereduce as nr  # lazy: keeps audio_io import light and optional

    original_dtype = audio_array.dtype
    if np.issubdtype(original_dtype, np.integer):
        y = audio_array.astype(np.float32) / 32768.0
    else:
        y = audio_array.astype(np.float32, copy=False)

    reduced = nr.reduce_noise(y=y, sr=sr, stationary=stationary, prop_decrease=prop_decrease)
    reduced = np.asarray(reduced, dtype=np.float32)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(np.int16)
        reduced = np.clip(reduced * 32768.0, info.min, info.max).astype(np.int16)
    return reduced

# ---------------------------------------------------------------------------
# WAV file I/O
# ---------------------------------------------------------------------------


def save_wav(
    audio_array: np.ndarray,
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """Save an audio array to a .wav file as 16-bit PCM mono.

    Args:
        audio_array: 1-D mono or 2-D (n_samples, channels) numpy array.
            float arrays in [-1.0, 1.0] are scaled to int16. int16 arrays
            are written as-is.
        path: Output file path. Parent directory must exist.
        sample_rate: Output sample rate in Hz. If None, uses
            audio.sample_rate from config.
        config_path: Optional explicit config file path.

    Raises:
        TypeError: if audio_array is not a numpy ndarray.
        ValueError: if audio_array is empty or has an unsupported dtype.
        FileNotFoundError: if path's parent directory does not exist.
    """
    if not isinstance(audio_array, np.ndarray):
        raise TypeError(
            f"audio_array must be np.ndarray, got {type(audio_array).__name__}"
        )
    if audio_array.size == 0:
        raise ValueError("audio_array is empty; nothing to save")

    cfg_sample_rate = _audio_cfg("sample_rate", 16000, config_path)
    sr = int(sample_rate if sample_rate is not None else cfg_sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sr}")

    out_path = Path(path)
    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory does not exist: {out_path.parent}"
        )

    # Convert to int16 mono.
    arr = audio_array
    if arr.ndim == 2:
        # Downmix to mono by averaging channels.
        arr = arr.mean(axis=1)

    if np.issubdtype(arr.dtype, np.floating):
        # Assume float input is in [-1.0, 1.0]. Clip and scale.
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype == np.int16:
        arr = arr.astype(np.int16, copy=False)
    elif np.issubdtype(arr.dtype, np.integer):
        # Other integer widths — clip into int16 range.
        info = np.iinfo(np.int16)
        arr = np.clip(arr, info.min, info.max).astype(np.int16)
    else:
        raise ValueError(f"Unsupported dtype {arr.dtype}")

    # wave.open requires params set before writeframes — same pattern that
    # tripped Piper TTS earlier in this project (see SECTION 9 of master
    # context).
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())


def load_wav(path: Union[str, Path]) -> tuple[np.ndarray, int]:
    """Load a .wav file into a numpy array.

    Args:
        path: Path to the .wav file.

    Returns:
        Tuple of (audio_array, sample_rate). audio_array is int16 1-D
        for mono input or 2-D (n_samples, channels) for multi-channel.

    Raises:
        FileNotFoundError: if path does not exist.
        wave.Error: if the file is not a valid PCM WAV.
        ValueError: if the file uses a sample width other than 1 or 2 bytes.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"WAV file not found: {in_path}")

    with wave.open(str(in_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        # 8-bit WAV is unsigned; centre at zero and widen to int16 so the
        # rest of the pipeline gets a uniform dtype.
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        arr = (arr - 128) * 256
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype=np.int16)
    else:
        raise ValueError(
            f"Unsupported sample width {sample_width} bytes; expected 1 or 2"
        )

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels)

    return arr, sample_rate


# ---------------------------------------------------------------------------
# CLI for manual sanity checks
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== audio_io.py sanity check ===\n")
    print("Devices:")
    list_devices()

    print("\nConfig loaded from:", _config_path_loaded or _DEFAULT_CONFIG_PATH)
    cfg = _get_config()
    print("Audio section:", cfg.get("audio"))

    # Resample sanity: a 1-second 440 Hz sine at 8kHz -> 16kHz.
    t = np.linspace(0.0, 1.0, 8000, endpoint=False, dtype=np.float32)
    sine_8k = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sine_16k = resample_8_to_16k(sine_8k)
    print(
        f"\nResample check: 8kHz len={sine_8k.shape[0]} -> 16kHz len={sine_16k.shape[0]} "
        f"(expected {2 * sine_8k.shape[0] - 1})"
    )

    # Round-trip save_wav / load_wav.
    rec_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("recordings_dir", "recordings")
    rec_dir.mkdir(parents=True, exist_ok=True)
    test_path = rec_dir / "audio_io_selftest.wav"
    save_wav(sine_16k, test_path, sample_rate=16000)
    loaded, loaded_sr = load_wav(test_path)
    print(
        f"Round-trip check: saved {sine_16k.shape[0]} samples @16kHz, "
        f"loaded {loaded.shape[0]} samples @{loaded_sr}Hz"
    )