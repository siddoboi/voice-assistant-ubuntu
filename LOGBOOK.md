# LOGBOOK - Voice Assistant (Offline LLM on Raspberry Pi 5)

All times are WSL2 x86_64 unless explicitly marked **[Pi]**.
RTF = Real-Time Factor (processing time ÷ audio duration). Lower is better.
Pi 5 has **not** arrived as of Week 2 Day 1. Pi-required tasks deferred.

---

## WEEK 1 - WSL2 Environment & Benchmarking

**Goal:** Full dev environment on WSL2, all models installed and benchmarked,
all core module skeletons written, model choices locked.

**Status: ✅ COMPLETE**

---

### Day 1 - WSL2 Setup

**Theme:** Base environment only. No models yet.

#### Done
- Confirmed Debian 13 Trixie WSL2 (x86_64, 3.1 GB RAM available to WSL2)
- Python 3.13.5 installed from Trixie repos (`python3.13`, `python3.13-venv`, `python3.13-dev`)
- All system dependencies installed via apt:
  - Audio: `libportaudio2`, `portaudio19-dev`, `alsa-utils`, `sox`, `ffmpeg`
  - Build: `build-essential`, `pkg-config`, `libssl-dev`, `libffi-dev`
  - Storage: `libsqlite3-dev`
  - Utilities: `git`, `curl`, `wget`, `zstd`
- Project folder structure created (`src/`, `configs/`, `tests/`, `recordings/`, `docs/`)
- Python virtual environment created and activated (`venv/`)
- Ollama 0.24.0 installed via official install script

#### Issues
- None on Day 1.

#### Carries to Day 2
- Pull LLM models
- Write `llm_client.py` skeleton
- Benchmark LLM latency and RAM

---

### Day 2 - LLM Install & Test

**Theme:** Pull both LLM models, write client, benchmark.

#### Done
- Pulled `llama3.2:1b-instruct-q4_K_M` (807 MB disk) via `ollama pull`
- Pulled `tinyllama:1.1b` (637 MB disk) via `ollama pull`
- Both models verified with `ollama list`
- `src/llm_client.py` written with:
  - `generate(prompt, model)` → `{text, latency_s}`
  - `stream_generate(prompt, model)` → yields tokens
  - `measure_latency(model, prompt)` → `{model, first_token_latency_s, total_latency_s, response}`
  - `__main__` block runs both models and prints results

#### Benchmark Results (WSL2 x86_64)

| Model | First Token | Total Latency | RAM Delta |
|---|---|---|---|
| llama3.2:1b-instruct-q4_K_M | **0.297s** ✓ | 5.084s* | +1,020 MB |
| tinyllama:1.1b | **0.116s** ✓ | 5.981s* | not measured |

*Total latency on WSL2 is not representative of Pi (x86 Ollama ≠ ARM64 Ollama).*
*Only first_token_latency matters - both are under the 600ms budget.*

RAM detail: baseline 454 MB → after Llama 3.2 1B load 1,474 MB = **+1,020 MB**.
Well within the 3.5 GB Pi target.

#### Decisions (locked)
- **Primary LLM:** `llama3.2:1b-instruct-q4_K_M` - cleaner, more concise responses; 128K context window vs TinyLlama's 2K.
- **Fallback LLM:** `tinyllama:1.1b` - faster first token (0.116s), lower RAM, emergency fallback only.

#### Issues
- None.

---

### Day 3 - ASR Install & Test

**Theme:** faster-whisper install, tiny.en benchmark on pre-recorded .wav files.

#### Done
- `faster-whisper 1.2.1` installed inside venv
- `tiny.en` model downloaded and cached at `~/.cache/huggingface/hub/` (~75 MB)
- `src/asr.py` written with:
  - `load_model()` - loads WhisperModel, `device='cpu'`, `compute_type='int8'`, module-level cache
  - `transcribe(audio_path)` → `{text, duration_s, latency_s}`
  - `__main__` block runs on `sample1.wav` and `sample2.wav`, prints RTF

#### Benchmark Results (WSL2 x86_64)

| File | Audio Duration | ASR Latency | RTF | vs Budget |
|---|---|---|---|---|
| sample1.wav (8kHz mono) | 51.03s | 3.278s | 0.064 | ✓ |
| sample2.wav (8kHz mono) | 49.17s | 2.889s | 0.059 | ✓ |
| **Average** | 50.1s | 3.08s | **0.062** | 16× faster than real-time |

RTF 0.062 → for a 3–5s utterance, expected ASR latency ~200–300ms. Well under 800ms budget.
Transcription quality on telephony-quality 8kHz audio: clean, no obvious errors on both samples.

#### Decisions (locked)
- **ASR model:** `tiny.en` confirmed. No need to evaluate `base.en` for Phase 1.

#### Issues
- None.

---

### Day 4 - TTS Install & Test

**Theme:** Piper TTS install, voice model download, benchmark on synthesised sentences.

#### Done
- `piper-tts 1.4.2` installed inside venv
- `en_US-amy-medium.onnx` (61 MB) and `.onnx.json` (4.8 KB) downloaded to `models/piper/`
- `src/tts.py` written with:
  - `load_voice()` - loads `PiperVoice`, module-level cache
  - `synthesize(text, output_path)` → `{output_path, duration_s, latency_s, rtf}`
  - `__main__` block synthesises 5 sentences to `recordings/tts_test_1.wav` through `_5.wav`
- **Critical method:** `synthesize_wav()` - both `synthesize()` and `synthesize_stream_raw()` fail in piper-tts 1.4.2. Wave params (`nchannels`, `sampwidth`, `framerate`) must be set before calling.

#### Benchmark Results (WSL2 x86_64)

| Sentence | Audio Duration | TTS Latency | RTF |
|---|---|---|---|
| 1 | 4.029s | 0.316s | 0.078 |
| 2 | 4.272s | 0.290s | 0.068 |
| 3 | 4.365s | 0.262s | 0.060 |
| 4 | 3.913s | 0.236s | 0.060 |
| 5 | 2.844s | 0.199s | 0.070 |
| **Average** | 3.885s | **0.261s** | **0.067** |

Average latency 261ms - slightly over 250ms budget but acceptable. Streaming to speaker starts before full synthesis completes, so perceived latency is lower.

#### Decisions (locked)
- **TTS voice:** `en_US-amy-medium` confirmed. Performance exceeds expectations on WSL2.

#### Issues
- `synthesize()` and `synthesize_stream_raw()` both fail in piper-tts 1.4.2. Discovered via `dir(PiperVoice)`. **Fix:** use `synthesize_wav()` and set wav params manually before calling.
- `wave.Error: # channels not specified` — fixed by adding `setnchannels(1)`, `setsampwidth(2)`, `setframerate(voice.config.sample_rate)` before `synthesize_wav()`.

---

### Day 5 - VAD + Lock-in

**Theme:** Silero VAD v4 install and benchmark; write `models.yaml`; commit Week 1.

#### Done
- `onnxruntime 1.26.0` installed inside venv (no PyTorch - avoids ~2 GB disk usage)
- `silero_vad_v4.onnx` (1.8 MB) downloaded from the v4.0 branch tag to `models/silero/silero_vad_v4.onnx`
- `src/vad.py` written with:
  - `load_model()` — loads ONNX session, initialises `_h` and `_c` as `np.zeros((2,1,64), float32)`
  - `reset_state()` — resets `_h` and `_c`; must be called at start of each utterance
  - `get_speech_prob(chunk)` — runs one 512-sample chunk, updates `_h`/`_c`, returns float 0–1
  - `is_speech(chunk)` → `bool` (threshold 0.5)
  - `test_on_file(wav_path)` — reads wav, resamples to 16kHz via `np.interp`, returns stats dict
  - `__main__` block tests on `sample1.wav`, `sample2.wav`, `tts_test_1.wav`
- `configs/models.yaml` written with all locked model paths, versions, and benchmark numbers
- `setup.sh` written — idempotent, runs on WSL2 and Pi OS (same Debian base)
- Week 1 committed to GitHub: `https://github.com/siddoboi/voice-assistant`

#### Benchmark Results (WSL2 x86_64)

| File | Sample Rate | Total Chunks | Speech Chunks | Speech Ratio | VAD Latency |
|---|---|---|---|---|---|
| sample1.wav | 8kHz → 16kHz | 1,594 | 1,404 | 88.1% | 0.408s |
| sample2.wav | 8kHz → 16kHz | 1,536 | 1,366 | 88.9% | 0.348s |
| tts_test_1.wav | 22050Hz → 16kHz | 125 | 103 | 82.4% | 0.031s |

All files resampled to 16kHz before VAD via `np.interp` (same approach carried into `audio_io.py`).

#### Decisions (locked)
- **VAD model:** Silero v4 ONNX. v5 is broken with onnxruntime 1.26.0 - all-zero probabilities regardless of audio.
- **VAD threshold:** 0.5 (default). Correctly separates speech from silence on all test files.

#### Issues

| Issue | Root cause | Fix |
|---|---|---|
| torch install OSError: no space | CUDA torch ~2 GB; WSL2 disk nearly full | Remove torch from deps — VAD only needs onnxruntime |
| Silero v5 ValueError on second call | v5 outputs `stateN` with wrong shape for onnxruntime 1.26.0 | Download v4 from tag `v4.0`: `github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx` |
| VAD 0% speech on 8kHz files | Silero requires 16kHz; 8kHz samples were passed directly | Add `np.interp` resampling in `test_on_file()` before chunking |

---

## WEEK 2 - Module Skeletons & Unit Tests

**Goal:** Write `audio_io.py`, full unit test suite for all modules, wire the
Day 3 hardcoded pipeline, then add live VAD-driven capture once Pi arrives.

**Status: 🔄 IN PROGRESS**
**Pi 5 status: ⏳ Not yet arrived - Day 4 and Day 5 Pi tasks deferred.**

---

### Day 1 - audio_io.py

**Theme:** Audio I/O module + config system + unit tests. WSL2 only, no live mic.

#### Done
- `configs/dev_config.yaml` created — config-driven device selection, all audio defaults, paths
- `src/audio_io.py` written (production-quality: type hints, docstrings, error handling):
  - `list_devices()` — enumerates all PortAudio devices; returns `[]` gracefully if no devices (common on bare WSL2)
  - `record(duration_sec, sample_rate, device, channels)` — all args config-driven; returns int16 numpy array, 1-D for mono
  - `play(audio_array, sample_rate, device, blocking=True)` — accepts int16/float32/float64; downcasts float64 silently; validates dtype and sample rate
  - `resample_8_to_16k(audio_array)` — `np.interp` linear interpolation; output length `2n-1`; preserves int16 dtype for raw PCM; float input → float32
  - `save_wav(audio_array, path, sample_rate)` — sets `nchannels/sampwidth/framerate` before `writeframes` (lesson from Day 4 Piper bug); downmixes stereo to mono; clips floats before int16 cast
  - `load_wav(path)` → `(np.ndarray, int)` — handles 8-bit unsigned (widens to int16) and 16-bit PCM; mono and multi-channel
- `tests/test_audio_io.py` written — 34 tests, no live mic required:
  - `save_wav` / `load_wav` round-trip: int16 exact equality, float32 within 1 LSB, stereo downmix, float clipping, default sample rate from config
  - `save_wav` error paths: non-ndarray, empty, invalid sample rate, missing parent dir, unsupported dtype
  - `load_wav` error paths: missing file, 8-bit unsigned widening
  - `resample_8_to_16k`: output length `2n-1`, float32→float32, int16→int16, endpoint preservation, all error paths, wav round-trip
  - `play()`: int16 passthrough, float64 downcast, config defaults, non-blocking mode, all error paths — via monkeypatched sounddevice (no hardware)
  - `list_devices()`: graceful `[]` on PortAudio failure, correct dict list when devices present
  - Config loading: defaults applied for missing keys, `VOICE_ASSISTANT_CONFIG` env var, explicit path override

#### Test Results (WSL2, Python 3.13.5)
```
34 passed in 0.41s
```

#### Issues
- None.

#### Carries to Day 2
- `tests/test_asr.py`
- `tests/test_tts.py`
- `tests/test_vad.py`

---

### Day 2 - Unit Tests for ASR, TTS, VAD

**Theme:** Write unit tests for all three core modules. WSL2 only, no live mic, no model downloads required for the default suite.

#### Done
- `tests/conftest.py` created - shared pytest infrastructure:
  - `--run-integration` CLI flag registers the `integration` marker
  - Auto-skips integration tests unless flag is passed
  - Adds project root to `sys.path` so all test files can `from src import ...` cleanly
  - Shared fixtures: `sine_chunk_16k`, `silence_chunk_16k`, `short_wav`, `stereo_wav_8k`
- `tests/test_asr.py` written — 10 unit tests + 3 integration tests:
  - Module constants, `load_model()` caching (`device=cpu`, `compute_type=int8`)
  - `transcribe()` contract: keys, segment concatenation, beam_size/language params, latency, error propagation, empty transcript
- `tests/test_tts.py` written - 15 unit tests + 2 integration tests:
  - Module constants, `load_voice()` caching with correct model + config paths
  - `synthesize()`: mono/16-bit wav, framerate from voice config, RTF math, output path, error propagation
  - Critical regression test: wave params (`nchannels=1`, `sampwidth=2`, `framerate`) verified to be set **before** `synthesize_wav()` is called — guards against the Day 4 Piper bug
- `tests/test_vad.py` written — 24 unit tests + 4 integration tests:
  - Module constants including v4-only path check
  - `load_model()`: `_h`/`_c` initialised as `(2,1,64)` zeros, session cached
  - `reset_state()`: zeros both tensors unconditionally, does **not** trigger model load (by design)
  - `get_speech_prob()`: shape, int16→float32 cast, `_h`/`_c` state update from session output
  - `is_speech()`: threshold logic including `>=` edge case at exactly 0.5
  - `test_on_file()`: 8kHz→16kHz resampling, stereo downmix, state reset per file, chunk count consistency
  - Critical v5 regression test (integration): runs 20 random chunks and asserts max prob > 0.001 - catches accidental v5 load

#### Design decision
Two-layer test architecture across all modules:
- **Unit tests** (default, ~0.4s): all external models mocked - fast, CI-friendly, no downloads
- **Integration tests** (opt-in via `--run-integration`): load real models end-to-end

#### Issues hit and fixed

| Issue | Root cause | Fix |
|---|---|---|
| `patch("faster_whisper.WhisperModel")` not intercepting | `asr.py` binds `WhisperModel` at module top via `from faster_whisper import WhisperModel` — patching the library doesn't affect the already-bound name in `src.asr` | Added `_patch_whisper_model()` helper that introspects `src.asr` and patches the right binding |
| Same for `patch("piper.PiperVoice")` | `tts.py` binds `PiperVoice` via `from piper.voice import PiperVoice` at module top | Added `_patch_piper_voice()` helper |
| Same for `patch("onnxruntime.InferenceSession")` | `vad.py` uses `import onnxruntime as ort` at module top | Added `_patch_inference_session()` helper — handles all 3 import styles + lazy-import fallback |
| `test_uses_cpu_provider` asserting `providers=["CPUExecutionProvider"]` | `vad.py` calls `ort.InferenceSession(MODEL_PATH)` with no `providers` kwarg — relies on onnxruntime default | Dropped `providers` assertion; test now only verifies model path |
| `test_reset_state_loads_model_if_unloaded` failing | `reset_state()` intentionally only zeros `_h`/`_c` — it never calls `load_model()` | Replaced with `test_zeros_h_and_c_when_session_not_loaded` which tests what the function actually does |
| `speech_ratio` precision mismatch | `vad.py` stores `round(speech_chunks/total_chunks, 3)` - 3dp; test used full float precision | Widened tolerance to `abs=1e-3` |

#### Test Results (WSL2, Python 3.13.5)
- 95 passed, 0 skipped in 13.68s   (--run-integration)
- 86 passed, 9 skipped in ~0.4s    (unit only)

#### Carries to Day 3
- `src/pipeline.py` — hardcoded 5s record → ASR → LLM → TTS → play

---

### Day 3 - pipeline.py (Hardcoded Chain)

**Theme:** Wire all modules into a single end-to-end chain. No VAD, no streaming.

#### Done
- `src/pipeline.py` written with five stages:
  - `_stage_record()` — live mic or `--input` pre-recorded wav (bypasses recording on WSL2)
  - `_stage_asr()` - calls `asr.transcribe()`, applies fallback for empty transcript
  - `_stage_llm()` - calls `llm_client.generate()`, applies fallback for empty reply
  - `_stage_tts()` - calls `tts.synthesize()`, logs RTF
  - `_stage_play()` - `load_wav` + `play` via `audio_io`; skipped by `--no-play`
- CLI flags: `--duration`, `--input`, `--output`, `--no-play`, `--model`
- `run()` returns structured dict with per-stage latencies - reusable by Day 5 VAD wrapper
- `tests/test_pipeline.py` written - 20 unit tests + 1 integration test:
  - Wiring, playback flag, model override, fallbacks, return contract, errors, CLI

#### Live Run Results (WSL2, sample1.wav - 51s file)
- rec=0.000s  asr=4.674s  llm=26.327s  tts=10.987s  play=0.000s
- Total: 41.988s
- TTS RTF: 0.073  |  TTS audio duration: 123.09s

*WSL2 numbers not representative of Pi. LLM latency inflated because sample1.wav
is 51s of dense speech - real utterances will be 3-5s. No system prompt yet.*

#### Test Results (WSL2, Python 3.13.5)
- 116 passed in 42.68s   (--run-integration)
- 106 passed in 0.49s    (unit only)

#### Notes
- LLM receives raw transcript with no system prompt → verbose 400-word replies on long input. System prompt ("be a brief voice assistant") is a Week 3 task.
- `--input recordings/sample1.wav --no-play` is the standard WSL2 smoke-test command until the Pi arrives.

#### Carries to Day 4
- Pi-dependent: ALSA setup, `pi_config.yaml`, USB audio adapter, live mic test
- Blocked until Pi 5 arrives

---

## WEEK 3 - Streaming & Multi-Turn Memory

**Goal:** Sub-3s perceived latency end-to-end via LLM streaming → TTS streaming pipeline,
plus multi-turn conversation memory with SQLite session persistence.

**Status: 🔄 IN PROGRESS**
**Pi 5 status: ⏳ Not yet arrived - Day 5 fallback task scheduled.**

---

### Day 1 - ConversationManager

**Theme:** Multi-turn conversation memory with rolling history, system prompt, SQLite persistence.

#### Done
- `configs/dev_config.yaml` extended with new `conversation:` section:
  - `system_prompt` - terse phone-call persona (one-to-two sentence replies, no markdown)
  - `max_history_turns: 6` - max (user, assistant) pairs kept in memory; fits both Llama 3.2 1B (128K ctx) and TinyLlama (2K ctx)
  - `db_path: recordings/conversations.db`
- `src/conversation.py` written — `ConversationManager` class (production-quality: type hints, docstrings, config-driven):
  - `__init__(config_path, session_id, system_prompt, max_history_turns, db_path)` - new session (UUID4) if `session_id=None`, else resume from SQLite
  - `add_user_turn(text)` / `add_assistant_turn(text)` - append to in-memory history, persist to SQLite immediately, evict oldest pair if over window
  - `build_messages()` → `list[{role, content}]` - system prompt first (omitted if empty), rolling window only, `timestamp` stripped (Ollama rejects extra keys)
  - `get_history()` — shallow copy of rolling window; caller mutation cannot affect internal state
  - `get_full_history()` — every turn for this session from SQLite (no window limit); used for post-call WER evaluation
  - `end_session()` — idempotent; stamps `ended_at` on session row, closes DB connection
  - Context manager (`__enter__` / `__exit__`) - `end_session()` called even on exception
  - Config resolution mirrors `audio_io.py`: explicit arg → `VOICE_ASSISTANT_CONFIG` env → `configs/dev_config.yaml`; result cached, reloads only on path change
- SQLite schema:
  - `sessions(id TEXT PK, started_at REAL, ended_at REAL, system_prompt TEXT)`
  - `turns(id INTEGER PK AUTOINCREMENT, session_id TEXT FK, role TEXT, content TEXT, timestamp REAL)`
  - `idx_turns_session` index on `(session_id, id)` for fast per-session lookups
  - WAL journal mode; `PRAGMA foreign_keys=ON`; schema auto-created on first use
- `tests/test_conversation.py` written - 54 unit tests across 8 test classes:
  - `TestConfigLoading` (8): resolution priority, missing file, invalid YAML, non-mapping root, cache hit/miss
  - `TestInit` (13): UUID shape, config values loaded, all 5 overrides, db parent dir creation, validation errors (zero/negative/non-int window, non-string prompt), missing conversation section fallback
  - `TestSchema` (3): tables created, session row inserted with correct fields, index present
  - `TestAddTurn` (7): append to history, persist to DB, empty/whitespace/non-string rejected, monotonic timestamps
  - `TestRollingWindow` (5): under/at/over/far-over capacity, SQLite unaffected by eviction
  - `TestBuildMessages` (6): system prompt first, order preserved, only `{role, content}` keys, empty history, empty prompt omitted, prompt survives eviction
  - `TestResume` (4): loads prior turns, stored prompt wins over override, window respected on resume, unknown session_id raises
  - `TestEndSession` (4): `ended_at` stamped, idempotent, context manager calls end, context manager ends on exception
  - `TestPersistence` (2): turns survive close+reopen, multiple sessions isolated
  - `TestGetHistory` (2): returns copy not reference, `get_full_history` chronological

#### Design Decisions
- **Evict in pairs** - never leaves a dangling user message with no assistant reply; coherence preserved for LLM context
- **SQLite retains everything** - eviction only affects what reaches the LLM; `get_full_history()` supports Week 5 WER evaluation
- **Resume uses stored prompt** - mid-conversation config changes cannot corrupt in-progress sessions; explicit `system_prompt=` override ignored on resume
- **WAL mode** - keeps reads non-blocking while call is actively writing turns

#### Test Results (WSL2, Python 3.13.5)
- tests/test_conversation.py: 54 passed in 0.30s
- Full suite (pytest tests/):  160 passed, 10 skipped in 1.33s
- Full suite (--run-integration): 170 passed in 22.08s

No regressions across existing tests.

#### Issues
- None.

#### Carries to Day 2
- `src/llm_client.py` - extend with `stream_generate()` that buffers tokens into sentences on `.?!`, measures time-to-first-sentence
- `tests/test_llm_streaming.py`

---

### Day 2 - LLM Streaming (Sentence-Buffered)

**Theme:** Extend `llm_client.py` with sentence-buffered streaming so TTS can begin synthesising the first sentence while the LLM is still generating the rest.

#### Done
- `src/llm_client.py` extended — four additions (existing `generate()`, `stream_generate()`, `measure_latency()` unchanged):
  - `_SENTENCE_TERMINATORS = ".?!"` — module-level constant
  - `_split_sentences(buffer: str) -> tuple[list[str], str]` — pure helper. Scans buffer for terminator runs; consecutive terminators (`?!`, `...`) count as one break; strips whitespace from each sentence; drops empty fragments; returns `(sentences, remainder)`.
  - `stream_sentences(prompt, model=None) -> Iterator[str]` — builds on `stream_generate()`. Buffers raw tokens, yields a complete sentence each time a terminator is hit, flushes trailing buffer at stream end (no-terminal-punctuation edge case). `model=None` calls `stream_generate(prompt)` with no model kwarg (PRIMARY_MODEL default); explicit model forwarded as kwarg. Generator return value (`StopIteration.value`) carries stats dict: `{first_token_latency_s, time_to_first_sentence_s, total_latency_s, num_sentences}`. `time_to_first_sentence_s` = first-token → first-sentence (maps to "First sentence assembled" line in the Pi latency budget, 800ms target). Empty chunks from Ollama handled safely.
  - `measure_first_sentence_latency(prompt, model=None) -> dict` — benchmark wrapper mirroring `measure_latency()`. Drains the generator via manual `next()` loop to capture `StopIteration.value`. Returns `{model, num_sentences, first_token_latency_s, time_to_first_sentence_s, total_latency_s, sentences}`.
- `tests/test_llm_streaming.py` written — 32 tests (30 unit + 2 integration) across 6 classes:
  - `TestSplitSentences` (7): single/multi/partial sentence, consecutive terminators, no-terminator, stripping, `?` and `!`
  - `TestStreamSentencesSplitting` (6): single token, multi-token span, multi-sentence in one token, terminator in own token, stripping
  - `TestStreamSentencesEdgeCases` (5): no-punctuation flush, trailing partial, empty stream, whitespace-only, empty token chunks
  - `TestStreamSentencesTiming` (5): stats keys present, num_sentences count, non-negative floats, assembly-gap bracketing (sleep-based), total ≥ first-sentence
  - `TestStreamSentencesModelForwarding` (2): default omits model kwarg, explicit model forwarded
  - `TestMeasureFirstSentenceLatency` (5): keys, sentence collection, default/override model field, empty-stream safety
  - `TestRealStream` (2 integration): live Ollama stream + `generate()` regression guard

#### Design Decisions
- **Built on `stream_generate()`** — reuses Ollama request logic; model-forwarding testable via `patch.object(llm_client, 'stream_generate', ...)` without touching Ollama.
- **`time_to_first_sentence_s` = first-token → first-sentence** (not request → first-sentence). Matches the "First sentence assembled" latency budget line (separate from "LLM first token"). Both lines exposed in stats.
- **Timing via generator return value** — `stream_sentences` stays a clean `Iterator[str]` for Day 4 pipeline; benchmark wrapper captures stats via `StopIteration.value`.
- **Punctuation-only splitting per spec** — decimals (`3.5`) and abbreviations (`Mr.`) may split early. Documented caveat; acceptable for short conversational replies. Refinable later without changing the public interface.
- **`model=None` omits model kwarg** (not passes PRIMARY_MODEL) — consistent with `pipeline._stage_llm` convention established in Week 2.

#### Test Results (WSL2, Python 3.13.5)
- tests/test_llm_streaming.py: 30 passed, 2 skipped in 0.64s
- Full suite (pytest tests/):  190 passed, 12 skipped in 1.81s
- Full suite (--run-integration): 202 passed in 42.45s

Zero regressions.

#### No Issues

#### Commit
`Week 3 Day 2: stream_sentences() with sentence buffering and first-sentence latency measurement`
Files changed: `src/llm_client.py` (modified), `tests/test_llm_streaming.py` (new)

---

### Day 3 - TTS Streaming + Noise Reduction

**Theme:** Wire sentence-level TTS streaming so PCM playback of sentence N overlaps with synthesis of sentence N+1. Add spectral noise suppression before VAD/ASR.

#### Done
- `src/tts.py` extended — `synthesize()` unchanged; three additions:
  - Imports added: `import io`, `from collections.abc import Iterable, Iterator`, `import numpy as np`
  - `output_sample_rate() -> int` — returns `int(load_voice().config.sample_rate)`. Exposes the voice's native rate (22050 Hz for amy-medium) up front so Day 4's pipeline can pass it to `play()` without coupling to Piper internals. Needed because TTS rate ≠ pipeline's 16000 Hz, and the rate must be known before the first PCM chunk arrives (stats via `StopIteration.value` too late for the first chunk).
  - `synthesize_stream(sentences, sample_rate=None) -> Iterator[np.ndarray]` — accepts any iterable of sentence strings (list, generator, or `stream_sentences()` output directly). For each non-empty sentence: synthesizes into an in-memory `io.BytesIO` WAV via the **identical `synthesize_wav` path** as batch `synthesize()` (same critical param order: `setnchannels`/`setsampwidth`/`setframerate` before `synthesize_wav`), decodes PCM back to int16 numpy, yields the array. Skips empty/whitespace sentences and zero-frame PCM. `sample_rate=None` defaults to `voice.config.sample_rate`. Raises `ValueError` for non-positive rate. Generator return value (`StopIteration.value`): `{time_to_first_audio_s, total_latency_s, num_sentences, sample_rate}`. `time_to_first_audio_s` = stream-start → first chunk ready (the "TTS first audio" latency budget line).

- `src/audio_io.py` extended — one new function:
  - `reduce_noise(audio_array, sample_rate=None, config_path=None) -> np.ndarray` — spectral noise suppression via noisereduce, applied before VAD/ASR. Validates: must be 1-D numpy, non-empty, positive sample rate. Config-toggleable: `noise_reduction.enabled: false` returns input unchanged (passthrough, same object). Dtype-preserving: int16 → normalise to float32 → denoise → rescale + clip → int16; float32 → denoise → float32. Passes `stationary` and `prop_decrease` from config to `noisereduce.reduce_noise`. Explicit `sample_rate` arg overrides config value. **Lazy import** (`import noisereduce as nr` inside the function) — keeps `audio_io` import light and non-breaking before the package is installed.

- `configs/dev_config.yaml` extended — new `noise_reduction:` section:
```yaml
  noise_reduction:
    enabled: true
    stationary: false    # adapts to changing noise; better for varied call audio
    prop_decrease: 1.0   # 0.0 = no reduction, 1.0 = maximum
```

- `pip install noisereduce` — installed Week 3 Day 3.

- `tests/test_tts_streaming.py` written — 21 tests (19 unit + 2 integration) across 5 classes:
  - `TestOutputSampleRate` (2): returns voice config rate, returns int
  - `TestSynthesizeStreamCore` (5): one chunk per sentence, int16 ndarray, non-empty, PCM roundtrip, **critical param order guard** (asserts `nchannels`/`sampwidth`/`framerate` set before `synthesize_wav` call)
  - `TestSynthesizeStreamInput` (4): skips empty/whitespace, accepts generator, empty iterable, strips sentence before synthesis
  - `TestSynthesizeStreamStats` (7): keys present, count, first-audio float, total ≥ first-audio, default sr, override sr, invalid sr rejected
  - `TestSynthesizeStreamChaining` (1): real `stream_sentences()` → `synthesize_stream()` with both LLM and voice mocked — confirms the two streaming layers chain cleanly end-to-end
  - `TestRealVoiceStream` (2 integration): real Piper voice + `synthesize()` regression guard

- `tests/test_audio_io.py` extended — three new classes appended (13 tests: 12 unit + 1 integration):
  - `TestReduceNoiseValidation` (4): non-ndarray, 2D array, empty, non-positive sample rate
  - `TestReduceNoiseBehaviour` (8): disabled passthrough (same object), int16→int16, float32→float32, sr forwarded, sr override, stationary+prop_decrease forwarded, missing section defaults to enabled, output length preserved
  - `TestReduceNoiseReal` (1 integration): real noisereduce on noisy 440Hz tone; asserts noise power drops post-reduction
  - Also added `from unittest.mock import patch` and `import yaml` to top-level imports; added `_write_config()` config helper fixture; removed duplicate `from __future__ import annotations` that caused SyntaxError at line 442

#### Design Decisions
- **`synthesize_stream` reuses `synthesize_wav` verbatim** via in-memory BytesIO round-trip. Guarantees byte-identical audio to batch path, preserves the "critical param order" contract, and adds zero new Piper API surface. The per-sentence WAV encode+decode is negligible (µs) vs synthesis time.
- **`output_sample_rate()` helper** instead of yielding `(pcm, rate)` tuples. Rate is constant per stream and needed before the first chunk. Day 4 pipeline calls it once.
- **Lazy noisereduce import** inside `reduce_noise()`. `audio_io` is imported by the whole pipeline; hard-depending at module level would slow startup and break import if noisereduce not yet installed.
- **Noise reduction is dtype-preserving** — mirrors `resample_8_to_16k()` convention. int16 in → int16 out; float32 in → float32 out. Preserves pipeline's dtype invariants.
- **Config toggle `enabled: false` returns input unchanged** — lets Pi disable noise reduction if latency budget is too tight (per the RNNoise fallback note in master doc Section 4.1).
- **`stationary: false`** default — non-stationary mode adapts to changing background noise, better for varied GSM call environments than stationary (constant-noise assumption).

#### Test Results (WSL2, Python 3.13.5) — expected
- tests/test_tts_streaming.py:  19 passed, 2 skipped
- tests/test_audio_io.py:       46 passed, 1 skipped
- Full suite (pytest tests/):   221 passed, 15 skipped
- Full suite (--run-integration): 236 passed

#### Carries to Day 4
- Wire `ConversationManager` + `stream_sentences()` + `synthesize_stream()` into `src/pipeline.py` in one loop
- `asyncio.Queue(maxsize=3)` between TTS synthesis and playback (buffer overflow prevention subtask)
- `pipeline.tts_buffer_max_chunks: 3` config key in `dev_config.yaml`
- `tts.output_sample_rate()` used for playback rate
- Full reply text accumulated from sentence stream → `conversation.add_assistant_turn()`
- `conversation.end_session()` called on hangup
- End-to-end latency measured on `sample1.wav`
- `tests/test_pipeline.py` updated

---

### Day 4 - Streaming Pipeline Wiring

**Theme:** Wire ConversationManager + streaming LLM + streaming TTS into one
loop with bounded back-pressure. First audio now plays while the LLM is still
generating later sentences.

#### Done
- `src/llm_client.py` refactored and extended:
  - Extracted `_sentences_from_tokens(token_iter)` — the shared buffering core
    (time.time/round, yields sentences, returns stats via StopIteration.value).
    `stream_sentences` now delegates via `return (yield from ...)` — externally
    identical, Day 2's 30 tests still pass unchanged.
  - `stream_generate_messages(messages, model=PRIMARY_MODEL)` — passes a full
    chat messages list straight to `ollama.chat(messages=...)`, enabling system
    prompt + conversation history. Used by the pipeline instead of the flat
    prompt path.
  - `stream_sentences_from_messages(messages, model=None)` — same buffering as
    `stream_sentences` but driven by a messages list from
    `ConversationManager.build_messages()`. This is the pipeline's LLM entry
    point for multi-turn.
  - 5 new tests in `tests/test_llm_streaming.py` (`TestStreamSentencesFromMessages`):
    sentence yield, default/override model forwarding, stats via StopIteration,
    no-punctuation flush.

- `src/pipeline.py` fully rewritten (streaming chain):
  - `_sentences_from_tokens` core shared with llm_client (no duplication).
  - `_recording_iter(sentence_iter, collected)` — tees the sentence stream: as
    TTS pulls each sentence, it's appended to `collected`, giving the full reply
    text for `add_assistant_turn()` without a second LLM pass.
  - `_safe_next(gen, stats_holder)` — calls next(gen), captures StopIteration.value
    as stats, returns sentinel on end — safe to call from run_in_executor.
  - `_produce(queue, sentence_iter, sample_rate, tts_stats)` async coroutine —
    drives `tts.synthesize_stream()` off-thread via `run_in_executor`; puts each
    PCM chunk onto the bounded queue. Sentinel placed in `finally` so the
    consumer always unblocks even if synthesis or LLM raises mid-stream
    (deadlock prevention — discovered and fixed during sandbox testing).
  - `_consume(queue, sample_rate, skip_play, t0, result)` async coroutine —
    drains the queue, plays each chunk off-thread via `run_in_executor`, records
    `first_audio_s` when the first chunk is dequeued. Runs concurrently with
    `_produce` via `asyncio.create_task`.
  - `_stream_and_play_async(sentence_iter, sample_rate, max_chunks, skip_play, t0)` —
    creates `asyncio.Queue(maxsize=max_chunks)`, runs producer as a task,
    awaits consumer, then awaits producer (re-raises any producer exception).
  - `_stream_and_play(...)` — synchronous wrapper calling `asyncio.run(...)`.
    Returns `(chunks, first_audio_s, tts_stats)`.
  - `_max_chunks(config_path)` — reads `pipeline.tts_buffer_max_chunks` from
    config (default 3); reuses `audio_io._get_config`. If value ≤ 0, returns 3.
  - `run(...)` — new signature adds `conversation`, `session_id`, `config_path`.
    Conversation lifecycle: if `conversation=None`, creates and ends its own
    session (single-turn); if passed in, caller owns end_session (multi-turn
    loop, called on hangup). `end_session` is in a `finally` so it fires on
    errors. Output WAV is the concatenated PCM from all chunks, written at
    `tts.output_sample_rate()`.
  - New result dict keys: `session_id`, `num_sentences`, `num_audio_chunks`.
  - New latency dict: `record_s`, `asr_s`, `first_audio_s` (stream-start →
    first chunk played, the Week 3 headline metric), `stream_s`, `perceived_s`
    (= asr_s + first_audio_s), `total_s` (= record + asr + stream). Old
    `llm_s`/`tts_s`/`play_s` keys removed.

- `configs/dev_config.yaml` extended — new `pipeline:` section:
```yaml
  pipeline:
    tts_buffer_max_chunks: 3
```

- `tests/test_pipeline.py` fully rewritten — 36 tests (35 unit + 1 integration)
  across 9 classes:
  - `TestWiring` (8): input skips record, recording triggered, ASR path, user
    turn added, LLM receives built messages, TTS stream invoked at voice rate,
    assistant turn added with full joined reply, reply_text in result.
  - `TestPlayback` (5): no-play skips, one play call per sentence, correct
    sample rate, all chunks played under buffer=1 (back-pressure exercise),
    reply WAV written.
  - `TestModelOverride` (2): default omits model kwarg, explicit forwarded.
  - `TestFallbacks` (3): empty transcript, whitespace transcript, empty reply.
  - `TestConversationLifecycle` (4): own session created+ended, caller session
    not touched, session_id in result, session_id forwarded to manager.
  - `TestReturnValue` (5): keys, total = record+asr+stream, perceived =
    asr+first_audio, skipped-record zero, first_audio non-negative.
  - `TestMaxChunksConfig` (3): reads value, defaults to 3, non-positive → 3.
  - `TestErrors` (3): missing input WAV, ASR error propagates, own session
    ended even when LLM raises mid-stream (deadlock-guard).
  - `TestCli` (2): missing input returns 2, model flag forwarded.
  - `TestRealChain` (1 integration): real chain on sample1.wav, checks reply
    WAV exists, latencies positive, prints perceived latency for Week 3 goal.

#### Design Decisions
- **Messages-based LLM streaming** (`stream_sentences_from_messages`) instead of
  flattened prompt — preserves system prompt and multi-turn history as proper
  role-tagged messages to `ollama.chat`. This is why `stream_sentences` was
  refactored (shared buffering core) rather than duplicated.
- **Sentinel in `finally`** on the producer — first sandbox run deadlocked when
  a mid-stream LLM error killed the producer before it could signal end-of-stream.
  `finally: await queue.put(_SENTINEL)` always unblocks the consumer; the
  exception still surfaces via `await producer`. Pinned by
  `test_own_session_ended_even_on_error`.
- **`_recording_iter` tee** — sentence stream consumed by TTS but also recorded
  for reply text assembly. Avoids a second LLM call for `add_assistant_turn`.
- **Own vs borrowed conversation** — `run()` manages its own session for
  single-turn latency runs; multi-turn main loop (Week 4) passes a shared
  manager and calls `end_session()` on hangup.
- **`asyncio.Queue(maxsize=3)`** per v3.1 Subtask 1 spec. Back-pressure: producer
  pauses when 3 unplayed chunks queued, capping memory on the Pi.

#### Issues
- **Deadlock on producer exception** — discovered in first sandbox run. Fixed via
  sentinel-in-finally pattern on `_produce`. See above.

#### Test Results (WSL2, Python 3.13.5)
- tests/test_pipeline.py:      35 passed, 1 skipped in 1.23s
- tests/test_llm_streaming.py: 35 passed, 2 skipped in 0.34s
- Full suite (pytest tests/):  241 passed, 15 skipped in 2.92s
- Full suite (--run-integration): 256 passed in 23.49s

Zero regressions.

#### Carries to Day 5
- Pi not arrived → `src/telephony/gsm_adapter.py` skeleton
- SIM7600EI AT-command logic (`connect`, `disconnect`, `answer_call`, `hangup`,
  `send_at`, `wait_for_ring`) with mocked pyserial unit tests
- `tests/test_gsm_adapter.py`

---

### Day 5 - GSM Adapter Skeleton

**Theme:** SIM7600EI AT-command skeleton, fully unit-tested with mocked
pyserial. Call audio path out of scope — signalling only. Real-hardware
validation deferred to Week 4 Day 1 when the Pi arrives.

#### Done
- `src/telephony/gsm_adapter.py` written from scratch — `GSMAdapter` class:
  - Config-driven (`telephony:` section in dev_config.yaml; falls back to
    defaults silently so the skeleton works without config).
  - Self-contained config loader — does not import audio_io, keeping telephony
    decoupled from the audio stack.
  - `connect()` — opens serial port, sends `AT` (link check), `ATE0` (echo
    off), `AT+CLIP=1` (caller ID). Raises `GSMConnectionError` if port
    unavailable or module silent.
  - `disconnect()` — closes port. Idempotent.
  - `send_at(command, timeout)` — core I/O: writes `command\r\n`, reads lines
    until final result code, returns intermediate lines on `OK`, raises on
    `ERROR`/`BUSY`/`+CME ERROR`/etc., raises `GSMTimeout` if no final code
    arrives. Skips command echo. Resets input buffer before each write.
  - `check_sim()` / `check_signal()` / `check_registration()` /
    `is_call_active()` — status queries parsing `+CPIN`, `+CSQ`, `+CREG`,
    `+CLCC`.
  - `answer_call()` — `ATA`.
  - `hangup()` — `AT+CHUP` (clears all calls; more reliable than ATH on
    SIM7600).
  - `dial(number)` — `ATD<number>;` (semicolon = voice call, not data).
  - `wait_for_ring(timeout)` — blocks reading lines for unsolicited `RING`,
    capturing caller number from preceding `+CLIP:`. Returns
    `{"event": "RING", "caller": <num|None>}` or `None` on timeout.
  - Context manager (`__enter__`/`__exit__`) — connect on enter, disconnect
    on exit even on exception.
  - Exception hierarchy: `GSMError` (base) → `GSMConnectionError`,
    `GSMTimeout`, `GSMCommandError`.
  - Module-level parsers: `_parse_csq`, `_parse_creg`, `_parse_clip`.

- `tests/test_gsm_adapter.py` written — 46 unit tests, 0 integration, 8
  classes. All tests use `FakeSerial` (a scripted deque of response lines;
  `readline()` returns `b""` when exhausted, mimicking pyserial's timeout):
  - `TestConstruction` (4): defaults, explicit overrides, config section read,
    not-connected before connect.
  - `TestConnect` (6): port/baudrate/timeout passed to serial.Serial, init
    sequence (AT/ATE0/CLIP=1), port unavailable, silent module, disconnect
    closes, disconnect idempotent.
  - `TestSendAt` (10): terminator written, intermediate lines returned, OK-only
    returns [], ERROR raises, +CME ERROR raises, BUSY raises, timeout,
    echo-skip, input-reset, not-connected raises.
  - `TestStatusQueries` (9): SIM ready/not-ready, signal RSSI, registration
    home/roaming/unregistered, call active/inactive.
  - `TestCallControl` (5): ATA sent, AT+CHUP sent, ATD with semicolon, strips
    whitespace from number, empty number rejected.
  - `TestWaitForRing` (4): RING detected, caller captured from +CLIP, timeout
    returns None, not-connected raises.
  - `TestContextManager` (2): connects on enter / disconnects on exit,
    disconnects on exception.
  - `TestParsers` (6): +CSQ, +CSQ bad, +CREG, +CREG bad, +CLIP, +CLIP empty.

- `pip install pyserial` — added to project dependencies.
