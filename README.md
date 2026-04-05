# Real-Time Conversational Voice Agent

A fully offline-capable, low-latency voice agent that enables natural spoken conversations with an LLM — including **barge-in interruption**, **emotion-aware speech**, and **persistent memory**.

---

## Features

- **Streaming pipeline** — STT, LLM, and TTS all run in parallel (no stage waits for the previous)
- **Barge-in interruption** — speak over the agent mid-response and it immediately stops and listens
- **Emotion-controlled speech** — the LLM selects an emotion per response; TTS adjusts prosody accordingly
- **Instant reactions** — 25+ trigger phrases bypass the LLM entirely for sub-10ms responses
- **Persistent memory** — remembers facts, feelings, and preferences across sessions
- **Offline-capable** — STT (Vosk) and TTS (Piper) run fully on-device; only the LLM requires internet

---

## Architecture

```
You speak → Vosk STT (offline)
                ↓
           EventBus (pub/sub)
                ↓
         VoiceAgent Orchestrator
          ├── ReactionEngine   → instant regex-based replies (no LLM)
          ├── InterruptibleLLM → Gemini 2.5 Flash (streaming tokens)
          └── ParallelTTSPipeline → Piper VITS (2-worker, ordered playback)
                                          ↓
                                    Audio output (sounddevice)
```

All components communicate through a **thread-safe EventBus**. Nothing blocks; everything is event-driven.

### Agent State Machine

```
IDLE → LISTENING → THINKING → SPEAKING → IDLE
          ↑                       |
          └──── (interruption) ───┘
```

---

## Project Structure

```
voice-agent/
├── main.py                   # Entry point — run this
├── config.py                 # All settings (models, API keys, tuning params)
├── requirements.txt
├── memory.json               # Auto-created: persisted user memories
├── model/                    # Vosk ASR model directory
├── models/                   # Piper TTS ONNX model files
│
├── modules/                  # Core engine wrappers
│   ├── stt.py                # Vosk STT (blocking + streaming)
│   ├── llm.py                # Gemini LLM (streaming, emotion-tagged responses)
│   └── tts.py                # Piper TTS (parallel synthesis, emotion presets)
│
└── core/                     # Real-time coordination layer
    ├── orchestrator.py       # VoiceAgent — main state machine
    ├── events.py             # EventBus + 12 typed EventTypes
    ├── stt_controller.py     # ContinuousSTT — background ASR thread
    ├── llm_streamer.py       # InterruptibleLLM — cancelable streaming
    ├── tts_pipeline.py       # ParallelTTSPipeline — multi-worker + ordered playback
    ├── reactions.py          # ReactionEngine — instant pattern-matched responses
    └── memory.py             # MemoryStore — keyword-based, disk-persisted
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch`, `torchaudio`, and `torchvision` are large. If you only need the voice agent (no vision features), you can skip those.

### 2. Download the Vosk STT model

```bash
python setup_vosk.py
```

This downloads a small English model (~50MB) into the `model/` directory. For best latency, use the small model. Larger models improve accuracy but increase latency.

> Or download manually from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract to `model/`.

### 3. Download the Piper TTS model

```bash
python setup_piper.py
```

This downloads the `en_US-lessac-medium` voice model into `models/`. An optional Hindi fallback (`hi_IN-rohan-medium`) is also supported.

### 4. Set your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 5. Run

```bash
python main.py
```

---

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Gemini model to use |
| `LLM_MAX_TOKENS` | `512` | Max response length (keep short for voice) |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `STT_BLOCKSIZE` | `800` | Audio chunk size (800 = 50ms at 16kHz) |
| `STT_SILENCE_TIMEOUT` | `0.5` | Seconds of silence to finalize a turn |
| `TTS_STREAM_MIN_CHARS` | `20` | Min chars before sending a chunk to TTS |
| `TTS_STREAM_MAX_CHARS` | `90` | Max chars before forcing a TTS flush |
| `LATENCY_LOGS` | `True` | Print per-turn latency stats |

---

## How It Works

### Speech-to-Text (STT)
Vosk processes audio in **50ms chunks** in a background thread. It emits two event types:
- **`SPEECH_PARTIAL`** — fires in real-time as you speak (used for reactions + interruption detection)
- **`SPEECH_FINAL`** — fires after silence is detected (triggers LLM response)

### LLM Generation
Gemini streams tokens one-by-one. A stop flag enables **clean mid-stream cancellation** — when you interrupt, generation halts immediately without waiting for the full response.

### Text-to-Speech (TTS)
The LLM stream is split into **speakable units** (20–90 chars) at punctuation boundaries. Two worker threads synthesize these units in parallel using Piper; a priority queue ensures **ordered sequential playback**. First audio plays while later chunks are still synthesizing.

### Interruption (Barge-In)
While TTS is playing, the STT controller keeps listening. If **≥3 words** of new speech are detected, an `INTERRUPTION` event fires and the orchestrator:
1. Stops audio playback immediately (`sd.stop()`)
2. Cancels LLM generation (stop flag)
3. Flushes all TTS queues
4. Routes your new speech to a fresh response

### Reaction Engine
Before the LLM is even involved, partial speech is matched against compiled regex patterns. Phrases like `"guess what"`, `"I need help"`, or `"bad news"` trigger instant spoken reactions with appropriate emotion — no LLM round-trip.

### Memory
The agent extracts memorable content (facts, feelings, preferences) from each turn using regex patterns and stores them in `memory.json`. On subsequent turns, relevant memories are retrieved via keyword overlap scoring and injected into the LLM prompt.

---

## Emotion Presets

The LLM returns an emotion tag with each response. TTS adjusts prosody accordingly:

| Emotion | Effect |
|---------|--------|
| `neutral` | Normal pace and expressiveness |
| `happy` | Slightly faster, more variation |
| `excited` | Fast, high variation |
| `calm` | Slower, smoother |
| `sad` | Slower, subdued |
| `serious` | Normal pace, minimal variation |
| `whisper` | Slow, very quiet feel |
| `curious` | Slightly slower, mid variation |

---

## Latency Breakdown

With the small Vosk model and Piper medium:

| Stage | Typical |
|-------|---------|
| STT finalization | ~100–300ms after you stop speaking |
| LLM first token (TTFT) | ~300–600ms |
| First audio playback | ~500–900ms from end of speech |

Enable `LATENCY_LOGS = True` in `config.py` to see per-turn stats printed to console.

---

## Planned Extensions

- [ ] Vision modality — scene-aware responses using webcam (OpenCV + DeepFace already in deps)
- [ ] Fully local LLM — replace Gemini with Ollama/LLaMA 3 for 100% offline operation
- [ ] Wake word detection — activate only on a trigger phrase
- [ ] Semantic memory — replace keyword matching with lightweight sentence embeddings

---

## Tech Stack

| Component | Library |
|-----------|---------|
| ASR | [Vosk](https://alphacephei.com/vosk/) (Kaldi-based, offline) |
| LLM | [Google Gemini](https://aistudio.google.com) via `google-generativeai` |
| TTS | [Piper](https://github.com/rhasspy/piper) (VITS neural, offline) |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Concurrency | Python `threading`, `queue.PriorityQueue` |

---

## License

MIT
