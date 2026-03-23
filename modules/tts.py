"""Piper-only TTS module with ordered streaming playback and emotion control."""

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig

import config


# Emotion presets: maps emotion name -> synthesis parameters
# length_scale: higher = slower (good for sad/calm)
# noise_scale: higher = more variation (good for happy/excited)
# noise_w_scale: phoneme variation
EMOTION_PRESETS = {
    "neutral": SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8),
    "happy": SynthesisConfig(length_scale=0.95, noise_scale=0.8, noise_w_scale=0.9),
    "sad": SynthesisConfig(length_scale=1.2, noise_scale=0.5, noise_w_scale=0.6),
    "calm": SynthesisConfig(length_scale=1.15, noise_scale=0.5, noise_w_scale=0.7),
    "excited": SynthesisConfig(length_scale=0.9, noise_scale=0.9, noise_w_scale=1.0),
    "serious": SynthesisConfig(length_scale=1.0, noise_scale=0.4, noise_w_scale=0.5),
    "whisper": SynthesisConfig(length_scale=1.3, noise_scale=0.3, noise_w_scale=0.4),
    "curious": SynthesisConfig(length_scale=1.05, noise_scale=0.75, noise_w_scale=0.85),
}


@dataclass(order=True)
class OrderedChunk:
    """Audio chunk with sequence for ordered playback."""

    sequence: int
    audio_data: Optional[np.ndarray] = field(default=None, compare=False)
    sample_rate: int = field(default=24000, compare=False)
    text: str = field(default="", compare=False)
    error: Optional[str] = field(default=None, compare=False)
    synth_ms: float = field(default=0.0, compare=False)
    backend: str = field(default="piper", compare=False)
    emotion: str = field(default="neutral", compare=False)


END_SENTINEL = OrderedChunk(sequence=999999)


class TTS:
    """Piper TTS engine with queued synthesis and ordered playback."""

    def __init__(self):
        self.synth_timeout_s = float(getattr(config, "TTS_SYNTH_TIMEOUT_S", 20.0))
        self._latency_logs = bool(getattr(config, "LATENCY_LOGS", True))

        self.piper_warmup_text = str(getattr(config, "PIPER_WARMUP_TEXT", "hi")).strip() or "hi"

        self.voice, self.model_path, self.config_path = self._load_voice()

        self._audio_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._text_queue: queue.Queue = queue.Queue()

        self._next_sequence = 0
        self._lock = threading.Lock()
        self._last_queued_text = ""

        self._stop = threading.Event()
        self._synth_thread = None
        self._player_thread = None

        self._stream_stats = {}
        self._stats_lock = threading.Lock()

        # Allow config to override emotion presets
        self._load_emotion_presets()

        print(f"TTS: backend=piper model={self.model_path}")
        self._warmup()

    def _load_emotion_presets(self):
        """Load custom emotion presets from config if provided."""
        custom = getattr(config, "EMOTION_PRESETS", None)
        if custom and isinstance(custom, dict):
            for emotion, params in custom.items():
                if isinstance(params, dict):
                    EMOTION_PRESETS[emotion] = SynthesisConfig(
                        length_scale=params.get("length_scale", 1.0),
                        noise_scale=params.get("noise_scale", 0.667),
                        noise_w_scale=params.get("noise_w_scale", 0.8),
                    )

    @staticmethod
    def get_available_emotions():
        """Return list of available emotion presets."""
        return list(EMOTION_PRESETS.keys())

    def _existing(self, value):
        return bool(value) and os.path.exists(value)

    def _load_voice(self):
        primary_model = getattr(config, "PIPER_PRIMARY_MODEL_PATH", None)
        primary_config = getattr(config, "PIPER_PRIMARY_CONFIG_PATH", None)
        fallback_model = getattr(config, "PIPER_FALLBACK_MODEL_PATH", None)
        fallback_config = getattr(config, "PIPER_FALLBACK_CONFIG_PATH", None)

        # Backward compatibility with older config names.
        legacy_model = getattr(config, "PIPER_MODEL_PATH", None)
        legacy_config = getattr(config, "PIPER_MODEL_CONFIG_PATH", None)

        candidates = [
            (primary_model, primary_config),
            (legacy_model, legacy_config),
            (fallback_model, fallback_config),
        ]

        chosen_model = None
        chosen_config = None
        for model_path, config_path in candidates:
            if self._existing(model_path):
                chosen_model = model_path
                chosen_config = config_path if self._existing(config_path) else None
                break

        if not chosen_model:
            raise FileNotFoundError(
                "No Piper model found. Add model files under models/ and update PIPER_*_MODEL_PATH in config.py"
            )

        voice = PiperVoice.load(chosen_model, config_path=chosen_config)
        return voice, chosen_model, chosen_config

    def _warmup(self):
        try:
            # Consume the generator to warm up the model
            list(self.voice.synthesize(self.piper_warmup_text))
        except Exception:
            pass

    def _reset_stream_stats(self):
        with self._stats_lock:
            self._stream_stats = {
                "start_time": time.perf_counter(),
                "queued": 0,
                "synth_done": 0,
                "played": 0,
                "first_audio_ready_ms": None,
                "first_play_start_ms": None,
                "first_play_end_ms": None,
                "total_synth_ms": 0.0,
                "by_backend": {},
                "end_time": None,
            }

    def _mark_queued(self):
        with self._stats_lock:
            if self._stream_stats:
                self._stream_stats["queued"] += 1

    def _mark_synth_done(self, synth_ms: float, backend: str):
        with self._stats_lock:
            if not self._stream_stats:
                return
            self._stream_stats["synth_done"] += 1
            self._stream_stats["total_synth_ms"] += float(synth_ms)
            by_backend = self._stream_stats.setdefault("by_backend", {})
            by_backend[backend] = by_backend.get(backend, 0) + 1
            if self._stream_stats["first_audio_ready_ms"] is None:
                self._stream_stats["first_audio_ready_ms"] = (
                    time.perf_counter() - self._stream_stats["start_time"]
                ) * 1000

    def _mark_play_start(self):
        with self._stats_lock:
            if self._stream_stats and self._stream_stats["first_play_start_ms"] is None:
                self._stream_stats["first_play_start_ms"] = (
                    time.perf_counter() - self._stream_stats["start_time"]
                ) * 1000

    def _mark_play_done(self):
        with self._stats_lock:
            if not self._stream_stats:
                return
            self._stream_stats["played"] += 1
            if self._stream_stats["first_play_end_ms"] is None:
                self._stream_stats["first_play_end_ms"] = (
                    time.perf_counter() - self._stream_stats["start_time"]
                ) * 1000

    def get_stream_stats(self):
        with self._stats_lock:
            return dict(self._stream_stats) if self._stream_stats else {}

    def _synth_one(self, seq: int, text: str, emotion: str = "neutral") -> OrderedChunk:
        start = time.perf_counter()
        try:
            # Get emotion-specific synthesis config
            syn_config = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])

            # synthesize yields AudioChunk objects
            audio_chunks = []
            sample_rate = self.voice.config.sample_rate
            for chunk in self.voice.synthesize(text, syn_config=syn_config):
                audio_chunks.append(chunk.audio_float_array)
                sample_rate = chunk.sample_rate

            if audio_chunks:
                audio = np.concatenate(audio_chunks)
            else:
                audio = np.array([], dtype=np.float32)

            result = OrderedChunk(
                sequence=seq,
                audio_data=audio,
                sample_rate=int(sample_rate),
                text=text,
                backend="piper",
                emotion=emotion,
            )
        except Exception as exc:
            result = OrderedChunk(sequence=seq, error=str(exc), text=text, backend="piper", emotion=emotion)

        result.synth_ms = (time.perf_counter() - start) * 1000
        if result.error:
            if self._latency_logs:
                print(f"[TTS Error:{result.backend}] {result.error}")
        else:
            self._mark_synth_done(result.synth_ms, result.backend)
            if self._latency_logs and emotion != "neutral":
                print(f"[TTS] emotion={emotion} length_scale={syn_config.length_scale:.2f} noise={syn_config.noise_scale:.2f}")

        return result

    def _synth_loop(self):
        while not self._stop.is_set():
            try:
                item = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._audio_queue.put(END_SENTINEL)
                break

            seq, text, emotion = item
            self._audio_queue.put(self._synth_one(seq, text, emotion))

    def _player_loop(self):
        expected_seq = 0
        pending = {}
        got_end = False

        while not self._stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                if got_end and not pending:
                    break
                continue

            if chunk is END_SENTINEL:
                got_end = True
                self._audio_queue.task_done()
            else:
                pending[chunk.sequence] = chunk
                self._audio_queue.task_done()

            while expected_seq in pending:
                play_chunk = pending.pop(expected_seq)
                if play_chunk.audio_data is not None:
                    self._mark_play_start()
                    sd.stop()  # Ensure previous audio is stopped
                    sd.play(play_chunk.audio_data, samplerate=play_chunk.sample_rate)
                    sd.wait()
                    self._mark_play_done()
                expected_seq += 1

            if got_end and not pending:
                break

    def start_streaming(self):
        if self._synth_thread is None or not self._synth_thread.is_alive():
            self._stop.clear()
            self._next_sequence = 0
            self._last_queued_text = ""
            self._text_queue = queue.Queue()
            self._audio_queue = queue.PriorityQueue()
            self._reset_stream_stats()

            self._synth_thread = threading.Thread(target=self._synth_loop, daemon=True)
            self._player_thread = threading.Thread(target=self._player_loop, daemon=True)

            self._synth_thread.start()
            self._player_thread.start()

    def queue_sentence(self, text: str, emotion: str = "neutral"):
        if not text or not text.strip():
            return

        normalized = " ".join(text.strip().split())
        if normalized == self._last_queued_text:
            if self._latency_logs:
                print(f"[TTS] skipped duplicate chunk: {normalized[:80]}")
            return

        with self._lock:
            seq = self._next_sequence
            self._next_sequence += 1
            self._last_queued_text = normalized

        self._mark_queued()
        self._text_queue.put((seq, normalized, emotion))

    def finish_streaming(self, timeout: float = 60.0):
        self._text_queue.put(None)

        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=timeout)

        with self._stats_lock:
            if self._stream_stats:
                self._stream_stats["end_time"] = time.perf_counter()

        stats = self.get_stream_stats()
        if self._latency_logs and stats:
            total_ms = 0.0
            if stats.get("start_time") and stats.get("end_time"):
                total_ms = (stats["end_time"] - stats["start_time"]) * 1000

            backend_counts = stats.get("by_backend", {})
            backend_text = ",".join(f"{k}:{v}" for k, v in backend_counts.items()) if backend_counts else "none"

            print(
                "[Latency:TTS] "
                f"queued={stats.get('queued', 0)} "
                f"ready_first={stats.get('first_audio_ready_ms') or 0:.0f}ms "
                f"play_start={stats.get('first_play_start_ms') or 0:.0f}ms "
                f"played={stats.get('played', 0)} "
                f"backend={backend_text} "
                f"total={total_ms:.0f}ms"
            )

        return stats

    def speak(self, text: str, emotion: str = "neutral"):
        if not text or not text.strip():
            return
        self.start_streaming()
        self.queue_sentence(text, emotion=emotion)
        self.finish_streaming()

    def shutdown(self):
        self._stop.set()
        if self._synth_thread and self._synth_thread.is_alive():
            self._text_queue.put(None)
            self._synth_thread.join(timeout=1.0)
        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=1.0)
