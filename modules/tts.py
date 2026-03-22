"""
Production TTS module with dual backend (Piper local + Edge cloud) and
concurrent synthesis maintaining sentence order.
"""

import asyncio
import io
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf

import config


@dataclass(order=True)
class OrderedChunk:
    """Audio chunk with sequence for ordered playback."""

    sequence: int
    audio_data: Optional[np.ndarray] = field(default=None, compare=False)
    sample_rate: int = field(default=24000, compare=False)
    text: str = field(default="", compare=False)
    error: Optional[str] = field(default=None, compare=False)
    synth_ms: float = field(default=0.0, compare=False)
    backend: str = field(default="", compare=False)


END_SENTINEL = OrderedChunk(sequence=999999)


class TTS:
    """
    Dual-backend TTS with concurrent synthesis and ordered playback.

    Modes:
    - fast/auto: prefer local Piper, fallback to Edge cloud
    - local: local Piper only (unless fallback enabled)
    - quality/cloud: Edge cloud only
    """

    def __init__(self):
        self.voice = getattr(config, "EDGE_TTS_VOICE", "en-IN-NeerjaNeural")
        self.mode = str(getattr(config, "TTS_MODE", "fast")).strip().lower()
        self.local_enabled = bool(getattr(config, "TTS_LOCAL_ENABLED", True))
        self.cloud_enabled = bool(getattr(config, "TTS_CLOUD_ENABLED", True))
        self.fallback_to_cloud = bool(getattr(config, "TTS_FALLBACK_TO_CLOUD", True))
        self.synth_timeout_s = float(getattr(config, "TTS_SYNTH_TIMEOUT_S", 20.0))

        self.piper_executable = getattr(config, "PIPER_EXECUTABLE", "piper")
        self.piper_model = getattr(config, "PIPER_MODEL_PATH", None)
        self.piper_config = getattr(config, "PIPER_MODEL_CONFIG_PATH", None)
        self._piper_available = self._detect_piper()

        self._num_workers = int(getattr(config, "TTS_PARALLEL_WORKERS", 3))
        self._latency_logs = bool(getattr(config, "LATENCY_LOGS", True))

        self._audio_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._text_queue: queue.Queue = queue.Queue()

        self._next_sequence = 0
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._synth_thread = None
        self._player_thread = None

        self._loop = None

        self._stream_stats = {}
        self._stats_lock = threading.Lock()

        print(
            f"TTS: mode={self.mode} backend={self._describe_mode()} voice={self.voice} "
            f"(workers: {self._num_workers})"
        )

    def _detect_piper(self):
        if not self.local_enabled:
            return False
        if not self.piper_model:
            return False

        exe = self.piper_executable
        resolved = shutil.which(exe) if not os.path.isabs(exe) else exe
        if not resolved or not os.path.exists(resolved):
            return False
        if not os.path.exists(self.piper_model):
            return False
        if self.piper_config and not os.path.exists(self.piper_config):
            return False

        self.piper_executable = resolved
        return True

    def _describe_mode(self):
        if self.mode in ("quality", "cloud"):
            return "edge"
        if self.mode == "local":
            return "piper" if self._piper_available else "none"
        if self._piper_available:
            return "piper->edge-fallback"
        return "edge" if self.cloud_enabled else "none"

    def _choose_backend(self):
        if self.mode in ("quality", "cloud"):
            return "edge" if self.cloud_enabled else "none"

        if self.mode == "local":
            if self._piper_available:
                return "piper"
            if self.fallback_to_cloud and self.cloud_enabled:
                return "edge"
            return "none"

        # fast / auto / default
        if self._piper_available:
            return "piper"
        return "edge" if self.cloud_enabled else "none"

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

    def _ensure_loop(self):
        if self._loop is not None:
            return

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True, name="TTS-Loop")
        thread.start()

        while self._loop is None:
            time.sleep(0.01)

    async def _do_edge_synth(self, seq: int, text: str) -> OrderedChunk:
        try:
            buf = io.BytesIO()
            comm = edge_tts.Communicate(text, self.voice)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

            buf.seek(0)
            data, sr = sf.read(buf, dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            return OrderedChunk(
                sequence=seq,
                audio_data=data,
                sample_rate=sr,
                text=text,
                backend="edge",
            )
        except Exception as exc:
            return OrderedChunk(sequence=seq, error=str(exc), text=text, backend="edge")

    def _do_piper_synth(self, seq: int, text: str) -> OrderedChunk:
        if not self._piper_available:
            return OrderedChunk(
                sequence=seq,
                error="Piper is not available. Configure PIPER_EXECUTABLE and PIPER_MODEL_PATH.",
                text=text,
                backend="piper",
            )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()

        cmd = [
            self.piper_executable,
            "--model",
            self.piper_model,
            "--output_file",
            temp_file.name,
        ]
        if self.piper_config:
            cmd.extend(["--config", self.piper_config])

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=self.synth_timeout_s,
                check=False,
            )

            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "piper synthesis failed").strip()
                return OrderedChunk(sequence=seq, error=err, text=text, backend="piper")

            data, sr = sf.read(temp_file.name, dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            return OrderedChunk(
                sequence=seq,
                audio_data=data,
                sample_rate=sr,
                text=text,
                backend="piper",
            )
        except Exception as exc:
            return OrderedChunk(sequence=seq, error=str(exc), text=text, backend="piper")
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def _synth_one(self, seq: int, text: str) -> OrderedChunk:
        start = time.perf_counter()
        chosen = self._choose_backend()

        if chosen == "piper":
            result = self._do_piper_synth(seq, text)
            if result.error and self.fallback_to_cloud and self.cloud_enabled:
                self._ensure_loop()
                future = asyncio.run_coroutine_threadsafe(
                    self._do_edge_synth(seq, text),
                    self._loop,
                )
                result = future.result(timeout=self.synth_timeout_s)
        elif chosen == "edge":
            self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._do_edge_synth(seq, text),
                self._loop,
            )
            result = future.result(timeout=self.synth_timeout_s)
        else:
            result = OrderedChunk(
                sequence=seq,
                error="No TTS backend available. Check TTS_MODE/local/cloud config.",
                text=text,
                backend="none",
            )

        result.synth_ms = (time.perf_counter() - start) * 1000
        if result.error:
            if self._latency_logs:
                print(f"[TTS Error:{result.backend or chosen}] {result.error}")
        else:
            self._mark_synth_done(result.synth_ms, result.backend or chosen)

        return result

    def _synth_loop(self):
        executor = ThreadPoolExecutor(max_workers=self._num_workers)
        pending = []

        while not self._stop.is_set():
            try:
                item = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                done = [f for f in pending if f.done()]
                for future in done:
                    pending.remove(future)
                    try:
                        self._audio_queue.put(future.result())
                    except Exception:
                        pass
                continue

            if item is None:
                for future in pending:
                    try:
                        self._audio_queue.put(future.result(timeout=self.synth_timeout_s))
                    except Exception:
                        pass
                self._audio_queue.put(END_SENTINEL)
                break

            seq, text = item
            pending.append(executor.submit(self._synth_one, seq, text))

            done = [f for f in pending if f.done()]
            for future in done:
                pending.remove(future)
                try:
                    self._audio_queue.put(future.result())
                except Exception:
                    pass

        executor.shutdown(wait=False)

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

        with self._lock:
            seq = self._next_sequence
            self._next_sequence += 1

        self._mark_queued()
        self._text_queue.put((seq, text.strip()))

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
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
