"""
Parallel TTS Pipeline

Multi-threaded TTS synthesis with ordered playback and interruption support.
"""

import queue
import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import EventBus, Event, EventType


@dataclass
class AudioChunk:
    """Audio chunk with sequence number for ordered playback."""
    sequence: int
    audio: np.ndarray
    sample_rate: int
    text: str


class ParallelTTSPipeline:
    """
    Parallel TTS synthesis with interruptible playback.

    Features:
    - Multiple synthesis workers
    - Priority queue for ordered playback
    - Can be interrupted at any time
    - Emits events for state tracking
    """

    def __init__(self, tts_instance, event_bus: EventBus, num_workers: int = 2):
        self.tts = tts_instance
        self.event_bus = event_bus
        self.num_workers = num_workers

        # Queues
        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Control
        self._stop = threading.Event()
        self._is_playing = threading.Event()
        self._workers: list = []
        self._playback_thread: Optional[threading.Thread] = None

        # Sequence tracking
        self._next_seq = 0
        self._seq_lock = threading.Lock()

        # Emotion config
        from piper.config import SynthesisConfig
        self._emotion_configs = {
            "neutral": SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8),
            "happy": SynthesisConfig(length_scale=0.95, noise_scale=0.8, noise_w_scale=0.9),
            "excited": SynthesisConfig(length_scale=0.9, noise_scale=0.9, noise_w_scale=1.0),
            "calm": SynthesisConfig(length_scale=1.15, noise_scale=0.5, noise_w_scale=0.7),
            "sad": SynthesisConfig(length_scale=1.2, noise_scale=0.5, noise_w_scale=0.6),
            "serious": SynthesisConfig(length_scale=1.0, noise_scale=0.4, noise_w_scale=0.5),
            "curious": SynthesisConfig(length_scale=1.05, noise_scale=0.75, noise_w_scale=0.85),
        }

    def start(self):
        """Start worker and playback threads."""
        self._stop.clear()

        # Start synthesis workers
        for i in range(self.num_workers):
            w = threading.Thread(target=self._synth_worker, daemon=True)
            w.start()
            self._workers.append(w)

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def stop(self):
        """Stop all threads."""
        self._stop.set()
        self._text_queue.put(None)  # Sentinel

        for w in self._workers:
            w.join(timeout=0.5)
        self._workers.clear()

        if self._playback_thread:
            self._playback_thread.join(timeout=0.5)

    def queue_text(self, text: str, emotion: str = "neutral"):
        """Queue text for synthesis."""
        with self._seq_lock:
            seq = self._next_seq
            self._next_seq += 1

        self._text_queue.put((seq, text, emotion))

    def finish(self):
        """Signal end of input and wait for playback."""
        self._text_queue.put(None)
        if self._playback_thread:
            self._playback_thread.join(timeout=60)

    def interrupt(self):
        """Immediately stop playback."""
        self._is_playing.clear()
        import sounddevice as sd
        sd.stop()

        # Clear queues
        try:
            while True:
                self._text_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass

        # Reset sequence
        with self._seq_lock:
            self._next_seq = 0

    def _synth_worker(self):
        """Worker thread for synthesis."""
        while not self._stop.is_set():
            try:
                item = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # Sentinel - signal playback to end
                self._audio_queue.put((999999, None))
                break

            seq, text, emotion = item

            try:
                audio, sample_rate = self._synthesize(text, emotion)
                if audio is not None:
                    self._audio_queue.put((seq, AudioChunk(
                        sequence=seq,
                        audio=audio,
                        sample_rate=sample_rate,
                        text=text
                    )))
            except Exception as e:
                print(f"[TTS Worker] Synthesis error: {e}")

    def _synthesize(self, text: str, emotion: str) -> Tuple[Optional[np.ndarray], int]:
        """Synthesize a single text chunk."""
        syn_config = self._emotion_configs.get(emotion, self._emotion_configs["neutral"])

        audio_chunks = []
        sample_rate = self.tts.voice.config.sample_rate

        for chunk in self.tts.voice.synthesize(text, syn_config=syn_config):
            audio_chunks.append(chunk.audio_float_array)
            sample_rate = chunk.sample_rate

        if audio_chunks:
            return np.concatenate(audio_chunks), sample_rate
        return None, sample_rate

    def _playback_worker(self):
        """Playback thread - maintains order and handles interruption."""
        import sounddevice as sd

        expected_seq = 0
        pending = {}
        got_end = False

        self.event_bus.publish(Event(type=EventType.TTS_START))

        while not self._stop.is_set():
            try:
                seq, chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                if got_end and not pending:
                    break
                continue

            if chunk is None:  # Sentinel
                got_end = True
            else:
                pending[seq] = chunk

            # Play in order
            while expected_seq in pending:
                audio_chunk = pending.pop(expected_seq)

                if self._stop.is_set():
                    sd.stop()
                    break

                self._is_playing.set()
                sd.play(audio_chunk.audio, samplerate=audio_chunk.sample_rate)
                sd.wait()
                self._is_playing.clear()

                if self._stop.is_set():
                    break

                expected_seq += 1

            if got_end and not pending:
                break

        self.event_bus.publish(Event(type=EventType.TTS_DONE))

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing.is_set()