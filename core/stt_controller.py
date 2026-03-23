"""
Continuous STT Controller

Non-blocking speech recognition that emits partial and final events.
Monitors for interruption during TTS playback.
"""

import queue
import threading
import time
import json
from typing import Optional, Callable
import sounddevice as sd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import EventBus, Event, EventType


class ContinuousSTT:
    """
    Continuous speech recognition with real-time partial output.

    Features:
    - Runs in background thread
    - Emits SPEECH_PARTIAL events in real-time
    - Emits SPEECH_FINAL when sentence complete
    - Monitors for interruption during TTS
    """

    def __init__(self, vosk_model, event_bus: EventBus, sample_rate: int = 16000):
        from vosk import Model, KaldiRecognizer

        self.model = vosk_model
        self.sample_rate = sample_rate
        self.event_bus = event_bus

        # Audio capture
        self.audio_queue: queue.Queue = queue.Queue()
        self.blocksize = 4000  # 250ms chunks
        self.silence_timeout = 1.0  # Seconds of silence to finalize

        # Control
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_listening = False
        self._lock = threading.Lock()

        # Interruption detection
        self._tts_is_playing = False
        self._interrupt_threshold = 3  # Words needed to trigger interrupt
        self._on_interrupt: Optional[Callable] = None

    def set_interrupt_callback(self, callback: Callable):
        """Set callback for when user interrupts TTS."""
        self._on_interrupt = callback

    def on_tts_start(self):
        """Called when TTS starts playing."""
        with self._lock:
            self._tts_is_playing = True

    def on_tts_stop(self):
        """Called when TTS stops playing."""
        with self._lock:
            self._tts_is_playing = False

    def start(self):
        """Start continuous listening."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def clear_buffer(self):
        """Clear any buffered audio."""
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

    def _callback(self, indata, frames, time_info, status):
        """Audio callback - puts data in queue."""
        if status:
            print(f"[STT] Audio status: {status}")
        self.audio_queue.put(bytes(indata))

    def _listen_loop(self):
        """Main listening loop - runs in background."""
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        self._is_listening = True
        self.clear_buffer()

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._callback
        )

        last_partial = ""
        silence_start = None
        had_speech = False
        partial_words = []

        with stream:
            while not self._stop.is_set():
                try:
                    data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process with Vosk
                if recognizer.AcceptWaveform(data):
                    # Final result
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        self.event_bus.publish(Event(
                            type=EventType.SPEECH_FINAL,
                            data={"text": text}
                        ))

                    # Reset state
                    last_partial = ""
                    silence_start = None
                    had_speech = False
                    partial_words = []

                else:
                    # Partial result
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()

                    if partial_text:
                        had_speech = True
                        silence_start = None

                        if partial_text != last_partial:
                            last_partial = partial_text
                            partial_words = partial_text.split()

                            # Emit partial event
                            self.event_bus.publish(Event(
                                type=EventType.SPEECH_PARTIAL,
                                data={"text": partial_text}
                            ))

                            # Check for interruption during TTS
                            with self._lock:
                                if self._tts_is_playing:
                                    if len(partial_words) >= self._interrupt_threshold:
                                        self.event_bus.publish(Event(
                                            type=EventType.INTERRUPTION,
                                            data={"text": partial_text}
                                        ))
                    else:
                        # Silence
                        if had_speech:
                            if silence_start is None:
                                silence_start = time.perf_counter()
                            elif time.perf_counter() - silence_start >= self.silence_timeout:
                                # Force finalize on silence
                                result = json.loads(recognizer.FinalResult())
                                text = result.get("text", "").strip()

                                if text:
                                    self.event_bus.publish(Event(
                                        type=EventType.SPEECH_FINAL,
                                        data={"text": text}
                                    ))

                                # Reset
                                last_partial = ""
                                silence_start = None
                                had_speech = False
                                partial_words = []