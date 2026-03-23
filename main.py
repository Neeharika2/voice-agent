"""
Voice Agent - Real-Time Conversational System

A simpler, more robust implementation that integrates with existing modules.
"""

import signal
import sys
import time
import threading
import queue
import re
from typing import Optional, Dict, Any

sys.path.insert(0, __file__.rsplit('\\voice_agent.py', 1)[0])

from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS
import config
import sounddevice as sd


class VoiceAgent:
    """
    Real-time voice agent with barge-in support.

    Key features:
    - Continuous listening (non-blocking STT)
    - Interruptible TTS (user can barge in)
    - Cancelable LLM generation
    - Quick reactions to phrases like "guess what"
    """

    # Reaction patterns for mid-sentence responses
    REACTION_PATTERNS = [
        (r"\bguess what\b", "Wait—what?!", "excited"),
        (r"\byou know what\b", "What happened?", "curious"),
        (r"\byou won't believe", "No way…", "excited"),
        (r"\bsomething crazy\b", "Oh my god, what?!", "excited"),
        (r"\bcan i tell you\b", "Yes, tell me!", "excited"),
        (r"\bi have to tell you\b", "What is it?!", "excited"),
        (r"\bbad news\b", "Oh no… what happened?", "sad"),
        (r"\bgood news\b", "What? Tell me!", "excited"),
        (r"\bhelp me\b", "I'm here. What's wrong?", "serious"),
    ]

    def __init__(self):
        # Components
        self._stt: Optional[STT] = None
        self._llm: Optional[LLM] = None
        self._tts: Optional[TTS] = None

        # State
        self._state = "idle"  # idle, listening, thinking, speaking
        self._state_lock = threading.Lock()

        # Control flags
        self._stop_flag = threading.Event()
        self._interrupt_flag = threading.Event()
        self._llm_cancel_flag = threading.Event()

        # TTS sequence tracking
        self._tts_seq = 0
        self._tts_lock = threading.Lock()

        # Current conversation state
        self._current_emotion = "neutral"
        self._last_partial = ""
        self._reaction_cooldown = 0

        # Stream settings
        self._stream_min_chars = int(getattr(config, "TTS_STREAM_MIN_CHARS", 20))
        self._stream_max_chars = int(getattr(config, "TTS_STREAM_MAX_CHARS", 90))

        # Audio queue for interruptible playback
        self._audio_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._synth_queue: queue.Queue = queue.Queue()
        self._player_thread: Optional[threading.Thread] = None
        self._synth_threads = []

    def initialize(self):
        """Initialize all components."""
        print("Initializing STT...")
        self._stt = STT()

        print("Initializing LLM...")
        self._llm = LLM()

        print("Initializing TTS...")
        self._tts = TTS()

        print("\n" + "=" * 55)
        print("Voice Agent Ready!")
        print("Speak naturally. You can interrupt at any time.")
        print("Press Ctrl+C to exit")
        print("=" * 55 + "\n")

    def run(self):
        """Main event loop."""
        self._stop_flag.clear()

        while not self._stop_flag.is_set():
            try:
                self._conversation_turn()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

        self.shutdown()

    def _conversation_turn(self):
        """Single conversation turn with interruption support."""
        self._state = "idle"
        self._interrupt_flag.clear()
        self._llm_cancel_flag.clear()

        # 1. Listen for user input (with partial monitoring)
        print("Listening...")
        text = self._listen_with_partials()

        if not text or self._stop_flag.is_set():
            return

        print(f"\n{'='*55}")
        print(f"You: \"{text}\"")
        print(f"{'-'*55}")

        # 2. Generate response with streaming
        self._state = "thinking"
        print("Agent: ", end="", flush=True)

        self._start_interruptible_response(text)

    def _listen_with_partials(self) -> str:
        """
        Listen for speech while monitoring for:
        - Interruption (if TTS is playing)
        - Reaction triggers in partial text
        """
        self._stt._init_recognizer()
        self._stt.clear_audio_buffer()

        import json

        stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=self._stt.blocksize,
            dtype='int16',
            channels=1,
            callback=self._stt._callback
        )

        last_partial = ""
        silence_start = None
        had_speech = False

        with stream:
            while not self._stop_flag.is_set():
                try:
                    data = self._stt.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Check for interruption
                if self._interrupt_flag.is_set():
                    return ""

                # Process with Vosk
                if self._stt.recognizer.AcceptWaveform(data):
                    result = json.loads(self._stt.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        return text
                    last_partial = ""
                    had_speech = False
                    silence_start = None
                else:
                    # Partial result
                    partial = json.loads(self._stt.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()

                    if partial_text:
                        had_speech = True
                        silence_start = None

                        if partial_text != last_partial:
                            last_partial = partial_text
                            print(f"\r  Hearing: {partial_text}        ", end="", flush=True)

                            # Check for reaction triggers
                            if self._state == "idle":
                                reaction = self._check_reaction(partial_text)
                                if reaction:
                                    self._play_reaction(reaction)

                    else:
                        # Silence
                        if had_speech:
                            if silence_start is None:
                                silence_start = time.perf_counter()
                            elif time.perf_counter() - silence_start >= self._stt.silence_timeout:
                                result = json.loads(self._stt.recognizer.FinalResult())
                                text = result.get("text", "").strip()
                                if text:
                                    return text
                                last_partial = ""
                                had_speech = False
                                silence_start = None

        return ""

    def _check_reaction(self, text: str) -> Optional[tuple]:
        """Check if partial text matches a reaction pattern."""
        now = time.perf_counter()
        if now - self._reaction_cooldown < 3.0:  # Cooldown between reactions
            return None

        text_lower = text.lower()
        for item in self.REACTION_PATTERNS:
            if not isinstance(item, (tuple, list)):
                continue
            if len(item) == 3:
                pattern, reaction, emotion = item
            elif len(item) == 2 and isinstance(item[1], (tuple, list)) and len(item[1]) == 2:
                pattern = item[0]
                reaction, emotion = item[1]
            else:
                continue

            if re.search(pattern, text_lower):
                self._reaction_cooldown = now
                return (reaction, emotion)
        return None

    def _play_reaction(self, reaction: tuple):
        """Play a quick reaction (bypasses LLM)."""
        text, emotion = reaction
        print(f"\n[Reacting: {text}]")
        self._tts.speak(text, emotion=emotion)

    def _start_interruptible_response(self, user_text: str):
        """
        Start LLM generation and TTS with interruption support.

        Uses a separate thread for audio playback that can be interrupted.
        """
        buffer = ""
        first_chunk = True
        self._tts_seq = 0
        self._tts.start_streaming()

        try:
            for chunk, done, stats in self._llm.generate_stream(user_text, verbose=False):
                # Check for cancellation
                if self._llm_cancel_flag.is_set():
                    print("\n[Cancelled]")
                    return

                if chunk:
                    buffer += chunk

                    # Extract speakable units
                    ready, buffer = self._extract_speakable_units(
                        buffer, self._stream_min_chars, self._stream_max_chars
                    )

                    for unit in ready:
                        print(unit, end=" ", flush=True)

                        # Check for interruption before queuing
                        if self._interrupt_flag.is_set():
                            self._tts.finish_streaming()
                            return

                        self._tts.queue_sentence(unit, emotion=self._current_emotion)

            # Handle remaining text
            if buffer.strip():
                print(buffer, end="", flush=True)
                if not self._interrupt_flag.is_set():
                    self._tts.queue_sentence(buffer.strip(), emotion=self._current_emotion)

            print("\n")

            # Wait for audio to finish (with interruption monitoring)
            if not self._interrupt_flag.is_set():
                self._monitor_playback()

        except Exception as e:
            print(f"\n[Error: {e}]")

    def _monitor_playback(self):
        """Monitor playback while watching for interruption."""
        # Start a thread to wait for TTS completion
        done_event = threading.Event()

        def wait_for_tts():
            self._tts.finish_streaming()
            done_event.set()

        wait_thread = threading.Thread(target=wait_for_tts, daemon=True)
        wait_thread.start()

        # Monitor for user speech during playback
        import json
        self._stt._init_recognizer()
        self._stt.clear_audio_buffer()

        stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=self._stt.blocksize,
            dtype='int16',
            channels=1,
            callback=self._stt._callback
        )

        with stream:
            while not done_event.is_set() and not self._stop_flag.is_set():
                try:
                    data = self._stt.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Check for user speech
                if self._stt.recognizer.AcceptWaveform(data):
                    result = json.loads(self._stt.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self._handle_interruption(text)
                        return
                else:
                    partial = json.loads(self._stt.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text and len(partial_text) > 3:  # Meaningful speech
                        self._handle_interruption(partial_text)
                        return

        # Clear buffer for next turn
        self._stt.clear_audio_buffer()

    def _handle_interruption(self, text: str):
        """Handle user interruption during TTS playback."""
        print(f"\n[Interrupted! User said: \"{text}\"]")

        # Stop audio
        sd.stop()

        # Cancel LLM
        self._llm_cancel_flag.set()

        # Finish TTS (clears queues)
        try:
            self._tts.finish_streaming(timeout=1.0)
        except:
            pass

        # Clear audio buffer
        self._stt.clear_audio_buffer()

        # Process the interruption
        print(f"\n{'='*55}")
        print(f"You: \"{text}\"")
        print(f"{'-'*55}")

        # Reset and respond
        self._llm_cancel_flag.clear()
        self._start_interruptible_response(text)

    def _extract_speakable_units(self, text: str, min_chars: int, max_chars: int):
        """Split text into speakable chunks at punctuation."""
        if not text:
            return [], ""

        delimiters = ".!?;,:\n"
        units = []
        cursor = 0

        for idx, ch in enumerate(text):
            if ch in delimiters:
                segment = text[cursor:idx + 1].strip()
                if len(segment) >= min_chars or ch in ".!?\n":
                    units.append(segment)
                    cursor = idx + 1

        remaining = text[cursor:].lstrip()

        if len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars)
            if split_at > 0:
                units.append(remaining[:split_at].strip())
                remaining = remaining[split_at + 1:].lstrip()

        return [u for u in units if u], remaining

    def shutdown(self):
        """Clean shutdown."""
        print("\n\nShutting down...")
        self._stop_flag.set()
        self._llm_cancel_flag.set()

        if self._tts:
            sd.stop()
            self._tts.shutdown()


def main():
    agent = VoiceAgent()

    def signal_handler(sig, frame):
        agent.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    agent.initialize()
    agent.run()


if __name__ == "__main__":
    main()