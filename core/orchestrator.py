"""
Voice Agent Orchestrator

Main controller that coordinates all components via events.
Handles interruption, state management, and conversation flow.
"""

import queue
import threading
import time
import signal
import sys
from typing import Optional

# Path setup
sys.path.insert(0, __file__.rsplit('\\core', 1)[0])

from core.events import EventBus, Event, EventType
from core.reactions import ReactionEngine, ReactionPlayer
from core.stt_controller import ContinuousSTT
from core.llm_streamer import InterruptibleLLM
from core.tts_pipeline import ParallelTTSPipeline
from core.memory import MemoryStore

from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS
import config


class AgentState:
    """Agent state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceAgent:
    """
    Real-time voice agent with barge-in support.

    Event-driven architecture:
    - EventBus coordinates all components
    - ContinuousSTT emits speech events
    - ReactionEngine triggers instant responses
    - InterruptibleLLM generates with cancellation
    - ParallelTTSPipeline plays with interruption support
    - MemoryStore provides context
    """

    def __init__(self):
        # Core infrastructure
        self.event_bus = EventBus()

        # Components (lazy init)
        self._stt: Optional[STT] = None
        self._llm: Optional[LLM] = None
        self._tts: Optional[TTS] = None

        self._continuous_stt: Optional[ContinuousSTT] = None
        self._interruptible_llm: Optional[InterruptibleLLM] = None
        self._tts_pipeline: Optional[ParallelTTSPipeline] = None
        self._reaction_engine: Optional[ReactionEngine] = None
        self._reaction_player: Optional[ReactionPlayer] = None
        self._memory: Optional[MemoryStore] = None

        # State
        self._state = AgentState.IDLE
        self._state_lock = threading.Lock()

        # Control
        self._stop = threading.Event()

        # Stream settings
        self._stream_min_chars = int(getattr(config, "TTS_STREAM_MIN_CHARS", 20))
        self._stream_max_chars = int(getattr(config, "TTS_STREAM_MAX_CHARS", 90))

        # Current generation state
        self._llm_buffer = ""
        self._tts_seq = 0

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Subscribe to events."""
        self.event_bus.subscribe(EventType.SPEECH_PARTIAL, self._on_speech_partial)
        self.event_bus.subscribe(EventType.SPEECH_FINAL, self._on_speech_final)
        self.event_bus.subscribe(EventType.INTERRUPTION, self._on_interruption)
        self.event_bus.subscribe(EventType.LLM_DONE, self._on_llm_done)
        self.event_bus.subscribe(EventType.TTS_DONE, self._on_tts_done)

    def initialize(self):
        """Initialize all components."""
        print("Initializing STT...")
        self._stt = STT()

        print("Initializing LLM...")
        self._llm = LLM()

        print("Initializing TTS...")
        self._tts = TTS()

        # Core modules
        self._continuous_stt = ContinuousSTT(
            self._stt.model,
            self.event_bus,
            sample_rate=config.SAMPLE_RATE
        )
        self._continuous_stt.set_interrupt_callback(self._handle_interrupt)

        self._interruptible_llm = InterruptibleLLM(self._llm, self.event_bus)

        self._tts_pipeline = ParallelTTSPipeline(self._tts, self.event_bus, num_workers=2)

        self._reaction_engine = ReactionEngine()
        self._reaction_player = ReactionPlayer(self._tts)

        self._memory = MemoryStore(
            max_memories=50,
            storage_path="memory.json"
        )

        print("\n" + "=" * 55)
        print("Voice Agent Ready!")
        print("Speak naturally. Interrupt anytime.")
        print("Press Ctrl+C to exit")
        print("=" * 55 + "\n")

    def run(self):
        """Start the agent."""
        self._stop.clear()

        # Start event bus
        self.event_bus.start()

        # Start continuous STT
        self._continuous_stt.start()

        # Start TTS pipeline
        self._tts_pipeline.start()

        # Main loop
        try:
            while not self._stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _on_speech_partial(self, event: Event):
        """Handle partial speech - check for reactions."""
        text = event.data.get("text", "")

        with self._state_lock:
            if self._state == AgentState.IDLE:
                self._state = AgentState.LISTENING
                print(f"\r  Hearing: {text}        ", end="", flush=True)

        # Check for instant reactions (bypass LLM)
        reaction = self._reaction_engine.check(text)
        if reaction and self._state != AgentState.SPEAKING:
            response, emotion = reaction
            # Play reaction immediately
            self._reaction_player.queue(response, emotion)
            self._reaction_player.play_queued()

    def _on_speech_final(self, event: Event):
        """Handle final speech - process with LLM."""
        text = event.data.get("text", "").strip()
        if not text:
            return

        print(f"\n\n{'='*55}")
        print(f"You: \"{text}\"")
        print(f"{'-'*55}")

        # Extract and store memory
        self._memory.extract_and_store(text)

        with self._state_lock:
            self._state = AgentState.THINKING

        # Reset reaction cooldowns for new turn
        self._reaction_engine.reset_cooldowns()

        # Generate response
        self._generate_response(text)

    def _on_interruption(self, event: Event):
        """Handle user interruption during TTS."""
        text = event.data.get("text", "")

        print(f"\n[INTERRUPTED] User: \"{text}\"")

        # Stop TTS
        self._tts_pipeline.interrupt()

        # Cancel LLM
        self._interruptible_llm.cancel()

        # Reset state
        with self._state_lock:
            self._state = AgentState.IDLE

        # Clear buffers
        self._llm_buffer = ""
        self._tts_seq = 0

        # Process the interruption as new input
        print(f"\n{'='*55}")
        print(f"You: \"{text}\"")
        print(f"{'-'*55}")

        # Generate response to interruption
        self._generate_response(text)

    def _on_llm_done(self, event: Event):
        """Handle LLM completion."""
        pass  # State handled in _generate_response

    def _on_tts_done(self, event: Event):
        """Handle TTS completion."""
        with self._state_lock:
            self._state = AgentState.IDLE

        self._continuous_stt.on_tts_stop()

    def _generate_response(self, user_text: str):
        """Generate LLM response and queue TTS."""
        print("Agent: ", end="", flush=True)

        # Get memory context
        memory_context = self._memory.get_context_for_prompt(user_text, max_items=2)

        self._llm_buffer = ""
        self._tts_seq = 0

        # Notify STT that TTS is starting (for interruption detection)
        self._continuous_stt.on_tts_start()

        with self._state_lock:
            self._state = AgentState.THINKING

        try:
            for chunk, done, stats in self._interruptible_llm.start_generation(
                user_text,
                memory_context=memory_context
            ):
                if done:
                    # Handle remaining buffer
                    if self._llm_buffer.strip():
                        self._queue_tts(self._llm_buffer.strip())
                    break

                if chunk:
                    self._llm_buffer += chunk
                    self._process_buffer()

            print("\n")

            # Wait for TTS
            if self._tts_seq > 0:
                with self._state_lock:
                    self._state = AgentState.SPEAKING
                self._tts_pipeline.finish()

        except Exception as e:
            print(f"\n[Error: {e}]")

        # Clear audio buffer for next turn
        self._continuous_stt.clear_buffer()

    def _process_buffer(self):
        """Extract speakable units from LLM buffer."""
        ready, remaining = self._extract_speakable_units(
            self._llm_buffer,
            self._stream_min_chars,
            self._stream_max_chars
        )

        for unit in ready:
            print(unit, end=" ", flush=True)
            self._queue_tts(unit)

        self._llm_buffer = remaining

    def _extract_speakable_units(self, text: str, min_chars: int, max_chars: int):
        """Split text into speakable chunks."""
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

    def _queue_tts(self, text: str):
        """Queue text for TTS synthesis."""
        self._tts_pipeline.queue_text(text, emotion="neutral")

    def _handle_interrupt(self):
        """Callback for interruption detection."""
        self._tts_pipeline.interrupt()
        self._interruptible_llm.cancel()

    def shutdown(self):
        """Clean shutdown."""
        print("\n\nShutting down...")
        self._stop.set()

        if self._continuous_stt:
            self._continuous_stt.stop()

        if self._tts_pipeline:
            self._tts_pipeline.stop()

        self.event_bus.stop()

        if self._tts:
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