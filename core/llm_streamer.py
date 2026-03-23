"""
Interruptible LLM Streamer

LLM generation that can be cancelled mid-stream.
Uses a stop flag for clean cancellation.
"""

import threading
import time
from typing import Callable, Optional, Generator, Tuple, Dict, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import EventBus, Event, EventType


@dataclass
class LLMState:
    """State of current LLM generation."""
    is_generating: bool = False
    text_generated: str = ""
    chunks_produced: int = 0
    cancelled: bool = False


class InterruptibleLLM:
    """
    LLM wrapper with cancellation support.

    Features:
    - Stream generation with stop flag
    - Cancel mid-stream
    - Emit events for each chunk
    """

    def __init__(self, llm_instance, event_bus: EventBus):
        self.llm = llm_instance
        self.event_bus = event_bus

        self._stop_flag = threading.Event()
        self._state = LLMState()
        self._lock = threading.Lock()

    def start_generation(
        self,
        prompt: str,
        memory_context: str = "",
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[Tuple[str, bool, Dict], None, None]:
        """
        Start streaming generation.

        Yields: (chunk_text, is_done, stats)

        Can be cancelled by calling cancel() at any time.
        """
        self._stop_flag.clear()

        with self._lock:
            self._state = LLMState(is_generating=True)

        # Build prompt with memory context
        full_prompt = prompt
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {prompt}"

        # Emit start event
        self.event_bus.publish(Event(
            type=EventType.LLM_START,
            data={"prompt": prompt}
        ))

        start_time = time.perf_counter()
        first_chunk_time = None
        chunks = 0

        try:
            for chunk, done, stats in self.llm.generate_stream(full_prompt, verbose=False):
                # Check for cancellation
                if self._stop_flag.is_set():
                    with self._lock:
                        self._state.cancelled = True
                        self._state.is_generating = False

                    self.event_bus.publish(Event(
                        type=EventType.LLM_CANCELLED
                    ))
                    return

                if chunk:
                    # Track timing
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                    chunks += 1

                    with self._lock:
                        self._state.text_generated += chunk
                        self._state.chunks_produced = chunks

                    # Emit chunk event
                    self.event_bus.publish(Event(
                        type=EventType.LLM_CHUNK,
                        data={"chunk": chunk}
                    ))

                    # Call callback if provided
                    if on_chunk:
                        on_chunk(chunk)

                    yield chunk, False, stats

            # Done
            total_time = (time.perf_counter() - start_time) * 1000
            ttft = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0

            self.event_bus.publish(Event(
                type=EventType.LLM_DONE,
                data={
                    "total_ms": total_time,
                    "ttft_ms": ttft,
                    "chunks": chunks
                }
            ))

            with self._lock:
                self._state.is_generating = False

            yield "", True, {"total_ms": total_time, "ttft_ms": ttft}

        except Exception as e:
            self.event_bus.publish(Event(
                type=EventType.ERROR,
                data={"error": str(e), "source": "llm"}
            ))
            with self._lock:
                self._state.is_generating = False
            yield "", True, {"error": str(e)}

    def cancel(self):
        """Cancel the current generation."""
        self._stop_flag.set()

    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        with self._lock:
            return self._state.is_generating

    def get_state(self) -> LLMState:
        """Get current state."""
        with self._lock:
            return LLMState(
                is_generating=self._state.is_generating,
                text_generated=self._state.text_generated,
                chunks_produced=self._state.chunks_produced,
                cancelled=self._state.cancelled
            )