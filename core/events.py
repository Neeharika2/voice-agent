"""
Event System for Real-Time Voice Agent

Simple, thread-safe event bus using queues.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """All events in the voice agent system."""
    # Speech events
    SPEECH_START = auto()       # User started speaking
    SPEECH_PARTIAL = auto()     # Partial transcript (real-time)
    SPEECH_FINAL = auto()       # Final transcript ready

    # Interruption
    INTERRUPTION = auto()       # User interrupted TTS

    # Reaction (instant, no LLM)
    REACTION_TRIGGER = auto()   # Pattern matched, play reaction

    # LLM events
    LLM_START = auto()          # LLM started generating
    LLM_CHUNK = auto()          # LLM yielded text
    LLM_DONE = auto()           # LLM finished
    LLM_CANCELLED = auto()      # LLM was cancelled

    # TTS events
    TTS_START = auto()          # TTS started playing
    TTS_CHUNK_QUEUED = auto()   # Audio chunk queued
    TTS_DONE = auto()           # TTS finished playing

    # State
    IDLE_TIMEOUT = auto()       # No activity for a while
    ERROR = auto()              # Something went wrong


@dataclass
class Event:
    """Event object passed through the system."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.perf_counter)

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.timestamp < other.timestamp


class EventBus:
    """
    Central event distribution system.

    Thread-safe pub/sub pattern using a single queue.
    """

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Register a handler for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event):
        """Put an event on the bus (non-blocking)."""
        self._queue.put((event.timestamp, event))

    def start(self):
        """Start the event processing loop."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the event loop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        """Main event loop - runs in background thread."""
        while not self._stop.is_set():
            try:
                _, event = self._queue.get(timeout=0.1)
                handlers = self._subscribers.get(event.type, [])

                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"[EventBus] Handler error for {event.type}: {e}")

                self._queue.task_done()
            except queue.Empty:
                continue