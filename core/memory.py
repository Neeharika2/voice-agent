"""
Simple Memory System

Lightweight memory for storing important facts and emotional statements.
No vector DB - uses keyword matching for retrieval.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    category: str  # "fact", "emotion", "preference", "event"
    timestamp: float = field(default_factory=time.time)
    importance: int = 1  # 1-3, higher = more important

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(
            content=data["content"],
            category=data["category"],
            timestamp=data.get("timestamp", time.time()),
            importance=data.get("importance", 1)
        )


class MemoryStore:
    """
    Simple keyword-based memory system.

    Stores:
    - Emotional statements (user feelings)
    - Important facts (exam tomorrow, birthday, etc.)
    - Preferences (likes, dislikes)

    NOT stored:
    - Random chitchat
    - System messages
    - Every partial input
    """

    # Patterns to detect memorable content
    EMOTION_PATTERNS = [
        (r"\bi('m| am) (so |very )?(happy|sad|excited|scared|worried|angry|stressed|anxious)\b", "emotion"),
        (r"\bi feel (so |very )?(happy|sad|excited|scared|worried|angry|stressed|anxious)\b", "emotion"),
        (r"\bi('ve| have) been (feeling|having) a (hard|rough|good|great) (time|day|week)\b", "emotion"),
    ]

    FACT_PATTERNS = [
        (r"\bmy (name|birthday|anniversary) (is|'s) \w+", "fact"),
        (r"\bi have (a |an )?(test|exam|meeting|interview|appointment) (tomorrow|today|next week)\b", "fact"),
        (r"\bmy (mom|dad|brother|sister|friend|partner) ('s|is) \w+", "fact"),
        (r"\bi (work|study|live) (at|in) \w+", "fact"),
        (r"\btomorrow i (have|need to|will)\b", "event"),
    ]

    PREFERENCE_PATTERNS = [
        (r"\bi (love|like|enjoy|hate|dislike) \w+", "preference"),
        (r"\bmy favorite \w+ (is|'s) \w+", "preference"),
    ]

    # Keywords for retrieval matching
    KEYWORD_MAP = {
        "exam": ["test", "exam", "study", "school"],
        "work": ["job", "work", "meeting", "boss", "office"],
        "family": ["mom", "dad", "brother", "sister", "family", "parent"],
        "feeling": ["happy", "sad", "scared", "worried", "stressed", "anxious", "excited"],
        "tomorrow": ["tomorrow", "soon", "next day"],
    }

    def __init__(self, max_memories: int = 50, storage_path: Optional[str] = None):
        self.memories: List[Memory] = []
        self.max_memories = max_memories
        self.storage_path = storage_path

        if storage_path and os.path.exists(storage_path):
            self._load()

    def extract_and_store(self, text: str) -> Optional[Memory]:
        """
        Extract memorable content from text and store if found.

        Returns the stored Memory or None.
        """
        text_lower = text.lower()

        # Check emotion patterns
        for pattern, category in self.EMOTION_PATTERNS:
            if re.search(pattern, text_lower):
                return self._add(text, category, importance=2)

        # Check fact patterns
        for pattern, category in self.FACT_PATTERNS:
            if re.search(pattern, text_lower):
                return self._add(text, category, importance=3)

        # Check preference patterns
        for pattern, category in self.PREFERENCE_PATTERNS:
            if re.search(pattern, text_lower):
                return self._add(text, category, importance=2)

        return None

    def _add(self, content: str, category: str, importance: int = 1) -> Memory:
        """Add a memory."""
        memory = Memory(
            content=content,
            category=category,
            importance=importance
        )

        self.memories.append(memory)

        # Trim if needed
        if len(self.memories) > self.max_memories:
            # Remove least important, oldest
            self.memories.sort(key=lambda m: (-m.importance, -m.timestamp))
            self.memories = self.memories[:self.max_memories]

        self._save()
        return memory

    def retrieve(self, query: str, limit: int = 3) -> List[Memory]:
        """
        Retrieve relevant memories for a query.

        Uses keyword matching - simple but effective.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Expand query with related keywords
        expanded_words = set(query_words)
        for word in query_words:
            if word in self.KEYWORD_MAP:
                expanded_words.update(self.KEYWORD_MAP[word])

        # Score each memory
        scored = []
        for memory in self.memories:
            content_words = set(memory.content.lower().split())
            overlap = len(content_words & expanded_words)

            if overlap > 0:
                # Boost by importance and recency
                recency_boost = 1 if (time.time() - memory.timestamp) < 86400 else 0
                score = overlap + memory.importance + recency_boost
                scored.append((score, memory))

        # Sort by score and return top
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:limit]]

    def get_context_for_prompt(self, query: str, max_items: int = 2) -> str:
        """
        Get memory context formatted for LLM prompt.

        Returns a context string or empty string if no relevant memories.
        """
        memories = self.retrieve(query, limit=max_items)

        if not memories:
            return ""

        context_parts = ["[Remember about the user:]"]
        for m in memories:
            context_parts.append(f"- {m.content}")

        return "\n".join(context_parts)

    def _save(self):
        """Save memories to disk."""
        if self.storage_path:
            try:
                data = [m.to_dict() for m in self.memories]
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"[Memory] Save error: {e}")

    def _load(self):
        """Load memories from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            self.memories = [Memory.from_dict(d) for d in data]
        except Exception as e:
            print(f"[Memory] Load error: {e}")

    def clear(self):
        """Clear all memories."""
        self.memories = []
        self._save()