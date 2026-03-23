"""
Reaction Engine - Instant Rule-Based Responses

Detects trigger phrases in partial speech and emits instant reactions.
NO LLM - uses pattern matching for sub-10ms response time.
"""

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading


@dataclass
class Reaction:
    """A reaction to a trigger phrase."""
    trigger_pattern: str      # Regex pattern
    response: str             # What to say
    emotion: str              # How to say it
    cooldown_seconds: float = 5.0  # Avoid repeating


class ReactionEngine:
    """
    Rule-based reaction system for instant responses.

    Matches partial speech against trigger patterns and emits reactions
    immediately, bypassing the LLM entirely.
    """

    DEFAULT_REACTIONS = [
        # Excitement/Curiosity triggers
        (r"\bguess what\b", "Wait—what?!", "excited"),
        (r"\byou know what\b", "What happened?", "curious"),
        (r"\byou won't believe", "No way… tell me!", "excited"),
        (r"\bsomething crazy\b", "Oh my god, what?!", "excited"),
        (r"\bsomething happened\b", "What? What happened?", "curious"),
        (r"\bcan i tell you\b", "Yes! Tell me!", "excited"),
        (r"\bi have to tell you\b", "What is it?!", "excited"),
        (r"\bi need to tell you\b", "I'm listening!", "curious"),
        (r"\blisten to this\b", "Okay, I'm listening!", "curious"),

        # Emotional triggers
        (r"\bbad news\b", "Oh no… what happened?", "sad"),
        (r"\bgood news\b", "Oh nice! What is it?", "happy"),
        (r"\bi'm so (sad|upset|depressed)\b", "Oh no… I'm here for you.", "sad"),
        (r"\bi'm so (happy|excited)\b", "That's wonderful!", "happy"),
        (r"\bi'm (scared|afraid)\b", "Hey, it's okay. I'm here.", "calm"),
        (r"\bi'm (angry|mad)\b", "I hear you. What's going on?", "serious"),

        # Help/Support triggers
        (r"\bhelp me\b", "I'm here. What's wrong?", "serious"),
        (r"\bi need help\b", "Of course. What do you need?", "serious"),
        (r"\bi don't know what to do\b", "Let's figure it out together.", "calm"),

        # Attention triggers
        (r"\bare you there\b", "I'm here!", "neutral"),
        (r"\bhello\b", "Hi there!", "happy"),
        (r"\bhey\b", "Hey!", "happy"),

        # Surprise triggers
        (r"\boh my god\b", "What?! What happened?", "excited"),
        (r"\bno way\b", "I know, right?!", "excited"),
        (r"\byou're not gonna believe", "Try me!", "curious"),
    ]

    def __init__(self):
        self.reactions: List[Tuple[re.Pattern, str, str, float]] = []
        self._last_triggered: dict = {}  # pattern -> timestamp
        self._lock = threading.Lock()

        # Compile default reactions
        for pattern, response, emotion in self.DEFAULT_REACTIONS:
            self.add_reaction(pattern, response, emotion)

    def add_reaction(self, pattern: str, response: str, emotion: str, cooldown: float = 5.0):
        """Add a new reaction rule."""
        compiled = re.compile(pattern, re.IGNORECASE)
        self.reactions.append((compiled, response, emotion, cooldown))

    def check(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Check if text matches any trigger pattern.

        Returns (response, emotion) if match found, None otherwise.
        Respects cooldown to avoid repeating reactions.
        """
        now = time.perf_counter()
        text_lower = text.lower().strip()

        with self._lock:
            for pattern, response, emotion, cooldown in self.reactions:
                if pattern.search(text_lower):
                    # Check cooldown
                    pattern_str = pattern.pattern
                    last_time = self._last_triggered.get(pattern_str, 0)

                    if now - last_time < cooldown:
                        continue  # Still cooling down

                    # Trigger!
                    self._last_triggered[pattern_str] = now
                    return (response, emotion)

        return None

    def reset_cooldowns(self):
        """Clear all cooldowns for a new conversation."""
        with self._lock:
            self._last_triggered.clear()


class ReactionPlayer:
    """
    Plays reactions immediately using TTS.

    Runs in a separate thread to not block anything.
    """

    def __init__(self, tts_instance):
        self.tts = tts_instance
        self._queue: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        self._playing = False

    def queue(self, response: str, emotion: str):
        """Queue a reaction to play immediately."""
        with self._lock:
            self._queue.append((response, emotion))

    def play_queued(self):
        """Play all queued reactions."""
        with self._lock:
            to_play = self._queue.copy()
            self._queue.clear()

        for text, emotion in to_play:
            self._play_sync(text, emotion)

    def _play_sync(self, text: str, emotion: str):
        """Play a single reaction synchronously."""
        import numpy as np
        from piper.config import SynthesisConfig

        # Get emotion config
        presets = {
            "neutral": SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8),
            "happy": SynthesisConfig(length_scale=0.95, noise_scale=0.8, noise_w_scale=0.9),
            "excited": SynthesisConfig(length_scale=0.9, noise_scale=0.9, noise_w_scale=1.0),
            "calm": SynthesisConfig(length_scale=1.15, noise_scale=0.5, noise_w_scale=0.7),
            "sad": SynthesisConfig(length_scale=1.2, noise_scale=0.5, noise_w_scale=0.6),
            "serious": SynthesisConfig(length_scale=1.0, noise_scale=0.4, noise_w_scale=0.5),
            "curious": SynthesisConfig(length_scale=1.05, noise_scale=0.75, noise_w_scale=0.85),
        }
        syn_config = presets.get(emotion, presets["neutral"])

        # Synthesize
        audio_chunks = []
        sample_rate = self.tts.voice.config.sample_rate

        for chunk in self.tts.voice.synthesize(text, syn_config=syn_config):
            audio_chunks.append(chunk.audio_float_array)
            sample_rate = chunk.sample_rate

        if audio_chunks:
            import sounddevice as sd
            audio = np.concatenate(audio_chunks)
            print(f"\n[Reaction] {text}")
            sd.play(audio, samplerate=sample_rate)
            sd.wait()