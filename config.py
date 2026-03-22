import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# STT Settings (Vosk)
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "model")  # Path to the Vosk model directory
SAMPLE_RATE = 16000
STT_BLOCKSIZE = 800  # 50ms chunks for low latency
STT_SILENCE_TIMEOUT = 0.5  # Seconds of silence before finalizing (reduced for faster response)
# For sub-100ms latency, use the small model: vosk-model-small-en-us-0.15 (~50MB)
# Download from: https://alphacephei.com/vosk/models

# TTS Settings
# Modes: fast/auto (prefer local), local (Piper only), quality/cloud (Edge only)
TTS_MODE = "fast"
TTS_LOCAL_ENABLED = True
TTS_CLOUD_ENABLED = True
TTS_FALLBACK_TO_CLOUD = True
TTS_SYNTH_TIMEOUT_S = 20.0

# Edge TTS (cloud quality)
EDGE_TTS_VOICE = "en-IN-NeerjaNeural"
# Other voices: en-US-JennyNeural, en-US-GuyNeural, en-GB-SoniaNeural

# Piper local (fast mode) - set both to enable local synthesis
PIPER_EXECUTABLE = "piper"
PIPER_MODEL_PATH = None  # Example: r"C:\\models\\en_US-lessac-medium.onnx"
PIPER_MODEL_CONFIG_PATH = None  # Optional config json path

TTS_PARALLEL_WORKERS = 3  # Concurrent synthesis threads
TTS_STREAM_MIN_CHARS = 20  # Lower -> earlier first audio, higher -> smoother chunks
TTS_STREAM_MAX_CHARS = 90  # Force flush if no punctuation appears

# Observability
LATENCY_LOGS = True

# Vision Settings
VISION_INTERVAL = 10 # Seconds between scene analysis

# LLM Settings
LLM_MODEL = "gemini-2.5-flash-lite"  # Fast model for voice
LLM_MAX_TOKENS = 512  # Keep responses short for voice
