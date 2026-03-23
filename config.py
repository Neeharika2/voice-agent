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

# TTS Settings (Piper only)
TTS_SYNTH_TIMEOUT_S = 20.0

# Preferred: English model
PIPER_PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "en_US-lessac-medium.onnx")
PIPER_PRIMARY_CONFIG_PATH = os.path.join(BASE_DIR, "models", "en_US-lessac-medium.onnx.json")

# Fallback: Hindi model for Indian-native feel
PIPER_FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "hi_IN-rohan-medium.onnx")
PIPER_FALLBACK_CONFIG_PATH = os.path.join(BASE_DIR, "models", "hi_IN-rohan-medium.onnx.json")

# Backward-compatible aliases (used by older code paths)
PIPER_MODEL_PATH = PIPER_PRIMARY_MODEL_PATH
PIPER_MODEL_CONFIG_PATH = PIPER_PRIMARY_CONFIG_PATH

PIPER_WARMUP_TEXT = "hi"
PIPER_SPEAKER_ID = None

TTS_STREAM_MIN_CHARS = 20  # Lower -> earlier first audio, higher -> smoother chunks
TTS_STREAM_MAX_CHARS = 90  # Force flush if no punctuation appears

# Observability
LATENCY_LOGS = True

# Vision Settings
VISION_INTERVAL = 10 # Seconds between scene analysis

# LLM Settings
LLM_MODEL = "gemini-2.5-flash-lite"  # Fast model for voice
LLM_MAX_TOKENS = 512  # Keep responses short for voice
