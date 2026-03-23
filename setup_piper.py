import os
import numpy as np
from piper import PiperVoice
import sounddevice as sd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PRIMARY_MODEL = os.path.join(MODELS_DIR, "en_US-lessac-medium.onnx")
PRIMARY_CONFIG = os.path.join(MODELS_DIR, "en_US-lessac-medium.onnx.json")
FALLBACK_MODEL = os.path.join(MODELS_DIR, "hi_IN-rohan-medium.onnx")
FALLBACK_CONFIG = os.path.join(MODELS_DIR, "hi_IN-rohan-medium.onnx.json")


def pick_model():
    if os.path.exists(PRIMARY_MODEL):
        config_path = PRIMARY_CONFIG if os.path.exists(PRIMARY_CONFIG) else None
        return PRIMARY_MODEL, config_path

    if os.path.exists(FALLBACK_MODEL):
        config_path = FALLBACK_CONFIG if os.path.exists(FALLBACK_CONFIG) else None
        return FALLBACK_MODEL, config_path

    raise FileNotFoundError(
        "No Piper model found. Add model files to models/: "
        "en_US-lessac-medium.onnx (+ .json) or hi_IN-rohan-medium.onnx (+ .json)."
    )


def play_chunked(voice, chunks):
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        # synthesize yields AudioChunk objects
        audio_chunks = []
        sample_rate = voice.config.sample_rate
        for audio_chunk in voice.synthesize(text):
            audio_chunks.append(audio_chunk.audio_float_array)
            sample_rate = audio_chunk.sample_rate
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()


if __name__ == "__main__":
    model_path, config_path = pick_model()
    print(f"Loading Piper model: {model_path}")
    voice = PiperVoice.load(model_path, config_path=config_path)

    # Warm-up first call to avoid initial latency on the first real sentence.
    voice.synthesize("hi")

    chunks = [
        "Hello.",
        "How are you?",
        "This is your assistant.",
    ]
    print("Speaking test chunks...")
    play_chunked(voice, chunks)
    print("Done.")
