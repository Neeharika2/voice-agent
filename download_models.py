"""Download Piper TTS voice models automatically."""
import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

MODEL_URL = f"{HF_BASE}/en_US-lessac-medium.onnx"
CONFIG_URL = f"{HF_BASE}/en_US-lessac-medium.onnx.json"

MODEL_FILENAME = "en_US-lessac-medium.onnx"
CONFIG_FILENAME = "en_US-lessac-medium.onnx.json"


def download_file(url, dest):
    """Download a file with progress indication."""
    print(f"Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest}")


def download_model():
    """Download the Piper voice model files."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    config_path = os.path.join(MODELS_DIR, CONFIG_FILENAME)

    if os.path.exists(model_path) and os.path.exists(config_path):
        print(f"Model already exists: {model_path}")
        return

    print(f"Downloading Piper model from HuggingFace...")

    try:
        if not os.path.exists(model_path):
            download_file(MODEL_URL, model_path)
        if not os.path.exists(config_path):
            download_file(CONFIG_URL, config_path)
        print("Model ready!")
    except Exception as e:
        print(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    download_model()