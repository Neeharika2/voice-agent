import os
import zipfile
import requests
from tqdm import tqdm

def download_model(url, save_path):
    print(f"Downloading model from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_model(zip_path, extract_to):
    print(f"Extracting model to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Vosk models usually extract into a folder named after the zip
    # We want the contents to be in the 'model' folder directly or adjust config
    extracted_folder = zip_path.replace('.zip', '')
    if os.path.exists(extracted_folder):
        print(f"Model extracted to {extracted_folder}")
        # If 'model' exists, we might want to rename or move
        if os.path.exists('model') and os.path.isdir('model'):
             import shutil
             print("Cleaning up existing 'model' directory...")
             shutil.rmtree('model')
        
        os.rename(extracted_folder, 'model')
        print("Moved model to 'model/' directory.")

if __name__ == "__main__":
    model_url = "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"
    zip_name = "vosk-model-en-in-0.5.zip"
    
    try:
        download_model(model_url, zip_name)
        extract_model(zip_name, ".")
        os.remove(zip_name)
        print("Vosk model setup complete!")
    except Exception as e:
        print(f"Error during setup: {e}")
