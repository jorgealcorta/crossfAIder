import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from hifi_gan.models import Generator
from types import SimpleNamespace
import torch
import json

# === 1. Load the audio ===

TRACKS_PATH = os.path.join(ROOT_DIR, "tracks")
OUTPUTS_PATH = os.path.join(ROOT_DIR, "outputs")
FILE_INPUT_NAME = "test_transition.mp3"
FILE_OUTPUT_NAME = "reconstructed_transition.mp3"
INPUT_PATH = os.path.join(TRACKS_PATH, FILE_INPUT_NAME)
OUTPUT_PATH = os.path.join(OUTPUTS_PATH, FILE_OUTPUT_NAME)

# Weights and configuration for HIFI-GAN
HIFIGAN_WEIGHTS = os.path.join(ROOT_DIR, "hifi_gan", "weights", "LJ_V3", "generator_v3")
HIFIGAN_CONFIG = os.path.join(ROOT_DIR, "hifi_gan", "weights", "LJ_V3", "config.json")

# Method: 0 = Griffin-Lim; 1 = HIFI-GAN; 2 = melGAN inverse vocoder
RECONSTRUCTION_METHOD = 0

# === Mel Spectrogram parameters ===

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 128    # 128 for Griffin-Lim, 80 for HIFI-GAN and and melGAN,
n_iter = 1024   # number of iterations, only if using Griffin-Lim

# === 2. Convert to Mel-Spectrogram ===

y, _ = librosa.load(INPUT_PATH, sr=sr)
print(f"Loaded {INPUT_PATH}, duration: {len(y)/sr:.2f}s")

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# === 3. Plot the Spectrogram ===
def plot_spectrogram(mel_spec, save : bool = True):
    
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    if save:
        plt.savefig("mel_spectrogram.png")
    # plt.show()

plot_spectrogram(mel, False)

# === 4. Convert back to audio ===

if RECONSTRUCTION_METHOD == 2:
    # First reconstruction method: MELGAN INVERSE VOCODER (github: https://github.com/seungwonpark/melgan)
    vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
    vocoder.eval()

    # Convert mel (NumPy array) to PyTorch tensor
    mel = torch.from_numpy(mel).float()

    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        mel = mel.cuda()

    # Ensure mel has batch dimension: [1, n_mels, timesteps]
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    with torch.no_grad():
        reconstructed_audio = vocoder.inference(mel).squeeze().cpu().numpy()

elif RECONSTRUCTION_METHOD == 1:
    # Load the HiFi-GAN configuration ------------------------------------------------------------------
    # The HIFI-GAN repository must be in the project's root dir (clone repo -> https://github.com/jik876/hifi-gan)
    # The weights of the model can also be downloaded following a link provided in the github page.
    with open(HIFIGAN_CONFIG, 'r') as f:
        config_dict = json.load(f)

    # Convert the dictionary to an object with attributes
    config = SimpleNamespace(**config_dict)

    # Initialize Generator with the configuration
    generator = Generator(h=config)
    checkpoint = torch.load(HIFIGAN_WEIGHTS)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    # Convert mel to tensor
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # Shape: [1, n_mels, T]

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    mel_tensor = mel_tensor.to(device)

    # Generate audio
    with torch.no_grad():
        reconstructed_audio = generator(mel_tensor).squeeze().cpu().numpy()

else:
    # Base reconstruction method: Griffin-Lim
    # mel = mel / np.max(mel) # Normalized mel - might work better, not using rn

    reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        mel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter
    )

# === 5. Save Reconstructed Audio ===
sf.write(OUTPUT_PATH, reconstructed_audio, sr)
print(f"✅ Audio reconstructed and saved to '{OUTPUT_PATH}'")