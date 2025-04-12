import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from hifi_gan.models import utils, Generator
import torch

# === 1. Load the audio ===

TRACKS_PATH = os.path.join(ROOT_DIR, "tracks")
FILE_INPUT_NAME = "test_transition.mp3"
FILE_OUTPUT_NAME = "reconstructed_transition.mp3"
INPUT_PATH = os.path.join(TRACKS_PATH, FILE_INPUT_NAME)
OUTPUT_PATH = os.path.join(TRACKS_PATH, FILE_OUTPUT_NAME)

# Weights for HIFI-GAN
HIFIGAN_WEIGHTS = os.path.join(ROOT_DIR, "hifi_gan", "weights", "LJ_V3", "generator_v3")

# Method: 0 = Griffin-Lim; 1 = HIFI-GAN
RECONSTRUCTION_METHOD = 1 

# === Mel Spectrogram parameters ===

sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 128  # Matches HiFi-GAN (using 80 previously)

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
    plt.show()

plot_spectrogram(mel, False)

# === 4. Convert back to audio ===

if RECONSTRUCTION_METHOD == 1:
    generator = Generator()
    checkpoint = torch.load(HIFIGAN_WEIGHTS)  # Example
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    # Convert mel to tensor
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # [1, n_mels, T]
    mel_tensor = mel_tensor.cuda()  # if using GPU
    with torch.no_grad():
        reconstructed_audio = generator(mel_tensor).squeeze().cpu().numpy()

else:
    # Best quality with high number of 'n_iter', but takes a long time to generate
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        mel, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=1024
    )

# === 5. Save Reconstructed Audio ===
sf.write(OUTPUT_PATH, reconstructed_audio, sr)
print("✅ Audio reconstructed and saved to 'reconstructed_audio.wav'")