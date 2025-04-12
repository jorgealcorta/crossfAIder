import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# === 1. Load the audio ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
TRACKS_PATH = os.path.join(ROOT_DIR, "tracks")

FILE_INPUT_NAME = "test_transition.mp3"
FILE_OUTPUT_NAME = "reconstructed_transition.mp3"
INPUT_PATH = os.path.join(TRACKS_PATH, FILE_INPUT_NAME)
OUTPUT_PATH = os.path.join(TRACKS_PATH, FILE_OUTPUT_NAME)

y, sr = librosa.load(INPUT_PATH, sr=22050)
print(f"Loaded {INPUT_PATH}, duration: {len(y)/sr:.2f}s")

# === 2. Convert to Mel-Spectrogram ===
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

# === 3. Plot the Spectrogram ===
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.show()

# === 4. Convert back to audio ===
# Step 1: dB to power
mel_reversed = librosa.db_to_power(mel_db)

# Step 2: Reconstruct waveform using Griffin-Lim
reconstructed_audio = librosa.feature.inverse.mel_to_audio(
    mel_reversed, sr=sr, n_fft=1024, hop_length=256, n_iter=32
)

# === 5. Save Reconstructed Audio ===
sf.write(OUTPUT_PATH, reconstructed_audio, sr)
print("✅ Audio reconstructed and saved to 'reconstructed_audio.wav'")