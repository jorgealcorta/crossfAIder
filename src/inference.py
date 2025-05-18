import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from models.generator import UNetGenerator
from models.dataset import MelTransitionDataset
import json
import os

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RES_DIR = os.path.join(ROOT_DIR, "res")
MEL_DIR = os.path.join(RES_DIR, "mel_specs")
DATASET_DIR = os.path.join(RES_DIR, "datasets")
CONFIG_PATH = os.path.join(RES_DIR, "config", "config.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "improvedGAN100epoch", "G_epoch_100.pth")
DATASET_PATH = os.path.join(DATASET_DIR, "transition_dataset_processed_III.csv"),
TRANSITIONS_NAME = "enhancedGAN_transition"
TEST_INDICES = [0, 1]  # First two examples for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Dataset (to get mean/std/max_time) ---
train_dataset = MelTransitionDataset(
    dataset_path=DATASET_PATH,
    config_path=CONFIG_PATH,
    spectrogram_path=MEL_DIR
)
mean = train_dataset.mean
std = train_dataset.std
max_time = train_dataset.max_time

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

hop_length = config["hop_length"]
sr = config["sr"]
min_db = config["min_db"]
global_ref = config["global_ref"]

# --- Load Test Examples ---
full_df = pd.read_csv(os.path.join(DATASET_DIR, "transition_dataset_processed_III.csv"))
test_df = full_df.iloc[TEST_INDICES]

# --- Load Trained Generator ---
G = UNetGenerator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

# --- Helper Functions ---
def trim_spectrogram(spec, start_time, end_time):
    start_frame = int(start_time * sr / hop_length)
    end_frame = int(end_time * sr / hop_length)
    return spec[:, start_frame:end_frame]

def pad_with_min(spec, target_time):
    if spec.shape[1] < target_time:
        pad_width = ((0, 0), (0, target_time - spec.shape[1]))
        return np.pad(spec, pad_width, mode='constant', constant_values=min_db)
    return spec

# --- Inference Loop ---
for idx, row in test_df.iterrows():
    # Process Mel-A
    mel_a = np.load(os.path.join(MEL_DIR, row["mel_a_path"]))
    mel_a_trimmed = trim_spectrogram(mel_a, row["start_time_a"], row["end_time_a"])
    mel_a_padded = pad_with_min(mel_a_trimmed, max_time)
    mel_a_norm = (mel_a_padded - mean) / std

    # Process Mel-B
    mel_b = np.load(os.path.join(MEL_DIR, row["mel_b_path"]))
    mel_b_trimmed = trim_spectrogram(mel_b, row["start_time_b"], row["end_time_b"])
    mel_b_padded = pad_with_min(mel_b_trimmed, max_time)
    mel_b_norm = (mel_b_padded - mean) / std

    # Stack and Generate
    input_pair = torch.tensor(np.stack([mel_a_norm, mel_b_norm])).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        generated = G(input_pair).squeeze().cpu().numpy()

    # Denormalize and Revert dB Scale
    generated_denorm = (generated * std) + mean

    # Convert Mel to Audio (replace with your inversion logic)
    # Example using Griffin-Lim (adjust parameters as needed):
    S = librosa.db_to_power(generated_denorm, ref=global_ref)  # Revert to power scale
    audio = librosa.feature.inverse.mel_to_audio(
        S,
        sr=sr,
        n_fft=config["n_fft"],
        hop_length=hop_length,
        win_length=config["win_length"],
        n_iter=config["n_iter"]
    )
    out_path = os.path.join(INFERENCE_DIR, f"{TRANSITIONS_NAME}_{idx}.wav")
    sf.write(out_path, audio, sr)
    print(f"Saved transition in '{out_path}'")