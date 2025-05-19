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

AUDIO_DIR = os.path.join(RES_DIR, "audio")
TRACKS_DIR = os.path.join(AUDIO_DIR, "tracks")

MODEL_PATH = os.path.join(
    MODELS_DIR, 
    "extendedData_WGAN-GP", 
    "G_epoch_100.pth"
    )
DATASET_PATH = os.path.join(DATASET_DIR, "transition_dataset_processed_IV.csv")
TRANSITIONS_NAME = "extendedDataGAN_transition_100epoch"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Dataset (to get mean/std/max_time) ---
train_dataset = MelTransitionDataset(
    dataset_path=DATASET_PATH,
    config_path=CONFIG_PATH,
    spectrogram_path=MEL_DIR,
    pad_strat="silence"
)
# Divisible over 8 length - important for architecture
# picked once non divisible real len was known - which was 4305
train_dataset.max_time = 4312

mean = train_dataset.mean
std = train_dataset.std
max_time = train_dataset.max_time

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

hop_length = config["hop_length"]
sr = config["sr"]
min_db = config["min_db"]
global_ref = config["global_ref"]

# --- Load Trained Generator ---
G = UNetGenerator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

# --- Helper Functions ---
def audio_to_mel(path):
    y, _ = librosa.load(path, sr=config["sr"])
    mel_power = librosa.feature.melspectrogram(
        y = y, 
        sr = config["sr"], 
        n_fft = config["n_fft"], 
        hop_length = config["hop_length"], 
        n_mels = config["n_mels"]
    )
    mel_db = librosa.power_to_db(mel_power, ref=config["global_ref"], top_db = 80)
    
    return np.clip(mel_db, a_min=-80, a_max=0)

def trim_spectrogram(spec, start_time, end_time):
    start_frame = int(start_time * sr / hop_length)
    end_frame = int(end_time * sr / hop_length)

    # Ensure we don't exceed the spectrogram's length
    end_frame = min(end_frame, spec.shape[1])

    return spec[:, start_frame:end_frame]

def pad_with_silence(spec, target_time):
    if spec.shape[1] < target_time:
        pad_width = ((0, 0), (0, target_time - spec.shape[1]))
        return np.pad(spec, pad_width, mode='constant', constant_values=min_db)
    return spec

def pad_with_reflect(spectrogram, target_time):
        """Pad the spectrogram with it's own reflection"""
        current_time = spectrogram.shape[-1]

        if current_time < target_time:
            # Pad symmetrically with reflection
            pad_total = target_time - current_time
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            return np.pad(
                spectrogram,
                ((0, 0), (pad_left, pad_right)),
                mode='reflect'
            )
        else:
            # Trim to exact target_time
            return spectrogram[:, :target_time]

def outGAN_to_audio(generated_spec):
    # Denormalize and Revert dB Scale
    generated_denorm = (generated_spec * std) + mean

    # Convert Mel to Audio (replace with your inversion logic)
    S = librosa.db_to_power(generated_denorm, ref=global_ref)  # Revert to power scale
    return librosa.feature.inverse.mel_to_audio(
        S,
        sr=sr,
        n_fft=config["n_fft"],
        hop_length=hop_length,
        win_length=config["win_length"],
        n_iter=config["n_iter"]
    )

def dataframe_inference(test_indices):

    full_df = pd.read_csv(DATASET_PATH)
    test_df = full_df.iloc[test_indices]

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

        transition = outGAN_to_audio(generated)
        
        out_path = os.path.join(INFERENCE_DIR, f"{TRANSITIONS_NAME}_{idx}.wav")
        sf.write(out_path, audio, sr)
        print(f"Saved transition in '{out_path}'")

def audio_inference(track_a_path, start_a, end_a, track_b_path, start_b, end_b):
    path_a = os.path.join(TRACKS_DIR, track_a_path)
    path_b = os.path.join(TRACKS_DIR, track_b_path)

    mel_a_raw = audio_to_mel(path_a) 
    mel_b_raw = audio_to_mel(path_b) 

    mel_a_trimmed = trim_spectrogram(mel_a_raw, start_a, end_a)
    mel_b_trimmed = trim_spectrogram(mel_b_raw, start_b, end_b)
    
    mel_a_padded = pad_with_silence(mel_a_trimmed, max_time)
    mel_b_padded = pad_with_silence(mel_b_trimmed, max_time)

    mel_a_norm = (mel_a_padded - mean) / std
    mel_b_norm = (mel_b_padded - mean) / std

    input_pair = torch.tensor(np.stack([mel_a_norm, mel_b_norm])).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        generated = G(input_pair).squeeze().cpu().numpy()

    transition = outGAN_to_audio(generated)

    out_path = os.path.join(INFERENCE_DIR, f"{TRANSITIONS_NAME}.wav")
    sf.write(out_path, transition, sr)
    print(f"Saved transition in '{out_path}'")

if __name__ == "__main__":
    # TEST_INDICES = [0, 1]  # First two examples for testing
    # dataframe_inference(TEST_INDICES)
    filename_a = "Guy Contact - 90 Mile Straight (Nullarbor Mix) [BSR034].mp3"
    filename_b = "Mutfak Robotu.mp3"

    audio_inference(
        track_a_path = filename_a,
        track_b_path = filename_b,
        start_a=242,
        end_a=275,
        start_b=15,
        end_b=30
    )