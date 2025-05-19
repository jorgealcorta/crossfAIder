import os
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MEL_DIR = os.path.join(ROOT_DIR, "res", "mel_specs")
DATASET_DIR = os.path.join(ROOT_DIR, "res", "datasets")
AUDIO_DIR = os.path.join(ROOT_DIR, "res", "audio")
TRACKS_DIR = os.path.join(AUDIO_DIR, "tracks")
TRANSITIONS_DIR = os.path.join(AUDIO_DIR, "transitions")

PRE_TRANSITION_DURATION = 0

AUGMENTATIONS = [
    {'name': 'original', 'func': None},  # Base case
    {'name': 'noise', 'func': lambda y: add_noise(y, 0.003)},
    {'name': 'pitch_up', 'func': lambda y: pitch_shift(y, SAMPLE_RATE, 0.5)},
    {'name': 'pitch_down', 'func': lambda y: pitch_shift(y, SAMPLE_RATE, -0.5)},
    {'name': 'gain_+2db', 'func': lambda y: adjust_gain(y, 2)},
    {'name': 'gain_-2db', 'func': lambda y: adjust_gain(y, -2)},
]

os.makedirs(MEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Config
with open(os.path.join(ROOT_DIR, "res", "config", "config.json"), "r") as f:
    config = json.load(f)

SAMPLE_RATE = config["sr"] 
N_FFT = config["n_fft"]
HOP_LENGTH = config["hop_length"]
WIDTH_LENGTH = config["win_length"]
N_MELS = config["n_mels"]    # 128 for Griffin-Lim, 80 for HIFI-GAN and and melGAN,
N_ITER = config["n_iter"]   # number of iterations, only if using Griffin-Lim (128 gets good quality without taking ages
GLOBAL_REF = config["global_ref"]


def extract_segment(y, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]

def audio_to_mel(y):
    mel_power = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_power, ref=GLOBAL_REF, top_db=80)     # The reference here is crucial
    return np.clip(mel_db, a_min=-80, a_max=0)  # Hard clipping to [-80, 0] dB

def save_mel(mel_path, mel):
    np.save(mel_path, mel)

def get_global_ref(df):

    all_max_powers = []
    for _, row in df.iterrows():
        for path in [
            os.path.join(TRACKS_DIR, row['track_a_filename']), 
            os.path.join(TRACKS_DIR, row['track_b_filename']), 
            os.path.join(TRANSITIONS_DIR, row['transition_filename'])
        ]:
            y, _ = librosa.load(path)
            mel_power = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
            all_max_powers.append(np.max(mel_power))

    # Global reference (use 99.9% percentile to avoid outliers)
    return np.percentile(all_max_powers, 99.9) 

def time_to_seconds(time_str):
    """Convert MM:SS.ss or HH:MM:SS.ss format to seconds"""
    parts = list(map(float, time_str.replace(',', '.').split(':')))
    if len(parts) == 2:  # MM:SS.ss
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:  # HH:MM:SS.ss
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0

# --- Data augmentation functions ---------------------------------------------------- |
def add_noise(y, noise_level=0.005):
    noise = np.random.normal(0, noise_level, len(y))
    augmented = y + noise
    return np.clip(augmented, -1.0, 1.0)

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def adjust_gain(y, db):
    augmented = y * (10 ** (db / 20))
    return np.clip(augmented, -1.0, 1.0)

def process_row(row):
    all_rows = []
    
    for aug in AUGMENTATIONS:
        # Process audio with augmentation
        suffix = f"_{aug['name']}" if aug['name'] != 'original' else ''
        
        # Load and augment audio
        def process_audio(path, dir_type='tracks'):
            full_path = os.path.join(TRACKS_DIR if dir_type == 'tracks' else TRANSITIONS_DIR, path)
            y, _ = librosa.load(full_path, sr=SAMPLE_RATE)
            if aug['func']:
                y = aug['func'](y)
            return audio_to_mel(y)

        try:
            mel_a = process_audio(row['track_a_filename'])
            mel_b = process_audio(row['track_b_filename'])
            mel_tr = process_audio(row['transition_filename'], 'transitions')

            # Save MELs
            mel_a_path = os.path.join(MEL_DIR, f"mel_track_{Path(row['track_a_filename']).stem}{suffix}.npy")
            mel_b_path = os.path.join(MEL_DIR, f"mel_track_{Path(row['track_b_filename']).stem}{suffix}.npy")
            mel_tr_path = os.path.join(MEL_DIR, f"mel_transition_{Path(row['transition_filename']).stem}{suffix}.npy")

            for path, mel in [(mel_a_path, mel_a), (mel_b_path, mel_b), (mel_tr_path, mel_tr)]:
                if not os.path.exists(path):
                    save_mel(path, mel)

            # Create new row entry
            new_row = {
                'track_a_path': row['track_a_filename'],
                'track_b_path': row['track_b_filename'],
                'transition_path': row['transition_filename'],
                'start_time_a': time_to_seconds(row['start_time_a']),
                'end_time_a': time_to_seconds(row['end_time_a']),
                'start_time_b': time_to_seconds(row['start_time_b']),
                'end_time_b': time_to_seconds(row['end_time_b']),
                'key_a': row['key_a'],
                'bpm_a': row['bpm_a'],
                'key_b': row['key_b'],
                'bpm_b': row['bpm_b'],
                'transition_type': row['transition_type'],
                'mel_a_path': os.path.basename(mel_a_path),
                'mel_b_path': os.path.basename(mel_b_path),
                'mel_tr_path': os.path.basename(mel_tr_path),
            }
            all_rows.append(pd.Series(new_row))
            
        except Exception as e:
            print(f"Error processing {row['transition_filename']} with {aug['name']}: {str(e)}")
    
    return all_rows

def process_dataset(csv_path, output_csv_path):
    """
    | --- Expected Dataset Columns ----------------------------------------------------|
    | track_a_name | track_a_filename | track_b_name | track_b_filename | start_time_a | 
    | end_time_a | start_time_b | end_time_b | transition_filename | transition_type   |
    | transition_duration | transition_bpm | key_a | bpm_a | key_b | bpm_b             |
    |__________________________________________________________________________________|   
    """

    df = pd.read_csv(csv_path)
    df = df[df['transition_type'] == 'EQ-out']  # Filter non-EQ-out transitions
    
    processed_rows = []
    for _, row in df.iterrows():
        processed_rows.extend(process_row(row))
    
    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(output_csv_path, index=False)
    print(f"✅ Augmented dataset saved to {output_csv_path} (Total samples: {len(processed_df)} vs {len(df)} original samples)") 

if __name__ == "__main__":
    csv_path = os.path.join(DATASET_DIR, "transition_dataset_raw.csv")
    output_path = os.path.join(DATASET_DIR, "transition_dataset_processed_IV.csv")
    process_dataset(csv_path, output_path)