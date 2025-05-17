import os
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Config
with open("config.json", "r") as f:
    config = json.load(f)

SAMPLE_RATE = config["sr"] 
N_FFT = config["n_fft"]
HOP_LENGTH = config["hop_length"]
WIDTH_LENGTH = config["win_length"]
N_MELS = config["n_mels"]    # 128 for Griffin-Lim, 80 for HIFI-GAN and and melGAN,
N_ITER = config["n_iter"]   # number of iterations, only if using Griffin-Lim (128 gets good quality without taking ages
GLOBAL_REF = config["global_ref"]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MEL_DIR = os.path.join(ROOT_DIR, "res", "mel_specs")
DATASET_DIR = os.path.join(ROOT_DIR, "res", "datasets")
AUDIO_DIR = os.path.join(ROOT_DIR, "res", "audio")
TRACKS_DIR = os.path.join(AUDIO_DIR, "tracks")
TRANSITIONS_DIR = os.path.join(AUDIO_DIR, "transitions")

PRE_TRANSITION_DURATION = 0

os.makedirs(MEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def extract_segment(y, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]

def audio_to_mel(y):
    mel_power = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_power, ref=GLOBAL_REF)     # The reference here is crucial
    return mel_db 

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

def process_row(row):
    # Debug
    print(f"Processing transition '{row['transition_filename']}'")

    # Convert time formats to seconds
    times = {
        'start_time_a': time_to_seconds(row['start_time_a']),
        'end_time_a': time_to_seconds(row['end_time_a']),
        'start_time_b': time_to_seconds(row['start_time_b']),
        'end_time_b': time_to_seconds(row['end_time_b'])
    }
    
    # Process Track A
    track_a_path = os.path.join(TRACKS_DIR, row['track_a_filename'])
    y_a, _ = librosa.load(track_a_path, sr=SAMPLE_RATE)
    mel_a = audio_to_mel(y_a)
    
    # Process Track B
    track_b_path = os.path.join(TRACKS_DIR, row['track_b_filename'])
    y_b, _ = librosa.load(track_b_path, sr=SAMPLE_RATE)
    mel_b = audio_to_mel(y_b)
    
    # Process Transition
    transition_path = os.path.join(TRANSITIONS_DIR,  row['transition_filename'])
    y_tr, _ = librosa.load(transition_path, sr=SAMPLE_RATE)
    mel_tr = audio_to_mel(y_tr)
    
    # Save MEL spectrograms
    mel_a_path = os.path.join( MEL_DIR, f"mel_track_{Path(row['track_a_filename']).stem}.npy" )
    mel_b_path = os.path.join( MEL_DIR, f"mel_track_{Path(row['track_b_filename']).stem}.npy" )
    mel_tr_path = os.path.join( MEL_DIR, f"mel_transition_{Path(row['transition_filename']).stem}.npy" )
    
    for path_db, mel_db in [
        (mel_a_path, mel_a),
        (mel_b_path, mel_b), 
        (mel_tr_path, mel_tr)
        ]:

        if not os.path.exists(path_db):
            save_mel(str(path_db), mel_db) 
    
    return pd.Series({
        'track_a_path': str(row['track_a_filename']),
        'track_b_path': str(row['track_b_filename']),
        'transition_path': str(row['transition_filename']),
        'start_time_a': times['start_time_a'],
        'end_time_a': times['end_time_a'],
        'start_time_b': times['start_time_b'],
        'end_time_b': times['end_time_b'],
        'key_a': row['key_a'],
        'bpm_a': row['bpm_a'],
        'key_b': row['key_b'],
        'bpm_b': row['bpm_b'],
        'transition_type': row['transition_type'],
        'mel_a_path': str(os.path.basename(mel_a_path)),
        'mel_b_path': str(os.path.basename(mel_b_path)),
        'mel_tr_path': str(os.path.basename(mel_tr_path)),
    })

def process_dataset(csv_path, output_csv_path):
    """
    | --- Expected Dataset Columns ----------------------------------------------------|
    | track_a_name | track_a_filename | track_b_name | track_b_filename | start_time_a | 
    | end_time_a | start_time_b | end_time_b | transition_filename | transition_type   |
    | transition_duration | transition_bpm | key_a | bpm_a | key_b | bpm_b             |
    |__________________________________________________________________________________|   
    """

    df = pd.read_csv(csv_path)
    processed = df.apply(process_row, axis=1)
    processed.to_csv(output_csv_path, index=False)
    print(f"✅ Dataset procesado guardado en {output_csv_path}")

if __name__ == "__main__":
    csv_path = os.path.join(DATASET_DIR, "transition_dataset_raw.csv")
    output_path = os.path.join(DATASET_DIR, "transition_dataset_processed_III.csv")
    process_dataset(csv_path, output_path)