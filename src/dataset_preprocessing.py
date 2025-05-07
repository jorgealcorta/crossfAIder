import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Config
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
PRE_TRANSITION_DURATION = 0  # seconds before transition start

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MEL_DIR = os.path.join(ROOT_DIR, "res", "mel_specs")
DATASET_DIR = os.path.join(ROOT_DIR, "res", "datasets")
AUDIO_DIR = os.path.join(ROOT_DIR, "res", "audio")
TRACKS_DIR = os.path.join(AUDIO_DIR, "tracks")
TRANSITIONS_DIR = os.path.join(AUDIO_DIR, "transitions")

os.makedirs(MEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def extract_segment(y, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]

def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def save_mel(mel, output_path):
    np.save(output_path, mel)

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
    segment_a = extract_segment(y_a, SAMPLE_RATE, 
                               max(times['start_time_a'] - PRE_TRANSITION_DURATION, 0),
                               times['end_time_a'])
    mel_a = audio_to_mel(segment_a, SAMPLE_RATE)
    
    # Process Track B
    track_b_path = os.path.join(TRACKS_DIR, row['track_b_filename'])
    y_b, _ = librosa.load(track_b_path, sr=SAMPLE_RATE)
    segment_b = extract_segment(y_b, SAMPLE_RATE,
                               max(times['start_time_b'] - PRE_TRANSITION_DURATION, 0),
                               times['end_time_b'])
    mel_b = audio_to_mel(segment_b, SAMPLE_RATE)
    
    # Process Transition
    transition_path = os.path.join(TRANSITIONS_DIR,  row['transition_filename'])
    y_tr, _ = librosa.load(transition_path, sr=SAMPLE_RATE)
    mel_tr = audio_to_mel(y_tr, SAMPLE_RATE)
    
    # Save MEL spectrograms
    mel_a_path = os.path.join( MEL_DIR, f"mel_a_{Path(row['track_a_filename']).stem}.npy" )
    mel_b_path = os.path.join( MEL_DIR, f"mel_b_{Path(row['track_b_filename']).stem}.npy" )
    mel_tr_path = os.path.join( MEL_DIR, f"mel_tr_{Path(row['transition_filename']).stem}.npy" )
    
    for path, mel in [(mel_a_path, mel_a), (mel_b_path, mel_b), (mel_tr_path, mel_tr)]:
        if not os.path.exists(path):
            save_mel(mel, str(path))
    
    return pd.Series({
        'track_a_path': str(track_a_path),
        'track_b_path': str(track_b_path),
        'transition_path': str(transition_path),
        'start_time_a': times['start_time_a'],
        'end_time_a': times['end_time_a'],
        'start_time_b': times['start_time_b'],
        'end_time_b': times['end_time_b'],
        'key_a': row['key_a'],
        'bpm_a': row['bpm_a'],
        'key_b': row['key_b'],
        'bpm_b': row['bpm_b'],
        'transition_type': row['transition_type'],
        'mel_fragment_a_path': str(mel_a_path),
        'mel_fragment_b_path': str(mel_b_path),
        'mel_transition_path': str(mel_tr_path)
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
    output_path = os.path.join(DATASET_DIR, "transition_dataset_processed.csv")
    process_dataset(csv_path, output_path)