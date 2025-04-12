import librosa
import numpy as np
import pandas as pd
import os

# Config
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
PRE_TRANSITION_DURATION = 10  # seconds before transition start

OUTPUT_DIR = "mel_specs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_segment(y, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]

def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def save_mel(mel, output_path):
    np.save(output_path, mel)

def process_row(row, index):
    # Load Track A
    y_a, _ = librosa.load(row['track_a_path'], sr=SAMPLE_RATE)
    start_a = max(row['track_a_start'] - PRE_TRANSITION_DURATION, 0)
    segment_a = extract_segment(y_a, SAMPLE_RATE, start_a, row['track_a_end'])
    mel_a = audio_to_mel(segment_a, SAMPLE_RATE)

    # Load Track B
    y_b, _ = librosa.load(row['track_b_path'], sr=SAMPLE_RATE)
    start_b = max(row['track_b_start'] - PRE_TRANSITION_DURATION, 0)
    segment_b = extract_segment(y_b, SAMPLE_RATE, start_b, row['track_b_end'])
    mel_b = audio_to_mel(segment_b, SAMPLE_RATE)

    # Load Transition
    y_tr, _ = librosa.load(row['transition_path'], sr=SAMPLE_RATE)
    mel_tr = audio_to_mel(y_tr, SAMPLE_RATE)

    # Save all mel spectrograms
    mel_a_path = f"{OUTPUT_DIR}/mel_a_{index}.npy"
    mel_b_path = f"{OUTPUT_DIR}/mel_b_{index}.npy"
    mel_tr_path = f"{OUTPUT_DIR}/mel_transition_{index}.npy"
    
    save_mel(mel_a, mel_a_path)
    save_mel(mel_b, mel_b_path)
    save_mel(mel_tr, mel_tr_path)

    return pd.Series({
        'mel_a_path': mel_a_path,
        'mel_b_path': mel_b_path,
        'mel_transition_path': mel_tr_path,
        'key_a': row['key_a'],
        'bpm_a': row['bpm_a'],
        'key_b': row['key_b'],
        'bpm_b': row['bpm_b']
    })

def process_dataset(csv_path, output_csv_path):
    """
    | --- Expected Dataset Columns ---------------------------------------------|
    | track_a_path | track_b_path | track_a_start | track_a_end | track_b_start |
    | track_b_end | transition_path | key_a | bpm_a | key_b | bpm_b |           |
    |___________________________________________________________________________|   
    """

    df = pd.read_csv(csv_path)
    processed = df.apply(lambda row: process_row(row, row.name), axis=1)
    processed.to_csv(output_csv_path, index=False)
    print(f"✅ Dataset procesado guardado en {output_csv_path}")

if __name__ == "__main__":
    process_dataset("transiciones_original.csv", "dataset_procesado.csv")
