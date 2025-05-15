import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MelTransitionDataset(Dataset):
    def __init__(self, metadata_df):
        self.df = metadata_df
        # Precompute the maximum time steps across all spectrograms
        self.max_time = self._compute_max_time()

    def _compute_max_time(self):
        max_time = 0
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            # Load all spectrograms and track their time dimensions
            for key in ['mel_fragment_a_path', 'mel_fragment_b_path', 'mel_transition_path']:
                spec = np.load(row[key])
                max_time = max(max_time, spec.shape[-1])  # shape is (n_mels, time)
        return max_time

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_a = np.load(row['mel_fragment_a_path'])
        mel_b = np.load(row['mel_fragment_b_path'])
        mel_t = np.load(row['mel_transition_path'])

        # Pad to the precomputed global maximum time
        mel_a = self._pad_with_context(mel_a, self.max_time)
        mel_b = self._pad_with_context(mel_b, self.max_time)
        mel_t = self._pad_with_context(mel_t, self.max_time)

        # Convert to tensors and add channel dimension
        mel_a = torch.tensor(mel_a).unsqueeze(0).float()  # shape: (1, n_mels, time)
        mel_b = torch.tensor(mel_b).unsqueeze(0).float()
        mel_t = torch.tensor(mel_t).unsqueeze(0).float()

        input_pair = torch.cat([mel_a, mel_b], dim=0)  # shape: (2, n_mels, time)
        return input_pair, mel_t

    def _pad_with_context(self, spectrogram, target_time):
        """Pad the spectrogram with zeroes."""
        current_time = spectrogram.shape[-1]
        if current_time < target_time:
            # Zero-padding
            pad_width = ((0, 0), (0, target_time - current_time))
            padded_spec = np.pad(spectrogram, pad_width, mode='constant')
            return padded_spec
        else:
            return spectrogram 

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../../res/datasets/transition_dataset_processed.csv")
    dataset = MelTransitionDataset(df)
    sample_input, sample_target = dataset[0]
    print(f"Input shape: {sample_input.shape}")  # Should be (2, n_mels, max_time)
    print(f"Target shape: {sample_target.shape}")  # Should be (1, n_mels, max_time)