import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MelTransitionDataset(Dataset):
    def __init__(self, metadata_df):
        self.df = metadata_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_a = np.load(row['mel_a_path'])
        mel_b = np.load(row['mel_b_path'])
        mel_t = np.load(row['mel_transition_path'])

        mel_a = torch.tensor(mel_a).unsqueeze(0).float()
        mel_b = torch.tensor(mel_b).unsqueeze(0).float()
        mel_t = torch.tensor(mel_t).unsqueeze(0).float()

        input_pair = torch.cat([mel_a, mel_b], dim=0)
        return input_pair, mel_t
