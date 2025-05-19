import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os

class MelTransitionDataset(Dataset):
    def __init__(self, dataset_path, config_path, spectrogram_path, pad_strat):
        self.df = pd.read_csv(dataset_path)
        self.spec_path = spectrogram_path

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Precompute max spectrogram time, mean and std to normalize later
        self.max_time, self.mean, self.std = self._compute_statistics()
        assert pad_strat == "silence" or pad_strat == "reflect", f"{pad_strat} is not valid! Choose 'silence' or 'reflect'"
        self.pad_strat = pad_strat

    def _compute_statistics(self):
        max_time = 0
        all_specs = []

        for idx in range(self.df.shape[0]):
            row = self.df.iloc[idx]
            # Compute max time after trimming
            for key, start_key, end_key in [
                ('mel_a_path', 'start_time_a', 'end_time_a'),
                ('mel_b_path', 'start_time_b', 'end_time_b'),
            ]:
                mel_path = os.path.join(self.spec_path, row[key])
                spec = np.load(mel_path)
                trimmed = self._trim_spectrogram(
                    spec,
                    row[start_key],
                    row[end_key]
                )
                max_time = max(max_time, trimmed.shape[1])
                all_specs.append(trimmed.flatten())
            
            # The transition doesn't have to be trimmed
            mel_tr_path = os.path.join(self.spec_path, row['mel_tr_path'])
            spec = np.load(mel_tr_path)
            max_time = max(max_time, spec.shape[1])
            all_specs.append(spec.flatten())

        all_data = np.concatenate(all_specs)
        mean, std = np.mean(all_data), np.std(all_data)
        print(f"The maximum spectrogram lenght after trimming in the dataset is: {max_time}")
        print(f"Calculated mean and std are -> mean: {mean}, std: {std}")
        return max_time, mean, std

    def _load_trim_pad(self, path, start_time, end_time, max_time):
        spec = np.load(os.path.join(self.spec_path, path))
        trimmed_spec = self._trim_spectrogram(
            spec, 
            start_time=start_time,
            end_time=end_time,
        )
        if self.pad_strat == "silence":
            return self._pad_with_silence(trimmed_spec, max_time)
        else:
            return self._pad_with_reflect(trimmed_spec, max_time)

    def _trim_spectrogram(self, spectrogram, start_time, end_time):
        """
        Trim a precomputed mel-spectrogram to a specific time range.
        
        Args:
            spectrogram: (n_mels, time) array
            start_time: Start time in seconds
            end_time: End time in seconds
            hop_length: Hop length used during spectrogram computation
            sr: Sample rate (used to calculate frames)
        
        Returns:
            Trimmed spectrogram of shape (n_mels, trimmed_time)
        """
        # Convert time to frames
        start_frame = int(start_time * self.config["sr"] / self.config["hop_length"])
        end_frame = int(end_time * self.config["sr"] / self.config["hop_length"])
        
        # Ensure we don't exceed the spectrogram's length
        end_frame = min(end_frame, spectrogram.shape[1])
        
        return spectrogram[:, start_frame:end_frame] 

    def _pad_with_silence(self, spectrogram, target_time):
        """Pad the spectrogram with silence"""
        current_time = spectrogram.shape[-1]
        min_db = self.config["min_db"]

        if current_time < target_time:
            # min-padding
            pad_width = ((0, 0), (0, target_time - current_time))
            return np.pad(spectrogram, pad_width, mode='constant', constant_values=min_db)
        else:
            return spectrogram 

    def _pad_with_reflect(self, spectrogram, target_time):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        # Load, trim and pad all spectrograms using metadata
        mel_a = self._load_trim_pad(row['mel_a_path'], row['start_time_a'], row['end_time_a'], self.max_time)
        mel_b = self._load_trim_pad(row['mel_b_path'], row['start_time_b'], row['end_time_b'], self.max_time)
        # Load and pad transition (doesn't need trimming)
        mel_t = np.load(os.path.join(self.spec_path, row['mel_tr_path']))
        if self.pad_strat == "silence":
            mel_t = self._pad_with_silence(mel_t, self.max_time)
        else:
            mel_t = self._pad_with_reflect(mel_t, self.max_time)

        # Normalize
        mel_a = (mel_a - self.mean) / self.std
        mel_b = (mel_b - self.mean) / self.std
        mel_t = (mel_t - self.mean) / self.std

        # Convert to tensors and add channel dimension
        mel_a = torch.tensor(mel_a).unsqueeze(0).float()  # shape: (1, n_mels, time)
        mel_b = torch.tensor(mel_b).unsqueeze(0).float()
        mel_t = torch.tensor(mel_t).unsqueeze(0).float()

        input_pair = torch.cat([mel_a, mel_b], dim=0)  # shape: (2, n_mels, time)
        return input_pair, mel_t

if __name__ == "__main__":

    dataset = MelTransitionDataset(
        dataset_path="../../res/datasets/transition_dataset_processed_IV.csv", 
        config_path="../../res/config/config.json",
        spectrogram_path="../../res/mel_specs"
        )
    sample_input, sample_target = dataset[0]
    print(f"Input shape: {sample_input.shape}")  # Should be (2, n_mels, max_time)
    print(f"Target shape: {sample_target.shape}")  # Should be (1, n_mels, max_time)
    print(sample_input[0])

    all_spec_mins = []
    all_spec_max = []
    all_specs = []
    for idx in range(len(dataset)):
        all_spec_mins.append(torch.min(dataset[idx][0]))
        all_spec_max.append(torch.max(dataset[idx][0]))
        all_specs.append(dataset[idx][0].flatten())
    print(f"Minimum value in all specs is: {min(all_spec_mins)}")
    print(f"Maximum value in all specs is: {min(all_spec_max)}")

    import matplotlib.pyplot as plt
    plt.hist(np.concatenate(all_specs), bins=100)
    plt.title("Normalized Spectrogram Values")
    plt.show()