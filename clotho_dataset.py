import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

class ClothoDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with captions.
            audio_dir (str): Path to the directory with audio files.
            transform (callable, optional): Optional transform to be applied
                                            on an audio sample.
        """
        self.audio_dir = audio_dir
        self.transform = transform

        # Load captions and file names
        self.captions_df = pd.read_csv(csv_file)

        # Extracting relevant columns
        self.file_names = self.captions_df['file_name'].tolist()
        self.captions = self.captions_df[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.tolist()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load audio
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        # Get captions (all 5 captions for the audio file)
        captions = self.captions[idx]

        # Apply transform if specified
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, captions


# Example usage:
# dataset = ClothoDataset(audio_dir='../development',
#                         csv_file='../clotho_captions_development.csv')
# print(len(dataset))
# print(dataset[0])
