import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os

class AudioCapsDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, target_sample_rate=16000):
        """
        Args:
            csv_file (str): Path to the CSV file containing audio file names and captions.
            audio_dir (str): Directory containing audio files.
            transform (callable, optional): Optional transform to be applied to the audio.
            target_sample_rate (int, optional): Desired sample rate for audio (default: 16kHz).
        """
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)
        return waveform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['file_name'])
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.resample_if_needed(waveform, sample_rate)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        caption = row['caption']
        return waveform, caption