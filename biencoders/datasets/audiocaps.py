import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import shutil


class AudioCapsDataset(Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        transform=None,
        target_sample_rate=16000
    ):
        """
        Args:
            csv_file (str): Path to the CSV file containing
              audio file names and captions.
            audio_dir (str): Directory containing audio files.
            transform (callable, optional): Optional transform
              to be applied to the audio.
            target_sample_rate (int, optional): Desired sample
              rate for audio (default: 16kHz).
        """
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )(waveform)
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


if __name__ == "__main__":

    # Create sample audio directory
    os.makedirs("test_audio", exist_ok=True)

    # Generate synthetic audio files
    freqs = [440, 880, 1320]
    for f in freqs:
        # Create a simple sine wave
        duration = 2  # seconds
        t = np.linspace(0, duration, int(16000 * duration))
        audio_data = np.sin(2 * np.pi * f * t)
        # Convert to torch tensor and save
        waveform = torch.from_numpy(audio_data[None, :]).float()
        filename = f"sample_{f}.wav"
        torchaudio.save(f"test_audio/{filename}", waveform, 16000)

    # Create sample CSV
    sample_data = pd.DataFrame({
        'file_name': [f"sample_{f}.wav" for f in freqs],
        'caption': [f'A pure sine wave at {f} Hz' for f in freqs]
    })
    sample_data.to_csv('test_audiocaps.csv', index=False)

    try:
        # Test the dataset
        dataset = AudioCapsDataset(
            csv_file='test_audiocaps.csv',
            audio_dir='test_audio'
        )

        # Test loading items
        for i in range(len(dataset)):
            waveform, caption = dataset[i]
            print(f"\nSample {i}:")
            print(f"Waveform shape: {waveform.shape}")
            print(f"Caption: {caption}")

    finally:
        # Clean up
        shutil.rmtree('test_audio')
        os.remove('test_audiocaps.csv')
