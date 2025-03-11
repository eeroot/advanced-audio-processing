import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset



class ClothoDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        transform: Optional[Callable],
        target_sample_rate: int = 16000
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with captions.
            audio_dir (str): Path to the directory with audio files.
            transform (callable, optional): Optional transform to be applied
                                            on an audio sample.
            target_sample_rate (int, optional): Desired sample
              rate for audio (default: 16kHz).
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate

        # Load captions and file names
        self.captions_df = pd.read_csv(csv_file)

        # Extracting relevant columns
        self.file_names = self.captions_df['file_name'].tolist()
        self.captions = self.captions_df[[
            'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values.tolist()

    def resample_if_needed(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )(waveform)
        return waveform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load audio
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.resample_if_needed(waveform, sample_rate)

        # Get captions (all 5 captions for the audio file)
        captions = self.captions[idx]

        # Apply transform if specified
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, captions


if __name__ == "__main__":
    # Create sample audio directory and CSV file for testing
    import shutil  # Add this import at the top of the file

    # Create sample directories
    sample_dir = "sample_data"
    audio_dir = os.path.join(sample_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        # Generate sample audio files
        for i in range(3):
            # Create a simple sine wave
            sample_rate = 16000
            duration = 2  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

            # Save as wav file
            filename = f"audio_{i}.wav"
            waveform = torch.from_numpy(audio_data).float()
            torchaudio.save(os.path.join(audio_dir, filename),
                            waveform.unsqueeze(0), sample_rate)

        # Create sample CSV with captions
        data = {
            'file_name': [f'audio_{i}.wav' for i in range(3)],
            'caption_1': ['A dog barking', 'Birds chirping', 'Car passing by'],
            'caption_2': ['Loud bark', 'Morning birds', 'Vehicle noise'],
            'caption_3': ['Canine sound', 'Nature sounds', 'Traffic sound'],
            'caption_4': ['Pet noise', 'Wildlife audio', 'Street noise'],
            'caption_5': ['Animal sound', 'Forest ambience', 'Urban sound']
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(sample_dir, "sample_captions.csv")
        df.to_csv(csv_path, index=False)

        # Test the dataset
        dataset = ClothoDataset(csv_file=csv_path, audio_dir=audio_dir)
        print(f"Dataset size: {len(dataset)}")
        # Test loading items
        for i in range(len(dataset)):
            waveform, captions = dataset[i]
            print(f"\nSample {i}:")
            print(f"Waveform shape: {waveform.shape}")
            print(f"Sample captions: {captions}")

    finally:
        # Clean up: remove the sample directory and all its contents
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
            print(f"Cleaned up {sample_dir}")



# Example usage:
# dataset = ClothoDataset(audio_dir='../development',
#                         csv_file='../clotho_captions_development.csv')
# print(len(dataset))
# print(dataset[0])