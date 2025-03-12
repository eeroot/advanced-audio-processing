import os
import torch
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, ConcatDataset
from transformers import RobertaTokenizer, Wav2Vec2Processor

from utils import random_augment
from audiocaps import AudioCapsDataset
from clothov2 import ClothoDataset


logging.basicConfig(
    level=logging.DEBUG if os.environ.get('DEBUG') == '1' else logging.INFO,
    format='[%(levelname)s] %(filename)s | %(message)s'
)


class AggregatedDataset(Dataset):
    def __init__(
        self,
        audiocaps_dir: Path,
        split="train",
        clotho_csv: Path = None,
        clotho_dir: Path = None,
        transform=None,
        target_sample_rate=16000
    ):
        """
        Args:
            audiocaps_dir (str): Path to AudioCaps audio directory
            clotho_csv (Path): Path to Clotho CSV file
            clotho_dir (str): Path to Clotho audio directory
            transform (callable, optional): Transform to apply to audio
            target_sample_rate (int): Target sample rate for audio files
        """
        # Initialize individual datasets
        logging.info("Initializing AudioCaps dataset...")
        audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        # Load data
        csv = pd.read_csv(audiocaps_dir / f"{split}.csv")

        # Extract audio IDs and captions
        audio_ids = csv['audiocap_id'].tolist()
        captions = csv['caption'].tolist()

        self.audiocaps_dataset = AudioCapsDataset(
            audio_ids=audio_ids,
            captions=captions,
            audio_folder=audiocaps_dir / split,
            tokenizer=tokenizer,
            audio_processor=audio_processor
        )
        self.audiocaps_len = len(self.audiocaps_dataset)

        if clotho_csv and clotho_dir:
            logging.info("Initializing Clotho dataset...")
            self.clotho_dataset = ClothoDataset(
                csv_file=clotho_csv,
                audio_dir=clotho_dir,
                transform=transform
            )
            self.clotho_len = len(self.clotho_dataset)
        else:
            logging.warning("Clotho dataset not provided. Skipping...")
            self.clotho_dataset = None
            self.clotho_len = 0

    def __len__(self):
        return self.audiocaps_len + self.clotho_len

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (waveform, captions)
            For AudioCaps: captions is a single string
            For Clotho: captions is a list of 5 strings
        """
        if idx < self.audiocaps_len:
            # Get item from AudioCaps
            waveform, caption = self.audiocaps_dataset[idx]
            # Convert single caption to list format for consistency
            captions = [caption]
        else:
            # Get item from Clotho
            clotho_idx = idx - self.audiocaps_len
            waveform, captions = self.clotho_dataset[clotho_idx]

        return waveform, captions

    def get_dataset_identifier(self, idx):
        """
        Returns which dataset the index belongs to
        """
        return "audiocaps" if idx < self.audiocaps_len else "clotho"


if __name__ == "__main__":

    # Create test directories
    os.makedirs("test_audio_audiocaps", exist_ok=True)
    os.makedirs("test_audio_clotho", exist_ok=True)

    try:
        # Generate test audio files for AudioCaps
        freqs = [440, 880]
        duration = 2
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create AudioCaps test files
        audiocaps_files = []
        for f in freqs:
            audio_data = np.sin(2 * np.pi * f * t)
            waveform = torch.from_numpy(audio_data[None, :]).float()
            filename = f"sample_{f}.wav"
            torchaudio.save(
                f"test_audio_audiocaps/{filename}", waveform, sample_rate)
            audiocaps_files.append(filename)

        # Create AudioCaps CSV
        audiocaps_data = pd.DataFrame({
            'file_name': audiocaps_files,
            'caption': ['A pure sine wave', 'Another sine wave']
        })
        audiocaps_data.to_csv('test_audiocaps.csv', index=False)

        # Create Clotho test files and CSV
        clotho_files = []
        for f in freqs:
            audio_data = np.sin(2 * np.pi * f * 2 * t)
            waveform = torch.from_numpy(audio_data[None, :]).float()
            filename = f"clotho_{f}.wav"
            torchaudio.save(
                f"test_audio_clotho/{filename}", waveform, sample_rate)
            clotho_files.append(filename)

        # Create Clotho CSV
        clotho_data = pd.DataFrame({
            'file_name': clotho_files,
            'caption_1': ['First caption 1', 'First caption 2'],
            'caption_2': ['Second caption 1', 'Second caption 2'],
            'caption_3': ['Third caption 1', 'Third caption 2'],
            'caption_4': ['Fourth caption 1', 'Fourth caption 2'],
            'caption_5': ['Fifth caption 1', 'Fifth caption 2']
        })
        clotho_data.to_csv('test_clotho.csv', index=False)

        # Test aggregated dataset
        dataset = AggregatedDataset(
            audiocaps_csv='test_audiocaps.csv',
            audiocaps_dir='test_audio_audiocaps',
            clotho_csv='test_clotho.csv',
            clotho_dir='test_audio_clotho',
            transform=random_augment,
        )

        print(f"\nTotal dataset size: {len(dataset)}\n")

        # Test a few samples
        for i in [0, len(dataset)-1]:
            waveform, captions = dataset[i]
            dataset_type = dataset.get_dataset_identifier(i)
            print(f"Sample {i} from {dataset_type}:")
            print(f"Waveform shape: {waveform.shape}")
            print(f"Captions: {captions}")
            print()

    finally:
        # Clean up
        for dir_name in ['test_audio_audiocaps', 'test_audio_clotho']:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
        for file_name in ['test_audiocaps.csv', 'test_clotho.csv']:
            if os.path.exists(file_name):
                os.remove(file_name)
