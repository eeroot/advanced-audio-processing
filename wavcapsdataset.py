import torch
import torchaudio
import pandas as pd
from datasets import load_dataset
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np

class WavCapsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 16000,
        transform: Optional[Callable] = None,
        max_length: Optional[int] = None,
        subset: Optional[str] = None
    ):
        """
        WavCaps dataset loader using Hugging Face datasets.

        Args:
            split (str): Dataset split ('train', 'validation', or 'test')
            sample_rate (int): Target sample rate for audio
            transform (callable, optional): Transform to apply to the audio
            max_length (int, optional): Max length of audio in samples (will pad/trim)
            subset (str, optional): Dataset subset ('freesound', 'audioset', 'bbc', or 'soundbible')
        """
        self.sample_rate = sample_rate
        self.transform = transform
        self.max_length = max_length
        
        # Load the dataset from Hugging Face
        self.dataset = load_dataset("cvssp/WavCaps", split=split)
        
        # Filter by subset if specified
        if subset:
            if subset not in ['freesound', 'audioset', 'bbc', 'soundbible']:
                raise ValueError("subset must be one of: freesound, audioset, bbc, soundbible")
            self.dataset = self.dataset.filter(lambda x: x['source'] == subset)

    def __len__(self) -> int:
        return len(self.dataset)

    def _process_audio(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Process audio by resampling, converting to mono, and padding/trimming."""
        # Resample if necessary
        if orig_sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            )(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Handle max_length if specified
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                # Trim
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                # Pad with zeros
                pad_length = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset.

        Returns:
            dict containing:
                - waveform (torch.Tensor): Audio waveform
                - caption (str): Audio caption
                - duration (float): Audio duration in seconds
                - source (str): Source dataset
                - fname (str): Original filename
        """
        item = self.dataset[idx]
        
        try:
            # Load audio
            waveform = torch.from_numpy(item['audio']['array']).float()
            orig_sr = item['audio']['sampling_rate']

            # Process audio
            waveform = self._process_audio(waveform, orig_sr)

            # Apply transforms if specified
            if self.transform is not None:
                waveform = self.transform(waveform)

            return {
                'waveform': waveform,
                'caption': item['caption'],
                'duration': item['duration'],
                'source': item['source'],
                'fname': item['fname']
            }

        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Return zero tensor with correct shape in case of error
            shape = (1, self.max_length) if self.max_length else (1, self.sample_rate)
            return {
                'waveform': torch.zeros(shape),
                'caption': '',
                'duration': 0.0,
                'source': '',
                'fname': ''
            }

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = WavCapsDataset(
        split="train",
        sample_rate=16000,
        max_length=160000,  # 10 seconds at 16kHz
        subset="freesound",
        cache_dir="./cache"  # Cache downloads locally
    )

    # Get dataset statistics
    stats = get_dataset_stats(dataset)
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average duration: {stats['avg_duration']:.2f} seconds")
    print("Source distribution:")
    for source, count in stats['sources_distribution'].items():
        print(f"  {source}: {count}")

    # Get a single sample
    sample = dataset[0]
    print("\nSingle Sample Information:")
    print("Audio Shape:", sample['waveform'].shape)
    print("Caption:", sample['caption'])
    print("Duration:", sample['duration'])
    print("Source:", sample['source'])
    print("Filename:", sample['fname'])
    print("Sampling Rate:", sample['sampling_rate'])

    # Create dataloader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # Show batch example
    print("\nBatch Processing Example:")
    for batch in dataloader:
        print("Batch shapes:")
        print("  Waveforms:", batch['waveform'].shape)
        print("  Number of captions:", len(batch['caption']))
        print("First caption in batch:", batch['caption'][0])
        break  # Show only first batch
