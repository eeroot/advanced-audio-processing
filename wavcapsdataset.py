import torch
import torchaudio
import pandas as pd
import json
import os

class WavCapsBaseDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, transform=None):
        """
        Base dataset class for WavCaps datasets.

        Args:
            metadata_path (str): Path to the metadata file (JSON or CSV).
            audio_dir (str): Directory containing the audio files.
            sample_rate (int, optional): Target sample rate for resampling (default: 16000).
            transform (callable, optional): Optional transformation to apply to the waveform.
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.data = self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path):
        """Loads metadata from a JSON or CSV file."""
        if metadata_path.endswith('.json'):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return list(metadata.items())  # Convert dictionary to list of tuples
        elif metadata_path.endswith('.csv'):
            return pd.read_csv(metadata_path).values.tolist()
        else:
            raise ValueError("Metadata file must be in JSON or CSV format.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Loads an audio file and returns its waveform and caption."""
        audio_filename, caption = self.data[idx]
        audio_path = os.path.join(self.audio_dir, audio_filename)

        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)

            # Apply transformations if specified
            if self.transform:
                waveform = self.transform(waveform)

            return waveform, caption

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None


class FreesoundDataset(WavCapsBaseDataset):
    """Dataset class for the Freesound portion of WavCaps."""
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, transform=None):
        super().__init__(metadata_path, audio_dir, sample_rate, transform)


class BBCSoundEffectsDataset(WavCapsBaseDataset):
    """Dataset class for the BBC Sound Effects portion of WavCaps."""
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, transform=None):
        super().__init__(metadata_path, audio_dir, sample_rate, transform)


class SoundBibleDataset(WavCapsBaseDataset):
    """Dataset class for the SoundBible portion of WavCaps."""
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, transform=None):
        super().__init__(metadata_path, audio_dir, sample_rate, transform)


class AudioSetDataset(WavCapsBaseDataset):
    """Dataset class for the AudioSet (Strongly-labeled) portion of WavCaps."""
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, transform=None):
        super().__init__(metadata_path, audio_dir, sample_rate, transform)


if __name__ == "__main__":
    # Example usage
    metadata_paths = {
        "freesound": "path/to/freesound_metadata.json",
        "bbc": "path/to/bbc_metadata.json",
        "soundbible": "path/to/soundbible_metadata.json",
        "audioset": "path/to/audioset_metadata.json"
    }
    audio_dirs = {
        "freesound": "path/to/freesound_audio/",
        "bbc": "path/to/bbc_audio/",
        "soundbible": "path/to/soundbible_audio/",
        "audioset": "path/to/audioset_audio/"
    }

    # Load one of the datasets
    dataset = FreesoundDataset(metadata_paths["freesound"], audio_dirs["freesound"])

    # Fetch a sample
    waveform, caption = dataset[0]
    
    if waveform is not None:
        print("Audio Shape:", waveform.shape)
        print("Caption:", caption)
