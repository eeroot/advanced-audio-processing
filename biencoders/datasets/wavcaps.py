from dataclasses import dataclass, field
import json
import os
import torch
import torchaudio
from typing import Tuple, Optional, Dict
import numpy as np

@dataclass
class SoundBibleData:
    audio_id: str
    base_audio_dir: str
    metadata_file: str
    sample_rate: int = 16000  # Standard sample rate for most audio models
    max_length: float = 30.0  # Maximum audio length in seconds
    audio_path: str = field(init=False)
    metadata: dict = field(init=False)
    
    def __post_init__(self):
        self.audio_path = os.path.join(self.base_audio_dir, f"{self.audio_id}.flac")
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict:
        """Load and return metadata for the audio file."""
        with open(self.metadata_file, 'r') as file:
            metadata = json.load(file)
        return metadata.get(self.audio_id, {})
    
    def load_audio(self) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file.
        
        Returns:
            Tuple[torch.Tensor, int]: (audio_tensor, sample_rate)
        """
        waveform, sr = torchaudio.load(self.audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Trim or pad to max_length
        max_samples = int(self.max_length * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            pad_length = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        return waveform, self.sample_rate
    
    def get_features(self) -> Optional[torch.Tensor]:
        """
        Extract audio features (mel spectrogram) for model input.
        
        Returns:
            torch.Tensor: Mel spectrogram features
        """
        try:
            waveform, sr = self.load_audio()
            
            # Mel spectrogram transformation
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=1024,
                hop_length=512,
                n_mels=80,
                power=2.0
            )
            
            # Convert to mel spectrogram
            mel_spec = mel_transform(waveform)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Normalize
            mean = mel_spec.mean()
            std = mel_spec.std()
            mel_spec = (mel_spec - mean) / (std + 1e-9)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error processing audio file {self.audio_id}: {str(e)}")
            return None
    
    def get_training_item(self) -> Optional[Dict]:
        """
        Get complete item for training including audio features and metadata.
        
        Returns:
            Dict containing:
                - features: mel spectrogram
                - caption: text description
                - duration: audio duration
                - sample_rate: audio sample rate
                - audio_id: unique identifier
        """
        try:
            features = self.get_features()
            if features is None:
                return None
                
            return {
                'features': features,
                'caption': self.metadata.get('caption', ''),
                'duration': self.metadata.get('duration', 0.0),
                'sample_rate': self.sample_rate,
                'audio_id': self.audio_id
            }
            
        except Exception as e:
            print(f"Error creating training item for {self.audio_id}: {str(e)}")
            return None

""" # Example usage for training
sound_data = SoundBibleData(
    audio_id='49',
    base_audio_dir='path/to/audio',
    metadata_file='path/to/metadata.json'
)

# Get training item
training_item = sound_data.get_training_item()
if training_item:
    features = training_item['features']  # Shape: [n_mels, time]
    caption = training_item['caption']
    # Use these in your training loop
 """
