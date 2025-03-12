# Transformer decoder architecture
import os

import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


# Dataset Class for text-audio retrieval
class AudioCapsDataset(Dataset):
    def __init__(self,
                 audio_ids,
                 captions,
                 audio_folder,
                 tokenizer,
                 audio_processor
                 ):
        self.audio_ids = audio_ids
        self.captions = captions
        self.audio_folder = audio_folder
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
        audio_id = self.audio_ids[idx]
        caption = self.captions[idx]

        # Load audio using torchaudio
        audio_path = os.path.join(self.audio_folder, f"{audio_id}.wav")
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # **Padding or Truncation**
        max_length = 16000 * 10  # 10 seconds at 16kHz
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]  # Truncate
        else:
            pad_length = max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_length))  # Pad with zeros

        # Process audio to match the required input format for Wav2Vec2
        audio_input = self.audio_processor(waveform.squeeze(
            0), return_tensors="pt", sampling_rate=16000)

        # Tokenize the text
        text_input = self.tokenizer(
            caption, return_tensors="pt", padding=True, truncation=True)

        return audio_input, text_input
