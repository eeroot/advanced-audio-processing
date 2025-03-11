import os
import torch
import torchaudio
import yt_dlp
from datasets import load_dataset
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def download_audio(youtube_id, start_time, output_path):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    # Use yt-dlp to download the audio
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}.%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"},
        ],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        print(f"Skipping unavailable video: {youtube_id}")
        return None  # Indicate failure

    # Trim audio using ffmpeg
    trimmed_audio_path = f"{output_path}_trimmed.wav"
    os.system(f'ffmpeg -i "{output_path}.wav" -ss {start_time} -t 10 "{trimmed_audio_path}" -y')

    return trimmed_audio_path


# Function to preprocess text captions
def preprocess_text(captions, tokenizer):
    tokenized = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    return tokenized.input_ids, tokenized.attention_mask


def preprocess_audio(audio_path, feature_extractor, target_length=160000):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to 16kHz if needed
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Extract features
    features = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = features.input_values.squeeze(0)

    # Pad or truncate
    if input_values.shape[0] < target_length:
        pad_length = target_length - input_values.shape[0]
        input_values = F.pad(input_values, (0, pad_length))
    else:
        input_values = input_values[:target_length]

    return input_values

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    audio_features = [item["audio_features"] for item in batch]

    # Ensure uniform size for audio features
    max_audio_length = max([audio.size(0) for audio in audio_features])  # Find the max length

    # Pad audio features to the same length
    padded_audio_features = [
        F.pad(audio, (0, max_audio_length - audio.size(0)), "constant", 0) for audio in audio_features
    ]

    # Pad other sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Stack the audio features
    padded_audio_features = torch.stack(padded_audio_features)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_features": padded_audio_features,
    }

# PyTorch dataset class
class AudioCapsDataset(Dataset):
    def __init__(self, split="train", download_dir="audiocaps_audio"):
        self.dataset = load_dataset("d0rj/audiocaps", split=split)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        youtube_id = item["youtube_id"]
        start_time = item["start_time"]
        caption = item["caption"]

        # Generate a local filename
        output_path = os.path.join(self.download_dir, f"{youtube_id}_{start_time}")

        # Download and process the audio
        audio_file = download_audio(youtube_id, start_time, output_path)

        if audio_file is None:
          return self.__getitem__((idx + 1) % len(self.dataset))

        audio_features = preprocess_audio(audio_file, self.feature_extractor)

        # Process text
        input_ids, attention_mask = preprocess_text(caption, self.tokenizer)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "audio_features": audio_features,
        }
    
# Main function
def main():
    train_dataset = AudioCapsDataset(split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Validation dataset
    val_dataset = AudioCapsDataset(split="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Test dataset
    test_dataset = AudioCapsDataset(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Example for training data
    for batch in train_dataloader:
        print("Train - Input IDs shape:", batch["input_ids"].shape)
        print("Train - Attention Mask shape:", batch["attention_mask"].shape)
        print("Train - Audio Features shape:", batch["audio_features"].shape)
        break

    # Example for validation data
    for batch in val_dataloader:
        print("Validation - Input IDs shape:", batch["input_ids"].shape)
        print("Validation - Attention Mask shape:", batch["attention_mask"].shape)
        print("Validation - Audio Features shape:", batch["audio_features"].shape)
        break

    """
    # Example for test data
    for batch in test_dataloader:
        print("Test - Input IDs shape:", batch["input_ids"].shape)
        print("Test - Attention Mask shape:", batch["attention_mask"].shape)
        print("Test - Audio Features shape:", batch["audio_features"].shape)
        break
    """


# Run main function
if __name__ == "__main__":
    main()