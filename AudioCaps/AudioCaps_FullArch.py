#Transformer decoder architecture
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from tqdm import tqdm
import torchaudio
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# Dataset Class for text-audio retrieval
class AudioTextDataset(Dataset):
    def __init__(self, audio_ids, captions, audio_folder, tokenizer, audio_processor):
        self.audio_ids = audio_ids
        self.captions = captions
        self.audio_folder = audio_folder
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.audio_ids)

    def __getitem__(self, idx):
      while True:
        audio_id = self.audio_ids[idx]
        caption = self.captions[idx]

        # Load audio using torchaudio
        audio_path = os.path.join(self.audio_folder, f"{audio_id}.wav")
        if not os.path.exists(audio_path):
          print(f"Warning: {audio_path} not found. Skipping.")
          idx = (idx + 1) % len(self.audio_ids)  # Move to next index (avoid infinite loop)
          continue

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
          waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
          resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
          waveform = resampler(waveform)

        # **Padding or Truncation**
        max_length = 16000 * 10  # 10 seconds at 16kHz
        if waveform.shape[1] > max_length:
          waveform = waveform[:, :max_length]  # Truncate
        else:
          pad_length = max_length - waveform.shape[1]
          waveform = F.pad(waveform, (0, pad_length))  # Pad with zeros

        # Process audio to match the required input format for Wav2Vec2
        audio_input = self.audio_processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000)

        # Tokenize the text
        text_input = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)

        return audio_input, text_input

# Define the Wav2Vec2 + RoBERTa Encoder Model
class TextAudioBiencoder(nn.Module):
    def __init__(self, audio_encoder, text_encoder, embedding_dim):
        super(TextAudioBiencoder, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim

    def forward(self, audio_input, text_input):
        audio_embeds = self.audio_encoder(**audio_input).last_hidden_state.mean(dim=1)
        text_embeds = self.text_encoder(**text_input).last_hidden_state.mean(dim=1)
        return audio_embeds, text_embeds

# Transformer Decoder for Caption Generation
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, nhead=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, embedding_dim))  # max caption length 500
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        #self.transformer_decoder = nn.TransformerDecoder(
        #    d_model=embedding_dim, nhead=nhead, num_layers=num_layers
        #)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, audio_embeds, text_input):
        # Prepare text input for decoder
        text_input_ids = text_input['input_ids'].squeeze(1)
        embedded_text = self.embedding(text_input_ids) + self.positional_encoding[:, :text_input_ids.size(1), :]

        # Pass through Transformer Decoder
        memory = audio_embeds.unsqueeze(0)  # Adding batch dimension
        transformer_out = self.transformer_decoder(embedded_text, memory)

        # Output layer to get prediction
        output = self.fc_out(transformer_out)
        return output

# Contrastive loss function
def contrastive_loss(audio_embeds, text_embeds, margin=0.1):
    cosine_similarity = F.cosine_similarity(audio_embeds, text_embeds)
    loss = torch.mean(F.relu(margin - cosine_similarity))
    return loss

def collate_fn(batch):
    audio_inputs, text_inputs = zip(*batch)  # Unpack batch

    # Stack and pad audio inputs
    audio_inputs = {key: pad_sequence([a[key].squeeze(0) for a in audio_inputs], batch_first=True, padding_value=0)
                    for key in audio_inputs[0].keys()}

    # Stack and pad text inputs
    text_inputs = {key: pad_sequence([t[key].squeeze(0) for t in text_inputs], batch_first=True, padding_value=0)
                   for key in text_inputs[0].keys()}

    return audio_inputs, text_inputs

# Evaluation metrics: cosine similarity
def evaluate(model, dataloader, device):
    model.eval()
    all_cosine_similarities = []

    with torch.no_grad():
        for audio_input, text_input in tqdm(dataloader, desc="Evaluating"):
            audio_input = {key: val.squeeze(1).to(device) for key, val in audio_input.items()}
            text_input = {key: val.squeeze(1).to(device) for key, val in text_input.items()}

            audio_embeds, text_embeds = model(audio_input, text_input)
            cosine_sim = F.cosine_similarity(audio_embeds, text_embeds)
            all_cosine_similarities.append(cosine_sim.cpu().numpy())

    all_cosine_similarities = np.concatenate(all_cosine_similarities)
    return np.mean(all_cosine_similarities)

# Main function to train and evaluate the model
def train(model, train_loader, val_loader, test_loader, epochs=10, lr=1e-5, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = -1

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for audio_input, text_input in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio_input = {key: val.squeeze(1).to(device) for key, val in audio_input.items()}
            text_input = {key: val.squeeze(1).to(device) for key, val in text_input.items()}

            optimizer.zero_grad()
            audio_embeds, text_embeds = model(audio_input, text_input)

            # Generate predictions using the Transformer Decoder
            output = model.decoder(audio_embeds, text_input)
            target = text_input['input_ids'][:, 1:]  # Remove <bos> token
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

        # Validation loop
        val_score = evaluate(model, val_loader, device)
        print(f"Validation Cosine Similarity: {val_score}")

        if val_score > best_val_score:
            best_val_score = val_score
            # Save the model
            torch.save(model.state_dict(), "best_model.pth")

    # Test loop
    model.load_state_dict(torch.load("best_model.pth"))
    test_score = evaluate(model, test_loader, device)
    print(f"Test Cosine Similarity: {test_score}")

def evaluate_test(model, data_loader, device='cpu'):
    """Evaluates model using Recall@K, KL Divergence, and t-SNE."""
    model.eval()
    audio_embeddings, text_embeddings = [], []

    with torch.no_grad():
        for audio_input, text_input in tqdm(data_loader, desc="Evaluating"):
            audio_input = {key: val.squeeze(1).to(device) for key, val in audio_input.items()}
            text_input = {key: val.squeeze(1).to(device) for key, val in text_input.items()}

            audio_embeds, text_embeds = model(audio_input, text_input)
            audio_embeddings.append(audio_embeds)
            text_embeddings.append(text_embeds)

    # Convert to tensors
    audio_embeddings = torch.cat(audio_embeddings)
    text_embeddings = torch.cat(text_embeddings)

    # Compute cosine similarity
    similarity_matrix = torch.mm(text_embeddings, audio_embeddings.T).cpu().numpy()

    # Compute Recall@K
    recall_k1 = recall_at_k(similarity_matrix, 1)
    recall_k5 = recall_at_k(similarity_matrix, 5)
    recall_k10 = recall_at_k(similarity_matrix, 10)

    # Compute KL Divergence
    kl_div = kl_divergence(audio_embeddings, text_embeddings)

    # Print results
    print(f"Recall@1: {recall_k1:.4f}, Recall@5: {recall_k5:.4f}, Recall@10: {recall_k10:.4f}")
    print(f"KL Divergence: {kl_div:.4f}")

    # t-SNE Visualization
    plot_tsne(audio_embeddings, text_embeddings)

    return recall_k1, recall_k5, recall_k10, kl_div

# Other helper functions (recall_at_k, kl_divergence, plot_tsne) remain the same

if __name__ == "__main__":
    # Load pre-trained models
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    text_encoder = RobertaModel.from_pretrained("roberta-large")

    # Load data
    train_csv = pd.read_csv("train.csv")
    val_csv = pd.read_csv("val.csv")
    test_csv = pd.read_csv("test.csv")

    # Extract audio IDs and captions
    train_audio_ids = train_csv['audiocap_id'].tolist()
    train_captions = train_csv['caption'].tolist()

    val_audio_ids = val_csv['audiocap_id'].tolist()
    val_captions = val_csv['caption'].tolist()

    test_audio_ids = test_csv['audiocap_id'].tolist()
    test_captions = test_csv['caption'].tolist()

    # Prepare datasets

    train_dataset = AudioTextDataset(
        audio_ids=train_audio_ids,
        captions=train_captions,
        audio_folder='train/',
        tokenizer=tokenizer,
        audio_processor=audio_processor
    )

    val_dataset = AudioTextDataset(
        audio_ids=val_audio_ids,
        captions=val_captions,
        audio_folder='val/',
        tokenizer=tokenizer,
        audio_processor=audio_processor
    )

    test_dataset = AudioTextDataset(
        audio_ids=test_audio_ids,
        captions=test_captions,
        audio_folder='test/',
        tokenizer=tokenizer,
        audio_processor=audio_processor
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Instantiate model with transformer decoder
    model = TextAudioBiencoder(audio_encoder, text_encoder, embedding_dim=768)
    model.decoder = TransformerDecoder(embedding_dim=768, vocab_size=len(tokenizer))

    # Train the model
    train(model, train_loader, val_loader, test_loader, epochs=2, lr=1e-5)
    evaluate_test(model, test_loader, device="cpu")