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


# Define the Wav2Vec2 + RoBERTa Encoder Model
class TextAudioBiencoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TextAudioBiencoder, self).__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.text_encoder = RobertaModel.from_pretrained("roberta-large")
        self.embedding_dim = embedding_dim
        self.audio_projection = nn.Linear(1024, embedding_dim)  # Fix mismatch (1024 → 768)

    def forward(self, audio_input, text_input):
        audio_embeds = self.audio_encoder(**audio_input).last_hidden_state.mean(dim=1)
        audio_embeds = self.audio_projection(audio_embeds)  # Align with RoBERTa

        text_embeds = self.text_encoder(**text_input).last_hidden_state.mean(dim=1)
        return audio_embeds, text_embeds


# Transformer Decoder for Caption Generation
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, nhead=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        vocab_size = tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, embedding_dim))  # max caption length 500
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, audio_embeds, text_input):
        # Prepare text input for decoder
        text_input_ids = text_input['input_ids'].squeeze(1)
        embedded_text = self.embedding(text_input_ids) + self.positional_encoding[:, :text_input_ids.size(1), :]

        # Fix memory shape (ensure correct format for TransformerDecoder)
        memory = audio_embeds.unsqueeze(1)  # (batch_size, 1, d_model)
        memory = memory.permute(1, 0, 2)  # Ensure shape is (seq_len=1, batch_size, d_model=768)

        # Pass through Transformer Decoder
        transformer_out = self.transformer_decoder(embedded_text.permute(1, 0, 2), memory)  # Ensure proper dimensions

        # Output layer to get prediction
        output = self.fc_out(transformer_out.permute(1, 0, 2))  # Restore batch-first format
        return output
