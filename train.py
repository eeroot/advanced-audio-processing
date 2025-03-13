from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from biencoders.datasets.aggregated import AggregatedDataset
from biencoders.datasets.utils import random_augment
from biencoders.model.biencoder import TextAudioBiencoder, TransformerDecoder


def collate_fn(batch):
    audio_inputs, text_inputs = zip(*batch)  # Unpack batch

    # Stack and pad audio inputs
    audio_inputs = {
        key: pad_sequence(
            [a[key].squeeze(0) for a in audio_inputs],
            batch_first=True, padding_value=0
        )
        for key in audio_inputs[0].keys()
    }

    # Stack and pad text inputs
    text_inputs = {
        key: pad_sequence(
            [t[key].squeeze(0) for t in text_inputs],
            batch_first=True,
            padding_value=0
        )
        for key in text_inputs[0].keys()
    }

    return audio_inputs, text_inputs


# Contrastive loss function
def contrastive_loss(audio_embeds, text_embeds, margin=0.1):
    cosine_similarity = F.cosine_similarity(audio_embeds, text_embeds)
    loss = torch.mean(F.relu(margin - cosine_similarity))
    return loss


# Evaluation metrics: cosine similarity
def evaluate(model, dataloader, device):
    model.eval()
    all_cosine_similarities = []

    with torch.no_grad():
        for audio_input, text_input in tqdm(dataloader, desc="Evaluating"):
            audio_input = {key: val.squeeze(1).to(
                device) for key, val in audio_input.items()}
            text_input = {key: val.squeeze(1).to(device)
                          for key, val in text_input.items()}

            audio_embeds, text_embeds = model(audio_input, text_input)
            cosine_sim = F.cosine_similarity(audio_embeds, text_embeds)
            all_cosine_similarities.append(cosine_sim.cpu().numpy())

    all_cosine_similarities = np.concatenate(all_cosine_similarities)
    return np.mean(all_cosine_similarities)


# Main function to train and evaluate the model
def train(
    model, train_loader, val_loader, test_loader,
    epochs=10, lr=1e-5, device='cpu'
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = -1

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for audio_input, text_input in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio_input = {key: val.squeeze(1).to(
                device) for key, val in audio_input.items()}
            text_input = {key: val.squeeze(1).to(device)
                          for key, val in text_input.items()}

            optimizer.zero_grad()
            audio_embeds, text_embeds = model(audio_input, text_input)

            # Generate predictions using the Transformer Decoder
            output = model.decoder(audio_embeds, text_input)
            target = text_input['input_ids'][:, 1:]  # Remove <bos> token
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), target.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

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


# def evaluate_test(model, data_loader, device='cpu'):
#     """Evaluates model using Recall@K, KL Divergence, and t-SNE."""
#     model.eval()
#     audio_embeddings, text_embeddings = [], []

#     with torch.no_grad():
#         for audio_input, text_input in tqdm(data_loader, desc="Evaluating"):
#             audio_input = {key: val.squeeze(1).to(
#                 device) for key, val in audio_input.items()}
#             text_input = {key: val.squeeze(1).to(device)
#                           for key, val in text_input.items()}

#             audio_embeds, text_embeds = model(audio_input, text_input)
#             audio_embeddings.append(audio_embeds)
#             text_embeddings.append(text_embeds)

#     # Convert to tensors
#     audio_embeddings = torch.cat(audio_embeddings)
#     text_embeddings = torch.cat(text_embeddings)

#     # Compute cosine similarity
#     similarity_matrix = torch.mm(
#         text_embeddings, audio_embeddings.T).cpu().numpy()

#     # Compute Recall@K
#     recall_k1 = recall_at_k(similarity_matrix, 1)
#     recall_k5 = recall_at_k(similarity_matrix, 5)
#     recall_k10 = recall_at_k(similarity_matrix, 10)

#     # Compute KL Divergence
#     kl_div = kl_divergence(audio_embeddings, text_embeddings)

#     # Print results
#     print(
#         f"Recall@1: {recall_k1:.4f}, Recall@5: {recall_k5:.4f}, Recall@10: {recall_k10:.4f}")
#     print(f"KL Divergence: {kl_div:.4f}")

#     # t-SNE Visualization
#     plot_tsne(audio_embeddings, text_embeddings)

#     return recall_k1, recall_k5, recall_k10, kl_div


# Other helper functions (recall_at_k, kl_divergence, plot_tsne) remain the same

if __name__ == "__main__":
    epochs = 2
    lr = 1e-5
    # Check if CUDA is available, else use CPU
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Process on {device}", end="\n\n")

    # Load dataset splits using AggregatedDataset class
    audiocaps_dir = 'data/audiocaps'
    dataset_train = AggregatedDataset(
        audiocaps_dir=audiocaps_dir,
        transform=random_augment,
    )
    dataset_val = AggregatedDataset(
        split='val',
        audiocaps_dir=audiocaps_dir,
    )
    dataset_test = AggregatedDataset(
        split='test',
        audiocaps_dir=audiocaps_dir,
    )

    # DataLoader
    train_loader = DataLoader(
        dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=16,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=16,
                             shuffle=False, collate_fn=collate_fn)

    # Instantiate model with transformer decoder
    model = TextAudioBiencoder(embedding_dim=768)
    model.decoder = TransformerDecoder(embedding_dim=768)

    # Train the model
    train(model, train_loader, val_loader, test_loader, epochs, lr, device)
    # evaluate_test(model, test_loader, device=device)
