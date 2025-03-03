import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List

# Define the Word class
class Word:
    def __init__(self, id: str, file_path: Path, writer_id: str, transcription: str):
        self.id = id
        self.file_path = file_path
        self.writer_id = writer_id
        self.transcription = transcription

# Custom Dataset
class HandwritingDataset(Dataset):
    def __init__(self, words: List[Word], vocab, transform=None):
        self.words = words
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        image = cv2.imread(str(word.file_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))  # Normalize size
        if self.transform:
            image = self.transform(image)
        label = [self.vocab[char] for char in word.transcription]
        return image, torch.tensor(label, dtype=torch.long)

# Vocabulary creation
def create_vocab(words: List[Word]):
    unique_chars = set(char for word in words for char in word.transcription)
    vocab = {char: idx for idx, char in enumerate(sorted(unique_chars), start=1)}
    vocab['<blank>'] = 0  # For blank token
    return vocab

# RNN-T Model
def build_rnnt_model(vocab_size):
    class RNNTModel(nn.Module):
        def __init__(self, vocab_size):
            super(RNNTModel, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.prediction = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
            self.joint = nn.Sequential(
                nn.Linear(128 + 128, 256),
                nn.ReLU(),
                nn.Linear(256, vocab_size)
            )

        def forward(self, image, label):
            encoder_output = self.encoder(image)
            label_embed = torch.nn.functional.one_hot(label, num_classes=vocab_size).float()
            prediction_output, _ = self.prediction(label_embed)
            joint_input = torch.cat((encoder_output.unsqueeze(1).repeat(1, label.size(1), 1), prediction_output), dim=-1)
            output = self.joint(joint_input)
            return output

    return RNNTModel(vocab_size)

# Training Loop
def train_model(dataset, model, epochs=10, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images = images.unsqueeze(1).float()  # Add channel dimension
            optimizer.zero_grad()
            logits = model(images, labels)
            input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            loss = criterion(logits.permute(1, 0, 2), torch.cat(labels), input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Data Collate Function
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.tensor(np.stack(images), dtype=torch.float32) / 255.0
    return images, labels

# Sample Main Execution
def main():
    # Assuming `words` is your list of Word objects
    words = [
        Word(id='a01-000u', file_path=Path('./iam_words/words/a01/a01-000u/a01-000u-00-00.png'), writer_id='000', transcription='A')
    ]
    vocab = create_vocab(words)
    dataset = HandwritingDataset(words, vocab, transform=transforms.ToTensor())
    model = build_rnnt_model(len(vocab))

    train_model(dataset, model, epochs=10, batch_size=4)

    print("Model training completed.")


























    

if __name__ == "__main__":
    main()
