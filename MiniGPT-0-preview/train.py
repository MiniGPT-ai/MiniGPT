import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPTModel, MiniGPTConfig
from torchinfo import summary

# Hyperparameters
TOKENIZER_PATH = "/MiniGPT-0-preview/tokenizer"
DATA_PATH      = "/MiniGPT-0-preview/openassistant_raw.txt"
BATCH_SIZE     = 32
BLOCK_SIZE     = 128
NUM_EPOCHS     = 3
LR             = 3e-4

# Load Tokenizer
tokenizer = ByteLevelBPETokenizer(
    os.path.join(TOKENIZER_PATH, "vocab.json"),
    os.path.join(TOKENIZER_PATH, "merges.txt")
)
vocab_size = tokenizer.get_vocab_size()

# Model Config
config = MiniGPTConfig()

# Read & Tokenize
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
tokens = tokenizer.encode(text).ids

# Dataset
class GPTDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.block_size],      dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1: idx+1 + self.block_size],  dtype=torch.long)
        return x, y

dataset = GPTDataset(tokens, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MiniGPTModel(config).to(device)

# Summary
summary(
    model,
    input_size=(1, BLOCK_SIZE),
    dtypes=[torch.long],
    device=device
)

# Optimizer & Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = criterion(
            logits.view(B * T, C),
            y.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {step:4d} | Loss {loss.item():.4f}")

    avg = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} done | Avg Loss {avg:.4f}")

torch.save(model.state_dict(), "minigpt0-preview.pth")
print("Saved minigpt0-preview.pth")
