import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SimpleTransformer


class CopyDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.randint(1, self.vocab_size, (self.seq_len,))
        return seq, seq


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.stack(src)
    tgt = torch.stack(tgt)
    return src, tgt


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CopyDataset(args.vocab_size, args.seq_len, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleTransformer(args.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:,:-1])
            loss = criterion(output.reshape(-1, args.vocab_size), tgt[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train(args)
