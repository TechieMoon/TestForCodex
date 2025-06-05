import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """간단한 Transformer 인코더-디코더 모델"""

    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_dim * 4,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * (self.embedding.embedding_dim ** 0.5)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * (self.embedding.embedding_dim ** 0.5)
        tgt = self.pos_encoder(tgt)
        out = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1))
        out = self.fc_out(out.transpose(0, 1))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
