import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class TFModel(nn.Module):
    def __init__(self, iw, ow, ft_dim, d_model, nhead, nlayers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=nlayers,
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(
            nn.Linear(ft_dim, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model),
        )
        self.linear =  nn.Sequential(
            nn.Linear(iw*d_model, ow*16),
            nn.ReLU(),
            nn.Linear(ow*16, ow),
        )
        self.linear1 =  nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(16*iw, ow)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.linear1(output)
        output = self.linear2(output.flatten(start_dim=1))
        return output

