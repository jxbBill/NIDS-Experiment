from __future__ import annotations

import math

import torch
from torch import nn


def _nearest_power_of_two(value: int) -> int:
    return 2 ** round(math.log2(max(value, 2)))


class AutoencoderClassifier(nn.Module):
    """Autoencoder with a binary classifier head for tabular flow features."""

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        base = _nearest_power_of_two(input_dim)
        hidden = max(base // 2, 32)
        latent = max(base // 4, 16)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent, max(latent // 2, 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(latent // 2, 8), 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        logits = self.classifier(features).squeeze(-1)
        return features, reconstruction, logits
