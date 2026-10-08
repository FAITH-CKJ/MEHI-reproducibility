"""State-compatible multilayer perceptron used for the six-class mosaics."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer_id, layer in enumerate(self.layers):
            outputs = F.relu(layer(outputs)) if layer_id < self.num_layers - 1 else layer(outputs)
        return outputs


class LegacyLayerNorm(nn.Module):
    """Layer-normalization module retained to match the archived state dictionaries."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


class Model(nn.Module):
    """Three-layer MLP with legacy modules required by the retained checkpoints."""

    def __init__(self, input_dim: int, output_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = 1024
        self.featEmbedding = nn.Sequential(nn.Linear(14, self.emb_dim), nn.ReLU())
        self.fusionPred = nn.Linear(self.emb_dim, 9)
        self.proj_norm = LegacyLayerNorm(self.emb_dim)
        self.MLP = MLP(input_dim, 256, output_dim, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.MLP(inputs)
        return torch.sigmoid(outputs) if not self.training else outputs


def checkpoint_input_dim(checkpoint: str | Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return int(state["MLP.layers.0.weight"].shape[1])


def load_checkpoint(checkpoint: str | Path, device: torch.device) -> Model:
    checkpoint = Path(checkpoint)
    model = Model(checkpoint_input_dim(checkpoint), 6)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
