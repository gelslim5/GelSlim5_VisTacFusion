"""Projection + embeddings: encoder dim E -> trunk dim D.

Per branch: Linear(E -> D) per token + a learned modality embedding.
The 2D positional embedding lives in the model (not here) so the same instance can be added
to either tactile patch tokens or the RGB-only mask tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BranchProjection(nn.Module):
    """Project one modality's encoder tokens to the trunk dim and add its modality embedding."""

    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.modality_emb, std=0.02)

    def forward(self, tokens):
        """tokens: [B, N, in_dim] -> [B, N, out_dim] + modality embedding."""
        return self.proj(tokens) + self.modality_emb


class SpatialPosEmbedding(nn.Module):
    """Learned positional embedding for the spatial queries (=196).

    Required: the DPT Reassemble reshapes 196 -> 14x14 and needs positional structure. Added
    to whatever fills the spatial queries (tactile patches or RGB-only mask tokens).
    """

    def __init__(self, num_tokens=196, dim=768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos
