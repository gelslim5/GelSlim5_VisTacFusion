"""Misc utilities: seeding, parameter counting."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def param_count_str(module: torch.nn.Module) -> str:
    total, trainable = count_parameters(module)
    return f"{total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable"
