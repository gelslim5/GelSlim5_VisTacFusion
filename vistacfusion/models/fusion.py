"""Fusion trunk: bottleneck tokens + a 3-step layer, repeated L times.

Output rows follow the query count; K/V count is independent -- so the decoder input stays
fixed at 4x(B,196,D) + (B,1,D) in every config.

Each layer, in order:
  (1) Bottleneck <- RGB memory   [cross-attn, no FFN]
  (2) Queries    <- Bottleneck   [cross-attn + FFN]
  (3) Queries self-attention     [self-attn + FFN]

By config: both / rgb-only run (1)(2)(3); tactile-only runs only (3) (no RGB -> bottleneck idle).

Flags:
  bottleneck_continuity {carry, reset} -- carry the condensed bottleneck across layers, or
    reset it to the learnable init each layer.
  fusion_variant {asymmetric, symmetric_coattention} -- symmetric also lets RGB memory attend
    to the queries (the control for "tactile is the anchor").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Pre-norm (cross- or self-)attention + residual, with an optional FFN sub-block.

    output length follows ``query``; ``context`` length is independent.
    """

    def __init__(self, dim, num_heads, dropout=0.0, use_ffn=True, ffn_mult=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ffn = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ffn_mult, dim),
                nn.Dropout(dropout),
            )

    def forward(self, query, context=None):
        if context is None:
            context = query
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + attn_out
        if self.use_ffn:
            query = query + self.ffn(self.norm_ffn(query))
        return query


class FusionTrunkLayer(nn.Module):
    """One (1)(2)(3) layer (+ optional symmetric memory update)."""

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0, fusion_variant="asymmetric"):
        super().__init__()
        self.fusion_variant = fusion_variant
        # (1) bottleneck <- RGB memory (cross-attn, no FFN)
        self.bottleneck_from_rgb = AttentionBlock(
            dim, num_heads, dropout, use_ffn=False, ffn_mult=ffn_mult
        )
        # (2) queries <- bottleneck (cross-attn + FFN)
        self.queries_from_bottleneck = AttentionBlock(
            dim, num_heads, dropout, use_ffn=True, ffn_mult=ffn_mult
        )
        # (3) queries self-attn (+ FFN)
        self.queries_self = AttentionBlock(
            dim, num_heads, dropout, use_ffn=True, ffn_mult=ffn_mult
        )
        if fusion_variant == "symmetric_coattention":
            # control: RGB memory also attends to the queries (RGB produces queries too)
            self.memory_from_queries = AttentionBlock(
                dim, num_heads, dropout, use_ffn=True, ffn_mult=ffn_mult
            )

    def forward(self, queries, bottleneck, memory, use_rgb):
        if use_rgb:
            if self.fusion_variant == "symmetric_coattention":
                memory = self.memory_from_queries(memory, queries)
            bottleneck = self.bottleneck_from_rgb(bottleneck, memory)   # (1)
            queries = self.queries_from_bottleneck(queries, bottleneck)  # (2)
        queries = self.queries_self(queries)                            # (3)
        return queries, bottleneck, memory


class FusionTrunk(nn.Module):
    """Stack of L FusionTrunkLayers + the shared learnable bottleneck tokens.

    Returns:
      taps          : list of len(tap_layers) spatial-query maps, each [B, num_spatial, D]
      pose_token    : [B, 1, D]   (the pose query after the final layer)
      bottleneck    : [B, m, D]   (the condensed bottleneck after the final layer; used by DPT v2)
    """

    def __init__(self, cfg, dim):
        super().__init__()
        self.dim = dim
        self.num_layers = cfg.num_layers
        self.num_bottleneck = cfg.num_bottleneck_tokens
        self.continuity = cfg.bottleneck_continuity      # {carry, reset}
        self.tap_layers = list(cfg.tap_layers)
        assert len(self.tap_layers) == 4, "DPT needs exactly 4 taps."
        assert all(0 <= t < self.num_layers for t in self.tap_layers), "tap_layers out of range"

        self.bottleneck = nn.Parameter(torch.zeros(1, self.num_bottleneck, dim))
        nn.init.trunc_normal_(self.bottleneck, std=0.02)

        self.layers = nn.ModuleList(
            [
                FusionTrunkLayer(
                    dim=dim,
                    num_heads=cfg.num_heads,
                    ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                    fusion_variant=cfg.fusion_variant,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, queries, memory, use_rgb):
        B = queries.shape[0]
        bottleneck = self.bottleneck.expand(B, -1, -1)
        taps = []
        for i, layer in enumerate(self.layers):
            if self.continuity == "reset":
                bottleneck = self.bottleneck.expand(B, -1, -1)
            queries, bottleneck, memory = layer(queries, bottleneck, memory, use_rgb)
            if i in self.tap_layers:
                taps.append(queries[:, :-1])          # spatial queries (drop the pose query)
        pose_token = queries[:, -1:]                  # [B, 1, D]
        return taps, pose_token, bottleneck
