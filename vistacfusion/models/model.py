"""VisTacFusion — visuo-tactile multi-task model with decoupled dense / pose paths.

  DPT path:  tactile-encoder multiscale taps at native dim → optional gated RGB injection
             (cross-attention to the fusion bottleneck) → DPT decoder → depth / normal.
  Pose path: encoder tokens → projection (→ trunk dim) → asymmetric bottleneck fusion
             trunk → pose MLP → planar pose (cos θ, sin θ, t_x, t_y).

Modality dropout (both / tactile / rgb) acts on the pose path; RGB injection into the DPT
taps is sampled independently at training time. The ReZero gate of the injection starts at
0, so without RGB the dense path is exactly the tactile-only decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .fusion import AttentionBlock, FusionTrunk
from .heads.dpt import DPTHead
from .heads.pose import PoseHead
from .projection import BranchProjection, SpatialPosEmbedding

VALID_CONFIGS = ("both", "tactile", "rgb")


def _config_flags(config):
    """(use_rgb, use_tactile) for each modality config."""
    if config == "both":
        return True, True
    if config == "tactile":
        return False, True
    if config == "rgb":
        return True, False
    raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")


def _resample_tokens(tokens, num_target):
    """Resample a [B, N, C] square token grid to num_target tokens (bilinear on the grid)."""
    B, N, C = tokens.shape
    if N == num_target:
        return tokens
    g_in, g_out = int(round(N ** 0.5)), int(round(num_target ** 0.5))
    x = tokens.transpose(1, 2).reshape(B, C, g_in, g_in)
    x = nn.functional.interpolate(x, size=(g_out, g_out), mode="bilinear", align_corners=False)
    return x.reshape(B, C, g_out * g_out).transpose(1, 2)


class TapInjection(nn.Module):
    """Per-tap residual RGB injection: tap += gate * CrossAttn(Q=tap, K=V=bottleneck).

    Q is at encoder dim (e.g. 1024), KV at trunk dim (e.g. 768).
    ReZero gate init 0 → starts as pure encoder taps, learns to inject.
    """

    def __init__(self, q_dim, kv_dim, num_heads, dropout=0.0, gate_init=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, tap, bottleneck):
        B, N, _ = tap.shape
        q = self.q_proj(self.norm_q(tap))
        kv_normed = self.norm_kv(bottleneck)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        return tap + self.gate * self.out_proj(attn)


class VisuoTactileModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.trunk_dim = cfg.trunk_dim                              # 768

        # ---- Encoders (frozen) ----
        enc_cfg = dict(cfg.encoder)
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.tactile_encoder = build_encoder(enc_cfg, self.image_size)

        rgb_enc_cfg = cfg.get("rgb_encoder", None)
        if rgb_enc_cfg is not None:
            rgb_enc_cfg = dict(rgb_enc_cfg)
            if rgb_enc_cfg.get("multiscale_layers", None) is None:
                rgb_enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
            self.rgb_encoder = build_encoder(rgb_enc_cfg, self.image_size)
        elif enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, self.image_size)

        self.enc_dim = self.tactile_encoder.embed_dim
        self.rgb_enc_dim = self.rgb_encoder.embed_dim
        self.num_spatial = self.tactile_encoder.num_patches          # 196 or 256

        # ---- Pose path: projection (enc_dim→768) + trunk ----
        self.tactile_proj = BranchProjection(self.enc_dim, self.trunk_dim)
        self.rgb_proj = BranchProjection(self.rgb_enc_dim, self.trunk_dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)
        self.use_rgb_pos = cfg.projection.get("rgb_positional_embedding", False)
        if self.use_rgb_pos:
            self.rgb_spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)

        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, self.trunk_dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, self.trunk_dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        self.trunk = FusionTrunk(cfg.fusion_trunk, self.trunk_dim)

        # ---- Optional object embedding (sim+real co-training with known objects) ----
        self.use_obj_emb = cfg.tokens.get("object_embedding", False)
        if self.use_obj_emb:
            num_obj = cfg.tokens.get("num_objects", 20)
            self.obj_embedding = nn.Embedding(num_obj, self.trunk_dim)

        # ---- DPT path: direct encoder taps + RGB injection ----
        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)
        # rgb-only fallback: DPT runs on RGB encoder taps. If the RGB encoder dim differs
        # from the tactile dim (e.g. SITR 768 vs MAE 1024) a per-tap linear projection maps
        # them to enc_dim; it is used ONLY in the rgb-only config.
        self.rgb_tap_proj = None
        if self.rgb_enc_dim != self.enc_dim:
            self.rgb_tap_proj = nn.ModuleList(
                [nn.Linear(self.rgb_enc_dim, self.enc_dim) for _ in range(4)])

        self.tap_inject = nn.ModuleList([
            TapInjection(
                q_dim=self.enc_dim,
                kv_dim=self.trunk_dim,
                num_heads=cfg.fusion_trunk.num_heads,
                dropout=cfg.fusion_trunk.dropout,
                gate_init=cfg.heads.dpt.get("inject_gate_init", 0.0),
            )
            for _ in range(4)
        ])

        # ---- Heads ----
        self.dpt = DPTHead(
            embed_dim=self.enc_dim,
            features=cfg.heads.dpt.features,
            dropout=cfg.heads.dpt.dropout,
            out_depth_channels=cfg.heads.dpt.out_depth_channels,
            out_normal_channels=cfg.heads.dpt.out_normal_channels,
        )
        self.pose_head = PoseHead(
            dim=self.trunk_dim,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            pose_mode=cfg.heads.pose.pose_mode,
            rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
            use_spatial_pool=cfg.heads.pose.get("use_spatial_pool", True),
        )

    # ------------------------------------------------------------------ helpers

    def _build_pose_memory(self, rgb_patch, rgb_cls):
        """RGB -> pose memory M = [B, N+1, trunk_dim]."""
        patch = self.rgb_proj(rgb_patch)
        if rgb_cls is not None:
            cls = self.rgb_proj(rgb_cls)
        else:
            cls = self.pose_mask.expand(patch.shape[0], -1, -1)
        if self.use_rgb_pos:
            patch = self.rgb_spatial_pos(patch)
        return torch.cat([patch, cls], dim=1)

    def _build_pose_queries(self, tac_patch, tac_cls, use_tactile, B, device):
        """Build N+1 pose queries at trunk_dim."""
        if use_tactile:
            spatial = self.spatial_pos(self.tactile_proj(tac_patch))
            if tac_cls is not None:
                pose_q = self.tactile_proj(tac_cls)
            else:
                pose_q = self.pose_mask.expand(B, -1, -1)
        else:
            spatial = self.spatial_pos(self.spatial_mask.expand(B, -1, -1))
            pose_q = self.pose_mask.expand(B, -1, -1)
        return torch.cat([spatial, pose_q], dim=1)

    # ------------------------------------------------------------------ forward

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        """
        Args:
            config: modality config for the POSE path ("both"/"tactile"/"rgb").
            inject_rgb_to_dpt: whether to inject RGB into DPT taps.
                None = auto (inject when config has RGB).
                Can be overridden for decoupled dropout training.
            encoder_cache: pre-computed encoder outputs (skips frozen forward).
            object_ids: (B,) int tensor of object class indices (for co-training).
            domain_ids: (B,) int tensor (0=sim, 1=real); accepted for interface parity.
        """
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb, use_tactile = _config_flags(config)

        ref = tactile if use_tactile else rgb
        B, device = ref.shape[0], ref.device

        if inject_rgb_to_dpt is None:
            inject_rgb_to_dpt = use_rgb

        # ---- Encode (frozen) ----
        tac_enc = encoder_cache.get("tactile") if encoder_cache else None
        rgb_enc = encoder_cache.get("rgb") if encoder_cache else None

        tac_patch = tac_cls = rgb_patch = rgb_cls = None
        if use_tactile:
            tac_patch, tac_cls = tac_enc if tac_enc else self.tactile_encoder(tactile)
        if use_rgb:
            rgb_patch, rgb_cls = rgb_enc if rgb_enc else self.rgb_encoder(rgb)

        # ---- Pose path: projection(768) → trunk → pose head ----
        pose_memory = self._build_pose_memory(rgb_patch, rgb_cls) if use_rgb else None
        pose_queries = self._build_pose_queries(
            tac_patch, tac_cls, use_tactile, B, device)

        if self.use_obj_emb and object_ids is not None:
            obj_emb = self.obj_embedding(object_ids).unsqueeze(1)  # (B, 1, D)
            pose_queries = pose_queries + obj_emb

        trunk_taps, pose_token, bottleneck = self.trunk(
            pose_queries, pose_memory, use_rgb)

        spatial_queries = trunk_taps[-1]
        pose = self.pose_head(pose_token, spatial_queries=spatial_queries)

        # ---- DPT path: encoder multiscale taps ----
        tac_ms = encoder_cache.get("tactile_ms") if encoder_cache else None
        rgb_ms = encoder_cache.get("rgb_ms") if encoder_cache else None
        if use_tactile:
            ms = tac_ms if tac_ms is not None else self.tactile_encoder.forward_multiscale(tactile)
            dpt_taps = [self.dpt_pos(t) for t in ms]
        else:
            # rgb-only: DPT falls back to RGB encoder taps. A different patch grid (MAE
            # 14x14 vs DAv2 16x16) is bilinearly resampled to the tactile token grid; a
            # different dim (SITR/Sparsh 768 vs MAE 1024) goes through rgb_tap_proj.
            ms = rgb_ms if rgb_ms is not None else self.rgb_encoder.forward_multiscale(rgb)
            if self.rgb_tap_proj is not None:
                ms = [proj(t) for proj, t in zip(self.rgb_tap_proj, ms)]
            ms = [_resample_tokens(t, self.num_spatial) for t in ms]
            dpt_taps = [self.dpt_pos(t) for t in ms]

        if inject_rgb_to_dpt and use_rgb:
            dpt_taps = [inj(t, bottleneck)
                        for inj, t in zip(self.tap_inject, dpt_taps)]
        depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


class SingleEncoderModel(nn.Module):
    """Single-encoder baseline: frozen encoder → DPT + PoseHead, no fusion trunk.

    Architecture:
        input → encoder (frozen) → CLS [B,1,E] + patches [B,N,E]
                                      ↓                ↓
                                multiscale taps     CLS + mean(patches)
                                      ↓                ↓
                                 DPT decoder        Pose head
                                (depth+normal)      (SE(2) pose)

    `input_modality: rgb` in the model config feeds the RGB image instead of the tactile one.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size

        enc_cfg = dict(cfg.encoder)
        if enc_cfg.get("multiscale_layers", None) is None:
            enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.encoder = build_encoder(enc_cfg, self.image_size)

        self.enc_dim = self.encoder.embed_dim
        self.num_spatial = self.encoder.num_patches

        self.use_obj_emb = cfg.tokens.get("object_embedding", False)
        if self.use_obj_emb:
            num_obj = cfg.tokens.get("num_objects", 20)
            self.obj_embedding = nn.Embedding(num_obj, self.enc_dim)

        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)

        self.dpt = DPTHead(
            embed_dim=self.enc_dim,
            features=cfg.heads.dpt.features,
            dropout=cfg.heads.dpt.dropout,
            out_depth_channels=cfg.heads.dpt.out_depth_channels,
            out_normal_channels=cfg.heads.dpt.out_normal_channels,
        )
        self.pose_head = PoseHead(
            dim=self.enc_dim,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            pose_mode=cfg.heads.pose.pose_mode,
            rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72),
            use_spatial_pool=cfg.heads.pose.get("use_spatial_pool", True),
        )

    @property
    def tactile_encoder(self):
        return self.encoder

    @property
    def rgb_encoder(self):
        return self.encoder

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        # config / inject_rgb_to_dpt / domain_ids accepted for interface parity (unused).
        inp = rgb if self.cfg.get("input_modality") == "rgb" else tactile

        cache_key = "rgb" if self.cfg.get("input_modality") == "rgb" else "tactile"
        enc = encoder_cache.get(cache_key) if encoder_cache else None
        tac_patch, tac_cls = enc if enc else self.encoder(inp)

        if self.use_obj_emb and object_ids is not None:
            obj_emb = self.obj_embedding(object_ids).unsqueeze(1)
            tac_patch = tac_patch + obj_emb
            if tac_cls is not None:
                tac_cls = tac_cls + obj_emb

        ms_key = "rgb_ms" if self.cfg.get("input_modality") == "rgb" else "tactile_ms"
        cached_ms = encoder_cache.get(ms_key) if encoder_cache else None
        ms = cached_ms if cached_ms is not None else self.encoder.forward_multiscale(inp)
        dpt_taps = [self.dpt_pos(t) for t in ms]
        depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        if tac_cls is None:
            # encoders without a usable CLS token (keep_cls: false, e.g. Sparsh):
            # mean-pooled patch tokens stand in for the pose token
            tac_cls = tac_patch.mean(dim=1, keepdim=True)
        pose = self.pose_head(tac_cls, spatial_queries=tac_patch)

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


def build_model(cfg):
    model_type = cfg.get("model_type", "visuo_tactile")
    if model_type == "single_encoder":
        return SingleEncoderModel(cfg)
    if model_type == "mvitac":
        from .mvitac import MViTacPoseModel
        mc = cfg.mvitac
        return MViTacPoseModel(
            tactile_ckpt=mc.tactile_ckpt,
            vision_ckpt=mc.vision_ckpt,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            num_objects=cfg.tokens.get("num_objects", 20),
            use_obj_emb=cfg.tokens.get("object_embedding", True),
        )
    if model_type == "vital":
        from .vital import VITaLPoseModel
        vc = cfg.vital
        return VITaLPoseModel(
            tactile_ckpt=vc.tactile_ckpt,
            vision_ckpt=vc.vision_ckpt,
            hidden_dim=cfg.heads.pose.hidden_dim,
            dropout=cfg.heads.pose.dropout,
            num_objects=cfg.tokens.get("num_objects", 20),
            use_obj_emb=cfg.tokens.get("object_embedding", True),
            dpt_features=vc.get("dpt_features", 128),
            image_size=cfg.image_size,
        )
    return VisuoTactileModel(cfg)
