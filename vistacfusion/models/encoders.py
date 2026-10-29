"""Frozen image encoders behind a common interface.

Every encoder exposes:
    embed_dim        : int          (channel width, e.g. 1024 for ViT-L, 768 for ViT-B)
    num_patches      : int          (spatial tokens, e.g. 196 for p16@224, 256 for p14@224)
    multiscale_layers: list[int]    (layer indices tapped for multi-scale features)
    forward(x)            -> (patch [B, N, E], cls [B, 1, E] | None)
    forward_multiscale(x) -> list of K  [B, N, E]  tensors

Supported encoders
------------------
  dinov3_vitl16    DINOv3 ViT-L/16 (HF transformers, gated weights)
  mae_vitl16       MAE ViT-L/16 (Meta, standard ViT checkpoint)
  siglip_vitl16    SigLIP ViT-L/16 (Google, HF hub, no CLS)
  clip_vitl14      CLIP ViT-L/14 (OpenAI, HF hub)
  dav2_vitl14      Depth Anything v2 ViT-L/14 (fine-tuned DINOv2)
  t3_large         T3 sensor encoder + shared trunk (two-file checkpoint)
  sparsh_dino_base Sparsh DINOv2-B tactile encoder (768 dim, 6-ch input, Fourier PE)
  sitr_b18         SITR B18 tactile encoder (768 dim, 18 calibration images, cached)
  (no checkpoint)  MockEncoder (deterministic stand-in for testing)
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────

def auto_layer_indices(depth, k=4):
    """k evenly-spaced layer indices ending at the last layer.

    depth 12 (ViT-B) -> (2, 5, 8, 11); depth 24 (ViT-L) -> (5, 11, 17, 23).
    """
    if not depth or depth < k:
        return tuple(range(max(1, k)))
    step = depth / k
    idx = sorted({min(depth - 1, int(round((i + 1) * step)) - 1) for i in range(k)})
    while len(idx) < k:
        for cand in range(depth - 1, -1, -1):
            if cand not in idx:
                idx.append(cand)
                break
        idx = sorted(set(idx))
    return tuple(idx[-k:])


def _resolve_multiscale(layers, depth, k=4):
    """Return validated multiscale layer indices.  Falls back to auto if any index >= depth."""
    if layers is not None and all(i < depth for i in layers):
        return sorted(layers)
    return list(auto_layer_indices(depth, k=k))


def _safe_load(path):
    """Load a ``.pth`` checkpoint, handling ZIP-compressed files that PyTorch rejects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError:
        import shutil
        import tempfile
        import zipfile

        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            fixed = os.path.join(tmp, "__fixed.pth")
            with zipfile.ZipFile(fixed, "w", compression=zipfile.ZIP_STORED) as zo:
                for root, _, fnames in os.walk(tmp):
                    for fn in fnames:
                        if fn == "__fixed.pth":
                            continue
                        fp = os.path.join(root, fn)
                        zo.write(fp, os.path.relpath(fp, tmp))
            return torch.load(fixed, map_location="cpu", weights_only=True)
        finally:
            shutil.rmtree(tmp)


# ──────────────────────────────────────────────────────────────────────
#  Shared ViT block (matches standard MAE / DINOv2 state-dict keys)
# ──────────────────────────────────────────────────────────────────────

class _Block(nn.Module):
    """Pre-norm ViT block with fused QKV and optional LayerScale.

    State-dict keys per block::

        norm1.{weight,bias}  attn.qkv.{weight,bias}  attn.proj.{weight,bias}
        norm2.{weight,bias}  mlp.fc1.{weight,bias}    mlp.fc2.{weight,bias}
        [ls1.gamma  ls2.gamma]           # only when layer_scale=True
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, layer_scale=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim)
        self.attn.proj = nn.Linear(dim, dim)
        self._nh, self._hd = num_heads, dim // num_heads
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, h)
        self.mlp.fc2 = nn.Linear(h, dim)
        self._ls = layer_scale
        if layer_scale:
            self.ls1 = nn.Module()
            self.ls1.gamma = nn.Parameter(torch.ones(dim))
            self.ls2 = nn.Module()
            self.ls2.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        B, N, C = x.shape
        y = self.attn.qkv(self.norm1(x)).reshape(B, N, 3, self._nh, self._hd)
        q, k, v = y.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        y = self.attn.proj(y.transpose(1, 2).reshape(B, N, C))
        x = x + (self.ls1.gamma * y if self._ls else y)
        y = self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))
        x = x + (self.ls2.gamma * y if self._ls else y)
        return x


# ──────────────────────────────────────────────────────────────────────
#  DINOv3 (existing)
# ──────────────────────────────────────────────────────────────────────

def _infer_dinov3_config(sd):
    """Infer a DINOv3ViTConfig from an HF-format state dict (works for any ViT size)."""
    from transformers import DINOv3ViTConfig

    pe = sd["embeddings.patch_embeddings.weight"]          # [hidden, 3, patch, patch]
    hidden = pe.shape[0]
    patch = pe.shape[-1]
    num_register = sd["embeddings.register_tokens"].shape[1]
    num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layer."))
    intermediate = sd["layer.0.mlp.up_proj.weight"].shape[0]
    gated = any("gate" in k for k in sd if k.startswith("layer.0.mlp"))
    return DINOv3ViTConfig(
        patch_size=patch,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden // 64,                  # DINOv3 head_dim = 64
        num_register_tokens=num_register,
        image_size=224,
        use_gated_mlp=gated,
    )


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 (HuggingFace ``transformers``).

    Loads the gated HF-format checkpoint directly. The token layout is
    [CLS, register x R, patch x N]; we expose patches and CLS, and intermediate layers for v2.
    """

    def __init__(self, model_name="dinov3_vitl16", weights=None, multiscale_layers=None):
        super().__init__()
        if weights is None:
            raise ValueError(
                "DINOv3 weights are gated. Pass a local checkpoint path, or use MockEncoder "
                "(set encoder.checkpoint: null) for scaffolding/tests."
            )
        from transformers import DINOv3ViTModel

        print(f"  [encoder] loading DINOv3 weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        cfg = _infer_dinov3_config(sd)
        self.dinov3 = DINOv3ViTModel(cfg)
        # HF nests the transformer layers under `model.`; embeddings/norm stay top-level.
        remap = {(f"model.{k}" if k.startswith("layer.") else k): v for k, v in sd.items()}
        self.dinov3.load_state_dict(remap, strict=True)

        self.embed_dim = cfg.hidden_size
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.num_register = cfg.num_register_tokens
        self._patch_start = 1 + self.num_register               # skip CLS + registers
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, cfg.num_hidden_layers)

        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode regardless of the parent's train/eval.
        super().train(mode)
        self.dinov3.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        tokens = self.dinov3(x).last_hidden_state          # [B, 1+R+N, E]
        patch = tokens[:, self._patch_start:]              # [B, N, E]
        cls = tokens[:, :1]                                # [B, 1, E]
        return patch, cls

    @torch.no_grad()
    def forward_multiscale(self, x):
        # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i.
        hs = self.dinov3(x, output_hidden_states=True).hidden_states
        return [hs[i + 1][:, self._patch_start:] for i in self.multiscale_layers]


# ──────────────────────────────────────────────────────────────────────
#  MAE ViT-L/16
# ──────────────────────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """Frozen MAE ViT-L/16 (Meta, patch=16, dim=1024, 24 layers, has CLS).

    Checkpoint format: ``{model: {cls_token, pos_embed, patch_embed.proj.*, blocks.N.*, norm.*}}``.
    """

    def __init__(self, weights, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading MAE weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]

        dim, patch, depth, heads = 1024, 16, 24, 16
        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # MAE pretrain checkpoint may include decoder keys; load encoder keys only
        self.load_state_dict(sd, strict=False)

        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  SigLIP ViT-L/16
# ──────────────────────────────────────────────────────────────────────

class SigLIPEncoder(nn.Module):
    """Frozen SigLIP ViT-L/16 (Google, patch=16, dim=1024, 196 tokens, no CLS).

    Loaded from HuggingFace hub via ``transformers.SiglipVisionModel``.
    SigLIP uses global average pooling instead of a CLS token; ``forward`` returns cls=None.
    """

    def __init__(self, model_name="google/siglip-large-patch16-224", multiscale_layers=None,
                 image_size=224):
        super().__init__()
        from transformers import SiglipVisionModel

        print(f"  [encoder] loading SigLIP from {model_name}")
        self.vit = SiglipVisionModel.from_pretrained(model_name)
        cfg = self.vit.config
        self.embed_dim = cfg.hidden_size
        native_grid = cfg.image_size // cfg.patch_size
        target_grid = image_size // cfg.patch_size
        if native_grid != target_grid:
            emb_mod = self.vit.embeddings
            old_pe = emb_mod.position_embedding.weight.data
            new_pe = old_pe.reshape(1, native_grid, native_grid, -1).permute(0, 3, 1, 2)
            new_pe = torch.nn.functional.interpolate(
                new_pe, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
            new_pe = new_pe.permute(0, 2, 3, 1).reshape(target_grid * target_grid, -1)
            emb_mod.position_embedding = nn.Embedding(
                target_grid * target_grid, self.embed_dim)
            emb_mod.position_embedding.weight.data.copy_(new_pe)
            emb_mod.register_buffer(
                "position_ids", torch.arange(target_grid * target_grid).unsqueeze(0))
            print(f"  [encoder] SigLIP pos embed interpolated {native_grid}²→{target_grid}²")
        self.num_patches = target_grid * target_grid
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, cfg.num_hidden_layers)
        for p in self.vit.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.vit(x).last_hidden_state, None

    @torch.no_grad()
    def forward_multiscale(self, x):
        hs = self.vit(x, output_hidden_states=True).hidden_states
        return [hs[i + 1] for i in self.multiscale_layers]


# ──────────────────────────────────────────────────────────────────────
#  CLIP ViT-L/14
# ──────────────────────────────────────────────────────────────────────

class CLIPEncoder(nn.Module):
    """Frozen CLIP ViT-L/14 (OpenAI, patch=14, dim=1024, 256 tokens, has CLS).

    Loaded from HuggingFace hub via ``transformers.CLIPVisionModel``.
    Token layout: [CLS, 256 patches].
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14", multiscale_layers=None):
        super().__init__()
        from transformers import CLIPVisionModel

        print(f"  [encoder] loading CLIP from {model_name}")
        self.vit = CLIPVisionModel.from_pretrained(model_name)
        cfg = self.vit.config
        self.embed_dim = cfg.hidden_size
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, cfg.num_hidden_layers)
        for p in self.vit.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        tokens = self.vit(x).last_hidden_state          # [B, 1+N, E]
        return tokens[:, 1:], tokens[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        hs = self.vit(x, output_hidden_states=True).hidden_states
        return [hs[i + 1][:, 1:] for i in self.multiscale_layers]


# ──────────────────────────────────────────────────────────────────────
#  Depth Anything v2 (DINOv2 ViT-L/14)
# ──────────────────────────────────────────────────────────────────────

class DAv2Encoder(nn.Module):
    """Frozen Depth Anything v2 encoder (DINOv2 ViT-L/14, patch=14, dim=1024, has CLS).

    Checkpoint is the full DAv2 model; only the ``pretrained.*`` keys are loaded.
    Uses LayerScale.  Positional embeddings are interpolated from the pretrained
    resolution (518 px) to ``image_size``.
    """

    def __init__(self, weights, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading DAv2 weights from {weights}")
        raw = _safe_load(weights)
        sd = {k[len("pretrained."):]: v for k, v in raw.items() if k.startswith("pretrained.")}

        dim, patch, depth, heads = 1024, 14, 24, 16
        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, layer_scale=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # Interpolate pos_embed from pretrained resolution (e.g. 518 -> 224)
        orig_pe = sd["pos_embed"]
        if orig_pe.shape[1] != 1 + self.num_patches:
            cls_pe = orig_pe[:, :1]
            patch_pe = orig_pe[:, 1:]
            g0 = int(patch_pe.shape[1] ** 0.5)
            patch_pe = F.interpolate(
                patch_pe.reshape(1, g0, g0, dim).permute(0, 3, 1, 2),
                size=(grid, grid), mode="bicubic", align_corners=False,
            ).permute(0, 2, 3, 1).reshape(1, -1, dim)
            sd["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)

        self.load_state_dict(sd, strict=True)
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  T3 (sensor encoder + shared trunk)
# ──────────────────────────────────────────────────────────────────────

class T3Encoder(nn.Module):
    """Frozen T3 encoder (sensor-specific blocks + shared trunk, patch=16, dim=1024, has CLS).

    Loads from two files inside ``weights_dir``:
      - ``encoder_mini.pth``: patch embed, CLS, pos_embed, and the first N sensor blocks.
      - ``trunk.pth``: shared trunk blocks and final LayerNorm.
    Blocks are concatenated into a single ``nn.ModuleList`` with re-indexed keys.
    """

    def __init__(self, weights_dir, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading T3 from {weights_dir}")
        enc_sd = torch.load(os.path.join(weights_dir, "encoder_mini.pth"),
                            map_location="cpu", weights_only=True)
        trunk_sd = torch.load(os.path.join(weights_dir, "trunk.pth"),
                              map_location="cpu", weights_only=True)

        dim, patch, heads = 1024, 16, 16
        n_enc = 1 + max(int(k.split(".")[1]) for k in enc_sd if k.startswith("blocks."))
        n_trunk = 1 + max(int(k.split(".")[1]) for k in trunk_sd if k.startswith("blocks."))
        depth = n_enc + n_trunk
        grid = image_size // patch

        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # Merge: encoder blocks keep their indices, trunk blocks are shifted by n_enc
        combined = {k: v for k, v in enc_sd.items()}
        for k, v in trunk_sd.items():
            if k.startswith("blocks."):
                parts = k.split(".", 2)
                combined[f"blocks.{int(parts[1]) + n_enc}.{parts[2]}"] = v
            else:
                combined[k] = v  # norm.weight, norm.bias
        self.load_state_dict(combined, strict=True)

        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  Sparsh (DINOv2-Base tactile encoder)
# ──────────────────────────────────────────────────────────────────────

class SparshEncoder(nn.Module):
    """Frozen Sparsh DINOv2-Base tactile encoder (dim=768, 12 layers, no CLS).

    The pretrained model expects 6-channel input; when given 3-channel images,
    channels are repeated (``x.repeat(1, 2, 1, 1)``) to match.
    Positional encoding is Fourier-based (learned frequency bands, not a learned table).
    Has 1 register token; ``forward`` returns cls=None.
    """

    def __init__(self, weights, multiscale_layers=None, image_size=224,
                 ssl_name="dinov2"):
        super().__init__()
        print(f"  [encoder] loading Sparsh weights from {weights}")
        sd = self._load_sparsh_weights(weights, ssl_name)

        dim = int(sd["norm.weight"].shape[0])
        in_ch = int(sd["patch_embed.proj.weight"].shape[1])
        patch = int(sd["patch_embed.proj.weight"].shape[-1])
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        n_reg = int(sd["register_tokens"].shape[1]) if "register_tokens" in sd else 0
        heads = dim // 64  # DINOv2 head_dim = 64

        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid
        self._in_ch = in_ch
        self._patch_start = n_reg
        self._grid = grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(in_ch, dim, patch, patch)
        if n_reg > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, dim))

        # Fourier positional embedding: frequency_bands [2, dim//4]
        nb = dim // 4
        self.pos_embed = nn.Module()
        self.pos_embed.frequency_bands = nn.Parameter(torch.zeros(2, nb))

        self.blocks = nn.ModuleList([_Block(dim, heads, layer_scale=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        result = self.load_state_dict(sd, strict=False)
        if result.missing_keys:
            print(f"  [encoder] Sparsh missing {len(result.missing_keys)} keys: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"  [encoder] Sparsh unexpected {len(result.unexpected_keys)} keys: {result.unexpected_keys[:5]}")
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_sparsh_weights(weights, ssl_name="dinov2"):
        _SSL_ENCODER_KEY = {
            "dino": "teacher_encoder.backbone",
            "dinov2": "teacher_encoder.backbone",
            "mae": "encoder",
            "ijepa": "target_encoder",
        }
        if weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(weights)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        raw = ckpt.get("model", ckpt)
        prefix = _SSL_ENCODER_KEY.get(ssl_name, "teacher_encoder.backbone") + "."
        sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        sd = {k: v for k, v in sd.items() if "mask_token" not in k and "masktoken" not in k}
        if not sd:
            raise KeyError(f"No keys under '{prefix}' in checkpoint")
        return sd

    def train(self, mode=True):
        super().train(False)
        return self

    def _fourier_pe(self):
        """Compute Fourier positional encoding from learned frequency bands."""
        bands = self.pos_embed.frequency_bands          # [2, nb]
        g = self._grid
        ys = torch.arange(g, device=bands.device, dtype=bands.dtype).unsqueeze(1) / g
        xs = torch.arange(g, device=bands.device, dtype=bands.dtype).unsqueeze(1) / g
        y_enc = torch.cat([torch.sin(ys * bands[0]), torch.cos(ys * bands[0])], dim=-1)  # [g, 2*nb]
        x_enc = torch.cat([torch.sin(xs * bands[1]), torch.cos(xs * bands[1])], dim=-1)  # [g, 2*nb]
        return torch.cat([
            y_enc.unsqueeze(1).expand(-1, g, -1),
            x_enc.unsqueeze(0).expand(g, -1, -1),
        ], dim=-1).reshape(1, g * g, -1)               # [1, N, dim]

    def _embed(self, x):
        B = x.shape[0]
        if x.shape[1] < self._in_ch:
            x = x.repeat(1, self._in_ch // x.shape[1], 1, 1)
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2) + self._fourier_pe()
        if self._patch_start > 0:
            t = torch.cat([self.register_tokens.expand(B, -1, -1), t], dim=1)
        return t

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, self._patch_start:], None

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, self._patch_start:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  TVL ViT-Base (Touch-Vision-Language tactile encoder)
# ──────────────────────────────────────────────────────────────────────

class TVLEncoder(nn.Module):
    """Frozen TVL tactile encoder (ViT-B/16, dim=768, 12 layers, has CLS).

    Checkpoint format: ``{model: {tactile_encoder.*, logit_scale}}``.
    We load only the ``tactile_encoder.*`` keys, discarding the CLIP
    projection head (``head.*``, ``fc_norm.*``) and ``logit_scale``.
    """

    def __init__(self, weights, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading TVL weights from {weights}")
        raw = torch.load(weights, map_location="cpu", weights_only=False)
        sd_raw = raw["model"] if "model" in raw else raw

        prefix = "tactile_encoder."
        sd = {}
        for k, v in sd_raw.items():
            if k.startswith(prefix) and "head." not in k:
                clean = k[len(prefix):]
                clean = clean.replace("fc_norm.", "norm.")
                sd[clean] = v

        dim, patch, depth, heads = 768, 16, 12, 12
        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        self.load_state_dict(sd, strict=True)
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  MockEncoder (testing stand-in)
# ──────────────────────────────────────────────────────────────────────

class MockEncoder(nn.Module):
    """Deterministic frozen stand-in for DINOv3 with identical output shapes.

    Patch-embed conv -> N tokens; CLS = linear(mean of tokens). forward_multiscale returns K
    distinct projected copies. All params frozen, like the real encoder.
    """

    def __init__(self, embed_dim=1024, patch_size=16, image_size=224, multiscale_k=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.multiscale_layers = list(range(multiscale_k))

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_proj = nn.Linear(embed_dim, embed_dim)
        # per-scale linear so multiscale taps are distinct but deterministic
        self.scale_proj = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(multiscale_k)]
        )
        for p in self.parameters():
            p.requires_grad = False

    def _tokens(self, x):
        f = self.patch_embed(x)                       # [B, E, g, g]
        B, E, g, _ = f.shape
        return f.flatten(2).transpose(1, 2)           # [B, N, E]

    @torch.no_grad()
    def forward(self, x):
        patch = self._tokens(x)                       # [B, N, E]
        cls = self.cls_proj(patch.mean(dim=1, keepdim=True))   # [B, 1, E]
        return patch, cls

    @torch.no_grad()
    def forward_multiscale(self, x):
        patch = self._tokens(x)
        return [proj(patch) for proj in self.scale_proj]       # K x [B, N, E]


# ──────────────────────────────────────────────────────────────────────
#  SITR B18 (tactile encoder with calibration)
# ──────────────────────────────────────────────────────────────────────

class SITREncoder(nn.Module):
    """Frozen SITR encoder (ViT-B/16, dim=768, depth=12, 18 calibration images).

    Calibration is handled internally: the 18 calibration images from a
    reference sim unit are loaded once at init, embedded, and cached as a
    buffer.  Every forward pass appends the cached calibration tokens to the
    sequence (same mechanism as gsrl's ``encoder.cache_calibration()``), then
    strips them before returning — so the caller sees the standard
    ``(patch [B,N,768], cls [B,1,768])`` interface with no calibration args.
    """

    def __init__(self, weights, calibration_dir=None, num_calibration=18,
                 multiscale_layers=None, image_size=224, fixed_crop=None):
        super().__init__()
        # Calibration images get the same fixed center crop as the data (data cfg
        # `fixed_crop`); encoder cfg `fixed_crop` must match it. None -> FIXED_CROP.
        self._fixed_crop = fixed_crop
        print(f"  [encoder] loading SITR from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]

        dim, patch, heads, depth = 768, 16, 12, 12
        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid
        self._num_calibration = num_calibration

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        if num_calibration > 0:
            self.c_patch_embed = nn.Module()
            self.c_patch_embed.proj = nn.Conv2d(num_calibration * 3, dim, patch, patch)
            self.c_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        compatible = {k: v for k, v in sd.items()
                      if k in self.state_dict() and v.shape == self.state_dict()[k].shape}
        self.load_state_dict(compatible, strict=False)
        skipped = [k for k in sd if k not in compatible]
        if skipped:
            print(f"  [encoder] skipped {len(skipped)} keys: "
                  f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
        print(f"  [encoder] loaded {len(compatible)}/{len(sd)} keys "
              f"(depth={depth}, dim={dim}, calib={num_calibration})")

        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("_calib_cache", None)
        if calibration_dir and num_calibration > 0:
            self._build_calib_cache(calibration_dir, image_size)

    @torch.no_grad()
    def _build_calib_cache(self, calib_dir, image_size):
        from PIL import Image
        import numpy as np
        from ..data.transforms import FIXED_CROP, fixed_center_crop, ToTensorResize
        crop = FIXED_CROP if self._fixed_crop is None else float(self._fixed_crop)
        xf = ToTensorResize((image_size, image_size),
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375])
        ids = list(range(1, self._num_calibration + 1))
        imgs = []
        for i in ids:
            img = np.array(Image.open(os.path.join(calib_dir, f"{i:04d}.png")),
                           dtype=np.float32)
            imgs.append(xf(fixed_center_crop(img, crop=crop)))
        calib_tensor = torch.cat(imgs, dim=0).unsqueeze(0)
        c_enc = self.c_patch_embed.proj(calib_tensor).flatten(2).transpose(1, 2)
        c_enc = c_enc + self.c_pos_embed
        self._calib_cache = c_enc
        print(f"  [encoder] calibration cache built from {calib_dir} "
              f"-> {tuple(self._calib_cache.shape)}")

    def set_calibration_dir(self, calib_dir, image_size=224, fixed_crop=None):
        if fixed_crop is not None:
            self._fixed_crop = fixed_crop
        self._build_calib_cache(calib_dir, image_size)

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        t = t + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        t = torch.cat([cls, t], dim=1)
        if self._calib_cache is not None:
            t = torch.cat([t, self._calib_cache.expand(B, -1, -1)], dim=1)
        return t

    def _strip_calib(self, t):
        if self._num_calibration > 0:
            return t[:, :self.num_patches + 1, :]
        return t

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(self._strip_calib(t))
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(self._strip_calib(t)[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────


class SparshXEncoder(nn.Module):
    """Frozen Sparsh-X (image-only, base) tactile encoder pretrained with the official
    ``sparsh-multisensory-touch`` DINO code on the VisTacFusion co-training images.

    The architecture is instantiated from the Sparsh-X repo itself (``repo_path`` on
    sys.path): ``dit_base`` = 8 per-modality ViT blocks + 4 bottleneck-fusion blocks
    (embed 768, 12 heads, 1 register token, 4 bottleneck tokens, sinusoidal pos-embed).
    With a single modality the fusion blocks reduce to plain ViT blocks with 4 extra
    bottleneck tokens, i.e. a 12-layer ViT-B/16 -> 196 patch tokens of dim 768.
    Weights: EMA teacher backbone (``teacher_encoder.backbone.*``) of the DINO checkpoint,
    as is standard for DINO downstream use. Input: ImageNet-normalized [B,3,H,W] (as all
    VisTacFusion encoders); it is mapped back to [0,1] and normalized with the encoder's
    own stored ``img_avg/img_div`` buffers. ``forward`` returns cls=None.
    """

    def __init__(self, weights, multiscale_layers=None, image_size=224,
                 repo_path=None,
                 which="teacher", img_avg=None, img_std=None):
        super().__init__()
        import sys as _sys
        if repo_path and repo_path not in _sys.path:
            _sys.path.insert(0, repo_path)
        # requires the official sparsh-multisensory-touch repo (encoder cfg `repo_path`)
        from tactile_ssl.model.d360_transformer import dit_base
        from tactile_ssl.model.layers import Attention
        from omegaconf import OmegaConf
        print(f"  [encoder] loading Sparsh-X ({which}) from {weights}")
        norm = OmegaConf.create({"img": {"avg": list(img_avg or [0.5, 0.5, 0.5]),
                                         "std": list(img_std or [0.25, 0.25, 0.25]), "div": 1}})
        self.model = dit_base(
            use_img=True, use_mic=False, use_imu=False, use_pressure=False,
            sensor_sizes={"img": [image_size, image_size]}, sensor_chans={"img": 3},
            patch_sizes={"img": 16}, num_register_tokens=1, fusion_type="bottleneck",
            fusion_layer=8, num_bottlenecks=4, drop_path_rate=0.0, pos_embed_fn="sinusoidal",
            normalization=norm, attn_class=Attention)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        raw = ckpt.get("model", ckpt)
        prefix = f"{which}_encoder.backbone."
        sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        if not sd:                                   # bare backbone state dict
            sd = raw
        res = self.model.load_state_dict(sd, strict=True)
        self.embed_dim = 768
        self.num_patches = (image_size // 16) ** 2
        self._n_reg = 1
        self._n_bn = 4
        depth = len(self.model.blocks) + len(self.model.fusion_blocks)
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def _layers(self, x):
        """Yield per-layer token states (register + patches, bottleneck stripped)."""
        m = self.model
        x01 = (x * self._in_std + self._in_mean).clamp(0, 1)
        xs = m.pre_embed({"img": x01})
        xs = m.embed(xs)
        xs = m.prepare_tokens(xs, None)
        h = xs["img"]
        outs = []
        for blk in m.blocks:
            h = blk["img"](h)
            outs.append(h)
        bn = m.bottleneck.expand(h.shape[0], -1, -1)
        for blk in m.fusion_blocks:
            h = blk["img"](torch.cat([bn, h], dim=1))
            bn = h[:, : self._n_bn]
            h = h[:, self._n_bn:]
            outs.append(h)
        return outs

    @torch.no_grad()
    def forward(self, x):
        outs = self._layers(x)
        h = self.model.norm["img"](outs[-1])
        return h[:, self._n_reg:], None

    @torch.no_grad()
    def forward_multiscale(self, x):
        outs = self._layers(x)
        return [outs[i][:, self._n_reg:] for i in self.multiscale_layers]


def build_encoder(enc_cfg, image_size):
    """Dispatch on ``enc_cfg["name"]``; fall back to MockEncoder without a checkpoint."""
    checkpoint = enc_cfg.get("checkpoint", None)
    if not checkpoint:
        return MockEncoder(
            embed_dim=enc_cfg.get("embed_dim", 1024),
            patch_size=enc_cfg.get("patch_size", 16),
            image_size=image_size,
        )
    name = enc_cfg.get("name", "dinov3_vitl16")
    ms = enc_cfg.get("multiscale_layers", None)

    if name == "dinov3_vitl16":
        return DINOv3Encoder(model_name=name, weights=checkpoint, multiscale_layers=ms)
    if name == "mae_vitl16":
        return MAEEncoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size)
    if name == "siglip_vitl16":
        model_id = enc_cfg.get("model_id", "google/siglip-large-patch16-256")
        return SigLIPEncoder(model_name=model_id, multiscale_layers=ms, image_size=image_size)
    if name == "clip_vitl14":
        model_id = enc_cfg.get("model_id", "openai/clip-vit-large-patch14")
        return CLIPEncoder(model_name=model_id, multiscale_layers=ms)
    if name == "dav2_vitl14":
        return DAv2Encoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size)
    if name == "t3_large":
        return T3Encoder(weights_dir=checkpoint, multiscale_layers=ms, image_size=image_size)
    if name == "sparsh_dino_base":
        ssl_name = enc_cfg.get("ssl_name", "dinov2")
        return SparshEncoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size,
                             ssl_name=ssl_name)
    if name == "tvl_vitb16":
        return TVLEncoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size)
    if name == "sparshx_base":
        return SparshXEncoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size,
                              repo_path=enc_cfg.get("repo_path", None),
                              which=enc_cfg.get("which", "teacher"),
                              img_avg=enc_cfg.get("img_avg", None), img_std=enc_cfg.get("img_std", None))
    if name == "sitr_b18":
        calib_dir = enc_cfg.get("calibration_dir", None)
        num_calib = enc_cfg.get("num_calibration", 18)
        return SITREncoder(weights=checkpoint, calibration_dir=calib_dir,
                           num_calibration=num_calib, multiscale_layers=ms,
                           image_size=image_size,
                           fixed_crop=enc_cfg.get("fixed_crop", None))

    raise ValueError(
        f"Unknown encoder name {name!r}. Available: dinov3_vitl16, mae_vitl16, "
        f"siglip_vitl16, clip_vitl14, dav2_vitl14, t3_large, sparsh_dino_base, "
        f"tvl_vitb16, sitr_b18, sparshx_base"
    )
