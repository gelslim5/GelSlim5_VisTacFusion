"""VITaL baseline: CLIP-pretrained ResNet-18 (tactile + vision) → DPT + PoseHead.

Reimplemented from https://github.com/Abraham190137/TactileACT
Two-stream ResNet-18 encoders pretrained with CLIP contrastive loss.

Unlike MViTac (global vectors only), VITaL preserves spatial feature maps,
enabling dense prediction (depth + normal) via DPT in addition to pose.

Downstream: frozen tactile + vision encoders → multiscale taps for DPT,
global pooled features for PoseHead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _replace_bn_with_gn(module, features_per_group=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(
                num_groups=child.num_features // features_per_group,
                num_channels=child.num_features))
        else:
            _replace_bn_with_gn(child, features_per_group)
    return module


def make_resnet18_encoder(pretrained=False, features_per_group=16):
    """ResNet-18 with GroupNorm (VITaL convention), returns full model for
    extracting multiscale features from layer1-4."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    _replace_bn_with_gn(resnet, features_per_group)
    return resnet


class ResNet18MultiScale(nn.Module):
    """Wraps a ResNet-18 to expose multiscale feature maps from layer1-4
    and a global pooled vector, matching the encoder interface we need."""

    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 56×56, 64
        self.layer2 = resnet.layer2   # 28×28, 128
        self.layer3 = resnet.layer3   # 14×14, 256
        self.layer4 = resnet.layer4   # 7×7, 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        """Returns (global_vec [B,512], multiscale_maps list of 4 tensors)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)    # [B, 64, 56, 56]
        f2 = self.layer2(f1)   # [B, 128, 28, 28]
        f3 = self.layer3(f2)   # [B, 256, 14, 14]
        f4 = self.layer4(f3)   # [B, 512, 7, 7]
        g = self.avgpool(f4).flatten(1)  # [B, 512]
        return g, [f1, f2, f3, f4]


class ResNetDPTHead(nn.Module):
    """Simple DPT-style decoder for ResNet multiscale features.
    Projects each scale to a common dim, then fuses coarse-to-fine."""

    def __init__(self, features=128, out_depth=1, out_normal=3):
        super().__init__()
        in_channels = [64, 128, 256, 512]
        self.proj = nn.ModuleList([
            nn.Conv2d(c, features, 1) for c in in_channels
        ])
        self.fuse = nn.ModuleList([
            self._fuse_block(features) for _ in range(4)
        ])
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, out_depth, 1),
        )
        self.normal_head = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, out_normal, 1),
        )

    @staticmethod
    def _fuse_block(features):
        return nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(True),
        )

    def forward(self, multiscale, out_hw):
        # Project each scale
        p = [proj(f) for proj, f in zip(self.proj, multiscale)]
        # Coarse-to-fine fusion (4→3→2→1)
        x = self.fuse[3](p[3])
        x = F.interpolate(x, size=p[2].shape[2:], mode="bilinear", align_corners=False) + p[2]
        x = self.fuse[2](x)
        x = F.interpolate(x, size=p[1].shape[2:], mode="bilinear", align_corners=False) + p[1]
        x = self.fuse[1](x)
        x = F.interpolate(x, size=p[0].shape[2:], mode="bilinear", align_corners=False) + p[0]
        x = self.fuse[0](x)
        # Predict at the finest fused scale, then resize to output
        depth = F.interpolate(self.depth_head(x), size=out_hw, mode="bilinear", align_corners=False)
        normal = F.interpolate(self.normal_head(x), size=out_hw, mode="bilinear", align_corners=False)
        return depth, normal


class VITaLPoseModel(nn.Module):
    """Downstream: frozen VITaL ResNet-18 encoders → DPT + PoseHead.

    Both encoders output [B, 512] global + [B, C, H, W] multiscale maps.
    Pose: concat global → [B, 1024] → MLP → SE(2).
    Dense: tactile multiscale taps → ResNetDPTHead → depth + normal.
    """

    def __init__(self, tactile_ckpt, vision_ckpt, hidden_dim=256, dropout=0.1,
                 num_objects=20, use_obj_emb=True, dpt_features=128,
                 image_size=224, features_per_group=16):
        super().__init__()
        self.image_size = image_size
        tac_resnet = make_resnet18_encoder(pretrained=False, features_per_group=features_per_group)
        vis_resnet = make_resnet18_encoder(pretrained=False, features_per_group=features_per_group)

        self._load_encoder(tac_resnet, tactile_ckpt)
        self._load_encoder(vis_resnet, vision_ckpt)

        self.tactile_enc = ResNet18MultiScale(tac_resnet)
        self.vision_enc = ResNet18MultiScale(vis_resnet)

        for p in self.tactile_enc.parameters():
            p.requires_grad = False
        for p in self.vision_enc.parameters():
            p.requires_grad = False

        enc_dim = 512
        self.use_obj_emb = use_obj_emb
        if use_obj_emb:
            self.obj_embedding = nn.Embedding(num_objects, enc_dim * 2)

        # Pose head (same structure as MViTac)
        in_dim = enc_dim * 2
        self.pose_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        # DPT decoder from tactile multiscale
        self.dpt = ResNetDPTHead(features=dpt_features)

    @staticmethod
    def _load_encoder(resnet, ckpt_path):
        """Load weights saved by the CLIP pretraining (nn.Sequential of resnet layers)."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # The pretrain saves nn.Sequential(*children()[:-2]); our ResNet18MultiScale
        # has named layers. Map sequential keys → named keys.
        new_sd = {}
        layer_map = {"0": "conv1", "1": "bn1", "4": "layer1",
                     "5": "layer2", "6": "layer3", "7": "layer4"}
        for k, v in sd.items():
            parts = k.split(".", 1)
            if parts[0] in layer_map:
                new_key = layer_map[parts[0]] + ("." + parts[1] if len(parts) > 1 else "")
                new_sd[new_key] = v
        # Load with strict=False (avgpool/relu/maxpool have no learned params)
        missing, unexpected = resnet.load_state_dict(new_sd, strict=False)
        ok = [k for k in missing if not any(x in k for x in ("avgpool", "relu", "maxpool", "fc"))]
        if ok:
            print(f"  [vital] WARNING: missing keys after load: {ok}")
        print(f"  [vital] loaded encoder from {ckpt_path} ({len(new_sd)} keys)")

    def train(self, mode=True):
        super().train(mode)
        self.tactile_enc.eval()
        self.vision_enc.eval()
        return self

    @torch.no_grad()
    def _encode(self, tactile, rgb):
        t_global, t_ms = self.tactile_enc.forward_features(tactile)
        v_global, _ = self.vision_enc.forward_features(rgb)
        return torch.cat([t_global, v_global], dim=-1), t_ms

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        B, device = tactile.shape[0], tactile.device
        feat, tac_ms = self._encode(tactile, rgb)

        if self.use_obj_emb and object_ids is not None:
            feat = feat + self.obj_embedding(object_ids)

        out_pose = self.pose_head(feat)
        cos_sin = F.normalize(out_pose[:, :2], dim=-1)
        se2 = torch.cat([cos_sin, out_pose[:, 2:]], dim=-1)

        depth, normal = self.dpt(tac_ms, (self.image_size, self.image_size))

        return {"depth": depth, "normal": normal, "se2": se2}


class CLIPPretrainer(nn.Module):
    """CLIP contrastive pretraining for VITaL (vision ↔ tactile alignment)."""

    def __init__(self, clip_dim=512, features_per_group=16, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        self.vision_enc = nn.Sequential(
            *list(make_resnet18_encoder(False, features_per_group).children())[:-2])
        self.tactile_enc = nn.Sequential(
            *list(make_resnet18_encoder(False, features_per_group).children())[:-2])

        self.vision_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1),
            nn.Linear(512, clip_dim),
        )
        self.tactile_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1),
            nn.Linear(512, clip_dim),
        )

    def forward(self, vision, tactile):
        v = F.normalize(self.vision_proj(self.vision_enc(vision)), dim=-1)
        t = F.normalize(self.tactile_proj(self.tactile_enc(tactile)), dim=-1)

        logits = torch.mm(v, t.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return {"loss": loss, "v2t_acc": (logits.argmax(1) == labels).float().mean().item()}
