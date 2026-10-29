"""MViTac: Multi-modal MoCo contrastive learning for vision + tactile.

Reimplemented from https://github.com/ligerfotis/mvitac
Two-stream ResNet-18 encoders pretrained with in-batch InfoNCE (MoCo-style).

Pretraining output: frozen tactile + vision encoders (512-dim global vectors).
Downstream: concat [tactile; vision] → PoseHead.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _make_resnet18_encoder(pretrained=True):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    layers = list(resnet.children())[:-2]  # strip avgpool + fc
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)  # output: [B, 512]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MultiModalMoCo(nn.Module):
    """MViTac pretraining model: 4 ResNet-18 encoders + projection heads."""

    def __init__(self, momentum=0.99, temperature=0.07, proj_dim=128,
                 pretrained=True):
        super().__init__()
        self.m = momentum
        self.T = temperature

        self.vision_q = _make_resnet18_encoder(pretrained)
        self.vision_k = _make_resnet18_encoder(pretrained)
        self.tactile_q = _make_resnet18_encoder(pretrained)
        self.tactile_k = _make_resnet18_encoder(pretrained)

        self.vision_intra_q = ProjectionHead(512, proj_dim)
        self.vision_intra_k = ProjectionHead(512, proj_dim)
        self.vision_inter = ProjectionHead(512, proj_dim)

        self.tactile_intra_q = ProjectionHead(512, proj_dim)
        self.tactile_intra_k = ProjectionHead(512, proj_dim)
        self.tactile_inter = ProjectionHead(512, proj_dim)

        # Init momentum encoders
        for p_q, p_k in zip(self.vision_q.parameters(), self.vision_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.tactile_q.parameters(), self.tactile_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.vision_intra_q.parameters(), self.vision_intra_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.tactile_intra_q.parameters(), self.tactile_intra_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.vision_q.parameters(), self.vision_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)
        for p_q, p_k in zip(self.tactile_q.parameters(), self.tactile_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)
        for p_q, p_k in zip(self.vision_intra_q.parameters(), self.vision_intra_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)
        for p_q, p_k in zip(self.tactile_intra_q.parameters(), self.tactile_intra_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)

    def _infonce(self, q, k):
        logits = torch.mm(q, k.T.detach()) / self.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, vis_q, vis_k, tac_q, tac_k):
        # Query encodings
        v_feat_q = self.vision_q(vis_q)
        t_feat_q = self.tactile_q(tac_q)

        v_intra_q = self.vision_intra_q(v_feat_q)
        t_intra_q = self.tactile_intra_q(t_feat_q)
        v_inter_q = self.vision_inter(v_feat_q)
        t_inter_q = self.tactile_inter(t_feat_q)

        # Key encodings (no grad)
        with torch.no_grad():
            v_feat_k = self.vision_k(vis_k)
            t_feat_k = self.tactile_k(tac_k)
            v_intra_k = self.vision_intra_k(v_feat_k)
            t_intra_k = self.tactile_intra_k(t_feat_k)
            v_inter_k = self.vision_inter(v_feat_k)
            t_inter_k = self.tactile_inter(t_feat_k)

        # 4 contrastive losses
        loss_intra_v = self._infonce(v_intra_q, v_intra_k)
        loss_intra_t = self._infonce(t_intra_q, t_intra_k)
        loss_inter_vt = self._infonce(v_inter_q, t_inter_k)
        loss_inter_tv = self._infonce(t_inter_q, v_inter_k)

        self._momentum_update()

        total = loss_intra_v + loss_intra_t + loss_inter_vt + loss_inter_tv
        return {
            "loss": total,
            "intra_v": loss_intra_v.item(),
            "intra_t": loss_intra_t.item(),
            "inter_vt": loss_inter_vt.item(),
            "inter_tv": loss_inter_tv.item(),
        }


class MViTacPoseModel(nn.Module):
    """Downstream: frozen MViTac ResNet-18 encoders → PoseHead.

    Both encoders output [B, 512] global vectors.
    Concat → [B, 1024] → MLP → SE(2).
    """

    def __init__(self, tactile_ckpt, vision_ckpt, hidden_dim=256, dropout=0.1,
                 num_objects=20, use_obj_emb=True):
        super().__init__()
        self.tactile_enc = _make_resnet18_encoder(pretrained=False)
        self.vision_enc = _make_resnet18_encoder(pretrained=False)

        self._load_encoder(self.tactile_enc, tactile_ckpt)
        self._load_encoder(self.vision_enc, vision_ckpt)

        for p in self.tactile_enc.parameters():
            p.requires_grad = False
        for p in self.vision_enc.parameters():
            p.requires_grad = False

        enc_dim = 512
        self.use_obj_emb = use_obj_emb
        if use_obj_emb:
            self.obj_embedding = nn.Embedding(num_objects, enc_dim * 2)

        in_dim = enc_dim * 2  # concat tactile + vision
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

    @staticmethod
    def _load_encoder(encoder, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(sd, strict=True)
        print(f"  [mvitac] loaded encoder from {ckpt_path}")

    def train(self, mode=True):
        super().train(mode)
        self.tactile_enc.eval()
        self.vision_enc.eval()
        return self

    @torch.no_grad()
    def _encode(self, tactile, rgb):
        t_feat = self.tactile_enc(tactile)
        v_feat = self.vision_enc(rgb)
        return torch.cat([t_feat, v_feat], dim=-1)  # [B, 1024]

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        # domain_ids accepted for interface parity with VisuoTactileModel (unused).
        B, device = tactile.shape[0], tactile.device
        feat = self._encode(tactile, rgb)

        if self.use_obj_emb and object_ids is not None:
            feat = feat + self.obj_embedding(object_ids)

        out = self.pose_head(feat)
        cos_sin = F.normalize(out[:, :2], dim=-1)
        se2 = torch.cat([cos_sin, out[:, 2:]], dim=-1)

        depth = torch.zeros(B, 1, 224, 224, device=device)
        normal = torch.zeros(B, 3, 224, 224, device=device)

        return {"depth": depth, "normal": normal, "se2": se2}
