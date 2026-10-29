"""Eval metrics: depth absrel/rmse, normal mean angle, pose rotation/translation error.

All dense metrics (depth, normal) are computed only on the contact region (mask > 0)
to avoid inflating errors on the zero-depth background.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def depth_absrel(pred, gt, mask=None, eps=1e-6):
    if mask is None:
        mask = (gt.abs() > eps).float()
    valid = mask.sum().clamp_min(1)
    return ((pred - gt).abs() * mask / gt.abs().clamp_min(eps)).sum().item() / valid.item()


@torch.no_grad()
def depth_rmse(pred, gt, mask=None, eps=1e-6):
    if mask is None:
        mask = (gt.abs() > eps).float()
    valid = mask.sum().clamp_min(1)
    return torch.sqrt(((pred - gt) ** 2 * mask).sum() / valid).item()


@torch.no_grad()
def normal_mean_angle_deg(pred, gt, mask=None, eps=1e-6):
    p = F.normalize(pred, dim=1, eps=eps)
    g = F.normalize(gt, dim=1, eps=eps)
    cos = (p * g).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos)
    if mask is not None:
        valid = mask.sum().clamp_min(1)
        return (angle * mask).sum().item() / valid.item() * 180.0 / math.pi
    return angle.mean().item() * 180.0 / math.pi


@torch.no_grad()
def pose_rot_deg(pred_se2, gt):
    cosp, sinp = pred_se2[:, 0], pred_se2[:, 1]
    cosg, sing = gt[:, 0], gt[:, 1]
    dcos = (cosp * cosg + sinp * sing).clamp(-1 + 1e-6, 1 - 1e-6)
    return (torch.acos(dcos).mean().item() * 180.0 / math.pi)


@torch.no_grad()
def pose_trans_l1(pred_se2, gt):
    return (pred_se2[:, 2:] - gt[:, 2:]).abs().mean().item()
