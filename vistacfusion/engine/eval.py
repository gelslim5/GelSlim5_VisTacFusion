"""Evaluation: report metrics per modality config (both / tactile / rgb).

Uses the SAME loss functions as training (MSE depth, MSE normal, 1-cos rot, L1 trans)
so train vs val numbers are directly comparable. Also reports pose_rot_deg (mean angular
error in degrees) and pose_rot_l1 (L1 on (cos, sin)).

The frozen encoders run once over the val set (encoder cache); the cached tokens are reused
across the three configs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CONFIGS = ("both", "tactile", "rgb")


@torch.no_grad()
def precompute_encoder_cache(model, loader, device):
    """Run the frozen encoder once on the full val set, cache patch/cls + multiscale (fp16, CPU)."""
    model.eval()
    tac_p, tac_c, rgb_p, rgb_c = [], [], [], []
    tac_ms = [[] for _ in range(4)]
    rgb_ms = [[] for _ in range(4)]
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        tactile = batch["tactile"].to(device, non_blocking=True)
        tp, tc = model.tactile_encoder(tactile)
        rp, rc = model.rgb_encoder(rgb)
        tac_p.append(tp.half().cpu())
        tac_c.append(tc.half().cpu() if tc is not None else None)
        rgb_p.append(rp.half().cpu())
        rgb_c.append(rc.half().cpu() if rc is not None else None)
        t_ms = model.tactile_encoder.forward_multiscale(tactile)
        r_ms = model.rgb_encoder.forward_multiscale(rgb)
        for i in range(4):
            tac_ms[i].append(t_ms[i].half().cpu())
            rgb_ms[i].append(r_ms[i].half().cpu())
    cache = {
        "tactile_patch": torch.cat(tac_p),
        "tactile_cls": torch.cat(tac_c) if tac_c[0] is not None else None,
        "rgb_patch": torch.cat(rgb_p),
        "rgb_cls": torch.cat(rgb_c) if rgb_c[0] is not None else None,
        "tactile_ms": [torch.cat(ms) for ms in tac_ms],
        "rgb_ms": [torch.cat(ms) for ms in rgb_ms],
    }
    n = cache["tactile_patch"].shape[0]
    mb = sum(v.nbytes for v in cache.values() if isinstance(v, torch.Tensor))
    mb += sum(v.nbytes for ms in [cache["tactile_ms"], cache["rgb_ms"]] for v in ms)
    mb /= 1024 ** 2
    print(f"  encoder cache: {n} samples, {mb:.0f} MB (fp16, CPU)")
    return cache


def _slice_cache(cache, start, end, device):
    def _sl(t):
        return t[start:end].to(device, dtype=torch.float32) if t is not None else None
    return {
        "tactile": (
            _sl(cache["tactile_patch"]),
            _sl(cache["tactile_cls"]),
        ),
        "rgb": (
            _sl(cache["rgb_patch"]),
            _sl(cache["rgb_cls"]),
        ),
        "tactile_ms": [ms[start:end].to(device, dtype=torch.float32)
                       for ms in cache["tactile_ms"]],
        "rgb_ms": [ms[start:end].to(device, dtype=torch.float32)
                   for ms in cache["rgb_ms"]],
    }


def _theta_to_bin(theta, num_bins=72):
    bin_size = 2 * math.pi / num_bins
    bins = ((theta + math.pi) / bin_size).long()
    return bins.clamp(0, num_bins - 1)


@torch.no_grad()
def evaluate(model, loader, cfg, device, configs=CONFIGS, encoder_cache=None):
    model.eval()
    report_per_config = cfg.eval.get("report_per_config", True)
    mt = cfg.get("model_type", "visuo_tactile")
    if mt == "single_encoder":
        configs = ("tactile",)
    elif mt in ("mvitac", "vital"):
        configs = ("both",)
    elif not report_per_config:
        configs = ("both",)
    pose_mode = cfg.heads.pose.pose_mode
    rot_num_bins = cfg.heads.pose.get("rot_num_bins", 72)

    acc = {c: {"depth_mse": 0.0, "normal_mse": 0.0,
               "pose_rot": 0.0, "pose_rot_l1": 0.0, "pose_trans": 0.0,
               "rot_deg": 0.0, "n": 0} for c in configs}

    sample_idx = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bs = batch["rgb"].shape[0]

        batch_enc = None
        if encoder_cache is not None:
            batch_enc = _slice_cache(encoder_cache, sample_idx, sample_idx + bs, device)
        sample_idx += bs

        gt_pose = batch["pose"]
        cos_gt, sin_gt = gt_pose[:, 0], gt_pose[:, 1]
        txy_gt = gt_pose[:, 2:]

        for c in configs:
            domain_ids = batch.get("domain")
            if domain_ids is None:
                domain_ids = torch.ones(bs, dtype=torch.long, device=device)
            out = model(batch["rgb"], batch["tactile"], config=c,
                        encoder_cache=batch_enc,
                        object_ids=batch.get("object"),
                        domain_ids=domain_ids)
            a = acc[c]

            a["depth_mse"] += F.mse_loss(out["depth"], batch["depth"]).item() * bs
            a["normal_mse"] += F.mse_loss(out["normal"], batch["normal"]).item() * bs

            if pose_mode == "classification" and "rot_logits" in out:
                theta_gt = torch.atan2(sin_gt, cos_gt)
                target_bins = _theta_to_bin(theta_gt, rot_num_bins)
                a["pose_rot"] += F.cross_entropy(out["rot_logits"], target_bins).item() * bs
                a["pose_trans"] += F.l1_loss(out["trans"], txy_gt).item() * bs
            elif "se2" in out:
                se2 = out["se2"]
                cos_p, sin_p = se2[:, 0], se2[:, 1]
                a["pose_rot"] += (1.0 - (cos_p * cos_gt + sin_p * sin_gt)).mean().item() * bs
                a["pose_rot_l1"] += ((cos_p - cos_gt).abs() + (sin_p - sin_gt).abs()).mean().item() * bs
                a["pose_trans"] += F.l1_loss(se2[:, 2:], txy_gt).item() * bs

            if "se2" in out:
                cos_p, sin_p = out["se2"][:, 0], out["se2"][:, 1]
                dcos = (cos_p * cos_gt + sin_p * sin_gt).clamp(-1 + 1e-6, 1 - 1e-6)
                a["rot_deg"] += (torch.acos(dcos).mean().item() * 180.0 / math.pi) * bs

            a["n"] += bs

    report = {}
    for c, a in acc.items():
        n = max(1, a["n"])
        report[c] = {
            "depth_mse": round(a["depth_mse"] / n, 6),
            "normal_mse": round(a["normal_mse"] / n, 6),
            "pose_rot": round(a["pose_rot"] / n, 4),
            "pose_rot_l1": round(a["pose_rot_l1"] / n, 4),
            "pose_trans": round(a["pose_trans"] / n, 4),
            "pose_rot_deg": round(a["rot_deg"] / n, 3),
        }
    return report
