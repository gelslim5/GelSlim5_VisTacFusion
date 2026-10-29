"""VITaL CLIP pretraining: contrastive learning on paired RGB + tactile.

Usage:
  python -m vistacfusion.engine.pretrain_vital \
    --data-config configs/data/default.yaml \
    --output-dir outputs/vital_pretrain --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), "..", ".."))
from vistacfusion.models.vital import CLIPPretrainer


class PairedDataset(Dataset):
    """Load paired (tactile, RGB) images for CLIP pretraining.
    Images are loaded as float32 [0,1], resized to image_size, ImageNet-normalised."""

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, samples, image_size=224):
        self.samples = samples  # list of (tactile_path, rgb_path)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tac_path, rgb_path = self.samples[idx]
        tac = self._load(tac_path)
        rgb = self._load(rgb_path)
        return rgb, tac

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        return (t - self.MEAN) / self.STD


def paired_paths_from_data_config(cfg_path):
    """Same logic as pretrain_mvitac.py — reuse VisTacFusion's dataset to extract
    (tactile_path, rgb_path) for every sim + real TRAIN sample."""
    from vistacfusion.data.dataset import SimVisuoTactileDataset
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
    except ImportError:
        import yaml
        class _A(dict):
            def __getattr__(self, k):
                try: return self[k]
                except KeyError as e: raise AttributeError(k) from e
            @classmethod
            def wrap(cls, o):
                if isinstance(o, dict): return cls({k: cls.wrap(v) for k,v in o.items()})
                if isinstance(o, list): return [cls.wrap(v) for v in o]
                return o
        with open(cfg_path) as f:
            cfg = _A.wrap(yaml.safe_load(f))
    pairs = []
    def _add(ds):
        for unit, idx in ds.samples:
            pairs.append((osp.join(unit, ds.tactile_subdir, f"{idx:04d}.png"),
                          osp.join(unit, ds.rgb_subdir, f"{idx:04d}.png")))
    n_sim = n_real = 0
    spp = cfg.sim.get("train_samples_per_session", None)
    if spp is None or int(spp) > 0:
        inc = cfg.sim.get("include_objects", None)
        sim_ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="train",
                                        include_objects=list(inc) if inc else None)
        _add(sim_ds); n_sim = len(sim_ds.samples)
    if cfg.get("dataset") == "sim+real":
        rspp = cfg.real.get("train_samples_per_session", None)
        if rspp is None or int(rspp) > 0:
            real_ds = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False,
                                             split="train", data_section="real")
            _add(real_ds); n_real = len(real_ds.samples)
    print(f"  data-config pairs: sim={n_sim} real={n_real} total={len(pairs)}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--output-dir", default="outputs/vital_pretrain")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--clip-dim", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    pairs = paired_paths_from_data_config(args.data_config)
    train_ds = PairedDataset(pairs, image_size=224)
    print(f"Train samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = CLIPPretrainer(
        clip_dim=args.clip_dim, temperature=args.temperature,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"\nVITaL CLIP Pretraining | epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} T={args.temperature}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses, accs = [], []
        for rgb, tac in train_loader:
            rgb, tac = rgb.to(device), tac.to(device)
            with torch.cuda.amp.autocast():
                out = model(rgb, tac)
                loss = out["loss"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            accs.append(out["v2t_acc"])
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] {time.time()-t0:.0f}s  loss={np.mean(losses):.4f}  "
              f"acc={np.mean(accs):.3f}  lr={lr:.2e}")

        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch},
                       osp.join(args.output_dir, f"checkpoint_{epoch:03d}.pt"))

    # Save final encoder weights (same format as MViTac: bare state_dict)
    torch.save(model.vision_enc.state_dict(),
               osp.join(args.output_dir, "vision_encoder.pt"))
    torch.save(model.tactile_enc.state_dict(),
               osp.join(args.output_dir, "tactile_encoder.pt"))
    print(f"\nPretraining complete. Encoders saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
