"""MViTac pretraining: MoCo contrastive learning on paired RGB + tactile.

Usage:
  # (a) all sim renders under a root
  python -m vistacfusion.engine.pretrain_mvitac \
    --data-root dataset/sim \
    --output-dir outputs/mvitac_pretrain \
    --device cuda:0 --epochs 240 --batch-size 256
  # (b) EXACTLY the sim+real train images of a ratio-ladder data config
  python -m vistacfusion.engine.pretrain_mvitac \
    --data-config configs/data/default.yaml \
    --output-dir outputs/mvitac_pretrain --device cuda:0
"""
import argparse
import os
import os.path as osp
import glob
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from PIL import Image
import torchvision.transforms as T

from ..models.mvitac import MultiModalMoCo


class PairedAugDataset(Dataset):
    """Load paired (tactile, RGB) from renders_v3/<obj>/<session>/sensor_0/*.png + rgb/*.png.
    Returns two augmented views of each modality for MoCo pretraining.
    """

    def __init__(self, root, transform, val_every=20, split="train", samples=None):
        self.transform = transform
        self.samples = []  # (tactile_path, rgb_path)
        if samples is not None:          # explicit (tactile_path, rgb_path) list
            self.samples = list(samples)
            return

        for obj_dir in sorted(glob.glob(osp.join(root, "*"))):
            if not osp.isdir(obj_dir):
                continue
            for session_dir in sorted(glob.glob(osp.join(obj_dir, "session_*"))):
                for sensor_dir in sorted(glob.glob(osp.join(session_dir, "sensor_*"))):
                    samples_dir = osp.join(sensor_dir, "samples")
                    rgb_dir = osp.join(sensor_dir, "rgb")
                    if not osp.isdir(samples_dir) or not osp.isdir(rgb_dir):
                        continue

                    tac_files = sorted(f for f in os.listdir(samples_dir)
                                       if f.endswith(".png"))
                    for idx, fn in enumerate(tac_files):
                        is_val = (idx % val_every == 0)
                        if (split == "train" and is_val) or (split == "val" and not is_val):
                            continue
                        tac_path = osp.join(samples_dir, fn)
                        rgb_path = osp.join(rgb_dir, fn)
                        if osp.exists(rgb_path):
                            self.samples.append((tac_path, rgb_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tac_path, rgb_path = self.samples[idx]
        tac = Image.open(tac_path).convert("RGB")
        rgb = Image.open(rgb_path).convert("RGB")

        tac_q = self.transform(tac)
        tac_k = self.transform(tac)
        rgb_q = self.transform(rgb)
        rgb_k = self.transform(rgb)

        return rgb_q, rgb_k, tac_q, tac_k


def paired_paths_from_data_config(cfg_path):
    """(tactile_path, rgb_path) for every sim + real TRAIN sample of a data config —
    the exact image set a co-training run on that config sees (same object filter,
    rotation-window session filter, per-session subsampling, tactile_subdir)."""
    from omegaconf import OmegaConf
    from ..data.dataset import SimVisuoTactileDataset
    cfg = OmegaConf.load(cfg_path)
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


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="all sim renders under this root (legacy mode)")
    ap.add_argument("--data-config", default=None,
                    help="VisTacFusion data yaml: pretrain on exactly its sim+real train images")
    ap.add_argument("--output-dir", default="outputs/mvitac_pretrain")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    transform = get_transform()
    if args.data_config:
        train_ds = PairedAugDataset(None, transform,
                                    samples=paired_paths_from_data_config(args.data_config))
    elif args.data_root:
        train_ds = PairedAugDataset(args.data_root, transform,
                                    val_every=args.val_every, split="train")
    else:
        ap.error("one of --data-root / --data-config is required")
    print(f"Train samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = MultiModalMoCo(
        momentum=args.momentum, temperature=args.temperature,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = GradScaler("cuda")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nMViTac Pretraining | epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} T={args.temperature}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0, "intra_v": 0, "intra_t": 0,
                   "inter_vt": 0, "inter_tv": 0}
        n_steps = 0

        for step, (vis_q, vis_k, tac_q, tac_k) in enumerate(train_loader):
            vis_q = vis_q.to(device, non_blocking=True)
            vis_k = vis_k.to(device, non_blocking=True)
            tac_q = tac_q.to(device, non_blocking=True)
            tac_k = tac_k.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda"):
                out = model(vis_q, vis_k, tac_q, tac_k)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k in running:
                running[k] += out[k] if k == "loss" else out[k]
            n_steps += 1

            if step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running["loss"] / (n_steps or 1)
                print(f"  [epoch {epoch:03d} | step {step:04d}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}")

        if epoch >= args.warmup_epochs:
            scheduler.step()

        dt = time.time() - t0
        avg = {k: v / n_steps for k, v in running.items()}
        lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] {dt:.0f}s  loss={avg['loss']:.4f}  "
              f"intra_v={avg['intra_v']:.3f}  intra_t={avg['intra_t']:.3f}  "
              f"inter_vt={avg['inter_vt']:.3f}  inter_tv={avg['inter_tv']:.3f}  "
              f"lr={lr:.2e}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "vision_encoder": model.vision_q.state_dict(),
            "tactile_encoder": model.tactile_q.state_dict(),
        }
        torch.save(ckpt, osp.join(args.output_dir, "latest.pt"))
        if (epoch + 1) % 20 == 0:
            torch.save(ckpt, osp.join(args.output_dir, f"epoch_{epoch:03d}.pt"))

    # Save final encoders separately for downstream use
    torch.save(model.vision_q.state_dict(),
               osp.join(args.output_dir, "vision_encoder.pt"))
    torch.save(model.tactile_q.state_dict(),
               osp.join(args.output_dir, "tactile_encoder.pt"))
    print(f"\nPretraining complete. Encoders saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
