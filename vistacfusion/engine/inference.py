"""Inference / evaluation — load checkpoints, predict depth / normal / pose, visualize.

Usage:
    # single tactile + RGB pair (all three input modes)
    python -m vistacfusion.engine.inference --train-dir checkpoints/vistacfusion_sitr_mae \
        --tactile /path/to/tactile.png --rgb /path/to/rgb.png

    # a whole sensor directory of the dataset (uses GT depth / pose when present)
    python -m vistacfusion.engine.inference --train-dir checkpoints/vistacfusion_sitr_mae \
        --session-dir dataset/real/pattern_32/session_000/sensor_0000

    # quantitative evaluation on the validation split of the data config
    python -m vistacfusion.engine.inference --train-dir checkpoints/vistacfusion_sitr_mae --eval

`--train-dir` must contain best_depth.pt (used for depth / normal) and optionally
best_pose.pt (used for pose); `--checkpoint` evaluates a single file for everything.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.model import build_model
from ..utils.config import merge_configs

CONFIGS_3 = ["both", "tactile", "rgb"]
CONFIG_LABELS = {"both": "Both (RGB+Tactile)", "tactile": "Tactile Only", "rgb": "RGB Only"}
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53])
IMAGENET_STD = np.array([58.395, 57.12, 57.375])


def load_model(cfg, checkpoint_path, device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing
               if not (k.startswith("tactile_encoder.") or k.startswith("rgb_encoder.")
                       or k.startswith("encoder."))]
    if missing:
        print(f"  [warn] missing keys: {len(missing)} ({missing[:3]} ...)")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}")
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def preprocess_image(img_path, image_size, fixed_crop=None):
    """Load image → same preprocessing as training: center square crop (non-square
    inputs), the fixed centre crop the model was trained with, then ImageNet norm."""
    from ..data.transforms import FIXED_CROP, fixed_center_crop

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

    # non-square (e.g. raw 640x480 frames): center square crop first, no aspect distortion
    h, w = img.shape[:2]
    if h != w:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]

    img = fixed_center_crop(img, out_size=image_size,
                            crop=FIXED_CROP if fixed_crop is None else fixed_crop)

    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def _extract_outputs(out):
    """Extract depth, normal, pose from model output dict."""
    depth = out["depth"][0, 0].cpu().numpy()
    normal = out["normal"][0].cpu().permute(1, 2, 0).numpy()
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    pose = out.get("se2", out.get("trans"))
    if pose is not None:
        pose = pose[0].cpu().numpy()
    else:
        pose = np.zeros(4)
    theta = np.degrees(np.arctan2(pose[1], pose[0]))
    return depth, normal, pose, theta


def predict(model, rgb_tensor, tactile_tensor, config="both", device="cpu",
            pose_model=None, object_ids=None):
    """Run inference. If pose_model is given, use it for pose and model for depth/normal."""
    rgb = rgb_tensor.to(device)
    tac = tactile_tensor.to(device)
    obj = object_ids.to(device) if object_ids is not None else None

    with torch.no_grad():
        out = model(rgb, tac, config=config, object_ids=obj)
        if pose_model is not None:
            pose_out = pose_model(rgb, tac, config=config, object_ids=obj)
            out["se2"] = pose_out.get("se2", pose_out.get("trans"))

    return _extract_outputs(out)


def center_crop_square(img):
    """Center-crop a HxW(xC) image to a square."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0+s, x0:x0+s]


def visualize_three_configs(tactile_img, rgb_img, results,
                            gt_depth=None, gt_normal=None, gt_pose=None,
                            save_path=None):
    """Plot 3-config comparison: rows = [both, tactile, rgb, (GT)], cols = [input, depth, normal, pose]."""
    has_gt = gt_depth is not None
    nrows = 4 if has_gt else 3
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.5 * nrows))

    for row, cfg_name in enumerate(CONFIGS_3):
        depth, normal, pose, theta = results[cfg_name]
        normal_vis = (normal * 0.5 + 0.5).clip(0, 1)

        axes[row, 0].imshow(tactile_img if cfg_name != "rgb" else rgb_img)
        axes[row, 0].set_ylabel(CONFIG_LABELS[cfg_name], fontsize=12, fontweight="bold")
        axes[row, 0].set_title("Input" if row == 0 else "")

        axes[row, 1].imshow(depth, cmap="viridis")
        axes[row, 1].set_title("Depth" if row == 0 else "")

        axes[row, 2].imshow(normal_vis)
        axes[row, 2].set_title("Normal" if row == 0 else "")

        pose_text = f"θ = {theta:.1f}°\ntx = {pose[2]:.3f}\nty = {pose[3]:.3f}"
        axes[row, 3].text(0.5, 0.5, pose_text,
                          transform=axes[row, 3].transAxes, fontsize=18,
                          ha="center", va="center", family="monospace")
        axes[row, 3].set_title("Pose" if row == 0 else "")
        axes[row, 3].axis("off")

    if has_gt:
        axes[3, 0].imshow(tactile_img)
        axes[3, 0].set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
        axes[3, 1].imshow(gt_depth, cmap="viridis")
        gt_normal_vis = (gt_normal * 0.5 + 0.5).clip(0, 1)
        axes[3, 2].imshow(gt_normal_vis)
        if gt_pose is not None:
            gt_theta = np.degrees(np.arctan2(gt_pose[1], gt_pose[0]))
            gt_text = f"θ = {gt_theta:.1f}°\ntx = {gt_pose[2]:.3f}\nty = {gt_pose[3]:.3f}"
            axes[3, 3].text(0.5, 0.5, gt_text,
                            transform=axes[3, 3].transAxes, fontsize=18,
                            ha="center", va="center", family="monospace",
                            color="green")
        axes[3, 3].axis("off")

    for ax in axes.flat:
        if ax.images or ax.texts:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.close(fig)


def run_single(model, rgb_path, tactile_path, cfg, device, output_dir,
               gt_depth=None, gt_normal=None, gt_pose=None,
               name="sample", pose_model=None, object_ids=None):
    """Run inference on a single pair with all 3 configs."""
    image_size = cfg.image_size
    rgb_t = preprocess_image(rgb_path, image_size, cfg.get("fixed_crop", None))
    tac_t = preprocess_image(tactile_path, image_size, cfg.get("fixed_crop", None))

    results = {}
    for config in CONFIGS_3:
        depth, normal, pose, theta = predict(model, rgb_t, tac_t, config=config,
                                             device=device, pose_model=pose_model,
                                             object_ids=object_ids)
        results[config] = (depth, normal, pose, theta)
        np.save(osp.join(output_dir, f"{name}_{config}_depth.npy"), depth)
        np.save(osp.join(output_dir, f"{name}_{config}_normal.npy"), normal)
        np.save(osp.join(output_dir, f"{name}_{config}_pose.npy"), pose)

    tactile_img = center_crop_square(np.array(Image.open(tactile_path).convert("RGB")))
    rgb_img = center_crop_square(np.array(Image.open(rgb_path).convert("RGB")))

    save_path = osp.join(output_dir, f"{name}.png")
    visualize_three_configs(tactile_img, rgb_img, results,
                            gt_depth=gt_depth, gt_normal=gt_normal,
                            gt_pose=gt_pose, save_path=save_path)

    for cfg_name in CONFIGS_3:
        _, _, pose, theta = results[cfg_name]
        print(f"  {name} [{cfg_name:8s}]: θ={theta:.1f}°  tx={pose[2]:.4f}  ty={pose[3]:.4f}")

    return results


def _object_ids_for(cfg, obj_name, device):
    """Object id of `obj_name` under the training id map (sorted sim.include_objects)."""
    objs = cfg.sim.get("include_objects", None)
    if not objs or obj_name not in objs:
        return None
    return torch.tensor([sorted(objs).index(obj_name)], device=device)


def run_session(model, session_dir, cfg, device, output_dir, max_samples=None,
                pose_model=None, tactile_subdir=None):
    """Run inference on all samples of one <object>/session_*/sensor_* directory."""
    from ..data.dataset import depth_to_normal
    from ..data.transforms import FIXED_CROP, fixed_center_crop

    tactile_subdir = tactile_subdir or cfg.real.get("tactile_subdir", "samples")
    samples_dir = osp.join(session_dir, tactile_subdir)
    rgb_dir = osp.join(session_dir, "rgb")
    raw_dir = osp.join(session_dir, "raw_data")
    obj_name = osp.basename(osp.dirname(osp.dirname(osp.abspath(session_dir))))
    obj_ids = _object_ids_for(cfg, obj_name, device)

    pngs = sorted(f for f in os.listdir(samples_dir) if f.endswith(".png"))
    if max_samples:
        pngs = pngs[:max_samples]

    print(f"Running inference on {len(pngs)} samples from {session_dir} (object {obj_name})")
    os.makedirs(output_dir, exist_ok=True)

    _crop = float(cfg.get("fixed_crop", FIXED_CROP))
    px = cfg.sim.get("gel_view_m", 0.017502) * _crop / cfg.image_size
    for png in pngs:
        idx = int(osp.splitext(png)[0])
        tactile_path = osp.join(samples_dir, png)
        rgb_path = osp.join(rgb_dir, png)
        if not osp.exists(rgb_path):
            print(f"  skipping {png}: no RGB")
            continue

        gt_depth = gt_normal = None
        gt_depth_path = osp.join(raw_dir, f"{idx:04d}_gt.npy")
        if osp.exists(gt_depth_path):
            gt_depth_raw = fixed_center_crop(
                np.load(gt_depth_path).astype(np.float32), crop=_crop)
            gt_depth = gt_depth_raw * 1000.0
            gt_normal = depth_to_normal(gt_depth_raw, px, px)

        run_single(model, rgb_path, tactile_path, cfg, device, output_dir,
                   gt_depth=gt_depth, gt_normal=gt_normal, name=f"{idx:04d}",
                   pose_model=pose_model, object_ids=obj_ids)


def run_eval(model, cfg, device, output_dir, pose_model=None, ckpt_info=None):
    """Quantitative eval on the validation split of the data config (3 configs) + one
    visualization per object. Writes eval_metrics.txt next to output_dir."""
    import math as _math
    from torch.utils.data import DataLoader
    from ..data.dataset import build_datasets, SimVisuoTactileDataset
    from .eval import precompute_encoder_cache, _slice_cache
    _, val_ds = build_datasets(cfg)

    os.makedirs(output_dir, exist_ok=True)

    # --- one representative sample per object for the visualisation ---
    obj_vis_idx = {}
    if isinstance(val_ds, SimVisuoTactileDataset):
        for i, (unit, _sample_idx) in enumerate(val_ds.samples):
            obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
            if obj_name not in obj_vis_idx:
                obj_vis_idx[obj_name] = i
    else:
        for i in range(min(len(val_ds), 20)):
            obj_vis_idx[f"sample_{i:03d}"] = i
    vis_indices = set(obj_vis_idx.values())

    eval_model = pose_model if pose_model is not None else model
    batch_size = 32
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("Pre-computing encoder cache for val set...")
    encoder_cache = precompute_encoder_cache(eval_model, val_loader, device)

    metrics = {cfg_name: {
        "depth_mse": 0.0, "normal_mse": 0.0,
        "depth_mae": [], "depth_rmse": [],
        "normal_ang_mean": 0.0,
        "pose_rot_l1": 0.0, "pose_rot_deg": 0.0, "pose_trans_l1": 0.0, "n": 0,
    } for cfg_name in CONFIGS_3}
    vis_results = {}

    print(f"Evaluating {len(val_ds)} val samples (3 configs, batched)...")
    sample_idx = 0
    for batch in val_loader:
        bs = batch["rgb"].shape[0]
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        batch_enc = _slice_cache(encoder_cache, sample_idx, sample_idx + bs, device)
        obj_ids = batch_dev.get("object")

        for cfg_name in CONFIGS_3:
            with torch.no_grad():
                out = model(batch_dev["rgb"], batch_dev["tactile"],
                            config=cfg_name, encoder_cache=batch_enc,
                            object_ids=obj_ids)
                if pose_model is not None:
                    pose_out = pose_model(batch_dev["rgb"], batch_dev["tactile"],
                                         config=cfg_name, encoder_cache=batch_enc,
                                         object_ids=obj_ids)
                    out["se2"] = pose_out.get("se2", pose_out.get("trans"))

            m = metrics[cfg_name]
            m["depth_mse"] += F.mse_loss(out["depth"], batch_dev["depth"]).item() * bs
            m["normal_mse"] += F.mse_loss(out["normal"], batch_dev["normal"]).item() * bs

            pred_depth = out["depth"].cpu()
            p_n = F.normalize(out["normal"].cpu().float(), dim=1)
            g_n = F.normalize(batch["normal"].float(), dim=1)
            cos_sim = (p_n * g_n).sum(dim=1).clamp(-1, 1)
            m["normal_ang_mean"] += (torch.acos(cos_sim) * (180.0 / _math.pi)).mean().item() * bs

            if "se2" in out:
                se2 = out["se2"].float().cpu()
                gt_pose = batch["pose"]
                cos_p, sin_p = se2[:, 0], se2[:, 1]
                cos_g, sin_g = gt_pose[:, 0], gt_pose[:, 1]
                rot_err = (torch.atan2(sin_p, cos_p) - torch.atan2(sin_g, cos_g)).abs()
                rot_err = torch.min(rot_err, 2 * _math.pi - rot_err)
                m["pose_rot_l1"] += ((cos_p - cos_g).abs() + (sin_p - sin_g).abs()).mean().item() * bs
                m["pose_rot_deg"] += (rot_err * 180.0 / _math.pi).mean().item() * bs
                m["pose_trans_l1"] += F.l1_loss(se2[:, 2:], gt_pose[:, 2:]).item() * bs

            for j in range(bs):
                pd = pred_depth[j, 0].numpy()
                gd = batch["depth"][j, 0].numpy()
                mask = gd > 0
                if mask.sum() > 0:
                    abs_diff = np.abs(pd[mask] - gd[mask])
                    m["depth_mae"].append(float(abs_diff.mean()))
                    m["depth_rmse"].append(float((abs_diff ** 2).mean()) ** 0.5)

            for j in range(bs):
                global_idx = sample_idx + j
                if global_idx in vis_indices:
                    d = out["depth"][j, 0].cpu().numpy()
                    n = out["normal"][j].cpu().permute(1, 2, 0).numpy()
                    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
                    p = out["se2"][j].float().cpu().numpy() if "se2" in out else np.zeros(4)
                    th = float(np.degrees(np.arctan2(p[1], p[0])))
                    vis_results.setdefault(global_idx, {})[cfg_name] = (d, n, p, th)

            m["n"] += bs

        sample_idx += bs
        if sample_idx % (batch_size * 10) == 0:
            print(f"  {sample_idx}/{len(val_ds)} samples done")

    summary = {}
    for cfg_name in CONFIGS_3:
        m = metrics[cfg_name]
        n = max(1, m["n"])
        summary[cfg_name] = {
            "depth_mse": m["depth_mse"] / n,
            "normal_mse": m["normal_mse"] / n,
            "depth_mae": float(np.mean(m["depth_mae"])) if m["depth_mae"] else float("nan"),
            "depth_rmse": float(np.mean(m["depth_rmse"])) if m["depth_rmse"] else float("nan"),
            "normal_ang_mean": m["normal_ang_mean"] / n,
            "pose_rot_l1": m["pose_rot_l1"] / n,
            "pose_rot_deg": m["pose_rot_deg"] / n,
            "pose_trans_l1": m["pose_trans_l1"] / n,
        }

    metrics_path = osp.join(osp.dirname(osp.abspath(output_dir)), "eval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("VisTacFusion evaluation\n")
        if ckpt_info:
            f.write(f"Checkpoint: {ckpt_info}\n")
        f.write(f"Val samples: {len(val_ds)}\n\n")
        for cfg_name in CONFIGS_3:
            s = summary[cfg_name]
            f.write(f"[{cfg_name}]\n")
            f.write(f"  Depth MSE (mm^2):          {s['depth_mse']:.6f}\n")
            f.write(f"  Depth MAE, contact (mm):   {s['depth_mae']:.6f}\n")
            f.write(f"  Depth RMSE, contact (mm):  {s['depth_rmse']:.6f}\n")
            f.write(f"  Normal MSE:                {s['normal_mse']:.6f}\n")
            f.write(f"  Normal angle mean (deg):   {s['normal_ang_mean']:.2f}\n")
            f.write(f"  Rotation L1 (cos,sin):     {s['pose_rot_l1']:.6f}\n")
            f.write(f"  Angular error (deg):       {s['pose_rot_deg']:.3f}\n")
            f.write(f"  Translation L1 (norm.):    {s['pose_trans_l1']:.6f}\n\n")
    print(f"\nMetrics saved -> {metrics_path}")

    print("\n" + "=" * 72)
    print(f"{'Metric':30s} {'Both':>12s} {'Tactile':>12s} {'RGB':>12s}")
    print("=" * 72)
    for key in ["depth_mse", "depth_mae", "depth_rmse", "normal_mse", "normal_ang_mean",
                "pose_rot_l1", "pose_rot_deg", "pose_trans_l1"]:
        vals = [summary[c].get(key, float("nan")) for c in CONFIGS_3]
        fmt = ".3f" if "deg" in key else ".6f"
        print(f"  {key:28s} {vals[0]:{fmt}} {vals[1]:{fmt}} {vals[2]:{fmt}}")
    print("=" * 72)

    print(f"\nSaving one visualization per object ({len(obj_vis_idx)} objects)...")
    for obj_name, idx in sorted(obj_vis_idx.items()):
        sample = val_ds[idx]
        results = vis_results.get(idx, {})
        if not results:
            continue
        gt_depth = sample["depth"][0].numpy()
        gt_normal = sample["normal"].permute(1, 2, 0).numpy()
        gt_pose = sample["pose"].numpy()
        tac_img = (sample["tactile"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN).clip(0, 255).astype(np.uint8)
        rgb_img = (sample["rgb"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN).clip(0, 255).astype(np.uint8)
        visualize_three_configs(tac_img, rgb_img, results,
                                gt_depth=gt_depth, gt_normal=gt_normal,
                                gt_pose=gt_pose, save_path=osp.join(output_dir, f"{obj_name}.png"))
    print(f"Visualizations saved -> {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="VisTacFusion inference / evaluation")
    ap.add_argument("--train-dir", required=True,
                    help="Directory with best_depth.pt (+ best_pose.pt), e.g. checkpoints/vistacfusion_sitr_mae or outputs/<run>")
    ap.add_argument("--checkpoint", default=None,
                    help="Single checkpoint file (overrides best_depth/best_pose)")
    ap.add_argument("--model", default="configs/model/vistacfusion.yaml")
    ap.add_argument("--train", default="configs/train/default.yaml")
    ap.add_argument("--data", default="configs/data/default.yaml")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tactile", help="Single tactile image path (pair with --rgb)")
    mode.add_argument("--session-dir", help="<object>/session_*/sensor_* directory of the dataset")
    mode.add_argument("--eval", action="store_true", help="Evaluate on the validation split")

    ap.add_argument("--rgb", help="Single RGB image path (pair with --tactile)")
    ap.add_argument("--output-dir", default="eval_results")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="Device (e.g. cuda:1). Default: first available GPU.")
    args = ap.parse_args()

    cfg = merge_configs(args.model, args.train, args.data)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        depth_model = load_model(cfg, args.checkpoint, device)
        pose_model = None
        ckpt_info = args.checkpoint
    else:
        depth_ckpt = osp.join(args.train_dir, "best_depth.pt")
        pose_ckpt = osp.join(args.train_dir, "best_pose.pt")
        if not osp.exists(depth_ckpt):
            raise FileNotFoundError(f"No best_depth.pt in {args.train_dir}")
        depth_model = load_model(cfg, depth_ckpt, device)
        pose_model = None
        if osp.exists(pose_ckpt):
            pose_model = load_model(cfg, pose_ckpt, device)
            ckpt_info = f"depth={depth_ckpt}, pose={pose_ckpt}"
        else:
            ckpt_info = depth_ckpt

    os.makedirs(args.output_dir, exist_ok=True)
    if args.tactile:
        if not args.rgb:
            ap.error("--rgb is required when using --tactile")
        run_single(depth_model, args.rgb, args.tactile, cfg, device, args.output_dir,
                   name="prediction", pose_model=pose_model)
    elif args.session_dir:
        run_session(depth_model, args.session_dir, cfg, device, args.output_dir,
                    max_samples=args.max_samples, pose_model=pose_model)
    else:
        run_eval(depth_model, cfg, device, osp.join(args.output_dir, "val_vis"),
                 pose_model=pose_model, ckpt_info=ckpt_info)

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
