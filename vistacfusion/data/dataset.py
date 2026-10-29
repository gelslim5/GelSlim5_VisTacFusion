"""Datasets.

Every sample is a dict:
    rgb [3,H,W], tactile [3,H,W], depth [1,H,W] (mm), normal [3,H,W],
    mask [1,H,W] (contact region), pose [4] = (cos θ, sin θ, t_x, t_y),
    object: int (object id, shared between sim and real), domain: 0 = sim, 1 = real.

- SyntheticVisuoTactileDataset: deterministic random tensors, so the whole pipeline runs
  without data (tests / smoke runs).
- SimVisuoTactileDataset: loader for the released dataset layout (sim and real share it):
      <root>/<object>/session_*/sensor_*/
          samples/ (or samples_g/)   tactile PNGs
          rgb/                       visual PNGs (proximity mode)
          raw_data/                  NNNN_gt.npy depth [m] + NNNN_pose.json
          norms/                     NNNN_gt.png rendered normals (sim only, optional)
"""
from __future__ import annotations

import glob as _glob
import json
import math
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import (FIXED_CROP, RGBPhotometricAug, TactileAugment,
                         ToTensorResize, fixed_center_crop, rotate_gel_spin)


def zoom_center(img, zoom):
    """Zoom a PIL image by `zoom` (>= 1): center-crop side/zoom and resize back (bilinear).

    Used to align the sim RGB camera's mm->pixel scale to the real camera (sim.rgb_zoom).
    """
    w, h = img.size
    c = int(round(w / zoom))
    o = (w - c) // 2
    return img.crop((o, o, o + c, o + c)).resize((w, h), Image.BILINEAR)


def rotate_pose_theta(pose, dtheta_rad, rotate_xy=False):
    """Shift the pose theta by dtheta_rad (gel spins in place around the press point).

    pose: tensor [4] = (cos θ, sin θ, x, y).
    Sign convention: image rotated by +φ (cv2) → θ' = θ − φ (pose is sensor-relative-
    to-object: object appearing rotated +φ ⟺ sensor rotated −φ). Caller passes
    dtheta_rad = −radians(φ_cv2).

    rotate_xy: what to do with (x, y) depends on the label frame:
      * object-frame label (legacy sim branch, R(delta_rz) applied in _load_pose):
        the object does not move, so (x, y) is unchanged → rotate_xy=False.
      * camera-frame label (real data / align_real_rotation, tx = sx/half):
        (x, y) is the object centre seen in the camera; rotating the image by +φ
        rotates that vector by R(+φ) = R(−dtheta) → rotate_xy=True.
    """
    cos_t, sin_t, x, y = (v.item() for v in pose)
    c, s = math.cos(dtheta_rad), math.sin(dtheta_rad)
    cos_new = cos_t * c - sin_t * s
    sin_new = sin_t * c + cos_t * s
    if rotate_xy:  # R(−dtheta) = [[c, s], [−s, c]]
        x, y = c * x + s * y, -s * x + c * y
    return torch.tensor([cos_new, sin_new, x, y], dtype=torch.float32)


def depth_to_normal(depth, pixel_size_x, pixel_size_y):
    """Compute unit surface normals from a depth map via central finite differences.

    depth: (H, W) float32.  Returns (H, W, 3) float32 unit normals.
    """
    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / (2.0 * pixel_size_x)
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / (2.0 * pixel_size_y)
    dz_dx[:, 0] = (depth[:, 1] - depth[:, 0]) / pixel_size_x
    dz_dx[:, -1] = (depth[:, -1] - depth[:, -2]) / pixel_size_x
    dz_dy[0, :] = (depth[1, :] - depth[0, :]) / pixel_size_y
    dz_dy[-1, :] = (depth[-1, :] - depth[-2, :]) / pixel_size_y

    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
    return (normal / norm).astype(np.float32)


class SyntheticVisuoTactileDataset(Dataset):
    """Deterministic random-tensor stub. Each index yields a fixed sample (seeded by index),
    so a single batch can be memorized by the overfit test."""

    def __init__(self, num_samples=256, image_size=224, num_objects=8, seed=0):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_objects = num_objects
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        H = W = self.image_size
        rgb = torch.randn(3, H, W, generator=g)
        tactile = torch.randn(3, H, W, generator=g)
        depth = torch.rand(1, H, W, generator=g)
        normal = torch.randn(3, H, W, generator=g)
        normal = normal / normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
        mask = (depth > 0.3).float()
        theta = torch.rand(1, generator=g).item() * 2 * math.pi - math.pi
        tx = torch.rand(1, generator=g).item() * 2 - 1
        ty = torch.rand(1, generator=g).item() * 2 - 1
        pose = torch.tensor([math.cos(theta), math.sin(theta), tx, ty], dtype=torch.float32)
        return {
            "rgb": rgb, "tactile": tactile, "depth": depth, "normal": normal,
            "mask": mask, "pose": pose, "object": idx % self.num_objects,
        }


class SimVisuoTactileDataset(Dataset):
    """Loader for the released dataset layout (used for both the sim and the real section):
        <root>/<object>/session_*/sensor_*/{samples, rgb, raw_data[, norms]}

    Depth is loaded from raw_data/NNNN_gt.npy (metres), normals from norms/NNNN_gt.png when
    `use_rendered_normals` is set (sim) and otherwise computed from depth by finite
    differences (real). The SE(2) pose label is (cos θ, sin θ, tx, ty) with
    θ = rotation_euler[2] and (tx, ty) = (sample_x, sample_y) / half_target_size, see
    `_load_pose`.
    """

    def __init__(self, cfg_data, image_size, augment=False, include_objects=None,
                 split="all", seed=0, data_section="sim", shared_obj_map=None):
        """split: 'train' | 'val' | 'all'. Per-session split: every val_every-th sample
        (idx % val_every == 0) is val, the rest train. Deterministic — no split file.
        data_section: 'sim' or 'real' — which config section to read paths from.
        shared_obj_map: if provided, use this obj_name→id mapping (for sim+real alignment)."""
        self.image_size = image_size
        self.augment = augment
        self.split = split
        sim = getattr(cfg_data, data_section)
        norm = cfg_data.norm
        self.val_every = sim.get("val_every", 20)
        self.root = sim.root
        self.mesh_dir = sim.get("mesh_dir", osp.join(osp.dirname(self.root), "meshes"))
        self.rgb_subdir = sim.get("rgb_subdir", "rgb")
        self.tactile_subdir = sim.get("tactile_subdir", "samples")
        self.use_gt_depth = sim.get("use_gt_depth", True)
        # Sim RGB zoom (sim-to-real camera scale alignment): center-crop 1/zoom of the
        # sim RGB and resize back to the original size, so the sim RGB camera's mm->pixel
        # scale matches the real camera. Applied on load, BEFORE the fixed crop and any
        # augmentation. Labels unchanged (physical pose is the same).
        self.rgb_zoom = float(sim.get("rgb_zoom", 1.0))
        if self.rgb_zoom < 1.0:
            raise ValueError(f"{data_section}.rgb_zoom must be >= 1.0 (got {self.rgb_zoom})")
        # Tactile camera view is fixed & square for ALL objects (half-width 0.008751 m ->
        # 17.5 mm). The same fixed centre crop is applied to every sample (train, val,
        # inference), so the pixel size is one constant.
        gel_view_m = sim.get("gel_view_m", 0.017502)
        self.fixed_crop = float(sim.get("fixed_crop", cfg_data.get("fixed_crop", FIXED_CROP)))
        if not (0.0 < self.fixed_crop <= 1.0):
            raise ValueError(f"fixed_crop must be in (0, 1], got {self.fixed_crop}")
        self.pixel_size = gel_view_m * self.fixed_crop / image_size
        # Gel-spin rotation augmentation (train only): rotate tactile+rgb+depth+normal by a
        # random angle and shift the pose label accordingly.
        self.rot_aug = augment and sim.get("rot_augment", True)
        self.rot_aug_max_deg = sim.get("rot_augment_max_deg", 180.0)
        self.use_rendered_normals = sim.get("use_rendered_normals", False)
        self.data_section = data_section
        # Optional per-session train subsampling (real:sim ratio ladder).
        # Applies to the TRAIN split only; val keeps every val_every-th sample.
        self.train_samples_per_session = sim.get("train_samples_per_session", None)
        # Sim-to-real rotation alignment: use the ABSOLUTE session rotation as pose GT
        # (matching the real-data convention) and keep only sessions whose base rotation
        # falls inside the object's real rotation window (rotation_windows json).
        self.align_real_rotation = sim.get("align_real_rotation", False)
        # Label frame of (tx, ty): camera frame for real data and rotation-aligned sim
        # (tx = sx/half in _load_pose); object frame for the legacy sim branch. Gel-spin
        # rot aug must rotate (x, y) together with the image ONLY for camera-frame labels.
        self._xy_is_camera_frame = (data_section == "real") or self.align_real_rotation
        # Border-free bound: a square rotated by phi keeps a centred square of side
        # 1/(cos phi + sin phi) free of padding.
        if augment and self.rot_aug:
            _phi = math.radians(min(abs(self.rot_aug_max_deg), 45.0))
            _max_crop = 1.0 / (math.cos(_phi) + math.sin(_phi))
            if self.fixed_crop > _max_crop + 1e-6:
                print(f"[WARN] {data_section}: fixed_crop={self.fixed_crop:.3f} exceeds the "
                      f"border-free bound {_max_crop:.3f} for gel-spin up to "
                      f"{math.degrees(_phi):.0f} deg -> rotated samples will contain "
                      f"reflect-padded corners")
        self._rotation_windows = None
        if self.align_real_rotation and data_section == "sim":
            win_path = sim.get("rotation_windows", None)
            if win_path and osp.exists(win_path):
                with open(win_path) as f:
                    self._rotation_windows = json.load(f)
        # Sim translation filter: keep only sim samples whose press offset lies within the
        # per-object real translation range (× margin), so sim and real share the label range.
        self._translation_bounds = None
        trans_bounds_path = sim.get("translation_bounds", None)
        if trans_bounds_path and osp.exists(trans_bounds_path) and data_section == "sim":
            with open(trans_bounds_path) as f:
                self._translation_bounds = json.load(f)
            self._trans_margin = sim.get("translation_margin", 1.5)

        if self.root is None:
            raise ValueError(f"data config {data_section}.root is null — set the data path.")

        self.img_xform = ToTensorResize((image_size, image_size),
                                        norm.imagenet_mean, norm.imagenet_std)
        self.tactile_aug = TactileAugment() if augment else None
        self.rgb_aug = RGBPhotometricAug() if augment else None

        # Pre-load per-object mesh info for pose computation
        self._obj_pose_info = {}
        self._load_object_pose_info()

        # Discover sensor units (each = one session × one sensor)
        units = sorted(_glob.glob(osp.join(self.root, "*", "session_*", "sensor_*")))
        if not units:
            units = sorted(_glob.glob(osp.join(self.root, "sensor_*")))
        units = [u for u in units if osp.isdir(osp.join(u, self.tactile_subdir))]
        if include_objects is not None:
            incl = set(include_objects)
            units = [u for u in units
                     if osp.basename(osp.dirname(osp.dirname(u))) in incl]

        self.samples = []
        self.unit_meta = {}
        for unit in units:
            sample_dir = osp.join(unit, self.tactile_subdir)
            pngs = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
            if not pngs:
                continue

            session_dir = osp.dirname(unit)
            session_json = osp.join(session_dir, "session.json")
            with open(session_json) as f:
                sess = json.load(f)

            obj_name = osp.basename(osp.dirname(session_dir))
            info = self._obj_pose_info.get(obj_name)
            if info is None:
                continue

            base_rot = sess["base_rotation"]
            delta_rz = base_rot[2] - info["rz0"]

            # Rotation-window filter (sim-to-real alignment): drop sessions whose
            # base rotation lies outside the object's real rotation window.
            if (self._rotation_windows is not None
                    and obj_name in self._rotation_windows):
                lo, hi = self._rotation_windows[obj_name]
                center = (lo + hi) / 2.0
                halfwidth = (hi - lo) / 2.0
                base_deg = math.degrees(base_rot[2])
                d = (base_deg - center) % 360.0
                circ = min(d, 360.0 - d)
                if circ > halfwidth:
                    continue

            session_center = self._get_session_center(
                info["vertices"], info["fixed_scale"], base_rot)

            self.unit_meta[unit] = {
                "delta_rz": delta_rz,
                "rz0": info["rz0"],
                "session_center": session_center,
                "half": info["half"],
            }

            selected = []
            for png in pngs:
                idx = int(osp.splitext(png)[0])
                is_val = (idx % self.val_every == 0)
                if self.split == "train" and is_val:
                    continue
                if self.split == "val" and not is_val:
                    continue
                selected.append(idx)

            # Evenly subsample train samples per session (ratio ladder)
            if (self.split == "train" and self.train_samples_per_session is not None
                    and len(selected) > self.train_samples_per_session):
                k = int(self.train_samples_per_session)
                if k <= 0:
                    selected = []
                else:
                    pos = np.linspace(0, len(selected) - 1, k).round().astype(int)
                    selected = [selected[i] for i in sorted(set(pos.tolist()))]

            trans_lim = None
            if self._translation_bounds is not None and obj_name in self._translation_bounds:
                tb = self._translation_bounds[obj_name]
                trans_lim = (tb["sx_max"] * self._trans_margin,
                             tb["sy_max"] * self._trans_margin)

            for idx in selected:
                suffix = "_gt" if self.use_gt_depth else ""
                rgb_ok = osp.exists(osp.join(unit, self.rgb_subdir, f"{idx:04d}.png"))
                depth_ok = osp.exists(osp.join(unit, "raw_data", f"{idx:04d}{suffix}.npy"))
                pose_ok = osp.exists(osp.join(unit, "raw_data", f"{idx:04d}_pose.json"))
                if rgb_ok and depth_ok and pose_ok:
                    if trans_lim is not None:
                        with open(osp.join(unit, "raw_data", f"{idx:04d}_pose.json")) as _pf:
                            _pd = json.load(_pf)
                        if abs(_pd["sample_x"]) > trans_lim[0] or abs(_pd["sample_y"]) > trans_lim[1]:
                            continue
                    self.samples.append((unit, idx))

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root} "
                               f"(include_objects={include_objects})")

        valid_units = [u for u in units if u in self.unit_meta]
        self.objects = sorted({osp.basename(osp.dirname(osp.dirname(u)))
                               for u in valid_units})
        if shared_obj_map is not None:
            self._obj_to_id = shared_obj_map
        else:
            self._obj_to_id = {o: i for i, o in enumerate(self.objects)}

    def _load_object_pose_info(self):
        """Pre-load mesh + first-session info for each object."""
        try:
            import trimesh
            from scipy.spatial.transform import Rotation  # noqa: F401
        except ImportError:
            print("[WARN] trimesh/scipy not available, pose labels will be zeros")
            return

        obj_dirs = sorted(d for d in os.listdir(self.root)
                          if osp.isdir(osp.join(self.root, d)))
        for obj_name in obj_dirs:
            mesh_path = osp.join(self.mesh_dir, f"{obj_name}.obj")
            s0_path = osp.join(self.root, obj_name, "session_000", "session.json")
            if not osp.exists(s0_path):
                sess_dirs = sorted(d for d in os.listdir(osp.join(self.root, obj_name))
                                   if d.startswith("session_"))
                s0_path = osp.join(self.root, obj_name, sess_dirs[0], "session.json") if sess_dirs else ""
            if not osp.exists(mesh_path) or not osp.exists(s0_path):
                continue
            mesh = trimesh.load(mesh_path, force="mesh")
            with open(s0_path) as f:
                d0 = json.load(f)
            fixed_scale = d0["fixed_scale"]
            target_size = d0.get("_target_size_mm", 82.0)
            half = target_size / 2.0 / 1000.0
            rz0 = d0["base_rotation"][2]
            self._obj_pose_info[obj_name] = {
                "vertices": mesh.vertices,
                "fixed_scale": fixed_scale,
                "half": half,
                "rz0": rz0,
            }

    @staticmethod
    def _get_session_center(vertices, fixed_scale, base_rotation):
        from scipy.spatial.transform import Rotation
        R_3d = Rotation.from_euler("xyz", base_rotation)
        v = R_3d.apply(vertices) / fixed_scale
        cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
        cy = (v[:, 1].min() + v[:, 1].max()) / 2.0
        return np.array([cx, cy])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self._load_sample(index)

    def _load_sample(self, index):
        unit, sample_idx = self.samples[index]
        meta = self.unit_meta[unit]

        # --- Load images ---
        tactile = np.array(
            Image.open(osp.join(unit, self.tactile_subdir, f"{sample_idx:04d}.png")),
            dtype=np.float32,
        )
        rgb_img = Image.open(osp.join(unit, self.rgb_subdir, f"{sample_idx:04d}.png"))
        if self.rgb_zoom != 1.0:
            rgb_img = zoom_center(rgb_img.convert("RGB"), self.rgb_zoom)
        rgb = np.array(rgb_img, dtype=np.float32)

        # --- Load depth (float32, H×W, metres) ---
        suffix = "_gt" if self.use_gt_depth else ""
        depth = np.load(
            osp.join(unit, "raw_data", f"{sample_idx:04d}{suffix}.npy")
        ).astype(np.float32)

        # --- Load normal GT from the rendered PNG (if enabled) ---
        normal = None
        if self.use_rendered_normals:
            norm_suffix = "_gt" if self.use_gt_depth else ""
            norm_path = osp.join(unit, "norms", f"{sample_idx:04d}{norm_suffix}.png")
            if osp.exists(norm_path):
                normal = np.array(Image.open(norm_path), dtype=np.float32)

        # --- Pose: SE(2) = (cos θ, sin θ, tx_norm, ty_norm) ---
        pose = self._load_pose(unit, sample_idx, meta)

        # --- Gel-spin rotation aug: tactile+rgb+depth+normal rotate together, θ -= φ ---
        if self.rot_aug:
            phi_deg = random.uniform(-self.rot_aug_max_deg, self.rot_aug_max_deg)
            if normal is not None:
                tactile, rgb, depth, normal = rotate_gel_spin(
                    tactile, rgb, depth, phi_deg, normal=normal)
            else:
                tactile, rgb, depth = rotate_gel_spin(tactile, rgb, depth, phi_deg)
            pose = rotate_pose_theta(pose, -math.radians(phi_deg),
                                     rotate_xy=self._xy_is_camera_frame)

        # --- Fixed center crop: EVERY sample (train/val, rotated or not) ---
        tactile = fixed_center_crop(tactile, crop=self.fixed_crop)
        rgb = fixed_center_crop(rgb, crop=self.fixed_crop)
        depth = fixed_center_crop(depth, crop=self.fixed_crop)
        if normal is not None:
            normal = fixed_center_crop(normal, crop=self.fixed_crop)

        # --- Photometric augmentation ---
        if self.tactile_aug is not None:
            tactile, _, depth, _, _ = self.tactile_aug(tactile, [], depth, None)
        if self.rgb_aug is not None:
            rgb = self.rgb_aug(rgb)

        # --- Normal: use PNG if available, else compute from depth ---
        if normal is not None:
            normal = normal / 127.5 - 1.0
        else:
            normal = depth_to_normal(depth, self.pixel_size, self.pixel_size)

        # --- Contact mask (before scaling) ---
        mask = (depth > 0).astype(np.float32)

        # --- Depth in mm (×1000) for numerical stability in fp16 ---
        depth = depth * 1000.0

        # --- Object ID ---
        obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
        obj_id = self._obj_to_id[obj_name]

        domain = 0 if self.data_section == "sim" else 1

        return {
            "rgb": self.img_xform(rgb),
            "tactile": self.img_xform(tactile),
            "depth": torch.from_numpy(depth).unsqueeze(0),       # (1, H, W)
            "normal": torch.from_numpy(np.ascontiguousarray(normal))
                      .permute(2, 0, 1),                         # (3, H, W)
            "mask": torch.from_numpy(mask).unsqueeze(0),          # (1, H, W)
            "pose": pose,                                         # (4,)
            "object": obj_id,
            "domain": domain,                                     # 0=sim, 1=real
        }

    def _load_pose(self, unit, sample_idx, meta):
        pose_path = osp.join(unit, "raw_data", f"{sample_idx:04d}_pose.json")
        with open(pose_path) as f:
            data = json.load(f)

        half = meta["half"]
        sx, sy = data["sample_x"], data["sample_y"]

        if self.data_section == "real" or self.align_real_rotation:
            # Real data (and rotation-aligned sim): rotation_euler[2] is the
            # absolute GT rotation; translation is normalized but not rotated
            # (camera-frame label: object centre as seen by the sensor).
            rz = data["rotation_euler"][2]
            cos_rz = math.cos(rz)
            sin_rz = math.sin(rz)
            tx = sx / max(half, 1e-8)
            ty = sy / max(half, 1e-8)
        else:
            # Legacy sim branch: delta rotation w.r.t. the first session and
            # translation rotated into the object frame.
            delta_rz = data["rotation_euler"][2] - meta["rz0"]
            cos_rz = math.cos(delta_rz)
            sin_rz = math.sin(delta_rz)
            tx = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
            ty = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)

        return torch.tensor([cos_rz, sin_rz, tx, ty], dtype=torch.float32)


def build_datasets(cfg):
    """Return (train_ds, val_ds) from the merged config (uses cfg.dataset switch)."""
    image_size = cfg.image_size
    which = cfg.dataset
    if which == "synthetic":
        s = cfg.synthetic
        n = s.num_samples
        n_val = max(1, n // 8)
        train = SyntheticVisuoTactileDataset(n - n_val, image_size, s.num_objects, seed=0)
        val = SyntheticVisuoTactileDataset(n_val, image_size, s.num_objects, seed=1)
        return train, val
    if which == "sim":
        sim_objects = cfg.sim.get("include_objects", None)
        sim_objects = list(sim_objects) if sim_objects else None
        train = SimVisuoTactileDataset(cfg, image_size, augment=True, split="train",
                                       include_objects=sim_objects)
        val = SimVisuoTactileDataset(cfg, image_size, augment=False, split="val",
                                     include_objects=sim_objects)
        print(f"  sim-only: train={len(train)} val={len(val)} (sim val split)")
        return train, val
    if which == "sim+real":
        from torch.utils.data import ConcatDataset

        all_real_objects = sorted(
            d for d in os.listdir(cfg.real.root)
            if osp.isdir(osp.join(cfg.real.root, d)))

        sim_objects = cfg.sim.get("include_objects", None)
        sim_objects = list(sim_objects) if sim_objects else None
        sim_val = SimVisuoTactileDataset(cfg, image_size, augment=False, split="val",
                                         include_objects=sim_objects)
        spp = cfg.sim.get("train_samples_per_session", None)
        if spp is not None and int(spp) <= 0:
            sim_train = None                     # real-only: no sim in training
            shared_obj_map = sim_val._obj_to_id
        else:
            sim_train = SimVisuoTactileDataset(cfg, image_size, augment=True, split="train",
                                               include_objects=sim_objects)
            shared_obj_map = sim_train._obj_to_id
        # Real objects absent from sim get fresh ids so the real datasets can index the map.
        missing = [o for o in all_real_objects if o not in shared_obj_map]
        if missing:
            shared_obj_map = dict(shared_obj_map)
            for o in missing:
                shared_obj_map[o] = len(shared_obj_map)
        real_augment = cfg.real.get("augment", True)
        rspp = cfg.real.get("train_samples_per_session", None)
        if rspp is not None and int(rspp) <= 0:
            real_train = None                    # sim-only baseline: no real in training
        else:
            real_train = SimVisuoTactileDataset(
                cfg, image_size, augment=real_augment, split="train",
                data_section="real", shared_obj_map=shared_obj_map,
                include_objects=all_real_objects)
        real_val = SimVisuoTactileDataset(
            cfg, image_size, augment=False, split="val",
            data_section="real", shared_obj_map=shared_obj_map,
            include_objects=all_real_objects)

        n_sim = len(sim_train) if sim_train is not None else 0
        n_real = len(real_train) if real_train is not None else 0
        if n_sim == 0 and n_real == 0:
            raise ValueError("Both sim and real train are empty — check "
                             "train_samples_per_session settings.")
        real_oversample = cfg.real.get("oversample", 1)
        if real_oversample <= 0:
            real_oversample = max(1, n_sim // max(1, n_real) // 2)
        print(f"  sim+real co-training: sim={n_sim}+{len(sim_val)}, "
              f"real_train={n_real}+{len(real_val)} (oversample {real_oversample}x)")
        print(f"  shared object classes: {list(shared_obj_map.keys())}")
        parts = ([sim_train] if sim_train is not None else [])
        if real_train is not None:
            parts += [real_train] * real_oversample
        train = ConcatDataset(parts)
        return train, real_val
    raise ValueError(f"Unknown dataset {which!r} (data config dataset:)")
