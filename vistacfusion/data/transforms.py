"""Augmentation + tensorization.

- TactileAugment   : sim2real domain randomization -- gain/bias/gradient/brightness/residual/
  noise (photometric only; geometric handled by rotate_gel_spin).
- rotate_gel_spin  : gel-spin rotation aug -- rotates tactile+rgb+depth by the same angle
  around the image center. GT theta shifts by MINUS the image angle (pose = sensor
  relative to object; verified against real cross-session pairs of asymmetric objects);
  (x, y) unchanged (rotation center = press point = image center).
- fixed_center_crop: the FIXED 1/sqrt(2) center crop applied to EVERY sample (train, val,
  augmented or not, and real inference). Covers the 45-deg worst-case inscribed square,
  so any rotation angle leaves no border artifacts, the scale is constant everywhere
  (no zoom<->angle leakage, no train/val scale gap), and the pixel<->physical mapping
  is a single constant.
- RGBPhotometricAug: photometric jitter on the RGB context (geometry kept stable).
- ToTensorResize   : HWC float32 -> normalized CHW tensor, resized to (H, W).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Fixed crop factor: worst-case (45 deg) inscribed square. Every sample lives in this
# cropped world; effective view = gel_view * FIXED_CROP (17.5mm -> 12.37mm).
# 1/sqrt(2) is the largest border-free square for ANY gel-spin angle; for a max spin of
# 15 deg, 1/(cos15+sin15) = 0.816 suffices (data cfg `fixed_crop`, default = FIXED_CROP).
FIXED_CROP = 1.0 / math.sqrt(2.0)


def fixed_center_crop(img, out_size=None, crop=FIXED_CROP):
    """Center-crop to `crop` of the side length (default FIXED_CROP), resize back to
    original (or out_size). `crop` comes from data cfg `fixed_crop` (same for sim & real)."""
    H, W = img.shape[:2]
    side = int(math.floor(min(H, W) * crop))
    off_y = (H - side) // 2
    off_x = (W - side) // 2
    crop = img[off_y:off_y + side, off_x:off_x + side]
    out = out_size or (W, H)
    if isinstance(out, int):
        out = (out, out)
    return cv2.resize(crop, out, interpolation=cv2.INTER_LINEAR)


def rotate_gel_spin(tactile, rgb, depth, angle_deg, normal=None):
    """Rotate tactile + rgb + depth (+ optional normal) by angle_deg around the image center.

    Simulates spinning the gel in place at the same press point: image content rotates,
    theta changes by -angle_deg (pose = sensor relative to object; verified on real
    cross-session pairs). Pose (x, y): unchanged for object-frame labels, rotated by
    R(+angle_deg) for camera-frame labels — see dataset.rotate_pose_theta(rotate_xy).
    """
    H, W = tactile.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    flags, border = cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
    tac = cv2.warpAffine(tactile, M, (W, H), flags=flags, borderMode=border)
    rgb_r = cv2.warpAffine(rgb, M, (W, H), flags=flags, borderMode=border)
    dep = cv2.warpAffine(depth, M, (W, H), flags=flags, borderMode=border)
    if normal is None:
        return tac, rgb_r, dep
    norm_r = cv2.warpAffine(normal, M, (W, H), flags=flags, borderMode=border)
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.float32(np.cos(rad)), np.float32(np.sin(rad))
    nx = norm_r[:, :, 0] / 127.5 - 1.0
    ny = norm_r[:, :, 1] / 127.5 - 1.0
    norm_r[:, :, 0] = np.clip((cos_a * nx + sin_a * ny + 1.0) * 127.5, 0, 255)
    norm_r[:, :, 1] = np.clip((-sin_a * nx + cos_a * ny + 1.0) * 127.5, 0, 255)
    return tac, rgb_r, dep, norm_r

DEFAULT_AUGMENT_PARAMS = {
    "gain": 0.5, "bias": 45.0, "grad": 0.7, "bright": 25.0,
    "resid": 20.0, "noise": 6.0, "rot_deg": 0.0,
    "hflip": False, "vflip": False,
}

LIGHT_AUGMENT_PARAMS = {
    "gain": 0.3, "bias": 20.0, "grad": 0.4, "bright": 15.0,
    "resid": 0.0, "noise": 4.0, "rot_deg": 0.0,
    "hflip": False, "vflip": False,
}


class TactileAugment:
    """Heavy tactile domain randomization. Operates on float32 HWC arrays in-place-ish.

    __call__(sample_diff, calib_diffs, depth, normal) -> same tuple, augmented.
    Geometric ops are applied to ALL images; normal vectors are corrected on flip/rotate.

    style="heavy" (default): per-channel gain/bias, strong params, residual noise.
    style="light": uniform gain/bias (like RGBPhotometricAug), no residual noise.
    """

    def __init__(self, params=None, style="heavy"):
        defaults = LIGHT_AUGMENT_PARAMS if style == "light" else DEFAULT_AUGMENT_PARAMS
        self.p = {**defaults, **(params or {})}
        self.uniform = (style == "light")

    def __call__(self, sample_diff, calib_diffs, depth, normal):
        p = self.p
        H, W = sample_diff.shape[:2]

        # ---- Photometric (sample only) ----
        if p["gain"] > 0:
            if self.uniform:
                g = np.float32(np.random.uniform(1 - p["gain"], 1 + p["gain"]))
            else:
                g = np.random.uniform(1 - p["gain"], 1 + p["gain"], size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff * g
        if p["bias"] > 0:
            if self.uniform:
                b = np.float32(np.random.uniform(-p["bias"], p["bias"]))
            else:
                b = np.random.uniform(-p["bias"], p["bias"], size=(1, 1, 3)).astype(np.float32)
            sample_diff = sample_diff + b
        if p["bright"] > 0:
            sample_diff = sample_diff + np.float32(np.random.uniform(-p["bright"], p["bright"]))
        if p["grad"] > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, H, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, -1)
            grad_map = np.float32(np.cos(angle)) * xs + np.float32(np.sin(angle)) * ys
            if self.uniform:
                amp = np.float32(np.random.uniform(0, p["grad"]))
                sample_diff = sample_diff + grad_map[..., None] * amp * np.float32(30.0)
            else:
                amp = np.random.uniform(0, p["grad"], size=(1, 1, 3)).astype(np.float32)
                sample_diff = sample_diff + grad_map[..., None] * amp * np.float32(50.0)
        if p["resid"] > 0:
            raw = np.random.randn(16, 16, 3).astype(np.float32)
            smooth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
            smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=H / 8.0)
            std = np.float32(smooth.std())
            if std > 1e-6:
                smooth = smooth / std * np.float32(p["resid"])
            sample_diff = sample_diff + smooth
        if p["noise"] > 0:
            noise = np.random.normal(0, p["noise"], sample_diff.shape).astype(np.float32)
            sample_diff = sample_diff + noise

        # ---- Geometric (all images) ----
        do_hflip = p["hflip"] and np.random.random() < 0.5
        do_vflip = p["vflip"] and np.random.random() < 0.5
        rot_angle = np.random.uniform(-p["rot_deg"], p["rot_deg"]) if p["rot_deg"] > 0 else 0.0

        if do_hflip:
            sample_diff = np.ascontiguousarray(sample_diff[:, ::-1])
            calib_diffs = [np.ascontiguousarray(c[:, ::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[:, ::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[:, ::-1])
                normal[:, :, 0] = 255.0 - normal[:, :, 0]

        if do_vflip:
            sample_diff = np.ascontiguousarray(sample_diff[::-1])
            calib_diffs = [np.ascontiguousarray(c[::-1]) for c in calib_diffs]
            if depth is not None:
                depth = np.ascontiguousarray(depth[::-1])
            if normal is not None:
                normal = np.ascontiguousarray(normal[::-1])
                normal[:, :, 1] = 255.0 - normal[:, :, 1]

        if abs(rot_angle) > 0.5:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_angle, 1.0)
            flags, border = cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
            sample_diff = cv2.warpAffine(sample_diff, M, (W, H), flags=flags, borderMode=border)
            calib_diffs = [cv2.warpAffine(c, M, (W, H), flags=flags, borderMode=border)
                           for c in calib_diffs]
            if depth is not None:
                depth = cv2.warpAffine(depth, M, (W, H), flags=flags, borderMode=border)
            if normal is not None:
                normal = cv2.warpAffine(normal, M, (W, H), flags=flags, borderMode=border)
                rad = np.radians(rot_angle)
                cos_a, sin_a = np.float32(np.cos(rad)), np.float32(np.sin(rad))
                nx = normal[:, :, 0] / 127.5 - 1.0
                ny = normal[:, :, 1] / 127.5 - 1.0
                normal[:, :, 0] = np.clip((cos_a * nx + sin_a * ny + 1.0) * 127.5, 0, 255)
                normal[:, :, 1] = np.clip((-sin_a * nx + cos_a * ny + 1.0) * 127.5, 0, 255)

        geo = {"hflip": do_hflip, "vflip": do_vflip, "rot_deg": rot_angle}
        return sample_diff, calib_diffs, depth, normal, geo


class RGBPhotometricAug:
    """Photometric jitter on the RGB context image (float32 HWC). No geometry.
    All gain/bias/brightness are UNIFORM across channels to preserve marker hue."""

    def __init__(self, gain=0.3, bias=20.0, bright=15.0, grad=0.4, noise=4.0):
        self.gain = gain
        self.bias = bias
        self.bright = bright
        self.grad = grad
        self.noise = noise

    def __call__(self, rgb):
        H, W = rgb.shape[:2]
        if self.gain > 0:
            g = np.float32(np.random.uniform(1 - self.gain, 1 + self.gain))
            rgb = rgb * g
        if self.bias > 0:
            b = np.float32(np.random.uniform(-self.bias, self.bias))
            rgb = rgb + b
        if self.bright > 0:
            rgb = rgb + np.float32(np.random.uniform(-self.bright, self.bright))
        if self.grad > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, H, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, -1)
            grad_map = np.float32(np.cos(angle)) * xs + np.float32(np.sin(angle)) * ys
            rgb = rgb + grad_map[..., None] * np.float32(self.grad * 30.0)
        if self.noise > 0:
            rgb = rgb + np.random.normal(0, self.noise, rgb.shape).astype(np.float32)
        return rgb


class ToTensorResize:
    """np.float32 HWC -> normalized CHW tensor, bilinear-resized to out_hw."""

    def __init__(self, out_hw, mean, std):
        self.out_hw = tuple(out_hw)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        t = t.permute(2, 0, 1)                          # HWC -> CHW
        if (t.shape[1], t.shape[2]) != self.out_hw:
            t = F.interpolate(t.unsqueeze(0), size=self.out_hw,
                              mode="bilinear", align_corners=False).squeeze(0)
        return (t - self.mean) / self.std
