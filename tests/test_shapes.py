"""Shape tests.

The three input modes (both / tactile-only / rgb-only) run through one shared model and must
produce identically-shaped outputs: depth [B,1,H,W], normal [B,3,H,W], se2 [B,4].
"""
import torch

from vistacfusion.models.model import build_model

B = 2


def _inputs(cfg):
    s = cfg.image_size
    return torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)


def test_output_shapes_all_configs(cfg):
    model = build_model(cfg).eval()
    rgb, tac = _inputs(cfg)
    s = cfg.image_size
    for config in ("both", "tactile", "rgb"):
        out = model(rgb, tac, config=config)
        assert out["depth"].shape == (B, 1, s, s), config
        assert out["normal"].shape == (B, 3, s, s), config
        assert out["se2"].shape == (B, 4), config
        # (cos, sin) must be unit-norm
        cs = out["se2"][:, :2]
        assert torch.allclose(cs.norm(dim=-1), torch.ones(B), atol=1e-4), config


def test_object_ids_change_pose_only_path(cfg):
    """Object embedding is added to the pose queries; it must not break any mode."""
    model = build_model(cfg).eval()
    rgb, tac = _inputs(cfg)
    ids = torch.tensor([0, 3])
    for config in ("both", "tactile", "rgb"):
        out = model(rgb, tac, config=config, object_ids=ids)
        assert out["se2"].shape == (B, 4), config


def test_injection_zero_without_rgb(cfg):
    """With the ReZero gate at its init (0) and RGB absent, the dense taps are pure encoder
    taps, so the tactile-only output must be invariant to the RGB tensor."""
    model = build_model(cfg).eval()
    _, tac = _inputs(cfg)
    rgb_a = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    rgb_b = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        out_a = model(rgb_a, tac, config="tactile")
        out_b = model(rgb_b, tac, config="tactile")
    assert torch.allclose(out_a["depth"], out_b["depth"], atol=1e-5)
    assert torch.allclose(out_a["se2"], out_b["se2"], atol=1e-5)


def test_frozen_encoders_have_no_trainable_params(cfg):
    model = build_model(cfg)
    for enc in (model.tactile_encoder, model.rgb_encoder):
        assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 0, "encoder must be frozen"
