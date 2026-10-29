"""Attention-routing tests: verify the cross/self-attention wiring is directionally correct,
not just shape-correct.

  - tactile-only must ignore RGB entirely (trunk steps (1)(2) skipped, no leak).
  - 'both' must let RGB influence the pose (RGB really flows through the bottleneck).
  - tactile-only must still respond to tactile (self-attention (3) is alive).
  - the ReZero injection gate gates RGB into the dense taps (0 -> no effect, >0 -> effect).
"""
import torch

from vistacfusion.models.model import build_model

B = 2


def _inputs(cfg):
    s = cfg.image_size
    return torch.randn(B, 3, s, s), torch.randn(B, 3, s, s), torch.randn(B, 3, s, s)


def test_tactile_only_ignores_rgb(cfg):
    model = build_model(cfg).eval()
    _, rgb_a, rgb_b = _inputs(cfg)
    tac = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        a = model(rgb_a, tac, config="tactile")
        b = model(rgb_b, tac, config="tactile")
    assert torch.allclose(a["depth"], b["depth"], atol=1e-6)
    assert torch.allclose(a["se2"], b["se2"], atol=1e-6)


def test_both_pose_depends_on_rgb(cfg):
    model = build_model(cfg).eval()
    _, rgb_a, rgb_b = _inputs(cfg)
    tac = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        a = model(rgb_a, tac, config="both")
        b = model(rgb_b, tac, config="both")
    assert not torch.allclose(a["se2"], b["se2"], atol=1e-5)


def test_tactile_only_depends_on_tactile(cfg):
    """Self-attention (3) must be alive: different tactile -> different output."""
    model = build_model(cfg).eval()
    _, rgb, _ = _inputs(cfg)
    s = cfg.image_size
    with torch.no_grad():
        a = model(rgb, torch.randn(B, 3, s, s), config="tactile")
        b = model(rgb, torch.randn(B, 3, s, s), config="tactile")
    assert not torch.allclose(a["depth"], b["depth"], atol=1e-4)


def test_injection_gate_controls_rgb_into_dense(cfg):
    model = build_model(cfg).eval()
    _, rgb_a, rgb_b = _inputs(cfg)
    tac = torch.randn(B, 3, cfg.image_size, cfg.image_size)

    # gate == 0 (init): RGB must not affect depth, but must affect pose (pose comes via trunk).
    with torch.no_grad():
        a = model(rgb_a, tac, config="both")
        b = model(rgb_b, tac, config="both")
    assert torch.allclose(a["depth"], b["depth"], atol=1e-4)
    assert not torch.allclose(a["se2"], b["se2"], atol=1e-5)

    # gate > 0: RGB now flows into the dense taps.
    for inj in model.tap_inject:
        torch.nn.init.constant_(inj.gate, 0.5)
    with torch.no_grad():
        a = model(rgb_a, tac, config="both")
        b = model(rgb_b, tac, config="both")
    assert not torch.allclose(a["depth"], b["depth"], atol=1e-4)
