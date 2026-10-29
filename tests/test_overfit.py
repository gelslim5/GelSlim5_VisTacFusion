"""One-batch overfit test: loss must drop sharply on a single synthetic batch -- proves the
model + losses + backprop are wired correctly."""
import torch

from vistacfusion.losses.total import MultiTaskLoss
from vistacfusion.models.model import build_model


def test_overfit_single_batch(cfg):
    torch.manual_seed(0)
    # Shrink to a tiny but shape-faithful model so a single batch overfits fast on CPU:
    # 64/16 = 4 -> 16-token square grid (DPT Reassemble still valid). L1 depth (SSI-normalizing
    # random targets is needlessly hard for a wiring check).
    cfg.image_size = 64
    cfg.tokens.num_spatial_queries = 16          # 64/16 = 4 -> 4x4 grid
    cfg.fusion_trunk.num_bottleneck_tokens = 8
    cfg.heads.dpt.features = 64
    cfg.loss.depth.type = "l1"

    model = build_model(cfg).train()
    criterion = MultiTaskLoss(cfg.loss, pose_mode=cfg.heads.pose.pose_mode,
                              rot_num_bins=cfg.heads.pose.get("rot_num_bins", 72))

    B, s = 2, cfg.image_size
    F = torch.nn.functional
    rgb = torch.randn(B, 3, s, s)
    tac = torch.randn(B, 3, s, s)
    # Low-frequency (representable) dense targets -- a real depth/normal map is smooth, not
    # per-pixel white noise, and a 4x4 token grid + DPT upsampling can only fit low frequencies.
    gt = {
        "depth": F.interpolate(torch.randn(B, 1, 8, 8), size=(s, s), mode="bilinear",
                               align_corners=False),
        "normal": F.normalize(F.interpolate(torch.randn(B, 3, 8, 8), size=(s, s),
                                            mode="bilinear", align_corners=False), dim=1),
        "pose": _rand_pose(B),
    }

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-3)

    first = last = None
    for step in range(250):
        opt.zero_grad(set_to_none=True)
        out = model(rgb, tac, config="both")
        loss, _ = criterion(out, gt, supervise_dense=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.3, f"loss did not drop enough: {first:.4f} -> {last:.4f}"


def _rand_pose(B):
    theta = torch.rand(B) * 6.283 - 3.1415
    txy = torch.rand(B, 2) * 2 - 1
    return torch.stack([theta.cos(), theta.sin(), txy[:, 0], txy[:, 1]], dim=1)
