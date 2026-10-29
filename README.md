# GelSlim 5.0 — VisTacFusion

Code and dataset for **GelSlim 5.0: Unifying Vision and Touch with a Single
Fingertip Camera** ([arXiv 2610.31334](https://arxiv.org/abs/2610.31334),
[project page](https://gelslim5.github.io/geslim5_website/)).

GelSlim 5.0 is a fingertip sensor whose single fixed camera acquires a through-membrane
**visual** image (UV off, proximity mode) and a contact-sensitive **tactile** image (UV on)
from the same viewpoint. **VisTacFusion** fuses the pair and predicts, in the tactile frame:

- contact **depth** and **surface normals** (DPT decoder on tactile multiscale features with a
  gated visual injection), and
- the planar **SE(2) object pose** `(cos θ, sin θ, t_x, t_y)` (asymmetric bottleneck fusion
  trunk → pose MLP).

One trained model runs with paired input, tactile only, or vision only (modality dropout).

```
tactile [B,3,224,224] ──► SITR-B/16 (frozen) ──► multiscale taps ──► gated RGB injection ──► DPT ──► depth, normal
                                 │ tokens                                     ▲
                                 ▼                                            │ 16 bottleneck tokens
visual  [B,3,224,224] ──► MAE-L/16 (frozen) ──► asymmetric bottleneck fusion trunk ──► pose MLP ──► (cos θ, sin θ, t_x, t_y)
```

## Results (real validation set, 12 objects, same model in all three input modes)

| Input            | Depth MSE (mm²) | Normal MSE | Rotation L₁ | Angular error (°) | Translation L₁ |
|------------------|----------------:|-----------:|------------:|------------------:|---------------:|
| vision + tactile | **0.0066**      | **0.0074** | **0.0129**  | **0.652**         | **0.0053**     |
| tactile only     | 0.0170          | 0.0122     | 0.0425      | 2.036             | 0.0122         |
| vision only      | 0.1124          | 0.0369     | 0.0134      | 0.676             | 0.0056         |

Translation is normalised by half the 82 mm object size (0.0053 ≈ 0.22 mm). Depth / normal are
read from the `best_depth.pt` of a run, rotation / translation from its `best_pose.pt`
(checkpoint selection in `vistacfusion/engine/train.py`). `bash scripts/train.sh` with the
default configs reproduces this setting (Table II of the paper, where the baselines are listed).

## Installation

```bash
git clone https://github.com/gelslim5/GelSlim5_VisTacFusion.git
cd GelSlim5_VisTacFusion
pip install -e .            # Python ≥ 3.10, PyTorch ≥ 2.2
pytest                      # CPU smoke tests with mock encoders, no weights needed
```

## Dataset

The dataset is hosted on Hugging Face (dataset repo `gelslim5/gelslim5-vistacfusion`, see
[docs/DATASET.md](docs/DATASET.md) for the layout and label conventions):

```bash
pip install huggingface_hub
bash scripts/download_dataset.sh         # real + sim + meshes (~20 GB)
bash scripts/download_dataset.sh real    # real data + meshes only (~1.3 GB)
```

| Part | Content | Size |
|---|---|---|
| `real/` | 12 objects × 1 session, 4 669 paired tactile / visual frames with mesh-rendered depth GT and robot pose GT | 1.3 GB |
| `sim/` | the same 12 objects × 12 sessions (30° rotation steps), 50 400 paired frames: Blender render, generated tactile appearance, visual render, depth / normal GT, pose | 18 GB |
| `meshes/` | the 12 object meshes (`.obj`, mm) | 3 MB |

## Encoder weights

The encoders are frozen and loaded from their original checkpoints, which you download
yourself (licences differ per model):

| Encoder | Config key | File expected at | Source |
|---|---|---|---|
| SITR-B/16 (tactile, paper) | `encoder.checkpoint` | `weights/SITR_B18.pth` | [SITR](https://github.com/hgupt3/gsrl) release |
| MAE ViT-L/16 (visual, paper) | `rgb_encoder.checkpoint` | `weights/mae_vitl16.pth` | [MAE](https://github.com/facebookresearch/mae) `mae_pretrain_vit_large.pth` |
| DINOv3 ViT-L/16 | `weights/dinov3_vitl16.pth` | gated, [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| Depth Anything v2 ViT-L | `weights/dav2_vitl14.pth` | [DAv2](https://github.com/DepthAnything/Depth-Anything-V2) |
| T3 | `weights/t3_large/` | [T3](https://github.com/alanzjl/t3) |
| Sparsh (MAE / DINOv2 base), Sparsh-X | `weights/sparsh_*`, `weights/sparshx_img_base.ckpt` | [Sparsh](https://github.com/facebookresearch/sparsh), [Sparsh-X](https://github.com/facebookresearch/sparsh-multisensory-touch) |
| TVL ViT-B/16 | `weights/tvl_enc_vitb.pth` | [TVL](https://github.com/Max-Fu/tvl) |
| CLIP ViT-L/14, SigLIP ViT-L/16 | `model_id` | downloaded through `transformers` |

Only SITR and MAE are needed for the paper model; the others are the encoder-ablation
baselines (`configs/model/tac_*_rgb_*.yaml`). The SITR encoder is calibrated on the 18
calibration images of `dataset/sim/pattern_01_2_lines_angle_1_2/session_000/sensor_0000/calibration/`,
which ship with the dataset.

## Evaluate a trained model

```bash
bash scripts/eval.sh --train-dir outputs/<run>          # real validation split, all three input modes
# -> eval_results/eval_metrics.txt and one visualisation per object in eval_results/val_vis/

# one sensor directory of the dataset (GT depth overlaid when present)
bash scripts/eval.sh --train-dir outputs/<run> --session-dir dataset/real/pattern_32/session_000/sensor_0000 --max-samples 20

# your own image pair
bash scripts/eval.sh --train-dir outputs/<run> --tactile my_tactile.png --rgb my_rgb.png
```

`--train-dir` is a training output folder (it holds `config.json`, `best_depth.pt`,
`best_pose.pt`). Inputs are 224×224 frames of the 17.5 mm tactile field of view; the model
works in the fixed 0.816 centre crop that `preprocess_image` applies.

## Train

```bash
bash scripts/train.sh                                   # paper setting (1 GPU, ~24 GB VRAM)
NGPUS=2 bash scripts/train.sh                           # DDP
MODEL=configs/model/tac_t3_rgb_mae.yaml bash scripts/train.sh     # encoder ablation
DATA=configs/data/ratio_sim34.yaml bash scripts/train.sh          # real:sim ratio ladder
TRAIN=configs/train/train_bs32_md_uniform.yaml bash scripts/train.sh   # modality-dropout ablation
```

A run writes `history.json`, TensorBoard logs, loss plots and `best_depth.pt` /
`best_pose.pt` / `latest.pt` to `outputs/<model>_<data>/`. Every config of the paper is
included:

| Folder | Files | Paper |
|---|---|---|
| `configs/model/` | `vistacfusion.yaml` (= `tac_sitr_rgb_mae.yaml`), `tac_{sitr,t3,dav2,dinov3,sparshv2,sparshmae,tvl,sparshx}_rgb_{mae,dinov3,clip,siglip}.yaml`, `*_single.yaml`, MViTac / VITaL baselines | Table II, encoder ablation |
| `configs/train/` | `default.yaml` (= `train_bs32_md_b70_inj075.yaml`) and the modality-dropout / injection sweep | Table II, dropout ablation |
| `configs/data/` | `default.yaml` (= `ratio_sim148.yaml`), `ratio_sim{34…760}.yaml`, `realonly.yaml`; `real_rotation_windows.json`, `real_translation_bounds.json` | real:sim ratio sweep |

Baselines that need their own pretraining (MViTac, VITaL) are trained with
`python -m vistacfusion.engine.pretrain_mvitac` / `pretrain_vital` first; Sparsh-X needs the
official repo on `encoder.repo_path`.

## Repository layout

```
configs/                model / train / data configs (see above)
vistacfusion/
  models/               encoders.py (frozen encoder zoo) · fusion.py · projection.py · model.py
    heads/              dpt.py (depth + normal) · pose.py (SE(2))
  data/                 dataset.py (sim + real loader, pose labels) · transforms.py (augmentation, fixed crop)
  losses/               depth · normal · pose · total (grouped uncertainty weighting)
  engine/               train.py (DDP, modality dropout) · eval.py · inference.py · pretrain_{mvitac,vital}.py
scripts/                train.sh · eval.sh · download_dataset.sh
tests/                  shape / routing / overfit tests with mock encoders
docs/DATASET.md         dataset layout, units and label conventions
```

## Citation

```bibtex
@article{gelslim5,
  title   = {GelSlim 5.0: Unifying Vision and Touch with a Single Fingertip Camera},
  journal = {arXiv preprint arXiv:2610.31334},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0. Third-party encoder weights
keep their own licences.
