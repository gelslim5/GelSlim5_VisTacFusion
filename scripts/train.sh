#!/usr/bin/env bash
# Train VisTacFusion (fusion trunk + DPT head + pose head; encoders frozen).
#
#   bash scripts/train.sh                                  # paper setting, 1 GPU
#   NGPUS=2 bash scripts/train.sh                          # DDP on 2 GPUs
#   MODEL=configs/model/tac_t3_rgb_mae.yaml bash scripts/train.sh   # encoder ablation
#   DATA=configs/data/ratio_sim34.yaml bash scripts/train.sh        # real:sim ratio ladder
#   bash scripts/train.sh --resume outputs/<run>/latest.pt
set -euo pipefail

MODEL=${MODEL:-configs/model/vistacfusion.yaml}
TRAIN=${TRAIN:-configs/train/default.yaml}
DATA=${DATA:-configs/data/default.yaml}
OUTDIR=${OUTDIR:-outputs/$(basename "$MODEL" .yaml)_$(basename "$DATA" .yaml)}
NGPUS=${NGPUS:-1}

ARGS=(-m vistacfusion.engine.train --model "$MODEL" --train "$TRAIN" --data "$DATA" --output-dir "$OUTDIR" "$@")

if [ "$NGPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$NGPUS" "${ARGS[@]}"
else
  python "${ARGS[@]}"
fi
