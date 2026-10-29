#!/usr/bin/env bash
# Evaluate the released checkpoints (or your own run) on the real validation split, in all
# three input modes (both / tactile-only / rgb-only).
#
#   bash scripts/eval.sh                                   # checkpoints/ from the paper
#   bash scripts/eval.sh --train-dir outputs/<run>          # your own training run
#   bash scripts/eval.sh --session-dir dataset/real/pattern_32/session_000/sensor_0000
#   bash scripts/eval.sh --tactile tac.png --rgb rgb.png
set -euo pipefail

MODEL=${MODEL:-configs/model/vistacfusion.yaml}
TRAIN=${TRAIN:-configs/train/default.yaml}
DATA=${DATA:-configs/data/default.yaml}

if [[ "$*" != *--train-dir* ]]; then set -- --train-dir checkpoints/vistacfusion_sitr_mae "$@"; fi
if [[ "$*" != *--tactile* && "$*" != *--session-dir* && "$*" != *--eval* ]]; then set -- "$@" --eval; fi

python -m vistacfusion.engine.inference --model "$MODEL" --train "$TRAIN" --data "$DATA" "$@"
