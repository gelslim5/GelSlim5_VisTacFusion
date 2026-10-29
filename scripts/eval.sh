#!/usr/bin/env bash
# Evaluate a training run on the real validation split, in all three input modes
# (both / tactile-only / rgb-only).  --train-dir is required (TRAIN_DIR env var or flag).
#
#   bash scripts/eval.sh --train-dir outputs/<run>
#   bash scripts/eval.sh --session-dir dataset/real/pattern_32/session_000/sensor_0000
#   bash scripts/eval.sh --tactile tac.png --rgb rgb.png
set -euo pipefail

MODEL=${MODEL:-configs/model/vistacfusion.yaml}
TRAIN=${TRAIN:-configs/train/default.yaml}
DATA=${DATA:-configs/data/default.yaml}

if [[ "$*" != *--train-dir* ]]; then
  if [ -n "${TRAIN_DIR:-}" ]; then set -- --train-dir "$TRAIN_DIR" "$@"; else echo "usage: bash scripts/eval.sh --train-dir outputs/<run> [...]" >&2; exit 1; fi
fi
if [[ "$*" != *--tactile* && "$*" != *--session-dir* && "$*" != *--eval* ]]; then set -- "$@" --eval; fi

python -m vistacfusion.engine.inference --model "$MODEL" --train "$TRAIN" --data "$DATA" "$@"
