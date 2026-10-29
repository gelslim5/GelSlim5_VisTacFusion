#!/usr/bin/env bash
# Download the GelSlim 5.0 visuo-tactile dataset (+ released checkpoints) from Hugging Face
# into ./dataset and ./checkpoints.  Requires `pip install huggingface_hub`.
#
#   bash scripts/download_dataset.sh            # everything (~21 GB)
#   bash scripts/download_dataset.sh real       # real data + meshes only (~1.3 GB)
set -euo pipefail

REPO=${HF_DATASET_REPO:-gelslim5/gelslim5-vistacfusion}
PART=${1:-all}

mkdir -p dataset checkpoints
case "$PART" in
  real) PATTERNS=("meshes/*" "real/*") ;;
  sim)  PATTERNS=("meshes/*" "sim/*") ;;
  *)    PATTERNS=("meshes/*" "real/*" "sim/*" "checkpoints/*") ;;
esac

for p in "${PATTERNS[@]}"; do
  hf download "$REPO" --repo-type dataset --include "$p" --local-dir dataset_release
done

# unpack per-object tarballs into the layout the configs expect
for t in dataset_release/real/*.tar dataset_release/sim/*.tar dataset_release/meshes/*.tar; do
  [ -f "$t" ] || continue
  tar -xf "$t" -C dataset
done
if [ -d dataset_release/checkpoints ]; then
  cp -r dataset_release/checkpoints/. checkpoints/
fi
echo "dataset -> ./dataset   checkpoints -> ./checkpoints"
