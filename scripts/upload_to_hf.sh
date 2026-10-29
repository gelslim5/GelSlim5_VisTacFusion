#!/usr/bin/env bash
# Maintainer script: upload dataset_release/ (per-object tarballs + checkpoints) to the
# Hugging Face dataset repo.  Needs `hf auth login` with an account that owns $REPO.
#
#   REPO=gelslim5/gelslim5-vistacfusion bash scripts/upload_to_hf.sh
set -euo pipefail
REPO=${REPO:-gelslim5/gelslim5-vistacfusion}
SRC=${SRC:-dataset_release}

hf repo create "$REPO" --repo-type dataset 2>/dev/null || true
cp docs/DATASET.md "$SRC/README.md"
# large-folder upload: resumable, parallel, one commit per chunk
hf upload-large-folder "$REPO" "$SRC" --repo-type dataset --num-workers 4
echo "uploaded -> https://huggingface.co/datasets/$REPO"
