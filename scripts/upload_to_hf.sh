#!/usr/bin/env bash
# Maintainer script: upload dataset_release/ (per-object tarballs + checkpoints) to the
# Hugging Face dataset repo.  Needs `hf auth login` with an account that owns $REPO.
#
#   REPO=gelslim5/gelslim5-vistacfusion bash scripts/upload_to_hf.sh
set -euo pipefail
REPO=${REPO:-gelslim5/gelslim5-vistacfusion}
SRC=${SRC:-dataset_release}

hf repo create "$REPO" --repo-type dataset 2>/dev/null || true
# dataset card = YAML header + docs/DATASET.md
{ printf -- '---\nlicense: cc-by-4.0\npretty_name: GelSlim 5.0 visuo-tactile dataset\ntask_categories:\n  - depth-estimation\n  - image-to-image\ntags:\n  - tactile\n  - robotics\n  - visuo-tactile\n  - sim-to-real\nsize_categories:\n  - 10K<n<100K\n---\n\n'; cat docs/DATASET.md; } > "$SRC/README.md"
# large-folder upload: resumable, parallel, one commit per chunk
hf upload-large-folder "$REPO" "$SRC" --repo-type dataset --num-workers 4
echo "uploaded -> https://huggingface.co/datasets/$REPO"
