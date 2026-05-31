#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-qingyi/wm811k-wafer-map}"
OUTPUT_DIR="${2:-archive}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI is not installed. Install it with: pip install kaggle" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading Kaggle dataset '$DATASET' to '$OUTPUT_DIR'..."
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR" --unzip

echo
echo "Download complete. Expected raw file: $OUTPUT_DIR/LSWMD.pkl"
echo "Next step: python data_loader.py"
