#!/usr/bin/env bash
set -euo pipefail

# usage: train_lora.sh <dataset_dir> <pretrained_ckpt> <out_dir> -- [extra args passed to sdxl_train.py]

DATASET_DIR=${1?Need dataset_dir}
PRETRAINED=${2?Need pretrained model path}
OUT_DIR=${3?Need output directory}
shift 3

PYTHON=${PYTHON:-python}

mkdir -p "$OUT_DIR"

echo "Starting LoRA training"
echo "Dataset: $DATASET_DIR"
echo "Pretrained: $PRETRAINED"
echo "Out: $OUT_DIR"

LOG="$OUT_DIR/train.log"

exec "$PYTHON" "thirdparty/sd-scripts/sdxl_train.py" \
  --pretrained_model_name_or_path "$PRETRAINED" \
  --train_data_dir "$DATASET_DIR/images" \
  --in_json "$DATASET_DIR/captions.jsonl" \
  --sdxl --network_module networks.lora --use_safetensors --save_model_as safetensors \
  --output_dir "$OUT_DIR" "$@" 2>&1 | tee "$LOG"
