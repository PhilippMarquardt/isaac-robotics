#!/usr/bin/env bash
# Multi-task GR00T N1.7 fine-tune (NEW_EMBODIMENT, G1 + 6-DoF hands) on several LeRobot v2.1
# datasets produced by ../data_prep/split_datasets.py (80/20 split, symlinked data/videos).
#
# Usage (inside an Isaac-GR00T checkout with its venv active):
#   GR00T_REPO=/path/to/Isaac-GR00T SPLIT_ROOT=/data/split_80_20 bash launch_multitask.sh
#
# This is the 4-task (Dex5-era) launcher. The six-task Inspire run described in the README used
# the same wrapper with the six Inspire datasets in TRAIN_PATHS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_REPO="${GR00T_REPO:-$HOME/Isaac-GR00T}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/split_80_20}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gr00t_alltasks_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_alltasks_v1}"
MODALITY="${MODALITY:-$HERE/../configs/g1_inspire_modality_config.py}"
BASE_MODEL=nvidia/GR00T-N1.7-3B

cd "$GR00T_REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$GR00T_REPO/.cache/hf_lerobot}"
mkdir -p "$HF_LEROBOT_HOME" "$OUTPUT_DIR"

# 80 % train split of each dataset
TRAIN_PATHS=(
  "$SPLIT_ROOT/train/G1_WB_Dex5_Pickup_Pillow"
  "$SPLIT_ROOT/train/G1_WB_Dex5_Collect_Clothes"
  "$SPLIT_ROOT/train/G1_WB_Dex5_Put_Clothes_into_Washing_Machine"
  "$SPLIT_ROOT/train/G1_WBT_Inspire_Put_Drinks_Into_Fridge"
)
# The stock launcher takes a single dataset; run_finetune_multi.py takes a colon-joined list,
# each entry with mix_ratio=1.0 so GR00T balances datasets by length.
DATASET_PATH="$(IFS=:; echo "${TRAIN_PATHS[*]}")"

exec python "$HERE/run_finetune_multi.py" \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag new_embodiment \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --modality-config-path "$MODALITY" \
  --max-steps 20000 \
  --save-steps 2000 \
  --save-total-limit 1 \
  --global-batch-size 32 \
  --dataloader-num-workers 4
