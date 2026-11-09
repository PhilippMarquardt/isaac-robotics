#!/bin/bash
# Extend cosmos-predict2 2B LoRA: add washing + fridge (drinks) tasks to the
# existing 2task (pillow+clothes) training set, then resume from iter_000004000.pt
# for another 4000 iters.
set -e
set -x

cd /workspace

DATA_DIR=/workspace/data/cosmos_train/g1_wbt_2task
CKPT=/workspace/runs/posttraining/video2world_lora/2b_g1_wbt_2task/checkpoints/model/iter_000004000.pt

echo "==== prep data $(date -u) ===="
python prep_local_g1.py --out $DATA_DIR --tasks pillow clothes washing fridge 2>&1

echo "==== t5 encode $(date -u) ===="
cd /workspace/repos/cosmos-predict2
source .venv/bin/activate
export PYTHONPATH=/workspace/repos/cosmos-predict2:$PYTHONPATH
python -m scripts.get_t5_embeddings_from_groot_dataset \
  --dataset_path $DATA_DIR \
  --meta_csv $DATA_DIR/metadata.csv 2>&1

echo "==== resume LoRA $(date -u) ===="
torchrun --nproc_per_node=1 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  -- experiment=predict2_video2world_lora_training_2b_g1_wbt_4task 2>&1

echo "==== DONE $(date -u) ===="
