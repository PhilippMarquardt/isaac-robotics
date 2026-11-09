#!/usr/bin/env bash
set -e
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=/workspace/repos/gr00t-dreams
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HOME=/workspace/.cache/hf
cd /workspace/repos/gr00t-dreams
/root/venvs/gr00t_venv/bin/python scripts/idm_training.py   --dataset-path /workspace/data/split_80_20/train/G1_WB_Dex5_Pickup_Pillow   --output-dir /workspace/runs/idm_g1_pillow   --data-config g1_full_body   --embodiment-tag g1_dex5   --video-backend torchvision_av   --batch-size 4   --max-steps 6000   --save-steps 1000   --num-gpus 1   --dataloader-num-workers 4   --learning-rate 1e-4   --report-to tensorboard   2>&1 | tee -a /workspace/runs/idm_g1_pillow.log
