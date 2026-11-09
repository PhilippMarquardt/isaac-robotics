#!/bin/bash
set -x
export VIRTUAL_ENV=/root/venvs/gr00t_venv
export PATH=$VIRTUAL_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH
export HF_HOME=/workspace/.cache/hf
# HF_TOKEN must be exported beforehand (the GR00T / Cosmos-Reason weights are gated).
export HF_HUB_ENABLE_HF_TRANSFER=1
export USE_WANDB=0 MAX_STEPS=10000 SAVE_STEPS=1000 GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=4
cd /workspace/dreamdojo/Isaac-GR00T
echo "PY=$(which python)  (should be $VIRTUAL_ENV/bin/python)"
python -c 'import sys; print("prefix=",sys.prefix,"exe=",sys.executable)'
bash examples/finetune.sh   --base-model-path /workspace/checkpoints/gr00t_n17   --dataset-path /workspace/data/pillow/G1_WB_Dex5_Pickup_Pillow   --embodiment-tag NEW_EMBODIMENT   --modality-config-path /workspace/data/pillow/g1_pillow_config.py   --output-dir /workspace/runs/gr00t_pillow   --experiment-name g1_pillow_v1
