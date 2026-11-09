#!/bin/bash
set -x
cd /workspace/repos/cosmos-predict2
source /workspace/repos/cosmos-predict2/.venv/bin/activate
export PYTHONPATH=/workspace/repos/cosmos-predict2:$PYTHONPATH
echo "==== LORA inference starting $(date -u) ===="
python -m examples.video2world_lora \
  --model_size 2B \
  --resolution 480 \
  --fps 16 \
  --use_lora \
  --dit_path /workspace/runs/posttraining/video2world_lora/2b_g1_wbt_2task/checkpoints/model/iter_000004000.pt \
  --lora_rank 32 --lora_alpha 32 \
  --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
  --batch_input_json /workspace/gen_out/batch_lora.json \
  --disable_guardrail --disable_prompt_refiner 2>&1 | tee /workspace/lora_infer.log
echo "==== LORA DONE $(date -u) ===="
