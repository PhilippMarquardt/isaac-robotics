#!/usr/bin/env bash
# Run all 3 encoder ablations sequentially after v6 data is ready.
# Usage: bash run_all_v6.sh [steps]
set -e
STEPS=${1:-1500}
DS=/root/lidar_dataset_v6
RUNS_DIR=/workspace/isaac-robotics/evals/lidar_recon

cd /workspace/diff_loco/lidar_perception

mkdir -p "$RUNS_DIR"

for ENC in hoeller pointnet ptv3; do
    OUT=$RUNS_DIR/v6_${ENC}
    echo "[RUN-ALL] training $ENC → $OUT"
    PYTHONUNBUFFERED=1 python train_v6.py \
        --encoder "$ENC" \
        --dataset_dir "$DS" \
        --out_dir "$OUT" \
        --steps "$STEPS" \
        --batch_size 4 \
        --lr 1e-3 \
        --log_every 25 --save_every 500 \
        2>&1 | tee "$OUT.log"
done

echo "[RUN-ALL] eval ablation"
python eval_v6.py \
    --runs $RUNS_DIR/v6_hoeller $RUNS_DIR/v6_pointnet $RUNS_DIR/v6_ptv3 \
    --dataset_dir "$DS" \
    --n_eval 64 \
    --out $RUNS_DIR/RESULTS_v6.md

echo "[RUN-ALL] DONE → $RUNS_DIR/RESULTS_v6.md"
