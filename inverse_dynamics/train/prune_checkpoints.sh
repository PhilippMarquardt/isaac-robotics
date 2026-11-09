#!/usr/bin/env bash
# Keep only the 2 newest checkpoint-* dirs inside idm_g1_pillow.
# Runs until manually stopped.
RUN_DIR=/workspace/runs/idm_g1_pillow
while true; do
  mapfile -t ckpts < <(ls -1d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -V)
  n=${#ckpts[@]}
  if (( n > 2 )); then
    for (( i=0; i < n-2; i++ )); do
      echo "[prune $(date -u +%H:%M:%S)] removing ${ckpts[$i]}"
      rm -rf "${ckpts[$i]}"
    done
  fi
  sleep 60
done
