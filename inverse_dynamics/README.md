# Inverse dynamics model: from dream videos to actions

A video alone cannot train a robot policy; it needs action labels. The inverse dynamics model (IDM)
takes a video chunk, the start state and the instruction and predicts the 41 joint actions (legs 2×6,
waist 3, arms 2×7, hands 2×6) that produced the motion. Architecture from NVIDIA GR00T-Dreams: SigLIP2-large
vision encoder + DiT flow-matching action head, 614M parameters, 16-step action chunks.

## Layout

| Path | What it does |
|---|---|
| `patches/gr00t_dreams_g1_idm.patch` | Diff against GR00T-Dreams `ec3881d` (8 files): `g1_dex5` embodiment tag, `G1FullBodyDataConfig` (`g1_full_body`), `action_dim` / `max_action_dim` 32 → 64 in `IDM_dump/base.yaml`, dataset / transform / trainer adjustments |
| `train/run_idm.sh` | Phase 1: 6,000 steps, batch 4, lr 1e-4, bf16, one A100 |
| `train/run_idm_resume.sh` | Phase 2: resume to 30,000 steps |
| `train/prune_checkpoints.sh` | Keeps only the two newest checkpoints during training (disk) |
| `eval/eval_idm.py`, `eval_idm_final.py` | Per-joint-group RMSE on 200 train/val samples, in normalised and physical (rad) units |
| `pipeline/idm_gen_traj.py` | Dream mp4 → sliding 16-frame windows → IDM → un-normalised `(N, 41)` trajectory `.npz` |
| `pipeline/render_idm_panels.py` | Trajectory + dream video → 3-panel mp4: dream \| MuJoCo replay \| per-joint values (uses `../gr00t_finetune/sim_viz/viz_mujoco.py`) |
| `results/eval_step6000.json` | Eval of the 6k-step checkpoint |
| `results/trainer_state_30k.json` | Hugging Face trainer state with the full 30k-step loss curve |

## Run

```bash
git clone https://github.com/NVIDIA/GR00T-Dreams.git && cd GR00T-Dreams && git checkout ec3881d
patch -p1 < /path/to/inverse_dynamics/patches/gr00t_dreams_g1_idm.patch

# training data: a LeRobot v2.1 dataset split from ../gr00t_finetune/data_prep/split_datasets.py
bash train/run_idm.sh          # 6k steps   (edit --dataset-path / --output-dir)
bash train/run_idm_resume.sh   # to 30k steps

# dream video -> actions -> MuJoCo panels
python pipeline/idm_gen_traj.py --video dream.mp4 --out traj.npz \
    --ckpt runs/idm_g1_pillow --val_path /data/split_80_20/val/G1_WB_Dex5_Pickup_Pillow \
    --cfg_yaml GR00T-Dreams/IDM_dump/base.yaml
python pipeline/render_idm_panels.py --traj traj.npz --video dream.mp4 --out idm_3panel.mp4 --h 480 --fps 16
```

## Results

Trained on the pillow task. After 6,000 steps the loss sat at 0.95, basically noise level, and the
predictions were useless. Continuing to 30,000 steps (~3.5 h on an A100), the loss broke through around
step 15k and ended at 0.13, 7.3x lower.

| Joint group | Error at 6k steps | Error at 30k steps |
|---|---|---|
| Left leg | 14.5° | **5.6°** |
| Right leg | 15.6° | **6.4°** |
| Waist | 9.6° | **5.0°** |
| Left arm | 16.7° | **8.5°** |
| Right arm | 19.8° | **9.7°** |
| Left hand | 6.2° | **1.5°** |
| Right hand | 14.7° | **10.0°** |
| **Overall** | **13.9°** | **6.7°** |

Legs and waist under 6° are usable; arms at 8-10° are fine for large reaching motions but too loose for
fine manipulation. The full loop (dream video → recovered actions → back into GR00T training) is built,
but it was not closed at scale: no GR00T model was retrained on IDM-labelled dreams.
[`../docs/media/idm_3panel.mp4`](../docs/media/idm_3panel.mp4) shows the pipeline on one dream.

The GR00T-Dreams script that converts raw videos + labels to LeRobot (`IDM_dump/raw_to_lerobot.py`) was
used unmodified and is not copied here.
