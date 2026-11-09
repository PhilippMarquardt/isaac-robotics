# GR00T N1.7 fine-tune on a new embodiment (G1 + 6-DoF hands)

Fine-tunes `nvidia/GR00T-N1.7-3B` on Unitree G1 whole-body teleoperation data. The pretrained
`UNITREE_G1` projector expects binary grippers plus `base_height_command` / `navigate_command`, which
these datasets do not have, so a `NEW_EMBODIMENT` modality is registered and the projector is trained
from scratch together with the DiT action head (`tune_projector=True`, `tune_diffusion_model=True`,
`tune_llm=False`, `tune_visual=False`: 1.62B of 3.14B parameters trainable).

## Layout

| Path | What it does |
|---|---|
| `configs/g1_inspire_modality_config.py` | Registers the `NEW_EMBODIMENT` modality: 7 state/action groups (legs, waist, arms, 6-D hands), 40-step action chunks, RELATIVE arms/legs, ABSOLUTE hands/waist |
| `configs/modality_cam0.json`, `modality_head_stereo.json` | Per-dataset `meta/modality.json`: slices `robot_q_current[36]` / `hand_state[12]` into the groups (root pose dropped); one per head-camera key |
| `data_prep/split_datasets.py` | Deterministic 80/20 episode split per dataset; writes filtered `meta/` and **symlinks** `data/` + `videos/` (no copy of ~600 GB source data) |
| `data_prep/reproduce_split.py` | Recomputes which episode indices landed in val (seed 42) from HF metadata |
| `data_prep/pull_val_episodes.py`, `pull_real_val.py` | Download individual held-out episodes (parquet + mp4) from the HF LeRobot v3 shards for local eval |
| `train/run_finetune_multi.py` | Wrapper around GR00T's experiment runner that takes a colon-joined list of datasets (`mix_ratio=1.0` each) |
| `train/launch_multitask.sh` | Multi-dataset launch (bs 32, lr 1e-4, save every 2k) |
| `train/launch_single_task.sh` | First single-task (pillow) run through the stock `examples/finetune.sh` |
| `eval/eval_pure_prediction.py` | Open-loop: ground-truth state + video in, 16-step chunk out, per-joint-group MSE vs teleop actions and vs the stand-still (identity) baseline |
| `eval/eval_closedloop_from_seed.py` | Closed-loop: seed MuJoCo from frame 0, feed the simulated joint state back to the policy, accumulate action MSE |
| `eval/eval_3ep_per_task.py`, `eval_all_tasks.py` | Per-task rollouts rendered side by side with the dataset video |
| `eval/gen_trajectories.py` | GPU-side dump of predicted vs ground-truth chunks to JSON for the MuJoCo viewer |
| `sim_viz/` | MuJoCo G1 + Inspire visualisation (`viz_mujoco.py` joint/hand maps, open-loop replay, GT replay, mp4 rendering) and the scene file |
| `results/inspire_6task/` | Six-task Inspire run: open-loop eval of the random-projector base and the step-18000 checkpoint, training curve |
| `results/dex5_4task/` | Earlier four-task Dex5-hand run: open-loop (sim and real val episodes) and closed-loop MuJoCo results |

## Data

Six Unitree LeRobot datasets, converted from LeRobot v3 to v2.1 (the only format GR00T's loader reads)
with Isaac-GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py`:

| Task | Size | Train episodes | Val episodes |
|---|---|---|---|
| Pick up pillow | 5.2 GB | 488 | 121 |
| Collect clothes | 2.0 GB | 269 | 67 |
| Put clothes into washing machine | 3.2 GB | 366 | 91 |
| Put drinks into fridge | 8.6 GB | 240 | 60 |
| Put clothes into basket | 12 GB | 240 | 60 |
| Put vegetables into basket | 20 GB | 240 | 60 |
| **Total** | **51 GB** | **1,843** (~847k frames) | **459** |

Practical issues that had to be solved: the videos are AV1, which the bundled OpenCV cannot decode
(use `torchcodec` for eval); `ffmpeg` is needed for the per-episode mp4 split during conversion; the
normalisation statistics (`relative_stats.json`) must be recomputed whenever the action horizon
(`delta_indices`, capped at 40) changes; each checkpoint is 24 GB with optimizer state, so
`save_total_limit` must be small.

## Run

```bash
# 0. convert each dataset v3 -> v2.1 with Isaac-GR00T's converter, then copy configs/modality_*.json
#    to <dataset>/meta/modality.json (cam_0 or head_stereo_left variant)

# 1. 80/20 split (symlinks)
python data_prep/split_datasets.py --sources /data/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly ... \
    --out-root /data/split_80_20 --val-ratio 0.2 --seed 42

# 2. train (inside an Isaac-GR00T checkout / venv); edit TRAIN_PATHS for the dataset list
GR00T_REPO=~/Isaac-GR00T SPLIT_ROOT=/data/split_80_20 OUTPUT_DIR=runs/gr00t_multitask \
    bash train/launch_multitask.sh
#    equivalent direct call:
python train/run_finetune_multi.py --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/split_80_20/train/A:/data/split_80_20/train/B \
    --embodiment-tag new_embodiment --modality-config-path configs/g1_inspire_modality_config.py \
    --output-dir runs/gr00t_multitask --experiment-name gr00t_multitask \
    --max-steps 60000 --save-steps 2000 --save-total-limit 2 --global-batch-size 32

# 3. eval: the eval scripts read GR00T_REPO and GR00T_EVAL_DIR
#    (GR00T_EVAL_DIR/ckpt_alltasks = checkpoint, GR00T_EVAL_DIR/_samples_other/val_episodes[_real] = pulled episodes)
export GR00T_REPO=~/Isaac-GR00T GR00T_EVAL_DIR=~/gr00t_eval
python data_prep/pull_real_val.py
python eval/eval_pure_prediction.py          # open loop
python eval/eval_closedloop_from_seed.py     # closed loop in MuJoCo
```

The MuJoCo scripts need the official Unitree G1 29-DoF + Inspire hand model
(`g1_29dof_inspire_DFQ.xml` and its `meshes/`) placed in `sim_viz/assets/` next to
`scene_pillow_inspire_official.xml`; it is not redistributed here.

## Results

Six-task Inspire run, 18k steps (`results/inspire_6task/`), open loop on held-out episodes:

- 50-66x lower error than the random-projector baseline (pretrained backbone + untrained new head, the
  only "zero-shot" model possible for this hand).
- Better than the stand-still baseline on all six tasks, at 40-80 % of its error. Pillow and basket are
  best; collecting clothes (bimanual, least data) is hardest.
- Hand error is about 5-10x the body error (the hand head is the only part trained from scratch).

Earlier Dex5-hand run (`results/dex5_4task/`): closed-loop errors in MuJoCo are 200-300x larger than
open loop and the model only barely beats standing still.

## Not included

The six-task Inspire run used variants of these scripts (`g1_wbt_config.py`, a six-dataset launcher and
an open-loop eval script) that existed only on the rented training machine and were not recovered. The
scripts here are the Dex5-era versions of the same pipeline (same wrapper, same modality layout); the
six-task run's outputs are in `results/inspire_6task/`.
