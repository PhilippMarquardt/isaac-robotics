# LiDAR terrain reconstruction for the G1

A real G1 cannot read a height map from the simulator; it has a LiDAR on its head. This part simulates a
Livox MID-360 (40 channels × 500 azimuths, 30 m range, mounted upside down on the head as on the real
robot) in IsaacLab and trains networks to reconstruct the 3D terrain around the robot from one scan. The
target is the true terrain mesh, voxelised into a 3.2 m cube of 5 cm voxels around the robot,
reproducing the setup of Hoeller et al. (2022), *Neural Scene Representation for Locomotion on
Structured Terrain*.

## Code map

| Group | Files |
|---|---|
| Simulation + collection | `lidar_env_cfg.py` (G1 rough env + MID-360 ray caster + dense ground-truth ray caster), `lidar_terrain_cfg.py` (10-type terrain mix), `collect_g1_lidar_dataset.py` (16 parallel G1s driven by a trained rough-terrain policy with random commands; chunked `.pt` output), `validate_dataset.py`, `combine_datasets.py` |
| Data | `dataset.py` (frame dataset, split by terrain tile), `augment.py` (range jitter, dropout and other sim-to-real augmentations), `dataset_temporal.py` (sliding window of past scans) |
| Single-frame models | `train_v6.py` (one script, three encoders with the same sparse decoder, data and loss: `hoeller` sparse 4D U-Net, `pointnet` PointNet-style encoder + sparse convs, `ptv3` simplified point-transformer), `hoeller_unet.py` + `train_hoeller.py` (first rollout-based Hoeller trainer) |
| Evaluation | `eval_v6.py` (IoU at several thresholds, Chamfer, FPS, VRAM), `eval_metrics.py`, `hist_distances.py` (LiDAR vs ground-truth distance histograms), `run_all_v6.sh` (train all three encoders + eval) |
| Temporal models (unfinished) | `model_temporal.py` + `train_temporal.py` (per-cell GRU), `model_temporal_unet.py` + `train_temporal_unet.py` (multi-frame early-fusion 3D U-Net), `eval_temporal_train_vs_val.py`, `bench_trivial_temporal.py` (copy-the-input baselines) |
| Visualisation | `viz_paper_pair.py`, `viz_paper_iso.py`, `viz_data.py` (paper-style voxel renders), `viz_open3d.py`, `viz_robot.py`, `viz_rerun.py`, `viz_reconstruction.py`, `viz_temporal*.py`, `export_*.py` (coloured `.ply` for CloudCompare) |

`results/ablation_v6_raw.md` is the raw output of `eval_v6.py` (the last row is a longer PointNet run).

## Run

```bash
# collect (IsaacLab); needs a trained G1 rough-terrain locomotion checkpoint to drive the robots
python collect_g1_lidar_dataset.py --num_envs 16 --num_frames 8000 --terrain diverse \
    --checkpoint <g1_rough_policy.pt> --output_dir /root/lidar_dataset_v6 --headless
python validate_dataset.py /root/lidar_dataset_v6

# train the three encoders (1,500 steps each, batch 4, lr 1e-3) and evaluate -> RESULTS_v6.md
bash run_all_v6.sh 1500
# or one encoder:
python train_v6.py --encoder pointnet --dataset_dir /root/lidar_dataset_v6 --out_dir runs/v6_pointnet --steps 1500
python eval_v6.py --runs runs/v6_hoeller runs/v6_pointnet runs/v6_ptv3 --dataset_dir /root/lidar_dataset_v6 --n_eval 64
```

Requires `spconv-cu120` for the sparse convolutions; the visualisers need matplotlib, Open3D or Rerun.

## Results

8,000 frames from 16 environments and 10 terrain types; the split is by terrain tile, so validation tiles
are never seen in training.

| Encoder (~7.2M parameters) | IoU @ 0.5 | IoU @ 0.7 | Chamfer (m) | FPS |
|---|---|---|---|---|
| Sparse 4D U-Net (Hoeller 2022 reproduction) | 0.202 | 0.158 | 0.233 | 52 |
| PointNet-style encoder + sparse convolutions | 0.203 | **0.167** | 0.233 | 60 |
| Point-transformer encoder (simplified, attention-based) | 0.189 | 0.112 | 0.243 | 72 |

All three land at the same IoU, and training twice as long did not help, which pointed to the sensor
rather than the model. A head-mounted LiDAR barely sees the ground right in front of the feet: in the
first setup the reconstruction cube got no LiDAR points at all because the median return was 15 m away.
Mounting the sensor upside down, as on the real robot, fixed most of that, but even in the final dataset
only about a quarter of the returns fall within the cube's 1.6 m half-width.

So the result is a sensor-geometry finding, not an architecture ranking: single-frame reconstruction is
capped by what the sensor can see. The fix is temporal (accumulate scans while walking); the temporal
models here were started but have no finished results, and the reconstruction was never plugged into the
locomotion policy.

Figures: `../docs/media/lidar_scan_rings.jpg`, `lidar_blind_input.jpg`, `lidar_distance_hist.png`.
