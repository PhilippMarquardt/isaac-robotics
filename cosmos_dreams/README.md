# Cosmos dreams: video world-model LoRA fine-tuning

LoRA fine-tuning of NVIDIA Cosmos video models on G1 teleoperation videos, so the model generates
("dreams") plausible robot videos from one first frame and a sentence. The dreams are meant as a data
source: an inverse dynamics model ([`../inverse_dynamics`](../inverse_dynamics)) turns them into
action labels.

Two generations of the experiment:

| | Cosmos-Predict2.5 2B Video2World (main) | Cosmos-Predict2 2B Video2World (earlier) |
|---|---|---|
| Data | Inspire-hand videos, 27 clips: pillow, clothes, washing machine; fridge held out | Dex5-hand videos: pillow + clothes, later extended with washing + fridge |
| LoRA | rank 32 / alpha 32 on `q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2` | same targets and rank |
| Clips | 93 frames, 432×768, 16 fps | 93 frames, 432×768, 16 fps |
| Training | 8,000 iterations, 1 H100, ~3.6 s/iteration (~296 epochs) | 4,000 iterations, then resumed with 4 tasks to 8,000 |
| Code | `predict2_5/` | `predict2/` |

## Layout

| Path | What it does |
|---|---|
| `data_prep/prep_cosmos_data.py` | Downloads episodes from the HF datasets and writes the flat Cosmos layout (`metadata.csv`, `videos/<task>_ep<idx>.mp4`, 93 frames at 16 fps, 432×768; re-encodes the AV1 sources with ffmpeg) |
| `data_prep/prep_local_g1.py` | Same output from locally converted LeRobot v2.1 datasets |
| `data_prep/split_cosmos25_train_val.py` | Splits the clips into `train/` (pillow, clothes, washing) and `val/` (fridge) with `metas/*.txt` captions, hard-linking the videos |
| `predict2_5/g1_wbt_lora_experiment.py` | Hydra experiment `predict2p5_lora_training_2b_g1_wbt`: datasets, LoRA settings, 8k iterations, validation + sample callbacks every 500 iterations |
| `predict2_5/apply_lora_key_remap_patch.py` | Patches `text2world_model_rectified_flow.py` so checkpoint keys are remapped onto the PEFT + activation-checkpoint-wrapped DiT (without it every LoRA target is silently unmatched and the DiT loads random weights) |
| `predict2_5/dataset_video_exact_length.patch` | Fixes `VideoDataset` crashing (`high <= 0`) when a clip has exactly `num_frames` frames |
| `predict2/lora_2task_config.yaml` | Resolved config of the first Cosmos-Predict2 LoRA run (pillow + clothes) |
| `predict2/g1_wbt_4task_lora_experiment.py`, `run_extend_4task.sh` | Resume that run from iteration 4000 with washing + fridge added, to 8000 |
| `predict2/inference/` | Base vs LoRA generation (`run_infer.sh`), held-out / OOD-prompt / scene-swap / under-the-table batches and their JSON manifests |
| `results/predict2_lora_training_loss.txt` | Loss log of the Cosmos-Predict2 run |

## Run (Cosmos-Predict2.5)

```bash
# clips -> train/val layout inside the cosmos-predict2.5 checkout
python data_prep/prep_cosmos_data.py --out /data/g1_wbt_4task --tasks pillow clothes washing fridge
python data_prep/split_cosmos25_train_val.py --src /data/g1_wbt_4task --dst ~/cosmos-predict2.5/datasets/g1_wbt

# patches (edit the path P at the top of the script first)
python predict2_5/apply_lora_key_remap_patch.py
patch -p0 ~/cosmos-predict2.5/cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py \
    < predict2_5/dataset_video_exact_length.patch

# register the experiment (import g1_wbt_lora_experiment.py from the experiment configs), then
cd ~/cosmos-predict2.5
torchrun --nproc_per_node=1 -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=predict2p5_lora_training_2b_g1_wbt
```

## Run (Cosmos-Predict2, earlier)

```bash
# 2-task run: predict2/lora_2task_config.yaml is the resolved config of that run
# (job posttraining/video2world_lora/2b_g1_wbt_2task, 4,000 iterations); register an experiment
# with these settings and launch it with scripts.train exactly as in run_extend_4task.sh.
# Extend to 4 tasks (data prep + T5 embeddings + resume from iter 4000); register
# predict2/g1_wbt_4task_lora_experiment.py in the cosmos-predict2 experiment configs first:
bash predict2/run_extend_4task.sh
# generate dreams (edit the JSON manifests: first frame, prompt, output path)
bash predict2/inference/run_infer.sh
```

## Results

- Training loss stayed flat around 0.05-0.13; validation loss at iteration 7000 (0.078) matched the
  training loss, so no overfitting on 27 clips despite ~296 epochs.
- Base model: generic motion, hand poses that do not belong to this robot, hands slipping off objects.
  Fine-tuned: correct grasp approach, plausible squatting and reaching, six-DoF fingers that move like
  the real ones. Held-out fridge task: weaker, but realistic body pose and non-random motion; usable for
  filtered data augmentation, not as gold-standard labels.
- Videos: [`../docs/media/`](../docs/media/) (`cosmos_*.mp4` for Cosmos-Predict2.5, `dream_*.mp4` for
  the earlier Cosmos-Predict2 run).

No quantitative video metric (FVD etc.) was computed. The exact inference command used for the
Cosmos-Predict2.5 base-vs-LoRA comparison videos was not kept.
