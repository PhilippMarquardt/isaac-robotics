# Latent-action world model (JEPA-style)

GR00T needs action labels, and most video has none. Latent-action models learn them from video alone: an
inverse model looks at two frames and compresses "what happened" into a small code Z, and a forward
(transition) model must predict the later frame's features from the earlier frame plus Z. If that works,
Z behaves like an action and can be learned from any video. Everything is predicted in a feature space,
not in pixels, with a ViT encoder: first a frozen ImageNet ViT-S, later one trained from scratch with the
V-JEPA recipe.

The evaluation is retrieval: given the current frame and a Z, does the predicted future match the true
future among all validation clips? With the true Z from the inverse model the answer is almost always
yes; the real question is whether a good Z can be *generated* without seeing the future (a prior).

## Code map

| Group | Files |
|---|---|
| Models | `models/`: ViT encoders (`encoder.py`, `vit3d.py`), reducers (`fixed_reducer.py`, `token_reducer.py`), `inverse_encoder.py`, `transition.py` / `transition_residual.py`, priors (`prior.py` deterministic, `flow_prior.py`, `flow_prior_goal.py`, `flow_prior_masked_goal.py`, `context_flow.py`), V-JEPA predictors, `la_modules.py` (multi-step / goal-conditioned modules), `ema.py`; `masks/multiblock3d.py` |
| Data | `scripts/download_prelinger.py` (public-domain films from archive.org), `scripts/build_cache_all.py`, `data/` frame-pair and clip caches, `precompute_bundle.py` / `precompute_multi_bundle.py` / `precompute_context_bundle.py` (encode films once with the frozen ViT into `H_t`, `H_t+k` bundles) |
| Stage 1: inverse + transition | `train_stage1.py`, `train_stage1_from_bundle.py`, `train_stage1_filmadv.py` (adversarial film classifier on Z via gradient reversal), `train_loo.py` (leave-one-film-out) |
| Stage 2: priors over Z | `train_stage2a.py` (deterministic prior), `train_stage2b.py` (flow-matching prior), `train_stage2b_goal*.py` (goal-conditioned, hard negatives, cross-film negatives), `train_stage2c_*.py` (rerankers), `train_context_flow.py`, `train_masked_goal_flow.py`, `train_c2_partial_goal_flow.py`, `eval_baseline_priors.py` (sample Z from the training set: random / nearest neighbour, best-of-K) |
| Diagnostics | `eval_diagnostics.py`, `eval_z_swap.py`, `eval_z_transfer*.py`, `eval_crossvideo.py`, `eval_per_film.py`, `eval_film_leakage.py`, `eval_filmadv_sweep.py`, `eval_goal_noise.py`, `eval_goal_corruption.py`, `eval_multistep_rollout.py`, `eval_z_step_diversity.py`, `eval_e1_zswap.py`, `eval_e2_v1_zswap.py`, `eval_e3_probes.py`, `aggregate_results.py` |
| End-to-end / V-JEPA variants | `train_vjepa.py`, `train_vjepa2.py` + `eval_vjepa2.py` (small-scale V-JEPA-2 reproduction), `train_e2e_jepa_la.py`, `train_e2e_flow_only.py`, `train_flow_jepa.py`, `train_flow_action_jepa*.py`, `train_la_jepa*.py`, `train_la_v1_*.py` (FSQ bottleneck, augmentation invariance), `eval_phase4_continuous_z.py`, `eval_phase5_adapter.py`, `phase_b_adapter.py` (synthetic action adapter tests) |
| Consolidated | `latent_action_jepa.py`: single-file version of the stage-1 / goal-conditioned flow pipeline |
| From scratch on robot video | `v6_from_scratch/`: `METHOD.md` (objectives and gradient topology), `train_v6.py` (ViT-3D trained from scratch with V-JEPA masking + latent-action chain + flow prior on BridgeData V2), `probe_v6.py` (linear probe Z → real robot action per dimension), `bridge_clips.py`; `charades_clips.py`, `train_v5_smoke.py`, `eval_v5_probes.py` (earlier version on Charades) |

## Run

```bash
cd latent_action_jepa
# films -> frozen ViT-S bundle -> stage 1 -> stage 2 prior -> sampling baselines
python scripts/download_prelinger.py --out_dir /data/films
python precompute_multi_bundle.py --films /data/films/a.mp4 /data/films/b.mp4 --names a b --out runs/bundle.pt
python train_stage1_from_bundle.py --bundle runs/bundle.pt --save_ckpt runs/stage1/ckpt.pt \
    --save_z runs/stage1/oracle_bundle.pt --action_tokens 4 --action_dim 128
python train_stage2b.py --ckpt runs/stage1/ckpt.pt --bundle runs/stage1/oracle_bundle.pt --out_dir runs/stage2b
python eval_baseline_priors.py --ckpt runs/stage1/ckpt.pt --bundle runs/stage1/oracle_bundle.pt --out_dir runs/baselines

# from-scratch model on BridgeData V2 (scripted_raw) and the action probe
python v6_from_scratch/train_v6.py --data_root /data/bridge/scripted_raw --out_dir runs/v6 --steps 6000
python v6_from_scratch/probe_v6.py --ckpt runs/v6/<ckpt>.pt --data_root /data/bridge/scripted_raw
```

Script defaults point at `/workspace/...` and `/root/...` run directories; pass explicit paths.

## Results

| Where Z comes from (100 validation clips) | Top-1 retrieval |
|---|---|
| Inverse model, sees the future (upper bound) | 0.97 |
| "Nothing changes" baseline | 0.15 |
| Learned deterministic prior | 0.15 |
| Learned flow-matching prior, best of 64 samples | 0.54 |
| Random training-set Z, best of 128 samples | **0.88** |
| Nearest-neighbour training-set Z, best of 512 samples | **0.91** |

- The learned priors lost to simply sampling Z from the training set. A goal-conditioned prior matched
  the upper bound (0.95), but it is given the true future as its goal, so that is close to re-learning
  the inverse model.
- On 13 real hand-object films, a tight Z (4×128) still ranks the right clip first 48 % of the time
  among 2,600 (about 1,200x chance). Z also leaked which film a clip came from; an adversarial loss
  reduced that leak.
- On BridgeData V2, where the real robot actions are known, a linear probe on Z predicts them with R²
  0.26 for the gripper position, 0.29 for open/close and 0.40 for vertical motion (shuffled control:
  about 0). The only knob that helped was Z capacity: a 4x smaller dataset gave about the same numbers,
  and bigger models (51M, 143M) did slightly worse than 22M.
- Training collapsed in every configuration after about 12k steps, so early stopping was part of the
  recipe.
