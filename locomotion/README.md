# G1 locomotion and parkour in IsaacLab

`diff_loco` is an IsaacLab extension package that registers G1 locomotion tasks (gym IDs
`Isaac-Velocity-*-G1-*`) and the custom actors, observations, rewards and curricula they use. Training
is PPO (rsl-rl) with 16,384-20,480 robots in parallel on one GPU at ~135k simulation frames per second.
The terrain is a curriculum of ten difficulty levels for each of eight tile types (flat, stairs up and
down, slopes up and down, boxes, stepping stones, gaps); a robot is promoted when it walks far enough and
demoted when it fails.

The package grew one experiment at a time: each `vXX_*.py` module subclasses an earlier config and
changes one thing, and its docstring records the hypothesis. Modules are kept because later versions
import earlier ones.

## Setup

```bash
bash setup_isaaclab.sh          # Isaac Sim 5.1 (pip) + IsaacLab pinned to 4df6560e1, editable install
export ISAACLAB_PATH=~/IsaacLab # train.py / play.py reuse IsaacLab's rsl_rl cli_args.py from there
cd locomotion                   # so that `diff_loco` is importable
```

## Run

```bash
# teacher (privileged observations); 16k-20k envs on an 80 GB GPU
python -m diff_loco.train --task Isaac-Velocity-Parkour-G1-V28c-Teacher-v0 --num_envs 16384 --max_iterations 15000 --headless
# student distillation: for the Distillation runner train.py loads --load_run/--checkpoint from
# logs/rsl_rl/<student experiment_name>/, so copy or symlink the teacher run there first
python -m diff_loco.train --task Isaac-Velocity-Parkour-G1-V28-Student-v0 --headless \
    --load_run <teacher_run_dir> --checkpoint model_12000.pt
# later baselines
python -m diff_loco.train --task Isaac-Velocity-Parkour-G1-VBaseV2-v0 --headless
python -m diff_loco.train --task Isaac-Velocity-Parkour-G1-V51-v0 --headless
# diffusion actor (pre-V28 series)
python -m diff_loco.train --task Isaac-Velocity-Rough-G1-Diffusion-v0 --num_envs 4096 --headless

# play / record (Play task variants fix the terrain level, e.g. ...-V28d-Teacher-Play-L9-v0)
python -m diff_loco.play --task Isaac-Velocity-Parkour-G1-V28c-Teacher-Play-v0 --num_envs 1 \
    --checkpoint logs/rsl_rl/<experiment>/<run>/model_last.pt --video --video_length 600
# drive a trained policy with the keyboard (WASD + yaw)
python -m diff_loco.play_keyboard --task Isaac-Velocity-Parkour-G1-V54-Play-v0 --checkpoint <model.pt>
# look at a terrain without a policy
python -m diff_loco.zero_agent --task Isaac-Velocity-Parkour-G1-V53-v0
```

## Code map

| Module(s) | Experiment |
|---|---|
| `train.py`, `play.py`, `play_keyboard.py`, `zero_agent.py` | IsaacLab rsl-rl train/play scripts with the `diff_loco` tasks registered |
| `tasks.py` | Task registry (`register_tasks()`), the original rough/parkour terrain mixes and all pre-V28 configs |
| `obs.py`, `priv_obs.py`, `rewards.py`, `visual_encoder.py`, `v16_depth.py` | Custom observation terms (noisy height scan, LiDAR distance, privileged contact/friction terms, depth features) and reward terms |
| `diffusion_model.py`, `diffusion_transformer.py` | Diffusion actors (DDPM denoiser MLP / transformer) usable as drop-in rsl-rl PPO actors |
| `v15_model.py` ... `v26_megaloco.py` | Pre-V28 series: flow-matching DiT actor (`v16_flow.py`), single-pass transformers, ViT height-map encoders, cross-attention variants |
| `v28_parkour.py`, `v28_curriculums.py` | V28 teacher-student parkour (privileged PPO teacher, proprioceptive GRU student via distillation), per-terrain level logging |
| `v28b_parkour.py` | Bigger networks, asymmetric critic with foot forces |
| `v28c_parkour.py` | Reward rebalance (`feet_air_time` lowered) that fixed the asymmetric gait |
| `v28d_parkour.py` | Sim-to-real randomisation: action latency, motor-gain and armature randomisation, torque-rate penalty |
| `moe_actor.py`, `v28e_bundle.py`, `v28f_parkour.py`, `v28g_parkour.py`, `v28h_parkour.py` | Mixture-of-experts actor, hysteresis curriculum (clip instead of wrap at max level), longer horizon, value-clip ablation |
| `v28n_clean.py` | Same environment with a plain MLP actor |
| `v28o_parkour.py` | Curriculum bug fixes (standing envs never promoted, demotion floor, spawn-locked inverted stairs) + warm-up |
| `v28p` ... `v28u`, `v29_parkour.py` | Gap-unlock attempts and reward-structure fixes on top of V28o |
| `v30_parkour.py`, `v30_full.py`, `v30_parkour_moe.py`, `v30a` ... `v30i` | V30: balanced reward (base-height and hip-deviation terms) on the V28o curriculum, then single-knob ablations |
| `v30_clean/` | Staged rebuild from stock: flat → slopes → stairs → full parkour → base-height terms |
| `g1_stock_parkour_terrain.py` | Stock G1 config + the same parkour terrain, for the apples-to-apples comparison against V30 |
| `v31_*.py`, `v40_parkour.py` | Terrain-conditioned reward + skill-based curriculum; reward rebuilt from an audit of V30 |
| `v50_parkour.py`, `vbasev1/2/3_parkour.py` | Stock G1 with only the terrain changed (V50 is a flat re-implementation of it); VBaseV2 = stronger long-step reward |
| `v51_parkour.py` | V50 + 5-frame proprioception history |
| `v52_transformer.py` | Transformer policy with a ViT height-map encoder |
| `v53_parkour.py`, `bd_zigzag_terrain.py` | Zigzag angled-plate terrain as a ninth tile type |
| `v54_parkour.py`, `v55_parkour.py` | Omnidirectional terrains (rough, wave, obstacles), MoE actor from scratch |
| `fasttd3.py`, `train_fasttd3.py` | From-scratch FastTD3 (off-policy, distributional critic) on the same envs; no results reported |

## Results

| Run | Outcome |
|---|---|
| V28 teacher | 12k iterations: 5.3 % fall rate, 0.28 m/s velocity error |
| V28 student | Did not work: fell 9x as often and never committed to actions. Its observations lacked angular velocity and had too short a history; distillation cannot copy behaviour the student cannot sense |
| V28b | Same reward 5x faster, no higher ceiling: the limit was the environment, not model capacity |
| V28c | Asymmetric gait (one leg hoisted very high) traced to an over-weighted `feet_air_time`; lowering it gave the best tracking of the series (0.23 m/s, 0.67 rad/s) |
| V28d | The torque-rate penalty dominated on hard terrain, "don't move" became optimal, reward collapsed by 79 % |
| V28e-g | MoE actor converged to a deep crouching duck walk; V28n with a plain network fixed it |
| V28o | Three bugs in the terrain curriculum found and fixed (e.g. robots told to stand still could never be promoted; IsaacLab's stock curriculum wraps robots at the max level back to a random level) |
| V30 | 5 of 8 terrain types saturated; gaps stuck at level 0.43 of 9 |
| V30-plain | **Retracted.** Mean terrain level 8.45 came from a relaxed promotion rule in the custom curriculum, not skill; under the stock curriculum the same family sits at about 5.6-5.8 |
| VBaseV2 | Mean level 5.61, gaps 5.91 of 9, fall rate halved to 3.5 % |
| V51 | Top level (9) on 7 of 8 terrain types within about 600 iterations; no final evaluation logged |
| V52 | Transformer + ViT height map never stood up (100 % falls) |
| V53 | Zigzag parkour plates were simply ignored by the policy |
| Diffusion / flow / transformer actors | Best: terrain level 3.50 of 10 with a 68 % fall rate, well below a plain MLP |

After the V28 fixes the policy maxes out flat ground, stairs, uphill slopes and boxes and gets far on
downhill slopes and stepping stones. Gaps remained the hard problem: crossing one needs a stepping-over
skill the curriculum never produced. All policies perceive terrain through the simulator's ray-cast
height map, not a real sensor, and none was deployed on hardware.

Figures and videos: `../docs/media/parkour_*.png|mp4`.
