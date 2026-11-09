"""V30-parkour-MoE — V30-parkour env + MoE actor.

Same env as V30-parkour (V28o + Unitree balance: base_height_safe -5,
hip_dev -1.0). Actor swapped from plain MLP [768,512,256,128] to soft-gated
MoE (4 experts of [512,256,128]) from V28e_bundle.

Hypothesis: MoE expert specialization helps with the chronic per-terrain
weakness (gaps + stairs_inv) — different experts can route per terrain
type instead of one MLP averaging across all 8.

Critic + PPO + DR unchanged from V30-parkour.

2026-04-26 patch: gap_width_range stock (0.1, 0.4) → (0.02, 0.4). Iter 2110
showed 7/8 terrains saturated near lvl 9 but gaps stuck at 0.05 — lvl 0
gap of 10cm was still too wide for the policy to ever clear, so curriculum
never started promoting. Tiny 2cm start at lvl 0 (skill-trainable) ramps
linearly to 40cm at lvl 9. Single terrain-cfg change; reward / actor / PPO /
hysteresis all preserved. Resume from iter 2110 ckpt.
"""
from __future__ import annotations

from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from diff_loco.v28e_bundle import _V28E_PPO_ALGO, RslRlMoEModelCfg, _gauss
from diff_loco.v30_parkour import (
    G1V30ParkourEnvCfg,
    G1V30ParkourEnvCfg_PLAY,
    G1V30ParkourTeacherPPORunnerCfg,
)


def _shrink_gap_start(self) -> None:
    """gap_width_range stock (0.1, 0.4) → (0.02, 0.4) — tiny lvl-0 gap."""
    gen_cfg = self.scene.terrain.terrain_generator
    if gen_cfg is not None and "gaps" in gen_cfg.sub_terrains:
        old_gap = gen_cfg.sub_terrains["gaps"]
        gen_cfg.sub_terrains["gaps"] = mesh.MeshGapTerrainCfg(
            proportion=old_gap.proportion,
            gap_width_range=(0.02, 0.4),
            platform_width=old_gap.platform_width,
        )


@configclass
class G1V30ParkourMoEEnvCfg(G1V30ParkourEnvCfg):
    """V30-parkour env with smaller lvl-0 gap to unblock curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _shrink_gap_start(self)


@configclass
class G1V30ParkourMoEEnvCfg_PLAY(G1V30ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _shrink_gap_start(self)


@configclass
class G1V30ParkourMoETeacherPPORunnerCfg(G1V30ParkourTeacherPPORunnerCfg):
    """V30-parkour runner with MoE actor."""

    experiment_name = "g1_v30_parkour_moe_teacher"
    num_steps_per_env = 32

    actor = RslRlMoEModelCfg(
        class_name="diff_loco.v28e_bundle:MoEActorModel",
        hidden_dims=[32],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
        num_experts=4,
        expert_hidden_dims=[512, 256, 128],
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(**_V28E_PPO_ALGO)


def register_v30_parkour_moe() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30-MoE-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-MoE-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_parkour_moe:G1V30ParkourMoEEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_parkour_moe:G1V30ParkourMoETeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-MoE-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_parkour_moe:G1V30ParkourMoEEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_parkour_moe:G1V30ParkourMoETeacherPPORunnerCfg",
        },
    )
