"""V30 — Unitree-balanced reward + V28o curriculum + V28d sim-to-real DR.

Single hypothesis: V28o's fence-sit attractor exists because alive=0.15 isn't
balanced by base_height penalty (which V28 disabled out of caution). Unitree's
verified G1 recipe pairs alive=0.15 with base_height=-10.0 + hip_pos=-1.0,
making poor standing posture costly enough to negate the alive subsidy.

V30 adds those two terms with parkour-safe implementation.

2026-04-26 patch: gap_width_range stock (0.1, 0.4) → (0.02, 0.4). machine E MoE
hit 9.0 on 7/8 terrains but gaps stuck at 0.05 — 10cm lvl-0 gap was
untrainable from cold. Tiny 2cm gap at lvl 0 unblocks the curriculum
(MoE resume run jumped 0.05 → 5.4 in 100 iters with this single change).
"""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
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


def base_height_safe_l2(
    env, target_height: float = 0.74,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Parkour-safe base height penalty.

    Uses robot.root_pos_w[:, 2] directly (NOT raycast against terrain), so
    gaps/pits don't produce -inf. Clipped to ±0.5m to prevent reward explosions
    when robot momentarily airborne over a pit.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    err = torch.clamp(height - target_height, min=-0.5, max=0.5)
    return torch.square(err)


@configclass
class G1V30ParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o + Unitree reward balance fix + tiny lvl-0 gap."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2 = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={"target_height": 0.74},
        )

        self.rewards.joint_deviation_hip.weight = -1.0

        _shrink_gap_start(self)


@configclass
class G1V30ParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_height_l2 = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={"target_height": 0.74},
        )
        self.rewards.joint_deviation_hip.weight = -1.0
        _shrink_gap_start(self)


@configclass
class G1V30ParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30_parkour_teacher"


def register_v30_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_parkour:G1V30ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_parkour:G1V30ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_parkour:G1V30ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_parkour:G1V30ParkourTeacherPPORunnerCfg",
        },
    )
