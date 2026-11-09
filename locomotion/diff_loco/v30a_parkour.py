"""V30a — V30-parkour env + projvel-only swap (stripped from full V28t).

Original V30a stack (full V28t pull-forward) cold-start failed — track 0.10,
base_contact 100% sustained at iter 368. Lesson: V28t reward shape assumes
warm gait. Strip to single change: world-frame projvel replacing exp(-L2)
tracking. Keep V30 alive bonus, no stall penalty, no stepping_stones,
no PPO-knob retunes.

Stack vs V30:
  - track_lin_vel_xy_exp → world-frame inner-product projvel (V28t fn)
  - everything else: V30 base unchanged (alive bonus kept, gaps stay
    MeshGap with 0.02–0.4 fix from V30)

Cold start. Reward shape change still forks the loss landscape — can't
resume from V30 ckpt that learned exp(-L2) tracking.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from diff_loco.v28t_parkour import track_lin_vel_inner_product_world
from diff_loco.v30_parkour import (
    G1V30ParkourEnvCfg,
    G1V30ParkourEnvCfg_PLAY,
    G1V30ParkourTeacherPPORunnerCfg,
)


def _apply_v30a_overrides(self) -> None:
    """Single-change overlay: projvel tracking only."""
    self.rewards.track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_inner_product_world,
        weight=2.0,
        params={"command_name": "base_velocity", "max_speed": 1.5},
    )


@configclass
class G1V30aParkourEnvCfg(G1V30ParkourEnvCfg):
    """V30 env + projvel-only reward swap."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v30a_overrides(self)


@configclass
class G1V30aParkourEnvCfg_PLAY(G1V30ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_v30a_overrides(self)


@configclass
class G1V30aParkourTeacherPPORunnerCfg(G1V30ParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30a_parkour_teacher"


def register_v30a_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30a-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30a-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30a_parkour:G1V30aParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30a_parkour:G1V30aParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30a-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30a_parkour:G1V30aParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30a_parkour:G1V30aParkourTeacherPPORunnerCfg",
        },
    )
