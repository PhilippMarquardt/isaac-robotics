"""V30b — V30-parkour env + feet_air_time weight 0.25 → 1.0.

V30-plain (machine E iter 442) reaches stairs 6.45, slope_up 6.91, mean reward
9.22 — but stuck on terrains needing real stepping: stairs_inv 1.66,
slope_down 2.40, gaps 3.01, stones 3.68. Policy walks forward fine, can't
descend / step over.

Bumping feet_air_time weight from 0.25 (V28c stock) to 1.0 forces longer
swing time per foot. V28b ran at 1.0 cleanly; V28p/q ran at 1.5/2.0 with
other co-knobs. 1.0 is the safe middle.

Single-knob change. Cold start, seed 7 (V30-plain on machine E uses 42).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30_parkour import (
    G1V30ParkourEnvCfg,
    G1V30ParkourEnvCfg_PLAY,
    G1V30ParkourTeacherPPORunnerCfg,
)


def _bump_feet_air_time(self) -> None:
    self.rewards.feet_air_time.weight = 1.0


@configclass
class G1V30bParkourEnvCfg(G1V30ParkourEnvCfg):
    """V30 + feet_air_time 0.25 → 1.0."""

    def __post_init__(self):
        super().__post_init__()
        _bump_feet_air_time(self)


@configclass
class G1V30bParkourEnvCfg_PLAY(G1V30ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _bump_feet_air_time(self)


@configclass
class G1V30bParkourTeacherPPORunnerCfg(G1V30ParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30b_parkour_teacher"


def register_v30b_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30b-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30b-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30b_parkour:G1V30bParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30b_parkour:G1V30bParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30b-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30b_parkour:G1V30bParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30b_parkour:G1V30bParkourTeacherPPORunnerCfg",
        },
    )
