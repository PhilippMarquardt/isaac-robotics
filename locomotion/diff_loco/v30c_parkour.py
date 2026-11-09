"""V30c — V30 with hip_deviation_hip removed (weight 0).

V30-plain (machine E, iter 2310, lvl 8.45) saturates 4 terrains at 9.00 but
stuck on gaps (4.91). Stock G1+parkour (machine B, iter 2488, lvl 5.72)
beats V30 on gaps (5.37 > 4.91) despite far less training.

Hypothesis: V30's joint_deviation_hip weight -1.0 (Unitree balance recipe,
hip_yaw + hip_roll lateral penalty) blocks the wide-stance step needed
to clear wide gaps. Stock has no hip-deviation cost. V30c removes the
penalty entirely (weight 0) to test whether gap level catches the rest.

Single-knob change vs V30-plain. Cold start fresh seed 7.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30_parkour import (
    G1V30ParkourEnvCfg,
    G1V30ParkourEnvCfg_PLAY,
    G1V30ParkourTeacherPPORunnerCfg,
)


def _drop_hip_deviation(self) -> None:
    self.rewards.joint_deviation_hip.weight = 0.0


@configclass
class G1V30cParkourEnvCfg(G1V30ParkourEnvCfg):
    """V30 - hip_deviation."""

    def __post_init__(self):
        super().__post_init__()
        _drop_hip_deviation(self)


@configclass
class G1V30cParkourEnvCfg_PLAY(G1V30ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _drop_hip_deviation(self)


@configclass
class G1V30cParkourTeacherPPORunnerCfg(G1V30ParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30c_parkour_teacher"


def register_v30c_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30c-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30c-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30c_parkour:G1V30cParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30c_parkour:G1V30cParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30c-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30c_parkour:G1V30cParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30c_parkour:G1V30cParkourTeacherPPORunnerCfg",
        },
    )
