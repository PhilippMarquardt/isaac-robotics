"""V30h — V30e with track_lin_vel_xy_exp weight 1.5 -> 2.5.

V30g (circling penalty walking_yaw_l2) was scrapped in favour of a
single-knob change: bump the velocity-tracking reward weight so
following the velocity command dominates whatever survival/avoidance
the policy can learn. Circling = low forward velocity = lower exp
reward, so the gradient pulls forward without an explicit anti-yaw
penalty term.

V28o had track_lin_vel_xy_exp = 1.5 (V28b set 2.0; V28o reverted).
V30h pushes to 2.5 — strongest pull seen in this lineage.

Single knob. Cold start fresh seed 23.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30e_parkour import (
    G1V30eParkourEnvCfg,
    G1V30eParkourEnvCfg_PLAY,
    G1V30eParkourTeacherPPORunnerCfg,
)


def _bump_track_lin(self) -> None:
    self.rewards.track_lin_vel_xy_exp.weight = 2.5


@configclass
class G1V30hParkourEnvCfg(G1V30eParkourEnvCfg):
    """V30e with track_lin_vel_xy_exp 1.5 -> 2.5."""

    def __post_init__(self):
        super().__post_init__()
        _bump_track_lin(self)


@configclass
class G1V30hParkourEnvCfg_PLAY(G1V30eParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _bump_track_lin(self)


@configclass
class G1V30hParkourTeacherPPORunnerCfg(G1V30eParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30h_parkour_teacher"


def register_v30h_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30h-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30h-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30h_parkour:G1V30hParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30h_parkour:G1V30hParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30h-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30h_parkour:G1V30hParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30h_parkour:G1V30hParkourTeacherPPORunnerCfg",
        },
    )
