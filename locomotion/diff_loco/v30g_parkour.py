"""V30g — V30e + circling penalty to break gap-bail.

V30e iter 1214 audit (machine B):
  gaps drift 5.11 (973) -> 4.81 -> 4.72 -> 4.32 (1214). Visual play
  confirms: small gaps -> step over; big gaps -> spin in circle to
  farm survival without committing forward. Curriculum oscillates
  around the demote bar.

V30g adds walking_yaw_l2: |ang_vel_z|^2 penalty active ONLY when
forward-commanded (|cmd_lin_xy| > 0.3) AND no yaw commanded
(|cmd_yaw| < 0.2). Does NOT touch commanded turns or stair yaw
correction. Gradient stays linear in yaw_rate (vs exp-saturated
track_ang_vel_z_world_exp).

Single knob added vs V30e. Cold start fresh seed 19.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from diff_loco import rewards as _rewards
from diff_loco.v30e_parkour import (
    G1V30eParkourEnvCfg,
    G1V30eParkourEnvCfg_PLAY,
    G1V30eParkourTeacherPPORunnerCfg,
)


def _add_walking_yaw(self) -> None:
    self.rewards.walking_yaw_l2 = RewTerm(
        func=_rewards.walking_yaw_l2,
        weight=-1.0,
        params={
            "min_lin_cmd": 0.3,
            "max_yaw_cmd": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class G1V30gParkourEnvCfg(G1V30eParkourEnvCfg):
    """V30e + walking_yaw_l2 (-1.0)."""

    def __post_init__(self):
        super().__post_init__()
        _add_walking_yaw(self)


@configclass
class G1V30gParkourEnvCfg_PLAY(G1V30eParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _add_walking_yaw(self)


@configclass
class G1V30gParkourTeacherPPORunnerCfg(G1V30eParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30g_parkour_teacher"


def register_v30g_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30g-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30g-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30g_parkour:G1V30gParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30g_parkour:G1V30gParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30g-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30g_parkour:G1V30gParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30g_parkour:G1V30gParkourTeacherPPORunnerCfg",
        },
    )
