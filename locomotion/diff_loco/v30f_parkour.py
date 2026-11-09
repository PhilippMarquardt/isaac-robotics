"""V30f — V30e with three low-value rewards zeroed.

Per-step audit at V30e iter 773 showed:
  joint_deviation_arms  -0.43  (29% of track_lin_vel reward)
  joint_deviation_fingers -0.13 (locomotion-irrelevant)
  feet_air_time         +0.007 (effectively zero, useless)

V30f zeros all three. Parkour teacher does not need:
  - locked arms (real arm swing aids balance on gaps + stairs)
  - finger pose enforcement (no manipulation task)
  - a feet_air_time bonus too small to encourage swing phase

Single ablation against V30e. Cold start fresh seed 17.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30e_parkour import (
    G1V30eParkourEnvCfg,
    G1V30eParkourEnvCfg_PLAY,
    G1V30eParkourTeacherPPORunnerCfg,
)


def _zero_low_value_rewards(self) -> None:
    self.rewards.joint_deviation_arms.weight = 0.0
    self.rewards.joint_deviation_fingers.weight = 0.0
    self.rewards.feet_air_time.weight = 0.0


@configclass
class G1V30fParkourEnvCfg(G1V30eParkourEnvCfg):
    """V30e minus arms_dev, fingers_dev, feet_air_time."""

    def __post_init__(self):
        super().__post_init__()
        _zero_low_value_rewards(self)


@configclass
class G1V30fParkourEnvCfg_PLAY(G1V30eParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _zero_low_value_rewards(self)


@configclass
class G1V30fParkourTeacherPPORunnerCfg(G1V30eParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30f_parkour_teacher"


def register_v30f_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30f-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30f-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30f_parkour:G1V30fParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30f_parkour:G1V30fParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30f-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30f_parkour:G1V30fParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30f_parkour:G1V30fParkourTeacherPPORunnerCfg",
        },
    )
