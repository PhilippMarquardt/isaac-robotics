"""V30i — V30f with lin_vel_z_l2 weight relaxed -2.0 -> -0.5.

V30f iter 1000 audit (machine E):
  pyramid_stairs       8.53  saturating
  pyramid_stairs_inv   3.80  STUCK
  geometric symmetry of up/down stairs => the only asymmetric reward
  is lin_vel_z_l2. Descending a stair forces a longer negative z-vel
  transient than the powered upward push of a climb, so |v_z|^2 sums
  larger going down than up. -2.0 weight crushes descent gradient.

Cited support: Extreme Parkour (Cheng et al. 2024), Humanoid Parkour
Learning (arxiv 2406.10759), Hiking-in-the-Wild (arxiv 2601.07718)
all down-weight z-vel penalties near stair edges.

V30i drops lin_vel_z_l2 to -0.5 (still bouncy-suppressing but
no longer dominant). Single knob vs V30f. Cold start fresh seed 29.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30f_parkour import (
    G1V30fParkourEnvCfg,
    G1V30fParkourEnvCfg_PLAY,
    G1V30fParkourTeacherPPORunnerCfg,
)


def _relax_lin_vel_z(self) -> None:
    self.rewards.lin_vel_z_l2.weight = -0.5


@configclass
class G1V30iParkourEnvCfg(G1V30fParkourEnvCfg):
    """V30f with lin_vel_z_l2 -2.0 -> -0.5."""

    def __post_init__(self):
        super().__post_init__()
        _relax_lin_vel_z(self)


@configclass
class G1V30iParkourEnvCfg_PLAY(G1V30fParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _relax_lin_vel_z(self)


@configclass
class G1V30iParkourTeacherPPORunnerCfg(G1V30fParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30i_parkour_teacher"


def register_v30i_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30i-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30i-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30i_parkour:G1V30iParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30i_parkour:G1V30iParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30i-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30i_parkour:G1V30iParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30i_parkour:G1V30iParkourTeacherPPORunnerCfg",
        },
    )
