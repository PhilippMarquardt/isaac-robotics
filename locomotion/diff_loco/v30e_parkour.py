"""V30e — V30d with flat_orientation_l2 weight reduced -2.5 → -1.0 (stock).

V30d (machine B, iter ~130) early base_contact 56% suggests aggressive
upright bias hurts cold-start gait discovery on parkour. V28c bumped
flat_orientation_l2 to -2.5 (2.5x stock) for vision/heightmap stability;
on parkour with gaps + boxes the same penalty over-restricts forward
lean and torso pitch needed to clear obstacles.

V30e drops it back to stock -1.0. Single-knob change vs V30d.

Cold start fresh seed 13.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30d_parkour import (
    G1V30dParkourEnvCfg,
    G1V30dParkourEnvCfg_PLAY,
    G1V30dParkourTeacherPPORunnerCfg,
)


def _relax_flat_orientation(self) -> None:
    self.rewards.flat_orientation_l2.weight = -1.0


@configclass
class G1V30eParkourEnvCfg(G1V30dParkourEnvCfg):
    """V30d with flat_orientation_l2 -2.5 -> -1.0 (stock)."""

    def __post_init__(self):
        super().__post_init__()
        _relax_flat_orientation(self)


@configclass
class G1V30eParkourEnvCfg_PLAY(G1V30dParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _relax_flat_orientation(self)


@configclass
class G1V30eParkourTeacherPPORunnerCfg(G1V30dParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30e_parkour_teacher"


def register_v30e_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30e-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30e-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30e_parkour:G1V30eParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30e_parkour:G1V30eParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30e-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30e_parkour:G1V30eParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30e_parkour:G1V30eParkourTeacherPPORunnerCfg",
        },
    )
