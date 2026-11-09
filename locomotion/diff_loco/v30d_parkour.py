"""V30d — V30c with base_height_l2 removed AND curriculum warmup disabled.

V30c (machine E, running) tests hip_deviation_hip removal alone. V30d removes
BOTH hip_deviation_hip AND base_height_l2, AND drops the V28o curriculum
warmup (3200 steps ≈ 100 iters of flat/slope-only spawning).

Hypotheses (two-knob test, deliberate):
  1. base_height_l2 (-5.0) blocks forward lean / CoM drop for wide gaps.
  2. Warmup wastes 100 iters on easy terrain that doesn't help gap policy.

Stock G1+parkour iter 4000 hit gaps 4.57 with NO base_height + NO warmup
(stock terrain_levels_vel doesn't gate spawn). V30-plain peaked 4.91 with
base_height -5.0 + warmup. V30d tests whether dropping both lifts gap lvl.

Cold start fresh seed 11.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v30c_parkour import (
    G1V30cParkourEnvCfg,
    G1V30cParkourEnvCfg_PLAY,
    G1V30cParkourTeacherPPORunnerCfg,
)


def _drop_base_height_and_warmup(self) -> None:
    self.rewards.base_height_l2.weight = 0.0
    if hasattr(self.curriculum, "terrain_levels"):
        params = self.curriculum.terrain_levels.params
        if params is not None and "warmup_steps" in params:
            params["warmup_steps"] = 0


@configclass
class G1V30dParkourEnvCfg(G1V30cParkourEnvCfg):
    """V30c - base_height - warmup."""

    def __post_init__(self):
        super().__post_init__()
        _drop_base_height_and_warmup(self)


@configclass
class G1V30dParkourEnvCfg_PLAY(G1V30cParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _drop_base_height_and_warmup(self)


@configclass
class G1V30dParkourTeacherPPORunnerCfg(G1V30cParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30d_parkour_teacher"


def register_v30d_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30d-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30d-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30d_parkour:G1V30dParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30d_parkour:G1V30dParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30d-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30d_parkour:G1V30dParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30d_parkour:G1V30dParkourTeacherPPORunnerCfg",
        },
    )
