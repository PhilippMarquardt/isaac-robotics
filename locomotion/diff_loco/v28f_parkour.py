"""V28f G1 Parkour — V28e + hysteresis terrain curriculum.

Single change vs V28e: ``terrain_levels`` curriculum swapped from stock
``terrain_levels_vel`` to ``terrain_levels_hysteresis`` with K=3 fail
threshold. Everything else (MoE actor, PPO stabilisation, V28d sim-to-real
DR, V28c reward rebalance) is inherited unchanged.

Why: V28e plateaus at terrain_levels ~5.7. Stock curriculum demotes on a
single failed episode, so a robot that just learned level 6 stairs falls
once on a level-6 gap and is sent back to level 5 — discarding the gradient
signal it just collected. K=3 lets PPO accumulate three attempts at a level
before demotion fires, which should clear the plateau if my theory holds.

Resume target: V28e model_last.pt at iter ~2280 (machine B,
``logs/rsl_rl/g1_v28e_parkour_teacher/2026-04-24_20-49-14/``).
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import (
    terrain_level_per_type_logger,
    terrain_levels_hysteresis,
)
from diff_loco.v28e_bundle import (
    G1V28eParkourEnvCfg,
    G1V28eParkourEnvCfg_PLAY,
    G1V28eParkourTeacherPPORunnerCfg,
    G1V28eParkourStudentDistillRunnerCfg,
)


_V28F_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


@configclass
class G1V28fParkourEnvCfg(G1V28eParkourEnvCfg):
    """V28e env + hysteresis curriculum (K=3) + per-terrain-type level loggers."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = CurrTerm(
            func=terrain_levels_hysteresis,
            params={"fail_threshold": 3},
        )
        # Per-terrain-type level loggers — V28b chain skips v28_parkour base,
        # so attach here. Indices follow V28_PARKOUR_TERRAINS_CFG dict order.
        for _i, _name in enumerate(_V28F_TERRAIN_TYPES):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28fParkourEnvCfg_PLAY(G1V28eParkourEnvCfg_PLAY):
    """V28f play cfg — hysteresis curriculum disabled at play (terrain pinned)."""

    def __post_init__(self):
        super().__post_init__()
        # Play already disables curriculum via num_rows=5 + curriculum=False;
        # terrain_levels term is harmless to keep but we leave parent's value.


@configclass
class G1V28fParkourTeacherPPORunnerCfg(G1V28eParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28f_parkour_teacher"


@configclass
class G1V28fParkourStudentDistillRunnerCfg(G1V28eParkourStudentDistillRunnerCfg):
    experiment_name = "g1_v28f_parkour_student"


def register_v28f_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28f-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28f-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28f-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28f-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28f-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28f_parkour:G1V28fParkourStudentDistillRunnerCfg",
        },
    )
