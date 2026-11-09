"""V28o G1 Parkour — V28n + curriculum fixes.

V28n exposed three curriculum-side bugs that capped progress:

  1. ``rel_standing_envs = 0.10`` — 10% of envs got cmd=(0,0,0). With
     cmd_norm=0 the demote_bar collapsed to 0 and promote_bar stayed at
     4 m, so these envs never moved off level 0 and dragged every
     per-type mean down by ~10%.
  2. Pure-yaw / near-zero lin cmds had the same blackhole pattern: low
     cmd_norm → tiny demote bar, high yaw cmd → robot spins in place
     and never crosses 4 m promote bar. Stuck at level 0.
  3. ``pyramid_stairs_inv`` was spawn-locked just like ``gaps``. Robot
     spawns on top platform, has to descend stairs then traverse the
     pit — Euclidean distance from spawn rarely exceeds 4 m even when
     traversal succeeds. Got stuck at 0.08 in V28n.

V28o changes vs V28n:

  - ``rel_standing_envs`` 0.10 → 0.02 (kills the standing blackhole).
  - ``terrain_levels_hysteresis_v2`` curriculum:
      * ``stairs_inv_promote_bar_m = 2.0`` (mirrors gap fix).
      * ``demote_bar_floor_m = 1.0`` — even pure-yaw envs demote if
        they make no progress.
      * ``warmup_steps = 3_200`` (= ~100 PPO iters at
        num_steps_per_env=32) — first 100 iters force resetting envs
        onto warmup tiles only (flat=0, slope_up=6, slope_down=7) so
        the policy bootstraps basic walking before stairs/gaps unlock.

Everything else inherited from V28n unchanged: plain MLP actor, V28d
sim-to-real bundle, V28c reward rebalance, V28e PPO stabilisation.

Trained from scratch — warmup is meaningful only from a cold policy.
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import (
    terrain_level_per_type_logger,
    terrain_levels_hysteresis_v2,
)
from diff_loco.v28n_clean import (
    G1V28nParkourEnvCfg,
    G1V28nParkourEnvCfg_PLAY,
    G1V28nParkourTeacherPPORunnerCfg,
)


_V28O_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


@configclass
class G1V28oParkourEnvCfg(G1V28nParkourEnvCfg):
    """V28n env + V28o curriculum + standing fix."""

    def __post_init__(self):
        super().__post_init__()
        # Standing-still blackhole fix.
        self.commands.base_velocity.rel_standing_envs = 0.02

        # Replace V28f hysteresis with v2: stairs_inv promote bar + demote
        # floor + 500-iter warmup on flat/slope only.
        self.curriculum.terrain_levels = CurrTerm(
            func=terrain_levels_hysteresis_v2,
            params={
                "fail_threshold": 3,
                "gap_terrain_idx": 4,
                "gap_promote_bar_m": 2.0,
                "stairs_inv_terrain_idx": 2,
                "stairs_inv_promote_bar_m": 2.0,
                "demote_speed_frac": 0.125,
                "demote_bar_floor_m": 1.0,
                "warmup_steps": 3_200,
                "warmup_terrain_idxs": (0, 6, 7),  # flat, slope_up, slope_down
            },
        )

        # Per-type level loggers (re-attach in case parent changes order).
        for _i, _name in enumerate(_V28O_TERRAIN_TYPES):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28oParkourEnvCfg_PLAY(G1V28nParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28oParkourTeacherPPORunnerCfg(G1V28nParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28o_parkour_teacher"


def register_v28o_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28o-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28o-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28o_parkour:G1V28oParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28o_parkour:G1V28oParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28o-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28o_parkour:G1V28oParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28o_parkour:G1V28oParkourTeacherPPORunnerCfg",
        },
    )
