"""V28q G1 Parkour — V28p + real lvl-0 gap + tighter range + air time.

V28p set ``gap_width_range = (0.0, 0.5)`` so lvl 0 had no gap (flat
platform). Policy walked across trivially → promoted to lvl 2 (~10cm
gap) → couldn't step over 15cm at lvl 3+ → hysteresis demoted back to
~2.4. The lvl-0 flat-platform shortcut meant the gap-stepping skill
was never actually trained — at lvl 0 the policy just walked, learning
nothing gap-specific, then hit a wall the moment it had to step over
real gaps at higher levels.

V28q forces a real gap at every level so the curriculum trains the
actual skill from the bottom:

  1. **Real lvl-0 gap** ``(0.0, 0.5) → (0.05, 0.3)``. Lvl 0 = 5cm gap
     (small but non-zero — policy MUST step over it to promote).
     Lvl 9 = 30cm. ~2.8cm increment per level, smooth skill ramp.

  2. **feet_air_time weight 1.0 → 2.0**. Direct stepping incentive,
     replaces the deferred-curriculum bump (was 1.5 @ iter 48k).
     Attacks the "feet stay too low" failure mode on gap tiles.

**Trained from scratch** (no resume). V28p policy may have learned the
"walk over flat lvl-0" shortcut and embedded it in the gait — fresh
weights force the gap-stepping skill to be learned from a 5cm gap up.
Warmup re-enabled (100 iters on flat/slope) since this is cold-start.
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import (
    terrain_level_per_type_logger,
    terrain_levels_hysteresis_v2,
)
from diff_loco.v28p_parkour import (
    G1V28pParkourEnvCfg,
    G1V28pParkourEnvCfg_PLAY,
    G1V28pParkourTeacherPPORunnerCfg,
)


_V28Q_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


@configclass
class G1V28qParkourEnvCfg(G1V28pParkourEnvCfg):
    """V28p env + narrower gap range + air time bump."""

    def __post_init__(self):
        super().__post_init__()

        # --- feet_air_time: 1.0 → 2.0 (encourage stepping over shuffling) ---
        # V28p showed the policy can't lift feet over 15cm+ gaps. Direct
        # incentive for non-zero air time should bias the gait toward
        # higher foot lift.
        self.rewards.feet_air_time.weight = 2.0

        # Drop the deferred-bump curriculum term (was 1.5 @ 48k steps).
        # We're already at 2.0 from iter 0 of V28q; the deferred curriculum
        # would clobber that back to 1.5 partway through training.
        if hasattr(self.curriculum, "v28_air_time"):
            delattr(self.curriculum, "v28_air_time")

        # --- real lvl-0 gap: (0.0, 0.5) → (0.05, 0.3) ---
        # V28p's lvl-0 flat platform let policy promote without learning
        # to step. Force a 5cm gap at lvl 0 so the skill is trained from
        # the bottom. Lvl 9 = 30cm (vs 50cm in V28p), ~2.8cm/lvl ramp.
        gen_cfg = self.scene.terrain.terrain_generator
        if gen_cfg is not None and "gaps" in gen_cfg.sub_terrains:
            old_gap = gen_cfg.sub_terrains["gaps"]
            gen_cfg.sub_terrains["gaps"] = mesh.MeshGapTerrainCfg(
                proportion=old_gap.proportion,
                gap_width_range=(0.05, 0.3),
                platform_width=old_gap.platform_width,
            )

        # Curriculum: V28o-style warmup re-enabled because V28q trains from
        # scratch (not resumed). 100 iters on flat/slope only lets the
        # policy learn basic walking before facing gaps/stairs/stones.
        self.curriculum.terrain_levels = CurrTerm(
            func=terrain_levels_hysteresis_v2,
            params={
                "fail_threshold": 3,
                "gap_terrain_idx": 4,
                "gap_promote_bar_m": 1.0,
                "stairs_inv_terrain_idx": 2,
                "stairs_inv_promote_bar_m": 2.0,
                "demote_speed_frac": 0.125,
                "demote_bar_floor_m": 1.0,
                "warmup_steps": 3_200,           # 100 PPO iters on flat+slope
                "warmup_terrain_idxs": (0, 6, 7),
            },
        )

        for _i, _name in enumerate(_V28Q_TERRAIN_TYPES):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28qParkourEnvCfg_PLAY(G1V28pParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28qParkourTeacherPPORunnerCfg(G1V28pParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28q_parkour_teacher"


def register_v28q_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28q-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28q-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28q_parkour:G1V28qParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28q_parkour:G1V28qParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28q-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28q_parkour:G1V28qParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28q_parkour:G1V28qParkourTeacherPPORunnerCfg",
        },
    )
