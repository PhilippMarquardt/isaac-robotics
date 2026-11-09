"""V28r G1 Parkour — V28q + harder gaps.

V28q at iter 282 had gaps level pinned at 8.28/9.0 (≈26cm gap, near max
of 30cm). Other terrains (stairs, boxes, stones) were at 1.8–2.9. Gaps
saturated the curriculum — making it impossible to learn harder
gap-stepping skill while other terrains catch up.

V28r raises the gap ceiling:

  ``gap_width_range = (0.05, 0.3) → (0.05, 0.4)``

  - Lvl 0 keeps the 5cm bootstrap (real skill from start).
  - Lvl 9 = 40cm (was 30cm) — ~33% harder ceiling.
  - ~3.9cm increment per level (was ~2.8cm).

**Resume from V28q model_last.pt** so the gap policy doesn't restart
from scratch. Warmup off (policy already walks). Everything else
inherited from V28q.
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import (
    terrain_level_per_type_logger,
    terrain_levels_hysteresis_v2,
)
from diff_loco.v28q_parkour import (
    G1V28qParkourEnvCfg,
    G1V28qParkourEnvCfg_PLAY,
    G1V28qParkourTeacherPPORunnerCfg,
)


_V28R_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


@configclass
class G1V28rParkourEnvCfg(G1V28qParkourEnvCfg):
    """V28q env + wider gap range (5cm → 40cm)."""

    def __post_init__(self):
        super().__post_init__()

        # Raise gap ceiling: (0.05, 0.3) → (0.05, 0.4).
        # V28q gaps saturated at lvl 8.28/9. Extend max to keep the
        # curriculum biting. Lvl 0 stays at 5cm (skill bootstrap).
        gen_cfg = self.scene.terrain.terrain_generator
        if gen_cfg is not None and "gaps" in gen_cfg.sub_terrains:
            old_gap = gen_cfg.sub_terrains["gaps"]
            gen_cfg.sub_terrains["gaps"] = mesh.MeshGapTerrainCfg(
                proportion=old_gap.proportion,
                gap_width_range=(0.05, 0.4),
                platform_width=old_gap.platform_width,
            )

        # Warmup off — V28r resumes from V28q ckpt, policy already walks.
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
                "warmup_steps": 0,
                "warmup_terrain_idxs": (0, 6, 7),
            },
        )

        for _i, _name in enumerate(_V28R_TERRAIN_TYPES):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28rParkourEnvCfg_PLAY(G1V28qParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28rParkourTeacherPPORunnerCfg(G1V28qParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28r_parkour_teacher"


def register_v28r_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28r-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28r-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28r_parkour:G1V28rParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28r_parkour:G1V28rParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28r-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28r_parkour:G1V28rParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28r_parkour:G1V28rParkourTeacherPPORunnerCfg",
        },
    )
