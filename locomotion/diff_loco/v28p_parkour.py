"""V28p G1 Parkour — V28o + gap unlock + yaw weight bump.

V28o cracked the curriculum bugs (rel_standing, demote-floor, stairs_inv
promote bar, warmup) and saturated 4 tiles (flat/stairs/slope_up/boxes).
Three open issues remain that V28p targets head-on.

Open issues from V28o @ iter ~1000:

  1. **gaps stuck at 0.10**. Even at level 0 the stock gap terrain has
     ``gap_width_range=(0.1, 0.5)`` so the robot must step over a 10cm
     gap from a cold gait. promote_bar 2.0m forced full platform-to-
     platform crossing. V28p:
       - ``gap_promote_bar_m = 1.0`` — reach platform edge = promote.
       - override gap cfg with ``gap_width_range=(0.0, 0.5)`` so level 0
         is a continuous platform (no gap), level 1 = 0.05m, etc.
  2. **err_vel_yaw 1.41 rad/s** — track_ang_vel_z weight 1.0 vs
     track_lin_vel_xy weight 2.0 starves yaw learning. V28p bumps yaw
     weight to 1.5.
  3. **stones 5.59 climbing slowly** — implicit benefit from yaw fix
     (precise turning helps stone-to-stone footing).

Resume from V28o ``model_last.pt`` to inherit all 4 saturated tiles
and the basic walking gait. Warmup off (policy already walks).
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import (
    terrain_level_per_type_logger,
    terrain_levels_hysteresis_v2,
)
from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)


_V28P_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


@configclass
class G1V28pParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o env + gap unlock + yaw bump."""

    def __post_init__(self):
        super().__post_init__()

        # --- yaw tracking weight bump (1.0 → 1.5) ---
        # V28o err_vel_yaw 1.41 rad/s. Lin weight 2.0 vs ang 1.0 = yaw starved.
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # --- gap terrain: extend lower bound to 0.0 so level 0 = no gap ---
        # Stock cfg has gap_width_range=(0.1, 0.5). At level 0 the easiest
        # gap is still 10cm — robot can't step over it from cold-walk gait.
        # Replace the gap sub-terrain with a 0.0-min variant.
        gen_cfg = self.scene.terrain.terrain_generator
        if gen_cfg is not None and "gaps" in gen_cfg.sub_terrains:
            old_gap = gen_cfg.sub_terrains["gaps"]
            gen_cfg.sub_terrains["gaps"] = mesh.MeshGapTerrainCfg(
                proportion=old_gap.proportion,
                gap_width_range=(0.0, 0.5),
                platform_width=old_gap.platform_width,
            )

        # --- curriculum: V28o curriculum + gap_promote_bar 2.0 → 1.0 ---
        # Reach platform edge (1m radial from spawn) counts as success.
        # Combined with 0.0-gap floor at level 0, gives the policy a
        # progression path on gap tiles instead of a brick wall.
        # Warmup off — V28p resumes a policy that already walks.
        self.curriculum.terrain_levels = CurrTerm(
            func=terrain_levels_hysteresis_v2,
            params={
                "fail_threshold": 3,
                "gap_terrain_idx": 4,
                "gap_promote_bar_m": 1.0,            # V28p: 2.0 → 1.0
                "stairs_inv_terrain_idx": 2,
                "stairs_inv_promote_bar_m": 2.0,
                "demote_speed_frac": 0.125,
                "demote_bar_floor_m": 1.0,
                "warmup_steps": 0,                   # V28p: warmup off (resume)
                "warmup_terrain_idxs": (0, 6, 7),
            },
        )

        # Per-type level loggers re-attached for clean naming.
        for _i, _name in enumerate(_V28P_TERRAIN_TYPES):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28pParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28pParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28p_parkour_teacher"


def register_v28p_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28p-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28p-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28p_parkour:G1V28pParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28p_parkour:G1V28pParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28p-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28p_parkour:G1V28pParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28p_parkour:G1V28pParkourTeacherPPORunnerCfg",
        },
    )
