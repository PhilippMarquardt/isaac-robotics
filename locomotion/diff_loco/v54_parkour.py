"""V54 — V53 minus zigzag, plus rough + wave omnidirectional terrains.

V53 zigzag had a directional ±x plate track. Robot couldn't learn it well —
even with a heading-lock command class, local play looked like the policy
ignored the plates and ran in random directions. Swap zigzag for two
omnidirectional terrains compatible with random heading commands:
  - rough (HfRandomUniformTerrainCfg) — uniform noise field
  - wave  (HfWaveTerrainCfg)         — sinusoidal undulations

Keeps V54's other contribution (`base_below_origin_holes` termination)
gated to terrains with actual holes (gaps + stepping_stones).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter
from isaaclab.terrains import height_field as hf
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.v50_parkour import G1V50ParkourPPORunnerCfg, _v50_build_col_to_type
from diff_loco.v53_parkour import (
    G1V53ParkourEnvCfg,
    G1V53ParkourEnvCfg_PLAY,
    v53_terrain_level_per_type_logger,
    v53_level_freq_logger,
)


# Terrain type indices (must match V54_PARKOUR_TERRAINS_CFG ordering):
# flat=0, p_stairs=1, p_stairs_inv=2, boxes=3, gaps=4, stones=5,
# slope_up=6, slope_down=7, rough=8, wave=9.
HOLE_TERRAIN_TYPES: tuple[int, ...] = (4, 5)


def base_below_origin_holes(
    env,
    threshold: float,
    hole_types: tuple[int, ...] = HOLE_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate envs on hole-bearing terrains whose pelvis dropped below origin.

    True iff env's terrain type is in hole_types AND pelvis_z is more than
    `threshold` meters below the env's terrain spawn z. Other terrains
    (stairs, slopes, rough, wave) return False unconditionally so descending
    them is not punished.
    """
    asset = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(env, "_v54_col_to_type"):
        env._v54_col_to_type = _v50_build_col_to_type(env)
    env_type = env._v54_col_to_type[terrain.terrain_types]
    on_hole_terrain = torch.zeros_like(env_type, dtype=torch.bool)
    for t in hole_types:
        on_hole_terrain |= env_type == t
    pelvis_z = asset.data.root_pos_w[:, 2]
    origin_z = terrain.env_origins[:, 2]
    fell = (pelvis_z - origin_z) < -threshold
    return fell & on_hole_terrain


def v54_max_level_ever_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int = -1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V53's max_lvl_ever logger but with 11-slot array (10 types + global)."""
    terrain: TerrainImporter = env.scene.terrain
    dev = terrain.terrain_levels.device
    if not hasattr(env, "_v54_max_lvl_ever"):
        env._v54_max_lvl_ever = torch.zeros(11, device=dev, dtype=torch.long)
    if type_idx < 0:
        cur = terrain.terrain_levels.max()
        slot = 10
    else:
        if not hasattr(env, "_v54_col_to_type"):
            env._v54_col_to_type = _v50_build_col_to_type(env)
        env_type = env._v54_col_to_type[terrain.terrain_types]
        mask = env_type == type_idx
        if not bool(mask.any()):
            return env._v54_max_lvl_ever[type_idx].float()
        cur = terrain.terrain_levels[mask].max()
        slot = type_idx
    env._v54_max_lvl_ever[slot] = torch.maximum(env._v54_max_lvl_ever[slot], cur.long())
    return env._v54_max_lvl_ever[slot].float()


# ============================================================================
# V54 terrain mix — V53 minus zigzag, + rough + wave
# Total proportion still 1.00. Old zigzag (0.21) split into rough (0.10) + wave (0.11).
# ============================================================================
V54_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": mesh.MeshPlaneTerrainCfg(proportion=0.05),
        "pyramid_stairs": mesh.MeshPyramidStairsTerrainCfg(
            proportion=0.14,
            step_height_range=(0.08, 0.32),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": mesh.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.14,
            step_height_range=(0.08, 0.32),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": mesh.MeshRandomGridTerrainCfg(
            proportion=0.12,
            grid_width=0.45,
            grid_height_range=(0.08, 0.30),
            platform_width=2.0,
        ),
        "gaps": mesh.MeshGapTerrainCfg(
            proportion=0.10,
            gap_width_range=(0.05, 0.65),
            platform_width=2.0,
        ),
        "stepping_stones": hf.HfSteppingStonesTerrainCfg(
            proportion=0.10,
            stone_height_max=0.10,
            stone_width_range=(0.25, 0.35),
            stone_distance_range=(0.10, 0.25),
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "slope_up": hf.HfPyramidSlopedTerrainCfg(
            proportion=0.07,
            slope_range=(0.0, 0.55),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": hf.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.07,
            slope_range=(0.0, 0.55),
            platform_width=2.0,
            border_width=0.25,
        ),
        "rough": hf.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "wave": hf.HfWaveTerrainCfg(
            proportion=0.11,
            amplitude_range=(0.05, 0.20),
            num_waves=5,
            border_width=0.25,
        ),
    },
)


# ============================================================================
# V54 curriculum — 10 types (rough at slot 8, wave at slot 9)
# ============================================================================
@configclass
class V54CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level_flat = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 0})
    terrain_level_pyramid_stairs = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 1})
    terrain_level_pyramid_stairs_inv = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 2})
    terrain_level_boxes = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 3})
    terrain_level_gaps = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 4})
    terrain_level_stepping_stones = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 5})
    terrain_level_slope_up = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 6})
    terrain_level_slope_down = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 7})
    terrain_level_rough = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 8})
    terrain_level_wave = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 9})
    level_freq_0 = CurrTerm(func=v53_level_freq_logger, params={"level": 0})
    level_freq_1 = CurrTerm(func=v53_level_freq_logger, params={"level": 1})
    level_freq_2 = CurrTerm(func=v53_level_freq_logger, params={"level": 2})
    level_freq_3 = CurrTerm(func=v53_level_freq_logger, params={"level": 3})
    level_freq_4 = CurrTerm(func=v53_level_freq_logger, params={"level": 4})
    level_freq_5 = CurrTerm(func=v53_level_freq_logger, params={"level": 5})
    level_freq_6 = CurrTerm(func=v53_level_freq_logger, params={"level": 6})
    level_freq_7 = CurrTerm(func=v53_level_freq_logger, params={"level": 7})
    level_freq_8 = CurrTerm(func=v53_level_freq_logger, params={"level": 8})
    level_freq_9 = CurrTerm(func=v53_level_freq_logger, params={"level": 9})
    max_lvl_ever_global = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": -1})
    max_lvl_ever_flat = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 0})
    max_lvl_ever_pyramid_stairs = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 1})
    max_lvl_ever_pyramid_stairs_inv = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 2})
    max_lvl_ever_boxes = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 3})
    max_lvl_ever_gaps = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 4})
    max_lvl_ever_stepping_stones = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 5})
    max_lvl_ever_slope_up = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 6})
    max_lvl_ever_slope_down = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 7})
    max_lvl_ever_rough = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 8})
    max_lvl_ever_wave = CurrTerm(func=v54_max_level_ever_logger, params={"type_idx": 9})


@configclass
class V54TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    base_below_origin_holes = DoneTerm(
        func=base_below_origin_holes,
        params={"threshold": 0.40, "hole_types": HOLE_TERRAIN_TYPES},
    )


@configclass
class G1V54ParkourEnvCfg(G1V53ParkourEnvCfg):
    terminations: V54TerminationsCfg = V54TerminationsCfg()
    curriculum: V54CurriculumCfg = V54CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = V54_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1V54ParkourEnvCfg_PLAY(G1V53ParkourEnvCfg_PLAY):
    terminations: V54TerminationsCfg = V54TerminationsCfg()
    curriculum: V54CurriculumCfg = V54CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = V54_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1V54ParkourPPORunnerCfg(G1V50ParkourPPORunnerCfg):
    experiment_name = "g1_v54_parkour"


def register_v54_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V54-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V54-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v54_parkour:G1V54ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v54_parkour:G1V54ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V54-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v54_parkour:G1V54ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v54_parkour:G1V54ParkourPPORunnerCfg",
        },
    )
