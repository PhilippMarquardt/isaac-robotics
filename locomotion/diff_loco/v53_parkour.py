"""V53 — V51 + Boston-Dynamics-style zigzag-plates as 9th sub-terrain.

V51 (V50 stock G1 + 5-frame proprio history) reaches lvl 9 on 6/8 parkour
terrains by iter ~600. Adding a harder zigzag-plate terrain forces
diagonal precision jumping (alternating L/R angled landing pads with
gaps), the kind of skill the BD Atlas demo shows.

Single change vs V51: terrain mix gets a 9th `zigzag_plates` sub-terrain,
loggers extended to 9 types. Obs/rewards/curriculum mechanic unchanged.

Terrain proportions (renormalized to 1.0):
  flat 0.08, p_stairs 0.16, p_stairs_inv 0.16, boxes 0.13, gaps 0.08,
  stones 0.08, slope_up 0.06, slope_down 0.06, **zigzag 0.20**
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter
from isaaclab.terrains import height_field as hf
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from diff_loco.bd_zigzag_terrain import HfZigzagPlatesTerrainCfg
from diff_loco.v50_parkour import (
    G1V50ParkourEnvCfg,
    G1V50ParkourEnvCfg_PLAY,
    G1V50ParkourPPORunnerCfg,
    V50CurriculumCfg,
    _v50_build_col_to_type,
)
from diff_loco.v51_parkour import G1V51ParkourEnvCfg, G1V51ParkourEnvCfg_PLAY


# ============================================================================
# V53 terrain mix — V50 8 + zigzag plates (9 types total)
# ============================================================================
V53_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
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
        "zigzag_plates": HfZigzagPlatesTerrainCfg(
            proportion=0.21,
            plate_size=(0.7, 1.0),
            plate_offset_y=0.7,
            gap_x_range=(0.20, 0.55),
            angle_range=(0.0, 0.32),
            plate_height_range=(0.05, 0.28),
            num_plates_per_side=3,
            platform_width=2.0,
            holes_depth=-0.5,
        ),
    },
)


# ============================================================================
# V53 loggers — extend V50 loggers to handle 9 types (slot 9 = global)
# ============================================================================
def v53_terrain_level_per_type_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(env, "_v53_col_to_type"):
        env._v53_col_to_type = _v50_build_col_to_type(env)
    env_type = env._v53_col_to_type[terrain.terrain_types]
    mask = env_type == type_idx
    if not bool(mask.any()):
        return torch.tensor(0.0, device=terrain.terrain_levels.device)
    return terrain.terrain_levels[mask].float().mean()


def v53_level_freq_logger(
    env,
    env_ids: Sequence[int],
    level: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    return (terrain.terrain_levels == level).float().mean()


def v53_max_level_ever_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int = -1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    dev = terrain.terrain_levels.device
    if not hasattr(env, "_v53_max_lvl_ever"):
        env._v53_max_lvl_ever = torch.zeros(10, device=dev, dtype=torch.long)  # 9 types + global
    if type_idx < 0:
        cur = terrain.terrain_levels.max()
        slot = 9
    else:
        if not hasattr(env, "_v53_col_to_type"):
            env._v53_col_to_type = _v50_build_col_to_type(env)
        env_type = env._v53_col_to_type[terrain.terrain_types]
        mask = env_type == type_idx
        if not bool(mask.any()):
            return env._v53_max_lvl_ever[type_idx].float()
        cur = terrain.terrain_levels[mask].max()
        slot = type_idx
    env._v53_max_lvl_ever[slot] = torch.maximum(env._v53_max_lvl_ever[slot], cur.long())
    return env._v53_max_lvl_ever[slot].float()


# ============================================================================
# V53 curriculum — V50 + zigzag entries (type_idx 8)
# ============================================================================
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class V53CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level_flat = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 0})
    terrain_level_pyramid_stairs = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 1})
    terrain_level_pyramid_stairs_inv = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 2})
    terrain_level_boxes = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 3})
    terrain_level_gaps = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 4})
    terrain_level_stepping_stones = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 5})
    terrain_level_slope_up = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 6})
    terrain_level_slope_down = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 7})
    terrain_level_zigzag = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 8})
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
    max_lvl_ever_global = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": -1})
    max_lvl_ever_flat = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 0})
    max_lvl_ever_pyramid_stairs = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 1})
    max_lvl_ever_pyramid_stairs_inv = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 2})
    max_lvl_ever_boxes = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 3})
    max_lvl_ever_gaps = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 4})
    max_lvl_ever_stepping_stones = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 5})
    max_lvl_ever_slope_up = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 6})
    max_lvl_ever_slope_down = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 7})
    max_lvl_ever_zigzag = CurrTerm(func=v53_max_level_ever_logger, params={"type_idx": 8})


# ============================================================================
# V53 env cfg
# ============================================================================
@configclass
class G1V53ParkourEnvCfg(G1V51ParkourEnvCfg):
    curriculum: V53CurriculumCfg = V53CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # swap terrain mix to V53 (9 types incl. zigzag)
        self.scene.terrain.terrain_generator = V53_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        # bump command range up to 2 m/s along x
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)


@configclass
class G1V53ParkourEnvCfg_PLAY(G1V51ParkourEnvCfg_PLAY):
    curriculum: V53CurriculumCfg = V53CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = V53_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        # PLAY override: hold lin_vel_x at high end so we can see fast running
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)


@configclass
class G1V53ParkourPPORunnerCfg(G1V50ParkourPPORunnerCfg):
    experiment_name = "g1_v53_parkour"


def register_v53_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V53-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V53-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v53_parkour:G1V53ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v53_parkour:G1V53ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V53-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v53_parkour:G1V53ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v53_parkour:G1V53ParkourPPORunnerCfg",
        },
    )
