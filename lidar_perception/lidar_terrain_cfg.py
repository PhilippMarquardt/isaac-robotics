"""Diverse terrain mix for Stage-3+ lidar dataset collection.

Wider sub-terrain distribution than ROUGH_TERRAINS_CFG, designed to cover the kinds
of structured environments a deployed humanoid actually walks through: stair landings,
warehouse aisles with isolated obstacles, broken sidewalks, ramps, low-clutter rooms.

Proportions sum to 1.0. Sub-terrain parameter ranges are picked to (a) match the
Hoeller-2022 paper's parkour parameter spread and (b) stay traversable by the stock
G1 rough-terrain locomotion policy.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


LIDAR_DIVERSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.30),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.02, 0.10),
            num_waves=3,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.05,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.5),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=10,
            platform_width=1.5,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.05,
            stone_height_max=0.05,
            stone_width_range=(0.4, 0.6),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-0.4,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Wider terrain mix used by Stage-3+ dataset collection.

Replace `env_cfg.scene.terrain.terrain_generator` with this and the existing
`G1RoughLidarCollectCfg` will produce a much more diverse dataset.
"""
