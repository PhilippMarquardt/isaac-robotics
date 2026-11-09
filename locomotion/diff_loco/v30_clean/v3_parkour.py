"""V30 stage v3 — full parkour: flat + slopes + stairs + boxes + gaps + stepping_stones.

v2 + add boxes (random_grid), gaps, stepping_stones. Add feet_slide penalty bump.
Lower max difficulty vs V28 per design spec, : gap_width cap 0.4 (not 0.5).
Forward-only commands. Stock PPO. Same MLP. = V30 base.

Acceptance: per-type level ≥ 4.0 each, base_contact ≤ 15%.
"""
from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


# 9 levels, 7 sub-terrains balanced. Lower difficulty caps than V28: gaps capped at 0.4m.
V30_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.20),
            platform_width=2.0,
            holes=False,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.075,
            gap_width_range=(0.0, 0.4),
            platform_width=1.0,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.075,
            stone_height_max=0.10,
            stone_width_range=(0.30, 0.45),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-1.0,
            platform_width=1.0,
        ),
    },
)


@configclass
class G1V30v3ParkourEnvCfg(G1RoughEnvCfg):
    """Stock G1Rough + full parkour terrain + forward-only cmds + feet_slide bump."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = V30_PARKOUR_TERRAINS_CFG

        # Design spec: add feet_slide penalty bump as the only extra reward.
        self.rewards.feet_slide.weight = -0.25  # stock G1 = -0.1; bump for parkour balance.

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1V30v3ParkourEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = V30_PARKOUR_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

        self.rewards.feet_slide.weight = -0.25

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1V30v3PPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v3_parkour"
    max_iterations = 4000


def register_v30_v3() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30v3-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v3-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v3_parkour:G1V30v3ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v3_parkour:G1V30v3PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v3-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v3_parkour:G1V30v3ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v3_parkour:G1V30v3PPORunnerCfg",
        },
    )
