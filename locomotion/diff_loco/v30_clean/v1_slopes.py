"""V30 stage v1 — flat + slope_up + slope_down with 9-level curriculum.

v0 + custom slope-only terrain generator. Stock terrain_levels_vel curriculum.
Forward-only commands. Stock rewards. Stock PPO. Same MLP. Nothing else changes.

Acceptance: mean terrain_lvl ≥ 5.0, base_contact ≤ 8%, err_vel_xy ≤ 0.30 on max.
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


# --- Terrain: flat + slope_up + slope_down, 9 levels ---
# slope_range=(0.0, 0.4) rad. At level 1 ≈ 0.04 rad (~2.3°). At level 9 ≈ 0.4 rad (~23°).
# 9 rows = 9 difficulty levels. 20 cols = within-level variety.
V30_SLOPES_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.4,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.4,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class G1V30v1SlopesEnvCfg(G1RoughEnvCfg):
    """Stock G1Rough + slope-only terrain + forward-only cmds."""

    def __post_init__(self):
        super().__post_init__()

        # --- Terrain: replace ROUGH_TERRAINS_CFG with slope-only ---
        self.scene.terrain.terrain_generator = V30_SLOPES_TERRAINS_CFG
        # Curriculum stock (terrain_levels_vel) — already enabled by G1RoughEnvCfg parent.

        # --- Forward-only commands (same as v0) ---
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1V30v1SlopesEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = V30_SLOPES_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1V30v1PPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v1_slopes"
    max_iterations = 2500


def register_v30_v1() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Slopes-G1-V30v1-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Slopes-G1-V30v1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v1_slopes:G1V30v1SlopesEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v1_slopes:G1V30v1PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Slopes-G1-V30v1-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v1_slopes:G1V30v1SlopesEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v1_slopes:G1V30v1PPORunnerCfg",
        },
    )
