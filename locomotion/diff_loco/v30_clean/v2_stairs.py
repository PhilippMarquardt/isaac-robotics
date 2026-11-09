"""V30 stage v2 — flat + slopes + pyramid_stairs (up/down).

v1 + add stairs to terrain mix. Stock terrain_levels_vel curriculum.
Forward-only commands. Stock rewards. Stock PPO. Same MLP. Nothing else changes.

Acceptance: per-type level ≥ 5.0 stairs (up AND inv), base_contact ≤ 12%.
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


# Step max 0.20m at level 9 (gentler than IsaacLab default 0.23). Step width 0.3m.
V30_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.225,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.225,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class G1V30v2StairsEnvCfg(G1RoughEnvCfg):
    """Stock G1Rough + slopes+stairs terrain + forward-only cmds."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = V30_STAIRS_TERRAINS_CFG

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1V30v2StairsEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = V30_STAIRS_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1V30v2PPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v2_stairs"
    max_iterations = 3000


def register_v30_v2() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Stairs-G1-V30v2-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Stairs-G1-V30v2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v2_stairs:G1V30v2StairsEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v2_stairs:G1V30v2PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Stairs-G1-V30v2-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v2_stairs:G1V30v2StairsEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v2_stairs:G1V30v2PPORunnerCfg",
        },
    )
