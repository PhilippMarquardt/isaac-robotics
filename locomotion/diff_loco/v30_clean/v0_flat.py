"""V30 stage v0 — flat ground walking baseline.

Stock G1RoughEnvCfg with terrain replaced by flat plane. No curriculum.
Forward-only commands. No standing envs. No DR. Stock G1 rewards. Stock PPO.
Plain MLP actor + critic [512, 256, 128].

Acceptance: track velocity well + near-zero falls in <1500 iters.
If it doesn't pass: env wiring is broken — debug here, do NOT add anything.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class G1V30v0FlatEnvCfg(G1RoughEnvCfg):
    """Flat plane, forward-only, stock rewards, no DR."""

    def __post_init__(self):
        super().__post_init__()

        # --- Flat terrain (replace generator with plane) ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Curriculum needs a generator; disable it.
        self.curriculum.terrain_levels = None
        # Height scanner harmless on plane (all zeros) — keep for obs consistency.

        # --- Forward-only commands. No standing. No reverse. No lateral. ---
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1V30v0FlatEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1V30v0PPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v0_flat"
    max_iterations = 1500


def register_v30_v0() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Flat-G1-V30v0-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Flat-G1-V30v0-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v0_flat:G1V30v0FlatEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v0_flat:G1V30v0PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Flat-G1-V30v0-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v0_flat:G1V30v0FlatEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v0_flat:G1V30v0PPORunnerCfg",
        },
    )
