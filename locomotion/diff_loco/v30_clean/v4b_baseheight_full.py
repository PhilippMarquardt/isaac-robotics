"""V30 v4b — same Booster recipe at full weights, cold-start with log_std.

Sister run to v4_baseheight (which is warm-start + soft weights):
  v4  : warm from v3-extend, base_height -3,  lin_vel_z -1, scalar std, LR 5e-4
  v4b : cold start,           base_height -10, lin_vel_z -2, log std,    LR 1e-3 (default)

Tests: full Booster weights (more aggressive shaping) vs gentler. Cold-start
sidesteps the std_param/log_std_param state-dict mismatch hit by warm v4.
"""
from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from diff_loco.rewards import base_height_safe_l2
from diff_loco.v30_clean.v3_parkour import (
    G1V30v3ParkourEnvCfg,
    G1V30v3ParkourEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class G1V30v4bBaseHeightFullEnvCfg(G1V30v3ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height = RewTerm(
            func=base_height_safe_l2,
            weight=-10.0,
            params={
                "target_height": 0.74,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "max_err": 0.5,
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.body_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )


@configclass
class G1V30v4bBaseHeightFullEnvCfg_PLAY(G1V30v3ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height = RewTerm(
            func=base_height_safe_l2,
            weight=-10.0,
            params={
                "target_height": 0.74,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "max_err": 0.5,
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.body_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )


@configclass
class G1V30v4bPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v4b_baseheight_full"
    max_iterations = 4000

    def __post_init__(self):
        # log_std avoids the negative-std crash that hits scalar std under
        # heavy reward shock. Cold start so no state_dict mismatch.
        self.policy.noise_std_type = "log"


def register_v30_v4b() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30v4b-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v4b-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v4b_baseheight_full:G1V30v4bBaseHeightFullEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v4b_baseheight_full:G1V30v4bPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v4b-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v4b_baseheight_full:G1V30v4bBaseHeightFullEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v4b_baseheight_full:G1V30v4bPPORunnerCfg",
        },
    )
