"""V30 stage v4 — v3 + Booster Gym reward terms to break v3 plateau (lvl ~4).

Adds (Booster Gym recipe):
  - base_height_l2 (weight -10, target 0.74m, terrain-relative via height_scanner)
  - lin_vel_z_l2 (weight -2.0)
  - body_lin_acc_l2 on torso_link (weight -1e-4)

Hypothesis: v3 plateau caused by crouched gait + vertical bouncing. Stock G1Rewards
omits base_height — pelvis sags on parkour terrain, foot clearance suffers,
gaps/stones become unsolvable. Adding terrain-aware base_height target should
unlock lvl 5+.

Acceptance: per-type level ≥ 5.0, base_contact ≤ 12%.
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
    V30_PARKOUR_TERRAINS_CFG,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


@configclass
class G1V30v4BaseHeightEnvCfg(G1V30v3ParkourEnvCfg):
    """v3 + base_height/lin_vel_z/body_lin_acc reward terms."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={
                "target_height": 0.74,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "max_err": 0.5,
            },
        )

        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

        self.rewards.body_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )


@configclass
class G1V30v4BaseHeightEnvCfg_PLAY(G1V30v3ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={
                "target_height": 0.74,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "max_err": 0.5,
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
        self.rewards.body_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )


@configclass
class G1V30v4PPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_v30_v4_baseheight"
    max_iterations = 4000

    def __post_init__(self):
        # base_height with raw mdp.base_height_l2 + height_scanner over gaps/pits
        # returns ±inf (rays miss everything → mean(-inf) → reward=-inf → NaN std).
        # We use base_height_safe_l2 instead, which masks non-finite hits and
        # clips the per-env error to ±0.5m.
        self.policy.noise_std_type = "log"


def register_v30_v4() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30v4-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v4-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v4_baseheight:G1V30v4BaseHeightEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v4_baseheight:G1V30v4PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30v4-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_clean.v4_baseheight:G1V30v4BaseHeightEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_clean.v4_baseheight:G1V30v4PPORunnerCfg",
        },
    )
