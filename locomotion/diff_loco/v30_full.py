"""V30-full — Unitree-balanced ground-up cfg (per design spec, 2026-04-26).

Inherits V28o (curriculum + DR + parkour terrain) but rewrites:
  - Architecture: smaller actor/critic [512, 256, 128] (down from 768/1024).
  - Rewards: drop dof_torques_rate, drop feet_contact_forces, soften
    termination -100→-50, weaken dof_torques_l2 -5e-6→-2e-6, weaken
    track_lin_vel_xy_exp 2.0→1.0, add base_height_safe_l2 -5,
    bump joint_deviation_hip -0.1→-1.0.
  - Commands: forward-only (lin_vel_x in (0,1), y=0, yaw=0), rel_standing 0.20,
    resample 10s, kill the 1→2 m/s curriculum ramp.
  - PPO: revert V28e tightening — clip 0.15→0.2, entropy 0.005→0.01,
    LR 2e-4→3e-4, kl 0.008→0.01, max_grad_norm 0.5→1.0, num_steps 32→24.

Hypothesis: V28e PPO tightening + over-rich reward stack starved gradients
and created the fence-sit attractor. Unitree's 15-min defaults + minimal
reward set + Booster-style base_height/hip discipline should beat the
V28o lvl 4 plateau.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from diff_loco.v28e_bundle import _gauss
from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)
from diff_loco.v30_parkour import base_height_safe_l2


@configclass
class G1V30FullParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o env + Unitree-balanced reward + forward-only cmd."""

    def __post_init__(self):
        super().__post_init__()

        # --- Rewards: Unitree-balanced ---
        # Tracking (back to Unitree weights, not V28b 2.0 bump)
        self.rewards.track_lin_vel_xy_exp.weight = 1.0

        # Energy: weaken to V28b baseline
        self.rewards.dof_torques_l2.weight = -2.0e-6

        # Drop V28e/V28d terms entirely (zero weight is the safest "remove")
        if hasattr(self.rewards, "dof_torques_rate"):
            self.rewards.dof_torques_rate.weight = 0.0
        if hasattr(self.rewards, "feet_contact_forces"):
            self.rewards.feet_contact_forces.weight = 0.0

        # Termination penalty: -100 → -50 (risk-aversion fix)
        self.rewards.termination_penalty.weight = -50.0

        # Booster-style discipline
        self.rewards.base_height_l2 = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={"target_height": 0.74},
        )
        self.rewards.joint_deviation_hip.weight = -1.0

        # --- Commands: forward-only, fixed range ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        # Kill the V28e cmd_lin_vel_x ramp — it would override our fixed range.
        if hasattr(self.curriculum, "v28_cmd_lin_vel_x"):
            delattr(self.curriculum, "v28_cmd_lin_vel_x")


@configclass
class G1V30FullParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        if hasattr(self.rewards, "dof_torques_rate"):
            self.rewards.dof_torques_rate.weight = 0.0
        if hasattr(self.rewards, "feet_contact_forces"):
            self.rewards.feet_contact_forces.weight = 0.0
        self.rewards.termination_penalty.weight = -50.0
        self.rewards.base_height_l2 = RewTerm(
            func=base_height_safe_l2,
            weight=-5.0,
            params={"target_height": 0.74},
        )
        self.rewards.joint_deviation_hip.weight = -1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        if hasattr(self.curriculum, "v28_cmd_lin_vel_x"):
            delattr(self.curriculum, "v28_cmd_lin_vel_x")


_V30_FULL_PPO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,           # V28e: 0.15 → reverted
    entropy_coef=0.01,        # V28e: 0.005 → bumped
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3.0e-4,     # V28e: 2e-4 → reverted
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,          # V28e: 0.008 → reverted
    max_grad_norm=1.0,        # V28e: 0.5 → loosened
)


@configclass
class G1V30FullParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v30_full_parkour_teacher"
    num_steps_per_env = 24    # V28e: 32 → smaller rollouts

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(**_V30_FULL_PPO)


def register_v30_full() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V30-Full-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-Full-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_full:G1V30FullParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_full:G1V30FullParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V30-Full-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v30_full:G1V30FullParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v30_full:G1V30FullParkourTeacherPPORunnerCfg",
        },
    )
