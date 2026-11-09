"""V40 — Fresh reward structure built from V30 audit. No reward inheritance.

V30 lineage carried forward 21 reward terms. Audit (vs V30h iter 1031 telemetry,
machine B 2026-04-26) found:
  - joint_deviation_arms (-0.44) + joint_deviation_fingers (-0.13) = LEAK on
    passive joints (V28 action manager dim 13 = legs+waist only). Removed.
  - dof_torques_l2 + dof_torques_rate + dof_acc_l2 — three correlated
    smoothness penalties; action_rate_l2 alone is the load-bearing one. Drop 3.
  - dof_pos_limits — termination_penalty handles actual limit violations.
  - feet_contact_forces — small noise term. Dropped.
  - base_height_l2 — V30d already zeroed (gaps incompatible). Dropped.
  - joint_deviation_hip — V30c zeroed (turning penalty). Dropped.
  - undesired_contacts — termination + base_contact handle this. Dropped.

V40 keeps only 11 load-bearing terms with V30i-validated weights:
  primary tracking + survival + gait + smoothness + posture (torso only).

V40 inherits V28o ONLY for env machinery (terrain / observations /
curriculum / DR / actions). RewardsCfg is overwritten with a fresh class.
Cold start fresh seed 40.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)


@configclass
class G1V40RewardsCfg:
    """11 load-bearing terms. NO inheritance from G1Rewards/RewardsCfg."""

    # --- primary tracking ---
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # --- survival ---
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # --- gait shaping ---
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # --- smoothness (single term, action_rate carries it) ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- bouncing + orientation ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # --- only active-joint posture term (torso/waist) ---
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class G1V40ParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o env machinery + clean V40 rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = G1V40RewardsCfg()


@configclass
class G1V40ParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = G1V40RewardsCfg()


@configclass
class G1V40ParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v40_parkour_teacher"


def register_v40_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V40-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V40-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v40_parkour:G1V40ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v40_parkour:G1V40ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V40-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v40_parkour:G1V40ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v40_parkour:G1V40ParkourTeacherPPORunnerCfg",
        },
    )
