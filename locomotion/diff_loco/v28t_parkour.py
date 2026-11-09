"""V28t G1 Parkour — root-cause reward structure fix.

Diagnosis carried over from V28e–V28r failures (deep crouch, fence-sit,
gap collapse). Root cause: track_lin_vel_xy_exp + alive bonus + high
feet_air_time weight create a reward landscape where "stand and shuffle"
beats "commit to motion." All prior V28 patches (curriculum tweaks, MoE,
torque-rate, gap range ping-pong) targeted symptoms not the structure.

Mechanical math at cmd=(1,0,0):
  exp(-||v - cmd||^2 / 0.25^2) at v=(0,0): exp(-1.0/0.0625) = 1e-7  (~0)
  exp(-||v - cmd||^2 / 0.5^2)  at v=(0,0): exp(-1.0/0.25)   = 0.018
  → standing earns ~0.018 × weight 2.0 = 0.037/step ≈ +37/episode
Plus alive +0.15/step × episode 1000 ≈ +150. Total ~+187 for fence-sit.
Termination -100 lets fence-sit win on net.

V28t inherits from V28o (last clean curriculum baseline, before reward
rot). One coherent structural change set:

  | term                  | V28o → V28t        | rationale                  |
  |-----------------------|--------------------|----------------------------|
  | track_lin_vel_xy_exp  | exp(-L2) → projvel | Extreme Parkour world-frame|
  | alive                 | +0.15 → REMOVED    | killed fence-sit subsidy   |
  | feet_air_time         | 1.0 → 0.25         | stock; no duck-walk reward |
  | stall_penalty         | (new) -1.0         | strict stall = neg reward  |
  | gaps terrain          | MeshGap → SteppingStones | multi-gap progressive |
  | entropy_coef          | 0.005 → 0.02       | retain exploration         |
  | desired_kl            | 0.008 → 0.012      | allow more drift           |

Resume from V28o ckpt. V28o gait was working walk; V28t reward changes
will bend it toward committed motion without teaching from zero.
"""
from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import height_field as hf
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)


# ============================================================================
# Custom reward 1: world-frame inner-product velocity tracking.
# ============================================================================
def track_lin_vel_inner_product_world(
    env,
    command_name: str,
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Project body lin-vel onto cmd direction in world frame.

    Standing still while commanded forward: 0
    Moving sideways at cmd speed: 0
    Moving backward: negative (capped at -max_speed)
    Moving forward at cmd speed: +cmd_speed (capped at max_speed)

    World-frame closes the Extreme-Parkour loophole where a base-frame
    tracking term lets the policy turn around obstacles instead of
    crossing them.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (B, 3) base-frame vx, vy, wz

    yaw = asset.data.heading_w
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cmd_world_x = cmd[:, 0] * cos_y - cmd[:, 1] * sin_y
    cmd_world_y = cmd[:, 0] * sin_y + cmd[:, 1] * cos_y
    cmd_norm = torch.sqrt(cmd_world_x.square() + cmd_world_y.square() + 1.0e-6)

    cmd_hat_x = cmd_world_x / cmd_norm
    cmd_hat_y = cmd_world_y / cmd_norm
    cmd_active = (cmd_norm > 0.1).float()

    v_world = asset.data.root_lin_vel_w[:, :2]
    projected = v_world[:, 0] * cmd_hat_x + v_world[:, 1] * cmd_hat_y
    capped = torch.clamp(projected, max=max_speed)
    return capped * cmd_active


# ============================================================================
# Custom reward 2: stall penalty.
# ============================================================================
def stall_penalty(
    env,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.2,
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """-1 per step while commanded but stationary.

    Inner-product reward gives 0 for standing; stall_penalty makes
    standing strictly negative so the gradient pulls toward motion.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :2], dim=-1)
    is_commanded = cmd_norm > cmd_threshold

    speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    is_stalled = speed < speed_threshold

    return -(is_commanded & is_stalled).float()


# ============================================================================
# PPO algorithm with bumped exploration.
# ============================================================================
_V28T_PPO_ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.15,
    entropy_coef=0.02,           # V28e: 0.005 — 4x bump, retain exploration
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=2.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.012,            # V28e: 0.008 — allow more KL before LR drop
    max_grad_norm=0.5,
)


@configclass
class G1V28tParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o env + reward structure fix (no curriculum changes)."""

    def __post_init__(self):
        super().__post_init__()

        # --- 1. Replace exp-of-L2 tracking with world-frame inner product ---
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=track_lin_vel_inner_product_world,
            weight=2.0,
            params={"command_name": "base_velocity", "max_speed": 1.5},
        )

        # --- 2. Drop alive bonus — killed the fence-sit subsidy. ---
        if hasattr(self.rewards, "alive"):
            self.rewards.alive = None

        # --- 3. Revert feet_air_time to stock 0.25 (V28b bumped to 1.0). ---
        self.rewards.feet_air_time.weight = 0.25
        if hasattr(self.curriculum, "v28_air_time"):
            delattr(self.curriculum, "v28_air_time")

        # --- 4. Stall penalty: strict negative for standing while commanded. ---
        self.rewards.stall_penalty = RewTerm(
            func=stall_penalty,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "speed_threshold": 0.2,
                "cmd_threshold": 0.1,
            },
        )

        # --- 5. Replace gaps single-block MeshGap with progressive SteppingStones. ---
        # Single-gap MeshGap = no learning gradient between "stop at edge" and
        # "successfully cross." Multi-gap stepping_stones = continuous skill
        # ramp via difficulty-scaled stone_distance_range.
        gen_cfg = self.scene.terrain.terrain_generator
        if gen_cfg is not None and "gaps" in gen_cfg.sub_terrains:
            old_gap = gen_cfg.sub_terrains["gaps"]
            gen_cfg.sub_terrains["gaps"] = hf.HfSteppingStonesTerrainCfg(
                proportion=old_gap.proportion,
                stone_height_max=0.0,
                stone_width_range=(0.40, 0.60),
                stone_distance_range=(0.05, 0.30),
                holes_depth=-0.5,
                platform_width=old_gap.platform_width,
            )


@configclass
class G1V28tParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28tParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28t_parkour_teacher"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm = RslRlPpoAlgorithmCfg(**_V28T_PPO_ALGO)


def register_v28t_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28t-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28t-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28t_parkour:G1V28tParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28t_parkour:G1V28tParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28t-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28t_parkour:G1V28tParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28t_parkour:G1V28tParkourTeacherPPORunnerCfg",
        },
    )
