"""V28e bundle — single-file overview of the V28e training run.

This file bundles the MoE actor, V28e env cfg overrides, V28e PPO runner cfg,
and task registration in one place so the full training stack can be reviewed
without chasing imports. The env cfg still subclasses V28d (which layers on
V28c → V28b → V28 base) — inlining the full chain would be thousands of
lines. The summary below lists every resolved parameter that matters.

============================================================================
INHERITANCE CHAIN
============================================================================
  v28_parkour.G1V28ParkourEnvCfg                      (base: terrain + obs)
    └── v28b_parkour.G1V28bParkourEnvCfg              (asymmetric critic)
          └── v28c_parkour.G1V28cParkourEnvCfg        (reward rebalance)
                └── v28d_parkour.G1V28dParkourEnvCfg  (sim-to-real DR)
                      └── v28e_parkour (THIS FILE)    (PPO stabilisation + MoE)

============================================================================
RESOLVED TRAINING CONFIG (what actually runs when you launch V28e)
============================================================================

OBSERVATIONS (asymmetric)
  actor  = priv   (proprio + heightmap + privileged terrain hints)
  critic = critic (= priv + foot wrench)

ACTION — DelayedJointPositionAction
  23 joints (legs + torso + arms)
  scale 0.25, offset = default pose
  max_delay_steps = 2  → uniform ∈ {0, 1, 2} policy steps = {0, 20, 40} ms
                        @ 50 Hz policy (resampled per-episode per env)

ACTUATION DR
  Calibrated armature (hip_roll, knee): 0.0251 (Unitree MIMIC)
  Calibrated armature (ankle pitch/roll): 0.0072
  Armature DR: ×[0.8, 1.2] on all leg + torso joints
  PD gain DR: Kp × [0.8, 1.2], Kd × [0.8, 1.2]
  Joint-vel obs noise: U(-1.0, 1.0) (bumped from default to match real encoder)

REWARDS (V28c rebalance, inherited unchanged except where noted)
  ~ tracking
    track_lin_vel_xy_exp         +1.0
    track_ang_vel_z_world_exp    +0.5
    feet_air_time                +0.125  (clipped)
    feet_slide                   -0.1
  ~ pose / style
    flat_orientation_l2          -0.5
    joint_deviation_hip_roll_yaw -0.1
    joint_deviation_arms         -0.1
    joint_deviation_torso        -0.5
  ~ regularisation
    action_rate_l2               (curriculum — see below)
    dof_torques_rate_l2          -2.0e-8   **V28e: softened 5× from -1e-7**
    termination_penalty          -200.0
  ~ safety
    undesired_contacts           -1.0 per offending link

CURRICULUM
  action_rate_l2: ramp to -0.03 over 200_000 steps  **V28e: was -0.05 @ 96k**
  terrain_levels_vel: 0..9, promote if radial > 4 m, demote if cmd·t < 0.5 × 4 m
  cmd_lin_vel_x: ramp max forward 1.0 → 2.0 m/s over 96_000 steps **V28e new**

COMMAND RANGES (**curriculum-ramped**)
  lin_vel_x   = (-0.5, 1.0) initial → (-0.5, 2.0) after 96k common steps
  lin_vel_y   = (-0.3, 0.3)
  ang_vel_z   = (-1.0, 1.0)
  heading     = (-3.14, 3.14)

TERRAIN (TerrainGenerator, curriculum mode)
  size       = (8.0, 8.0)
  num_rows   = 10, num_cols = 20   (during training)
  sub_terrains: parkour mix (stairs up/down, ramps, gaps, boxes)
  difficulty_range = (0.0, 1.0)

NETS (**MoE change is here**)
  actor : soft-gated MoE, K=4 experts, each MLP(512, 256, 128), elu
           router = Linear(in → 4), softmax weights, outputs summed over K
           obs normalization = True
  critic: MLP(1024, 512, 256, 128), elu
           obs normalization = True
  policy distribution: Gaussian, init_std=1.0, std_type="log", learnable

PPO ALGORITHM (**V28e stabilisation**)
  clip_param       = 0.15          (V28d: 0.2)
  learning_rate    = 2.0e-4        (V28d: 3.0e-4; adaptive KL schedule)
  desired_kl       = 0.008         (V28d: 0.01 — earlier LR drop on drift)
  schedule         = "adaptive"
  num_learning_epochs = 5
  num_mini_batches    = 4
  entropy_coef     = 0.005
  value_loss_coef  = 1.0
  use_clipped_value_loss = True
  gamma = 0.99, lam = 0.95
  max_grad_norm = 0.5
  num_steps_per_env = 32           (V28d: 24 — bigger rollouts)

TRAINING LAUNCH
  num_envs = 20480 (machine B, H100)
  max_iterations = 15000
  save_interval = 200

============================================================================
"""
from __future__ import annotations

from dataclasses import field

import torch
import torch.nn as nn

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP

from diff_loco.v28d_parkour import (
    G1V28dParkourEnvCfg,
    G1V28dParkourEnvCfg_PLAY,
    G1V28dParkourTeacherPPORunnerCfg,
    G1V28dParkourStudentDistillRunnerCfg,
)


# ============================================================================
# V28e command-velocity curriculum — ramp max forward speed 1.0 → 2.0 m/s
# ============================================================================
def cmd_lin_vel_x_ramp(env, env_ids, start: float = 1.0, end: float = 2.0, num_steps: int = 96_000) -> float:
    """Linear ramp of lin_vel_x upper bound over common_step_counter.

    Terrain curriculum promotes on distance-covered; at a flat cmd top of
    2.0 m/s from iter 0, terrain outpaces policy skill and reward stalls.
    Easing the top from 1.0 → 2.0 over the first ~3k iters lets the policy
    master easier tiles before the distance bar rises."""
    step = env.common_step_counter
    progress = min(step / num_steps, 1.0) if num_steps > 0 else 1.0
    current_max = start + (end - start) * progress
    term = env.command_manager.get_term("base_velocity")
    term.cfg.ranges.lin_vel_x = (-0.5, float(current_max))
    return float(current_max)


# ============================================================================
# Soft-gated Mixture-of-Experts actor
# ============================================================================
class _SoftMoEBlock(nn.Module):
    """Soft-gated MoE: out(x) = Σ_k softmax(gate(x))_k · E_k(x).

    All K experts are active every forward. Cost is K× a single expert MLP.
    No top-k routing + no load-balance aux loss — simple, avoids collapse
    in the common K ≤ 4 regime.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expert_hidden_dims: list[int],
        num_experts: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [MLP(input_dim, output_dim, expert_hidden_dims, activation) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_w = torch.softmax(self.gate(x), dim=-1)                    # (B, K)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, D)
        return (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)          # (B, D)


class MoEActorModel(MLPModel):
    """MLPModel whose backbone MLP is replaced by a soft-gated MoE.

    Matches MLPModel's shape contract: obs-group handling, normalizer,
    distribution head, JIT/ONNX export paths all inherited. Only the
    backbone MLP (self.mlp) is overwritten.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(32,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        expert_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    ) -> None:
        # Build base with a tiny placeholder backbone — we overwrite self.mlp.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=list(hidden_dims),
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # Pull mlp_output_dim (accounting for stochastic head's 2× stride)
        # from the placeholder's last Linear.
        last_linear = None
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                last_linear = m
        assert last_linear is not None, "placeholder MLP has no nn.Linear"
        mlp_output_dim = last_linear.out_features
        input_dim = self._get_latent_dim()

        self.mlp = _SoftMoEBlock(
            input_dim=input_dim,
            output_dim=mlp_output_dim,
            expert_hidden_dims=list(expert_hidden_dims),
            num_experts=num_experts,
            activation=activation,
        )

        if self.distribution is not None:
            try:
                self.distribution.init_mlp_weights(self.mlp)
            except Exception:
                # HeteroscedasticGaussian's init assumes nn.Sequential[-2]; N/A
                # for MoE. GaussianDistribution (scalar/log, state-independent
                # std, used here) has a no-op init — safe to skip.
                pass


@configclass
class RslRlMoEModelCfg(RslRlMLPModelCfg):
    """Cfg for the MoE actor. `class_name` points to MoEActorModel here."""

    class_name: str = "diff_loco.v28e_bundle:MoEActorModel"

    num_experts: int = 4
    expert_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])


# ============================================================================
# V28e env cfg — V28d with softened torque-rate + longer action-rate ramp
# ============================================================================
@configclass
class G1V28eParkourEnvCfg(G1V28dParkourEnvCfg):
    """V28e training env."""

    def __post_init__(self):
        super().__post_init__()
        # Soften torque-rate penalty 5× (primary culprit of V28d collapse).
        self.rewards.dof_torques_rate.weight = -2.0e-8
        # Gentler action-rate ramp: -0.03 over 200k steps (V28d: -0.05 @ 96k).
        self.curriculum.v28_action_rate.params["weight"] = -0.03
        self.curriculum.v28_action_rate.params["num_steps"] = 200_000
        # Start narrower; ramp top 1.0 → 2.0 m/s over 96k steps (see curriculum).
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.curriculum.v28_cmd_lin_vel_x = CurrTerm(
            func=cmd_lin_vel_x_ramp,
            params={"start": 1.0, "end": 2.0, "num_steps": 96_000},
        )


@configclass
class G1V28eParkourEnvCfg_PLAY(G1V28dParkourEnvCfg_PLAY):
    """V28e play env — inherits training deltas but pins cmd at final 2.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.dof_torques_rate.weight = -2.0e-8
        self.curriculum.v28_action_rate.params["weight"] = -0.03
        self.curriculum.v28_action_rate.params["num_steps"] = 200_000
        # Play: pin at final range, skip the ramp.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.0)


# ============================================================================
# V28e PPO runner cfg — stabilised PPO + MoE actor
# ============================================================================
_V28E_PPO_ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.15,             # V28d: 0.2
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=2.0e-4,        # V28d: 3.0e-4
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.008,            # V28d: 0.01
    max_grad_norm=0.5,
)


def _gauss():
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="GaussianDistribution",
        init_std=1.0,
        std_type="log",
    )


@configclass
class G1V28eParkourTeacherPPORunnerCfg(G1V28dParkourTeacherPPORunnerCfg):
    """V28e PPO runner: MoE actor, V28d critic, PPO stabilisation."""

    experiment_name = "g1_v28e_parkour_teacher"
    num_steps_per_env = 32
    algorithm = RslRlPpoAlgorithmCfg(**_V28E_PPO_ALGO)

    actor = RslRlMoEModelCfg(
        class_name="diff_loco.v28e_bundle:MoEActorModel",
        hidden_dims=[32],            # placeholder; MoEActorModel ignores it
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
        num_experts=4,
        expert_hidden_dims=[512, 256, 128],
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1V28eParkourStudentDistillRunnerCfg(G1V28dParkourStudentDistillRunnerCfg):
    """V28e student distill: same GRU as V28d; teacher is MoE."""

    experiment_name = "g1_v28e_parkour_student"

    teacher = RslRlMoEModelCfg(
        class_name="diff_loco.v28e_bundle:MoEActorModel",
        hidden_dims=[32],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
        num_experts=4,
        expert_hidden_dims=[512, 256, 128],
    )


# ============================================================================
# Gym task registration
# ============================================================================
def register_v28e_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28e-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28e-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28e-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28e-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28e-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28e_bundle:G1V28eParkourStudentDistillRunnerCfg",
        },
    )
