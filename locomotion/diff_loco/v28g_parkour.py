"""V28g G1 Parkour — V28f + longer rollout horizon.

Single change vs V28f: ``num_steps_per_env`` 32 → 64 (~2× horizon) and
``gamma`` 0.99 → 0.995 (longer effective discount). Keeps hysteresis
curriculum, MoE actor, all V28e PPO stabilisation.

Why: Andrychowicz 2024 "What Matters in On-Policy RL" — short rollouts
truncate GAE for sparse reward signals. V28e's 32-step rollout @ 50 Hz =
0.64s, shorter than one parkour stride. The critic literally can't see
the consequence of a step-up action before the rollout ends. Doubling
horizon should give the critic a full stride of context, sharpening
advantages on stair/gap terrain where reward materialises 0.5–1s after
foot placement.

Resume target: V28e model_last.pt at iter ~2280, same as V28f.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from diff_loco.v28f_parkour import (
    G1V28fParkourEnvCfg,
    G1V28fParkourEnvCfg_PLAY,
    G1V28fParkourTeacherPPORunnerCfg,
    G1V28fParkourStudentDistillRunnerCfg,
)


@configclass
class G1V28gParkourEnvCfg(G1V28fParkourEnvCfg):
    """V28f env unchanged (all PPO knobs are runner-side)."""


@configclass
class G1V28gParkourEnvCfg_PLAY(G1V28fParkourEnvCfg_PLAY):
    pass


_V28G_PPO_ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.15,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=2.0e-4,
    schedule="adaptive",
    gamma=0.995,        # V28e: 0.99
    lam=0.95,
    desired_kl=0.008,
    max_grad_norm=0.5,
)


@configclass
class G1V28gParkourTeacherPPORunnerCfg(G1V28fParkourTeacherPPORunnerCfg):
    """V28g PPO runner: longer rollout horizon + matching gamma."""

    experiment_name = "g1_v28g_parkour_teacher"
    num_steps_per_env = 64        # V28e: 32
    algorithm = RslRlPpoAlgorithmCfg(**_V28G_PPO_ALGO)


@configclass
class G1V28gParkourStudentDistillRunnerCfg(G1V28fParkourStudentDistillRunnerCfg):
    experiment_name = "g1_v28g_parkour_student"


def register_v28g_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28g-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28g-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28g-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28g-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28g-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28g_parkour:G1V28gParkourStudentDistillRunnerCfg",
        },
    )
