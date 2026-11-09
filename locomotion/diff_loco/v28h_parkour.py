"""V28h G1 Parkour — V28f + PPO value-clipping disabled.

Single change vs V28f: ``use_clipped_value_loss`` True -> False.
Keeps hysteresis curriculum, MoE actor, all V28e PPO stabilisation.

Why: Engstrom "Implementation Matters" / CleanRL ablations show that
clipped value loss can suppress large TD errors that arise after curriculum
jumps. V28e plateaus partly because the critic under-fits sudden value
changes when terrain levels rise. Disabling clipping lets the critic
absorb the full TD signal — at the cost of higher value-loss variance.

Plan called for SmoothL1 (Huber, beta=10.0) replacement of MSE value loss
in addition. Deferred — requires rsl_rl source patch (no cfg toggle for
loss family). V28h tests the cleaner single-knob first; if value clipping
alone moves the needle, Huber retry is a free re-run.

Resume target: V28e model_last.pt at iter ~2280, same as V28f / V28g.
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
class G1V28hParkourEnvCfg(G1V28fParkourEnvCfg):
    """V28f env unchanged (PPO knob is runner-side)."""


@configclass
class G1V28hParkourEnvCfg_PLAY(G1V28fParkourEnvCfg_PLAY):
    pass


_V28H_PPO_ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=False,   # V28e: True
    clip_param=0.15,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=2.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.008,
    max_grad_norm=0.5,
)


@configclass
class G1V28hParkourTeacherPPORunnerCfg(G1V28fParkourTeacherPPORunnerCfg):
    """V28h PPO runner: value clipping disabled."""

    experiment_name = "g1_v28h_parkour_teacher"
    algorithm = RslRlPpoAlgorithmCfg(**_V28H_PPO_ALGO)


@configclass
class G1V28hParkourStudentDistillRunnerCfg(G1V28fParkourStudentDistillRunnerCfg):
    experiment_name = "g1_v28h_parkour_student"


def register_v28h_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28h-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28h-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28h-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28h-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28h-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28h_parkour:G1V28hParkourStudentDistillRunnerCfg",
        },
    )
