"""V28n G1 Parkour — clean baseline: V28f env minus MoE actor.

V28e–V28g policies converged to a deep crouch / duck-walk despite
hysteresis curriculum + reward fixes. Hypothesis: the MoE actor introduced
in V28e is the proximate cause — soft-gated 4-expert routing destabilises
early policy gradient signal, and the policy collapses to a low-CoM
local optimum that survives curriculum demotions.

V28n keeps every fix accumulated through V28f and only reverts the
actor backbone to a plain 4-layer MLP (V28b/V28d shape):

  Inherited from V28f env (G1V28fParkourEnvCfg):
    - V28b parkour terrain + asymmetric obs groups
    - V28c reward rebalance
    - V28d sim-to-real bundle (action latency DR, joint torque rate
      penalty, armature values, torque limits)
    - V28e PPO stabilisation (clip 0.15, lr 2e-4, kl 0.008, ne=32)
    - V28f hysteresis curriculum (K=3) + clip-not-wrap fix in
      v28_curriculums.py

  Reverted vs V28e:
    - actor: plain MLP [768, 512, 256, 128] (was MoEActorModel
      4×[512,256,128] with Linear→4 gate)

  Critic + everything else: unchanged from V28f.

Trained from scratch — no resume from V28e/f/g checkpoints, since the
crouch is baked into those weights.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from diff_loco.v28e_bundle import _V28E_PPO_ALGO, _gauss
from diff_loco.v28f_parkour import (
    G1V28fParkourEnvCfg,
    G1V28fParkourEnvCfg_PLAY,
    G1V28fParkourTeacherPPORunnerCfg,
)


@configclass
class G1V28nParkourEnvCfg(G1V28fParkourEnvCfg):
    """Identical to V28f env. Subclass exists only for clean naming."""


@configclass
class G1V28nParkourEnvCfg_PLAY(G1V28fParkourEnvCfg_PLAY):
    pass


@configclass
class G1V28nParkourTeacherPPORunnerCfg(G1V28fParkourTeacherPPORunnerCfg):
    """V28f PPO runner with the actor swapped from MoE back to plain MLP."""

    experiment_name = "g1_v28n_parkour_teacher"
    num_steps_per_env = 32  # V28e knob, kept

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[768, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
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
    algorithm = RslRlPpoAlgorithmCfg(**_V28E_PPO_ALGO)


def register_v28n_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28n-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28n-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28n_clean:G1V28nParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28n_clean:G1V28nParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28n-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28n_clean:G1V28nParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28n_clean:G1V28nParkourTeacherPPORunnerCfg",
        },
    )
