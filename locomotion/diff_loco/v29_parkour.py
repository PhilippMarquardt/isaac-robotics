"""V29 G1 Parkour — V28o + drop alive bonus, resume from V28o ckpt.

V28o was the last good walking policy before the V28p/q/r/s/t series of narrow fixes.
Only flaw: avoid-death / fence-sit attractor — alive bonus +0.15/step makes
standing still net-positive (+150/episode) which beats trying-and-falling
(net -50 with -100 termination penalty).

V29 = V28o, drop alive bonus only. Keep termination penalty (V28u dropped
both + tightened track std and produced HORRIBLE policy when played).

Resume from V28o ckpt (iter 1200) — V28o gait is the asset; just re-shape
the steady-state attractor by removing the standing subsidy.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)


@configclass
class G1V29ParkourEnvCfg(G1V28oParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "alive") and self.rewards.alive is not None:
            self.rewards.alive = None


@configclass
class G1V29ParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "alive") and self.rewards.alive is not None:
            self.rewards.alive = None


@configclass
class G1V29ParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v29_parkour_teacher"


def register_v29_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V29-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V29-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v29_parkour:G1V29ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v29_parkour:G1V29ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V29-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v29_parkour:G1V29ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v29_parkour:G1V29ParkourTeacherPPORunnerCfg",
        },
    )
