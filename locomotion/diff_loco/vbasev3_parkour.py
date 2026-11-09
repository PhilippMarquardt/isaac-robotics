"""VBaseV3 — VBaseV2 + ONE structural change: joint_deviation_torso -0.1 -> -0.5.

VBaseV2 final iter 4000 (machine E s42, 2026-04-27 07:15): mean 5.61, stones 5.08
(+0.23 vs V1), gaps 5.91 (+0.51 vs V1), b_c 3.5% (HALVED vs V1 7-9%). BUT
p_stairs_inv collapsed to 5.06 (-0.34 vs V1 ~5.40). Longer feet_air_time
likely makes robot overcommit on descents -> torso pitches forward -> fall.

Hypothesis: 5x bump on torso-yaw deviation penalty enforces upright posture
on descending stairs without disturbing the swing-time gain V2 unlocked on
stones/gaps. Booster Gym 2025 torso/upright weights are typically -0.5 to
-1.0; -0.5 is the conservative midpoint.

Everything else identical to VBaseV2 (feet_air_time stays at 1.0).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.vbasev2_parkour import (
    G1VBaseV2ParkourEnvCfg,
    G1VBaseV2ParkourEnvCfg_PLAY,
    G1VBaseV2ParkourPPORunnerCfg,
)


@configclass
class G1VBaseV3ParkourEnvCfg(G1VBaseV2ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_torso.weight = -0.5


@configclass
class G1VBaseV3ParkourEnvCfg_PLAY(G1VBaseV2ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_torso.weight = -0.5


@configclass
class G1VBaseV3ParkourPPORunnerCfg(G1VBaseV2ParkourPPORunnerCfg):
    experiment_name = "g1_vbasev3_parkour"


def register_vbasev3_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-VBaseV3-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV3-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev3_parkour:G1VBaseV3ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev3_parkour:G1VBaseV3ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV3-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev3_parkour:G1VBaseV3ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev3_parkour:G1VBaseV3ParkourPPORunnerCfg",
        },
    )
