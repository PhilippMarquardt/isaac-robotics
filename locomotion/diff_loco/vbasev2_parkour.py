"""VBaseV2 — VBaseV1 + ONE structural change: feet_air_time weight 0.25 -> 1.0.

Audit at iter ~3700 (machine B 3888 / machine E 3698, 2026-04-27 02:50): VBaseV1 plateau
mean lvl 5.6-5.7 since iter 1721 (>2000 iters), stones (4.5-5.1) and
pyramid_stairs_inv (5.3-5.5) are chronic bottlenecks. Reward + b_c still
improving (b_c 4.5-5.6%, best yet) -> quality refining at lvl ceiling.

Hypothesis: stock G1Rewards feet_air_time weight 0.25 is too weak for high-
clearance terrain (stepping stones, raised box edges). Booster Gym 2025 default
is 1.0; Castro 2024 swing-time term is comparably weighted. Bumping to 1.0 is
the canonical ONE-change candidate per stuck protocol.

Everything else identical to VBaseV1 -> stock G1Rewards. machine B keeps VBaseV1
running as control; this runs on machine E from cold (seed 42).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.vbasev1_parkour import (
    G1VBaseV1ParkourEnvCfg,
    G1VBaseV1ParkourEnvCfg_PLAY,
    G1VBaseV1ParkourPPORunnerCfg,
)


@configclass
class G1VBaseV2ParkourEnvCfg(G1VBaseV1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 1.0


@configclass
class G1VBaseV2ParkourEnvCfg_PLAY(G1VBaseV1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 1.0


@configclass
class G1VBaseV2ParkourPPORunnerCfg(G1VBaseV1ParkourPPORunnerCfg):
    experiment_name = "g1_vbasev2_parkour"


def register_vbasev2_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-VBaseV2-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev2_parkour:G1VBaseV2ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev2_parkour:G1VBaseV2ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV2-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev2_parkour:G1VBaseV2ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev2_parkour:G1VBaseV2ParkourPPORunnerCfg",
        },
    )
