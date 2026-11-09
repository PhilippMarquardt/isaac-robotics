"""VBaseV1 — pure stock G1 + parkour terrain. Nothing else.

Verification baseline. Stock G1RoughEnvCfg machinery (rewards, actions, obs,
DR, terminations, commands, curriculum) is left untouched. Only the terrain
generator is swapped to the V28 parkour mix and per-terrain level loggers
attached so we can inspect uniform progress.

Identical in spirit to task #112 (`Isaac-Velocity-Rough-G1-StockParkourTerrain-v0`)
which reached iter 4000 with mean lvl 5.58 / gaps 4.57 / base_contact 11.1%.
This is the floor we must match before layering ANY iteration learning on
top. After ~1k iters of uniform climbing, future VBaseV{N} variants will
introduce ONE isolated change at a time.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.g1_stock_parkour_terrain import (
    G1StockParkourTerrainEnvCfg,
    G1StockParkourTerrainEnvCfg_PLAY,
    G1StockParkourTerrainPPORunnerCfg,
)


@configclass
class G1VBaseV1ParkourEnvCfg(G1StockParkourTerrainEnvCfg):
    pass


@configclass
class G1VBaseV1ParkourEnvCfg_PLAY(G1StockParkourTerrainEnvCfg_PLAY):
    pass


@configclass
class G1VBaseV1ParkourPPORunnerCfg(G1StockParkourTerrainPPORunnerCfg):
    experiment_name = "g1_vbasev1_parkour"


def register_vbasev1_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-VBaseV1-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev1_parkour:G1VBaseV1ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev1_parkour:G1VBaseV1ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-VBaseV1-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.vbasev1_parkour:G1VBaseV1ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.vbasev1_parkour:G1VBaseV1ParkourPPORunnerCfg",
        },
    )
