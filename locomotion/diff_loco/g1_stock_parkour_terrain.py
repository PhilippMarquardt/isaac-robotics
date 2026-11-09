"""Stock G1 + V30 parkour terrain + per-terrain logging.

For apples-to-apples baseline against V30-plain. Same 8 sub-terrains
(flat / pyramid_stairs / pyramid_stairs_inv / boxes / gaps / stepping_stones /
slope_up / slope_down) with V30's gap fix (gap_width_range 0.02-0.4).

Stock G1Rewards / G1RoughPPORunnerCfg / observations / commands all unchanged —
only terrain swapped + curriculum-level per-terrain loggers added.
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)

from diff_loco.v28_curriculums import terrain_level_per_type_logger
from diff_loco.v28_parkour import V28_PARKOUR_TERRAINS_CFG


_PARKOUR_TERRAIN_NAMES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


def _swap_terrain_and_log(self) -> None:
    """Replace stock rough generator with V28 parkour mix + V30 gap fix + per-type loggers."""
    self.scene.terrain.terrain_generator = V28_PARKOUR_TERRAINS_CFG

    gen = self.scene.terrain.terrain_generator
    if gen is not None and "gaps" in gen.sub_terrains:
        old_gap = gen.sub_terrains["gaps"]
        gen.sub_terrains["gaps"] = mesh.MeshGapTerrainCfg(
            proportion=old_gap.proportion,
            gap_width_range=(0.02, 0.4),
            platform_width=old_gap.platform_width,
        )

    for _i, _name in enumerate(_PARKOUR_TERRAIN_NAMES):
        setattr(
            self.curriculum,
            f"terrain_level_{_name}",
            CurrTerm(
                func=terrain_level_per_type_logger,
                params={"type_idx": _i},
            ),
        )


@configclass
class G1StockParkourTerrainEnvCfg(G1RoughEnvCfg):
    """Stock G1 cfg + parkour terrain mix + per-terrain level loggers."""

    def __post_init__(self):
        super().__post_init__()
        _swap_terrain_and_log(self)


@configclass
class G1StockParkourTerrainEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _swap_terrain_and_log(self)


@configclass
class G1StockParkourTerrainPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_stock_parkour_terrain"


def register_g1_stock_parkour_terrain() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Rough-G1-StockParkourTerrain-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Rough-G1-StockParkourTerrain-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.g1_stock_parkour_terrain:G1StockParkourTerrainEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.g1_stock_parkour_terrain:G1StockParkourTerrainPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-StockParkourTerrain-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.g1_stock_parkour_terrain:G1StockParkourTerrainEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.g1_stock_parkour_terrain:G1StockParkourTerrainPPORunnerCfg",
        },
    )
