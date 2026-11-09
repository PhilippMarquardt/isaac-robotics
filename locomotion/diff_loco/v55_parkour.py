"""V55 — V54b mix + 2 new terrains + MoE actor, fresh from scratch.

Changes from V54b:
  1. Add 2 new terrains: discrete_obstacles (pillars to dodge) and
     repeated_pyramids (small cones to step around). Both omnidirectional —
     compatible with random-heading commands.
  2. Bump end-of-curriculum difficulty on existing terrains:
       gaps         max width  0.65 → 0.80
       slopes       max slope  0.55 → 0.65
       boxes        max height 0.30 → 0.40
       stairs       max step   0.32 → 0.36
       waves        max amp    0.20 → 0.25
       stones       max dist   0.25 → 0.30
  3. Swap actor MLP→MoE (4 soft-gated experts, [512,256,128] each).
     Hypothesis: experts can specialize per terrain (stairs vs slopes vs
     obstacle-dodging) rather than one MLP averaging across 12 types.
  4. Cold start. No warm transfer from V54b — MoE backbone is structurally
     different from MLP so weights aren't transferable anyway.

Critic stays plain MLP. PPO algo unchanged from V54.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter
from isaaclab.terrains import height_field as hf
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.moe_actor import RslRlMoEModelCfg
from diff_loco.v50_parkour import _v50_build_col_to_type
from diff_loco.v53_parkour import (
    G1V53ParkourEnvCfg,
    G1V53ParkourEnvCfg_PLAY,
    v53_terrain_level_per_type_logger,
    v53_level_freq_logger,
)


# Terrain type indices (must match V55_PARKOUR_TERRAINS_CFG ordering):
# flat=0, p_stairs=1, p_stairs_inv=2, boxes=3, gaps=4, stones=5,
# slope_up=6, slope_down=7, rough=8, wave=9, obstacles=10, pyramids=11.
HOLE_TERRAIN_TYPES: tuple[int, ...] = (4, 5)


def base_below_origin_holes(
    env,
    threshold: float,
    hole_types: tuple[int, ...] = HOLE_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(env, "_v55_col_to_type"):
        env._v55_col_to_type = _v50_build_col_to_type(env)
    env_type = env._v55_col_to_type[terrain.terrain_types]
    on_hole = torch.zeros_like(env_type, dtype=torch.bool)
    for t in hole_types:
        on_hole |= env_type == t
    pelvis_z = asset.data.root_pos_w[:, 2]
    origin_z = terrain.env_origins[:, 2]
    fell = (pelvis_z - origin_z) < -threshold
    return fell & on_hole


def v55_max_level_ever_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int = -1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """13-slot array (12 terrain types + global)."""
    terrain: TerrainImporter = env.scene.terrain
    dev = terrain.terrain_levels.device
    if not hasattr(env, "_v55_max_lvl_ever"):
        env._v55_max_lvl_ever = torch.zeros(13, device=dev, dtype=torch.long)
    if type_idx < 0:
        cur = terrain.terrain_levels.max()
        slot = 12
    else:
        if not hasattr(env, "_v55_col_to_type"):
            env._v55_col_to_type = _v50_build_col_to_type(env)
        env_type = env._v55_col_to_type[terrain.terrain_types]
        mask = env_type == type_idx
        if not bool(mask.any()):
            return env._v55_max_lvl_ever[type_idx].float()
        cur = terrain.terrain_levels[mask].max()
        slot = type_idx
    env._v55_max_lvl_ever[slot] = torch.maximum(env._v55_max_lvl_ever[slot], cur.long())
    return env._v55_max_lvl_ever[slot].float()


# ============================================================================
# V55 terrain mix — 12 types, end-of-curriculum bumped harder
# ============================================================================
V55_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=24,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": mesh.MeshPlaneTerrainCfg(proportion=0.04),
        "pyramid_stairs": mesh.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.08, 0.36),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": mesh.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.08, 0.36),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": mesh.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.08, 0.40),
            platform_width=2.0,
        ),
        "gaps": mesh.MeshGapTerrainCfg(
            proportion=0.08,
            gap_width_range=(0.05, 0.80),
            platform_width=2.0,
        ),
        "stepping_stones": hf.HfSteppingStonesTerrainCfg(
            proportion=0.08,
            stone_height_max=0.10,
            stone_width_range=(0.25, 0.35),
            stone_distance_range=(0.10, 0.30),
            holes_depth=-0.5,
            platform_width=2.0,
        ),
        "slope_up": hf.HfPyramidSlopedTerrainCfg(
            proportion=0.06,
            slope_range=(0.0, 0.65),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": hf.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.06,
            slope_range=(0.0, 0.65),
            platform_width=2.0,
            border_width=0.25,
        ),
        "rough": hf.HfRandomUniformTerrainCfg(
            proportion=0.08,
            noise_range=(0.02, 0.18),
            noise_step=0.02,
            border_width=0.25,
        ),
        "wave": hf.HfWaveTerrainCfg(
            proportion=0.08,
            amplitude_range=(0.05, 0.25),
            num_waves=5,
            border_width=0.25,
        ),
        "discrete_obstacles": hf.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0),
            obstacle_height_range=(0.10, 0.40),
            num_obstacles=30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "repeated_pyramids": mesh.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.12,
            platform_width=2.0,
            object_params_start=mesh.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=20, height=0.05, radius=0.10, max_yx_angle=0.0,
            ),
            object_params_end=mesh.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=40, height=0.30, radius=0.20, max_yx_angle=0.0,
            ),
        ),
    },
)


# ============================================================================
# V55 curriculum — 12 types
# ============================================================================
@configclass
class V55CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level_flat = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 0})
    terrain_level_pyramid_stairs = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 1})
    terrain_level_pyramid_stairs_inv = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 2})
    terrain_level_boxes = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 3})
    terrain_level_gaps = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 4})
    terrain_level_stepping_stones = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 5})
    terrain_level_slope_up = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 6})
    terrain_level_slope_down = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 7})
    terrain_level_rough = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 8})
    terrain_level_wave = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 9})
    terrain_level_obstacles = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 10})
    terrain_level_pyramids = CurrTerm(func=v53_terrain_level_per_type_logger, params={"type_idx": 11})
    level_freq_0 = CurrTerm(func=v53_level_freq_logger, params={"level": 0})
    level_freq_1 = CurrTerm(func=v53_level_freq_logger, params={"level": 1})
    level_freq_2 = CurrTerm(func=v53_level_freq_logger, params={"level": 2})
    level_freq_3 = CurrTerm(func=v53_level_freq_logger, params={"level": 3})
    level_freq_4 = CurrTerm(func=v53_level_freq_logger, params={"level": 4})
    level_freq_5 = CurrTerm(func=v53_level_freq_logger, params={"level": 5})
    level_freq_6 = CurrTerm(func=v53_level_freq_logger, params={"level": 6})
    level_freq_7 = CurrTerm(func=v53_level_freq_logger, params={"level": 7})
    level_freq_8 = CurrTerm(func=v53_level_freq_logger, params={"level": 8})
    level_freq_9 = CurrTerm(func=v53_level_freq_logger, params={"level": 9})
    max_lvl_ever_global = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": -1})
    max_lvl_ever_flat = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 0})
    max_lvl_ever_pyramid_stairs = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 1})
    max_lvl_ever_pyramid_stairs_inv = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 2})
    max_lvl_ever_boxes = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 3})
    max_lvl_ever_gaps = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 4})
    max_lvl_ever_stepping_stones = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 5})
    max_lvl_ever_slope_up = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 6})
    max_lvl_ever_slope_down = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 7})
    max_lvl_ever_rough = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 8})
    max_lvl_ever_wave = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 9})
    max_lvl_ever_obstacles = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 10})
    max_lvl_ever_pyramids = CurrTerm(func=v55_max_level_ever_logger, params={"type_idx": 11})


@configclass
class V55TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    base_below_origin_holes = DoneTerm(
        func=base_below_origin_holes,
        params={"threshold": 0.40, "hole_types": HOLE_TERRAIN_TYPES},
    )


@configclass
class G1V55ParkourEnvCfg(G1V53ParkourEnvCfg):
    terminations: V55TerminationsCfg = V55TerminationsCfg()
    curriculum: V55CurriculumCfg = V55CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = V55_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1V55ParkourEnvCfg_PLAY(G1V53ParkourEnvCfg_PLAY):
    terminations: V55TerminationsCfg = V55TerminationsCfg()
    curriculum: V55CurriculumCfg = V55CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = V55_PARKOUR_TERRAINS_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


# ============================================================================
# V55 PPO runner — MoE actor + plain MLP critic
# ============================================================================
def _gauss_cfg() -> "RslRlMLPModelCfg.GaussianDistributionCfg":
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="GaussianDistribution",
        init_std=1.0,
        std_type="log",
    )


@configclass
class G1V55ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 50
    experiment_name = "g1_v55_parkour"

    actor = RslRlMoEModelCfg(
        class_name="diff_loco.moe_actor:MoEActorModel",
        hidden_dims=[32],  # placeholder — MoEActorModel ignores it
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss_cfg(),
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
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # half V50's 1e-3 — MoE has ~4x more params, lower LR for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ============================================================================
# V55-MLP runner — same V55 env, plain MLP actor (V50/V54 architecture).
# Cold-start comparison vs MoE actor on identical 12-terrain mix.
# ============================================================================
from diff_loco.v50_parkour import G1V50ParkourPPORunnerCfg


@configclass
class G1V55MLPParkourPPORunnerCfg(G1V50ParkourPPORunnerCfg):
    experiment_name = "g1_v55_mlp_parkour"
    max_iterations = 4000


def register_v55_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V55-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V55-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V55-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V55-MLP-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v55_parkour:G1V55MLPParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V55-MLP-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v55_parkour:G1V55ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v55_parkour:G1V55MLPParkourPPORunnerCfg",
        },
    )
