"""V50 — STOCK G1 + parkour terrain mix. FULLY FLAT REIMPLEMENTATION.

Constraint: V50 must be functionally IDENTICAL to upstream IsaacLab
G1RoughEnvCfg, with ONE deviation:
  • terrain swapped from ROUGH_TERRAINS_CFG to a V28-style parkour mix
    (8 sub-terrains: flat, p_stairs, p_stairs_inv, boxes, gaps, stones,
    slope_up, slope_down — gaps width (0.02, 0.4) for cold start) +
    8 per-type curriculum loggers so we can monitor each terrain.

Everything else (rewards, actions, obs, events, commands, terminations,
curriculum mechanic) mirrors upstream `G1RoughEnvCfg` byte-for-byte.

NO imports from any V28*/V30* file.
NO inheritance from `isaaclab_tasks...g1.rough_env_cfg.G1RoughEnvCfg`.
The only ancestor is `ManagerBasedRLEnvCfg` itself.

Why: VBaseV1 (machine B) hits ~5.86 mean lvl by iter ~1000 with stock G1
config; our previous V50 (V30i flavour, asymmetric obs + DR + custom
curriculum + reward overhaul) was at lvl 1.79 at iter 242 — clearly the
config soup is hurting, not helping. Strip everything back to upstream-
G1 exactly, and pure-reimplement so there is no hidden inheritance drift.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.terrains import height_field as hf
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets import G1_MINIMAL_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# V50 parkour terrain — V28 mix with V30 gap fix (0.02, 0.4)
# ============================================================================
V50_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": mesh.MeshPlaneTerrainCfg(proportion=0.10),
        "pyramid_stairs": mesh.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": mesh.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": mesh.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.20),
            platform_width=2.0,
        ),
        "gaps": mesh.MeshGapTerrainCfg(
            proportion=0.10,
            gap_width_range=(0.02, 0.4),
            platform_width=2.0,
        ),
        "stepping_stones": hf.HfSteppingStonesTerrainCfg(
            proportion=0.10,
            stone_height_max=0.05,
            stone_width_range=(0.30, 0.40),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-0.4,
            platform_width=2.0,
        ),
        "slope_up": hf.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.40),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": hf.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.40),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)

_V50_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


# ============================================================================
# Per-terrain level logger (curriculum-term wrapper)
# ============================================================================
def _v50_build_col_to_type(env) -> torch.Tensor:
    import numpy as np
    terrain: TerrainImporter = env.scene.terrain
    gen_cfg = terrain.cfg.terrain_generator
    proportions = np.array([s.proportion for s in gen_cfg.sub_terrains.values()])
    proportions = proportions / proportions.sum()
    cum = np.cumsum(proportions)
    num_cols = gen_cfg.num_cols
    col_to_type = np.zeros(num_cols, dtype=np.int64)
    for c in range(num_cols):
        col_to_type[c] = int(np.min(np.where(c / num_cols + 0.001 < cum)[0]))
    return torch.from_numpy(col_to_type).to(terrain.terrain_levels.device)


def v50_terrain_level_per_type_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(env, "_v50_col_to_type"):
        env._v50_col_to_type = _v50_build_col_to_type(env)
    env_type = env._v50_col_to_type[terrain.terrain_types]
    mask = env_type == type_idx
    if not bool(mask.any()):
        return torch.tensor(0.0, device=terrain.terrain_levels.device)
    return terrain.terrain_levels[mask].float().mean()


def v50_level_freq_logger(
    env,
    env_ids: Sequence[int],
    level: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of envs currently at terrain ``level`` (0..max).

    Random-reset on max means ``terrain_levels`` is a steady-state distribution
    over [0, max]. Frequency at top level (e.g. lvl 9) tells us how many envs
    currently sit at the hardest tier — a much sharper signal than the mean.
    """
    terrain: TerrainImporter = env.scene.terrain
    return (terrain.terrain_levels == level).float().mean()


def v50_max_level_ever_logger(
    env,
    env_ids: Sequence[int],
    type_idx: int = -1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Max ``terrain_levels`` ever reached across training.

    ``type_idx == -1`` => global; otherwise restricted to envs on the given
    sub-terrain type (matches ``v50_terrain_level_per_type_logger`` index).
    Persistent counter on env survives env_ids subsampling — read every step
    and only grows. Resets only when env is rebuilt.
    """
    terrain: TerrainImporter = env.scene.terrain
    dev = terrain.terrain_levels.device
    if not hasattr(env, "_v50_max_lvl_ever"):
        env._v50_max_lvl_ever = torch.zeros(9, device=dev, dtype=torch.long)  # 8 types + global
    if type_idx < 0:
        cur = terrain.terrain_levels.max()
        slot = 8
    else:
        if not hasattr(env, "_v50_col_to_type"):
            env._v50_col_to_type = _v50_build_col_to_type(env)
        env_type = env._v50_col_to_type[terrain.terrain_types]
        mask = env_type == type_idx
        if not bool(mask.any()):
            return env._v50_max_lvl_ever[type_idx].float()
        cur = terrain.terrain_levels[mask].max()
        slot = type_idx
    env._v50_max_lvl_ever[slot] = torch.maximum(env._v50_max_lvl_ever[slot], cur.long())
    return env._v50_max_lvl_ever[slot].float()


# ============================================================================
# Scene
# ============================================================================
@configclass
class V50SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=V50_PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# Commands — stock G1 (lin_vel_x in [0,1], heading-following yaw)
# ============================================================================
@configclass
class V50CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ============================================================================
# Actions — stock JointPositionAction, all joints, scale 0.5, default offset
# ============================================================================
@configclass
class V50ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ============================================================================
# Observations — single policy group, identical to stock
# ============================================================================
@configclass
class V50ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Events — stock G1 set (push_robot, add_base_mass, base_com all None)
# ============================================================================
@configclass
class V50EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


# ============================================================================
# Rewards — STOCK G1Rewards verbatim
# ============================================================================
@configclass
class V50RewardsCfg:
    # Tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # Penalties (stock RewardsCfg defaults, with G1 weight overrides)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # Joint deviation (stock G1)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
        )},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
        )},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_five_joint", ".*_three_joint", ".*_six_joint",
                ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint",
            ],
        )},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


# ============================================================================
# Terminations — stock (time_out + base_contact on torso_link)
# ============================================================================
@configclass
class V50TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
                "threshold": 1.0},
    )


# ============================================================================
# Curriculum — stock terrain_levels_vel + 8 per-type loggers
# ============================================================================
@configclass
class V50CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_level_flat = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 0}
    )
    terrain_level_pyramid_stairs = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 1}
    )
    terrain_level_pyramid_stairs_inv = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 2}
    )
    terrain_level_boxes = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 3}
    )
    terrain_level_gaps = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 4}
    )
    terrain_level_stepping_stones = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 5}
    )
    terrain_level_slope_up = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 6}
    )
    terrain_level_slope_down = CurrTerm(
        func=v50_terrain_level_per_type_logger, params={"type_idx": 7}
    )
    # Per-level frequency — how many envs are at each level NOW.
    # Random-reset on max means terrain_levels is a distribution; freq at lvl 9
    # tells us how many envs currently sit at the hardest tier.
    level_freq_0 = CurrTerm(func=v50_level_freq_logger, params={"level": 0})
    level_freq_1 = CurrTerm(func=v50_level_freq_logger, params={"level": 1})
    level_freq_2 = CurrTerm(func=v50_level_freq_logger, params={"level": 2})
    level_freq_3 = CurrTerm(func=v50_level_freq_logger, params={"level": 3})
    level_freq_4 = CurrTerm(func=v50_level_freq_logger, params={"level": 4})
    level_freq_5 = CurrTerm(func=v50_level_freq_logger, params={"level": 5})
    level_freq_6 = CurrTerm(func=v50_level_freq_logger, params={"level": 6})
    level_freq_7 = CurrTerm(func=v50_level_freq_logger, params={"level": 7})
    level_freq_8 = CurrTerm(func=v50_level_freq_logger, params={"level": 8})
    level_freq_9 = CurrTerm(func=v50_level_freq_logger, params={"level": 9})
    # Max level ever reached — global + per-terrain. Monotone counter that
    # tells us whether ANY env on terrain X has ever cleared lvl 9, regardless
    # of where the steady-state mean sits.
    max_lvl_ever_global = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": -1})
    max_lvl_ever_flat = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 0})
    max_lvl_ever_pyramid_stairs = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 1})
    max_lvl_ever_pyramid_stairs_inv = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 2})
    max_lvl_ever_boxes = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 3})
    max_lvl_ever_gaps = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 4})
    max_lvl_ever_stepping_stones = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 5})
    max_lvl_ever_slope_up = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 6})
    max_lvl_ever_slope_down = CurrTerm(func=v50_max_level_ever_logger, params={"type_idx": 7})


# ============================================================================
# Environment cfg
# ============================================================================
@configclass
class G1V50ParkourEnvCfg(ManagerBasedRLEnvCfg):
    scene: V50SceneCfg = V50SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: V50ObservationsCfg = V50ObservationsCfg()
    actions: V50ActionsCfg = V50ActionsCfg()
    commands: V50CommandsCfg = V50CommandsCfg()
    rewards: V50RewardsCfg = V50RewardsCfg()
    terminations: V50TerminationsCfg = V50TerminationsCfg()
    events: V50EventCfg = V50EventCfg()
    curriculum: V50CurriculumCfg = V50CurriculumCfg()

    def __post_init__(self):
        # general
        self.decimation = 4
        self.episode_length_s = 20.0
        # sim
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # robot
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # curriculum implies generator curriculum on
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1V50ParkourEnvCfg_PLAY(G1V50ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


# ============================================================================
# PPO runner — stock G1RoughPPORunnerCfg verbatim
# ============================================================================
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1V50ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_v50_parkour"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ============================================================================
# Gym registration
# ============================================================================
def register_v50_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V50-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V50-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v50_parkour:G1V50ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v50_parkour:G1V50ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V50-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v50_parkour:G1V50ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v50_parkour:G1V50ParkourPPORunnerCfg",
        },
    )
