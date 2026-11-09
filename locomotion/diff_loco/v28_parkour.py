"""V28 G1 Parkour — teacher-student whole-body control.

Architecture (AGILE / ANYmal-Parkour / Extreme-Parkour style):
  - Phase 1 (teacher): PPO with privileged observations
    (lin_vel, full-resolution height_scan, contact forces, friction, mass_perturb).
    Actor + critic both MLP over priv obs. Goal: learn oracle policy.
  - Phase 2 (student): DAgger distillation
    Student = GRU over proprio-only obs (no base_lin_vel, short height scan).
    Trained to imitate teacher; deployable on real G1 with only IMU + chest depth.

Lower-body policy only (arms locked at default pose via PD). Waist included.

Residual action: target = q_default + 0.25 · a_policy  (IsaacLab default).

Parkour terrains:
  - stairs up/down (mesh pyramid)
  - stepping stones (hf)
  - gaps
  - boxes (single-step obstacles)
  - random grid (rough)
  - slopes
  - flat (easy bootstrap)

Reference:
  - ANYmal Parkour  (Hoeller et al. 2024)
  - Extreme Parkour (Cheng et al. 2024)
  - HumanoidVerse   (NVIDIA 2025)
  - WBC-AGILE       (NVIDIA 2026)
"""
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import modify_reward_weight as mdp_modify_reward_weight  # noqa: F401
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains import height_field as hf
from isaaclab.terrains import trimesh as mesh
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)

from diff_loco.v28_curriculums import terrain_level_per_type_logger


# ============================================================================
# Parkour terrain generator — mix of obstacles at graded difficulty
# ============================================================================
V28_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
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
            gap_width_range=(0.1, 0.5),
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


# ============================================================================
# Observation groups
# ============================================================================
@configclass
class G1V28ObservationsCfg:
    """Three groups:
      - proprio (student): IMU + joint state + cmd + short height scan (~50 pts)
      - priv    (teacher): proprio + base_lin_vel + full height scan + priv info
      - critic  : priv (alias — critic is always priv-observed)
    """

    @configclass
    class ProprioCfg(ObsGroup):
        # IMU-derivable
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan_short = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_short"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10  # 10 frames of history fed into GRU

    @configclass
    class PrivCfg(ObsGroup):
        """Privileged obs for teacher + critic."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan_full = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    priv: PrivCfg = PrivCfg()
    critic: PrivCfg = PrivCfg()


# ============================================================================
# V28 env cfg
# ============================================================================
@configclass
class G1V28ParkourEnvCfg(G1RoughEnvCfg):
    """V28 Parkour env: G1 teacher-student with priv/proprio split.

    Inherits stock G1RoughEnvCfg (gives us scene, G1_MINIMAL_CFG asset, events,
    rewards, commands). Overrides:
      - terrain → parkour
      - observations → proprio + priv groups
      - height scanners → two (full 1.6m×1.0m for teacher, short 0.8m×0.4m for student)
      - action → lower body + waist only (arms locked at default)
      - rewards → parkour-tuned
    """

    observations: G1V28ObservationsCfg = G1V28ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain ---
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=V28_PARKOUR_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            ),
            debug_vis=False,
        )

        # --- dual height scanners ---
        # Full scanner already exists via parent cfg; rename to match group params.
        # Pattern below covers ~1.6m × 1.0m at 10cm resolution = 160 points.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=(1.6, 1.0)
        )
        # Short scanner: chest-depth-cam proxy, 0.8m × 0.4m = ~35 points.
        self.scene.height_scanner_short = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.8, 0.4)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # --- action: lower body + waist only (arms locked) ---
        self.actions.joint_pos.joint_names = [
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "torso_joint",
        ]
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True  # residual over q_default

        # --- commands ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)

        # --- domain randomization (on top of stock G1) ---
        # mass perturbation
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-2.0, 4.0),
                "operation": "add",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.08, 0.08), "y": (-0.05, 0.05), "z": (-0.03, 0.03)},
            },
        )
        # friction range
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        # random pushes
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 12.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # --- rewards (parkour-tuned) ---
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.termination_penalty.weight = -100.0
        self.rewards.joint_deviation_hip.weight = -0.15
        self.rewards.joint_deviation_arms.weight = -0.2   # stronger — arms locked, penalize PD error
        self.rewards.joint_deviation_torso.weight = -0.2
        if hasattr(self.rewards, "joint_deviation_fingers"):
            self.rewards.joint_deviation_fingers.weight = -0.05

        # base_height with sensor_cfg is unsafe on parkour terrain (rays miss gaps -> inf).
        # Upright posture is already covered by flat_orientation_l2 + joint_deviation_{hip,arms,torso}.
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)

        # --- progressive curriculum (kicks harder DR in later) ---
        # After iter 1000, tighten tracking & action smoothness.
        self.curriculum.v28_action_rate = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate_l2", "weight": -0.02, "num_steps": 24_000},
        )
        self.curriculum.v28_air_time = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "feet_air_time", "weight": 1.5, "num_steps": 48_000},
        )

        # --- per-terrain-type level loggers (read-only) ---
        # Column-to-type mapping follows V28_PARKOUR_TERRAINS_CFG dict order.
        # Logged as Curriculum/terrain_level_<type> in tfevents so we can see
        # whether mean=5.7 is uniform or dominated by easy types.
        for _i, _name in enumerate([
            "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
            "gaps", "stepping_stones", "slope_up", "slope_down",
        ]):
            setattr(
                self.curriculum,
                f"terrain_level_{_name}",
                CurrTerm(
                    func=terrain_level_per_type_logger,
                    params={"type_idx": _i},
                ),
            )


@configclass
class G1V28ParkourEnvCfg_PLAY(G1V28ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = 9
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.proprio.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        from isaaclab.envs import ViewerCfg
        self.viewer = ViewerCfg(
            eye=(5.0, 5.0, 3.5),
            lookat=(0.0, 0.0, 0.8),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
        )

        # Local-play lighting fix: stock sky_light pulls an HDRI from
        # ISAAC_NUCLEUS_DIR (s3 nucleus). On Windows w/o nucleus access the
        # texture URL is broken → DomeLight fails to load → fully dark scene.
        # Override with a textureless DomeLight at high intensity + a
        # DistantLight (sun) so play renders bright.
        from isaaclab.assets import AssetBaseCfg
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=2500.0,
                color=(0.9, 0.95, 1.0),
            ),
        )
        self.scene.distant_light = AssetBaseCfg(
            prim_path="/World/distantLight",
            spawn=sim_utils.DistantLightCfg(
                intensity=3000.0,
                color=(1.0, 0.98, 0.95),
                angle=0.53,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                rot=(0.866, 0.353, 0.0, 0.353),  # ~ sun from upper-front
            ),
        )


# ============================================================================
# Runner configs
# ============================================================================
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

_PPO_ALGO = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=0.5,
)


def _gauss():
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="GaussianDistribution",
        init_std=1.0,
        std_type="log",
    )


@configclass
class G1V28ParkourTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1: privileged-obs teacher via PPO.

    Actor + critic both eat the `priv` obs group (full height scan + lin_vel +
    friction/mass proxy). Feedforward MLP. Proven combo for parkour-style
    locomotion; once this converges we distill into a recurrent proprio-only
    student.
    """

    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 200
    experiment_name = "g1_v28_parkour_teacher"
    empirical_normalization = False

    obs_groups = {"actor": ["priv"], "critic": ["priv"]}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(**_PPO_ALGO)


@configclass
class G1V28ParkourStudentDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 2: DAgger distillation from teacher into recurrent proprio-only student.

    Student actor: GRU(256) over `proprio` obs (history_length=10 already baked in).
    Teacher actor: the trained MLP from Phase 1, loaded by ckpt path.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "g1_v28_parkour_student"
    empirical_normalization = False

    # obs routing: student input = proprio, teacher input = priv (full obs)
    obs_groups = {"student": ["proprio"], "teacher": ["priv"]}

    student = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=0.5,
        distribution_cfg=_gauss(),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_gauss(),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )


# ============================================================================
# Gym task registration
# ============================================================================
def register_v28_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28_parkour:G1V28ParkourStudentDistillRunnerCfg",
        },
    )
