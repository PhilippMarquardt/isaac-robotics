"""V28b G1 Parkour — scaled-up teacher w/ research-backed upgrades (v2 after review).

Changes vs V28:

  1. Symmetry aug: DISABLED. Broken half-mirror (actions mirrored, obs not) would
     collapse policy toward midline. Re-enable only after full obs mirror.
  2. Genuinely asymmetric critic: actor reads `priv` (no contact forces),
     critic reads `critic` = priv + foot_incoming_wrench. Matches Hoeller 2024.
  3. Bigger nets — actor [768,512,256,128], critic [1024,512,256,128].
  4. Adaptive KL — kept from V28 (schedule=adaptive, desired_kl=0.01).
  6. Priv history_length = 0 (teacher has full state; history wastes capacity).
     Proprio history stays 10 for the student.
  7. Action scale 0.25 -> 0.30 (modest bump; 0.40 too twitchy w/o action-delay DR).
     Action-delay DR left as TODO — add in V28c with a buffer wrapper.

Skipped:
  #5 stage-wise curriculum rewards, #8 waypoint commands.
  Symmetry aug (item 1 above) pending full obs mirror.
  Student priv-lin_vel aux regression head — that is Task #54 / V28b-student.

Ref: ANYmal-Parkour (Hoeller 2024), Extreme Parkour (Cheng 2024),
     HumanoidVerse (NVIDIA 2025), AGILE (NVIDIA 2026), PPOaug (Mittal 2024).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)

from diff_loco.v28_parkour import V28_PARKOUR_TERRAINS_CFG


# ============================================================================
# V28b observation cfg — three groups
#   proprio : student view (history 10)
#   priv    : teacher-actor view (no contact forces, no history)
#   critic  : critic view = priv + foot contact forces (asymmetric advantage)
# ============================================================================
@configclass
class G1V28bObservationsCfg:

    @configclass
    class ProprioCfg(ObsGroup):
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
            self.history_length = 10  # student's crutch for missing lin_vel

    @configclass
    class PrivCfg(ObsGroup):
        """Teacher-actor view. Full state, no contact forces, no history."""
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
            self.history_length = 0  # teacher has oracle obs; no history needed

    @configclass
    class CriticCfg(ObsGroup):
        """Critic view = priv + foot incoming wrench. Asymmetric advantage."""
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
        feet_contact_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    proprio: ProprioCfg = ProprioCfg()
    priv: PrivCfg = PrivCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================================
# V28b env cfg
# ============================================================================
@configclass
class G1V28bParkourEnvCfg(G1RoughEnvCfg):
    observations: G1V28bObservationsCfg = G1V28bObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- terrain (same as V28) ---
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
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=(1.6, 1.0)
        )
        self.scene.height_scanner_short = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.8, 0.4)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # --- action: lower body + waist; scale 0.30 (V28 was 0.25) ---
        self.actions.joint_pos.joint_names = [
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "torso_joint",
        ]
        self.actions.joint_pos.scale = 0.30
        self.actions.joint_pos.use_default_offset = True

        # --- commands ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)

        # --- DR (same as V28) ---
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
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 12.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # --- rewards (parkour-tuned, same as V28) ---
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
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.2
        if hasattr(self.rewards, "joint_deviation_fingers"):
            self.rewards.joint_deviation_fingers.weight = -0.05
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)

        # --- progressive curriculum ---
        self.curriculum.v28_action_rate = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate_l2", "weight": -0.02, "num_steps": 24_000},
        )
        self.curriculum.v28_air_time = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "feet_air_time", "weight": 1.5, "num_steps": 48_000},
        )


@configclass
class G1V28bParkourEnvCfg_PLAY(G1V28bParkourEnvCfg):
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
class G1V28bParkourTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """V28b teacher: asymmetric actor/critic + bigger nets. No sym-aug (disabled)."""

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 200
    experiment_name = "g1_v28b_parkour_teacher"
    empirical_normalization = False  # model does its own norm via obs_normalization=True

    # ASYMMETRIC: actor sees priv (no contact forces), critic sees critic (with).
    obs_groups = {"actor": ["priv"], "critic": ["critic"]}

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
    algorithm = RslRlPpoAlgorithmCfg(**_PPO_ALGO)


@configclass
class G1V28bParkourStudentDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """V28b student: same GRU recipe as V28; teacher is V28b MLP.

    NOTE: does NOT yet include priv-lin_vel aux regression head (Task #54).
    Expect fall-rate improvement from V28's 48% mainly via better teacher,
    not from any student-side fix. To truly crack the yaw-tracking gap,
    follow up with V28b-student-aux.
    """

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 200
    experiment_name = "g1_v28b_parkour_student"
    empirical_normalization = False

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
        hidden_dims=[768, 512, 256, 128],
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
def register_v28b_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28b-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28b-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28b-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28b-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28b-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28b_parkour:G1V28bParkourStudentDistillRunnerCfg",
        },
    )
