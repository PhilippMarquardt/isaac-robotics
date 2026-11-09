"""Register the G1 Rough locomotion task using a diffusion-policy actor."""
from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
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
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.12, noise_range=(0.02, 0.15), noise_step=0.02, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12, step_height_range=(0.05, 0.28), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12, step_height_range=(0.05, 0.28), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.08, amplitude_range=(0.05, 0.2), num_waves=5, border_width=0.25
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10, obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0), obstacle_height_range=(0.05, 0.2),
            num_obstacles=40, platform_width=2.0, border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.10, stone_height_max=0.0,
            stone_width_range=(0.3, 0.5), stone_distance_range=(0.05, 0.35),
            holes_depth=-1.0, platform_width=2.0, border_width=0.25,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05, gap_width_range=(0.1, 0.4), platform_width=2.0,
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05, pit_depth_range=(0.1, 0.6), platform_width=2.0, double_pit=False,
        ),
    },
)
"""Parkour terrain mix: rough, stairs (up/down), slopes, boxes, waves, discrete
obstacles, stepping stones, gaps, and pits. 10-level curriculum; level 9 is
~28cm stairs + ~40cm gaps + 30cm stone gaps + 60cm pits."""
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.obs import lidar_distance, noisy_height_scan
from diff_loco.rewards import alive_when_stationary, base_height_safe_l2, stalled_penalty
from diff_loco.visual_encoder import dinov3_depth_features
from isaaclab.sensors import RayCasterCameraCfg, TiledCameraCfg


N_ICE_TILES = 4


def _make_ice_tile(i: int) -> RigidObjectCfg:
    """One low-friction kinematic tile — sits on terrain, resets position per episode."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/IceTile_{i}",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.04,
                dynamic_friction=0.02,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.92, 1.0),
                metallic=0.1,
                roughness=0.15,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5 + i * 1.5, 0.0, 0.02)),
    )


@configclass
class G1RoughIcySceneCfg(MySceneCfg):
    """Scene extension — N ice tiles per env, kinematic so they teleport on reset."""

    ice_tile_0: RigidObjectCfg = _make_ice_tile(0)
    ice_tile_1: RigidObjectCfg = _make_ice_tile(1)
    ice_tile_2: RigidObjectCfg = _make_ice_tile(2)
    ice_tile_3: RigidObjectCfg = _make_ice_tile(3)


_SHARED_ALGO = dict(
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


def _make_critic() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


def _make_gaussian_cfg() -> "RslRlMLPModelCfg.GaussianDistributionCfg":
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="GaussianDistribution",
        init_std=1.0,
        std_type="log",  # stores log_std; exp() keeps std strictly positive
    )


@configclass
class G1RoughDiffusionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner cfg swapping the actor MLP for a conditional DDPM."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough_diffusion"
    empirical_normalization = False

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.diffusion_model:DiffusionModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = _make_critic()
    algorithm = RslRlPpoAlgorithmCfg(**_SHARED_ALGO)


@configclass
class G1RoughDiffusionRobustPPORunnerCfg(G1RoughDiffusionPPORunnerCfg):
    """Separate experiment name so the robust run doesn't overwrite the walking checkpoint."""

    experiment_name = "g1_rough_diffusion_robust"


@configclass
class G1RoughDiffusionIcyPPORunnerCfg(G1RoughDiffusionPPORunnerCfg):
    experiment_name = "g1_rough_diffusion_icy"


@configclass
class G1RoughDiffusionRealObsPPORunnerCfg(G1RoughDiffusionPPORunnerCfg):
    experiment_name = "g1_rough_diffusion_real_obs"


@configclass
class G1RoughDiffusionRobustV2PPORunnerCfg(G1RoughDiffusionPPORunnerCfg):
    experiment_name = "g1_rough_diffusion_robust_v2"


@configclass
class G1RoughTinyTFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """V25: 24x24 heightmap ViT + tiny 2-layer cross-attn transformer."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_rough_tinytf_v25"
    empirical_normalization = False

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v25_tinytf:V25TinyTFModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v25_tinytf:V25TinyTFModel",
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(**_SHARED_ALGO)


@configclass
class G1RoughDiffusionRealObsLidarPPORunnerCfg(G1RoughDiffusionPPORunnerCfg):
    experiment_name = "g1_rough_diffusion_real_obs_lidar"


@configclass
class G1RoughRobustEnvCfg(G1RoughEnvCfg):
    """G1 rough env with wider velocity commands + perturbation randomization.

    Re-enables pushes, mass/COM randomization, friction variation, and widens
    the commanded velocity range so the policy actually runs instead of just
    walking at 1 m/s.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- wider velocity commands (running + sideways + sharper turns) ---
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)

        # --- re-enable mid-episode pushes ---
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

        # --- re-enable mass & COM randomization (sim-to-real robustness) ---
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
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )

        # --- bigger impulse on reset (initial-state diversity) ---
        self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (-5.0, 5.0)

        # --- actual friction randomization (the default sets equal min=max) ---
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)


@configclass
class G1RoughRobustV2EnvCfg(G1RoughRobustEnvCfg):
    """Robust cfg + Booster-Gym-style reward to kill the crouched gait.

    Adds six reward terms missing from G1's stock cfg: terrain-aware base
    height, vertical velocity penalty, ang_vel_xy rate penalty, alive bonus,
    joint velocity smoothness, and torso linear-acceleration penalty.
    """

    def __post_init__(self):
        super().__post_init__()

        # 0.74m is G1's nominal standing pelvis height; sensor_cfg makes target
        # relative to the terrain under the robot (rough terrain follows ground).
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={
                "target_height": 0.74,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        # Stock G1 zeroes lin_vel_z_l2; re-enable to stop hopping / bouncing.
        self.rewards.lin_vel_z_l2.weight = -2.0

        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=mdp.ang_vel_xy_l2,
            weight=-0.05,
        )
        # Booster Gym uses 0.025; keep small so it doesn't dominate tracking.
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.025)
        self.rewards.joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-1.0e-4,
        )
        self.rewards.base_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )


@configclass
class G1RoughRobustV2EnvCfg_PLAY(G1RoughRobustV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughTinyTFEnvCfg(G1RoughRobustV2EnvCfg):
    """V25: same robust env + rewards, but 24x24 heightmap (576 pts) for ViT."""

    def __post_init__(self):
        super().__post_init__()
        # 24x24 = 576 cells at 0.1m = 2.3m x 2.3m around torso_link.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=[2.3, 2.3],
        )
        # RobustV2's -5.0 base_height crushed the policy into base_contact=1.0.
        # Relax to a nudge; velocity tracking does the lifting.
        self.rewards.base_height.weight = -1.0


@configclass
class G1RoughTinyTFEnvCfg_PLAY(G1RoughTinyTFEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughRealObsEnvCfg(G1RoughRobustV2EnvCfg):
    """Real-robot-only obs + Booster-Gym-style reward (V2).

    Inherits V2 reward shaping (kills the crouch via terrain-aware base_height)
    and robust randomization (pushes, mass, friction, PD gains). Strips the
    heightmap raycast from the OBSERVATION so the policy trains on the same
    sensor set the real G1 exposes via SDK. The heightmap sensor itself stays
    in the scene (used only by the base_height reward — privileged sim info,
    never observed by the policy, so no sim-to-real cheating).
    """

    def __post_init__(self):
        super().__post_init__()

        # --- strip the heightmap term; policy now blind to terrain ahead ---
        self.observations.policy.height_scan = None

        # --- bump obs noise to match real IMU / encoder / estimator drift ---
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.3, n_max=0.3)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.08, n_max=0.08)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)

        # --- PD gain randomization (real motor stiffness/damping drift ±20%) ---
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


@configclass
class G1RoughLidarSceneCfg(MySceneCfg):
    """Base G1 rough scene + Livox MID-360 style ray-caster on torso_link.

    Livox MID-360 real specs: 360deg H-FOV, -7 to +52deg V-FOV, non-repetitive.
    For RL we downsample to a uniform grid: 8 channels x 18 horiz beams = 144
    rays at 20deg horizontal resolution. Enough to "see" obstacles and terrain
    shape, small enough for the policy to chew each step.
    """

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-7.0, 52.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=20.0,
        ),
        max_distance=10.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class G1RoughRealObsLidarEnvCfg(G1RoughRealObsEnvCfg):
    """Real-robot obs + simulated Livox MID-360 lidar as policy input.

    Drops the privileged heightmap (already done by parent), adds a 144-ray
    lidar distance scan. This matches the G1 standard sensor set: IMU +
    encoders + Livox MID-360. RealSense D435 depth cam is deliberately
    deferred — it'd need an encoder net and ~5x the obs dims.
    """

    scene: G1RoughLidarSceneCfg = G1RoughLidarSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # raycaster update rate must match control decimation (mirror height_scanner)
        self.scene.lidar.update_period = self.decimation * self.sim.dt

        self.observations.policy.lidar_scan = ObsTerm(
            func=lidar_distance,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 10.0},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(0.0, 10.0),
        )


@configclass
class G1RoughRealObsLidarEnvCfg_PLAY(G1RoughRealObsLidarEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughRealObsEnvCfg_PLAY(G1RoughRealObsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughIcyEnvCfg(G1RoughRobustEnvCfg):
    """Robust cfg + patches of low-friction ice tiles randomized per reset."""

    scene: G1RoughIcySceneCfg = G1RoughIcySceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        for i in range(N_ICE_TILES):
            setattr(
                self.events,
                f"reset_ice_tile_{i}",
                EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (-3.5, 3.5),
                            "y": (-3.5, 3.5),
                            "yaw": (-3.14, 3.14),
                        },
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg(f"ice_tile_{i}"),
                    },
                ),
            )


@configclass
class G1RoughIcyEnvCfg_PLAY(G1RoughIcyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughRobustEnvCfg_PLAY(G1RoughRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # each env samples a different command — see variety of gaits / speeds
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.observations.policy.enable_corruption = False


@configclass
class G1RoughMLPBaselinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same hyperparams as the diffusion cfg but with a plain MLP actor."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough_mlp_baseline"
    empirical_normalization = False

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = _make_critic()
    algorithm = RslRlPpoAlgorithmCfg(**_SHARED_ALGO)


# =======================================================================
# Unitree-style baseline (mirrors unitreerobotics/unitree_rl_lab G1-29dof)
# =======================================================================
#   - Asymmetric actor-critic: actor sees real-robot obs only; critic adds
#     base_lin_vel (the single privileged term Unitree uses).
#   - Obs history length = 5 frames concatenated in the env (replaces
#     RNN / internal state estimator).
#   - Obs scales baked into ObsTerm (ang_vel x0.2, joint_vel x0.05).
#   - Action scale 0.25 (down from stock 0.5) + default-joint offset.
#   - Stock G1 conservative commands (0..1 m/s fwd, -1..1 rad/s yaw); no
#     pushes, no mass rand. Adds one-to-one sim-to-real contract.


@configclass
class G1UnitreeObservationsCfg:
    """Asymmetric obs: PolicyCfg real-obs-only, CriticCfg adds base_lin_vel."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1UnitreeStyleEnvCfg(G1FlatEnvCfg):
    """G1 locomotion env mirroring Unitree's canonical recipe.

    Base = G1FlatEnvCfg (plane terrain, no height_scanner, stock G1 flat
    rewards already tuned). On top of that we swap in asymmetric obs groups
    with history=5 and bake in obs scales.
    """

    observations: G1UnitreeObservationsCfg = G1UnitreeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- action scale 0.25 (Unitree uses smaller deltas than stock 0.5) ---
        self.actions.joint_pos.scale = 0.25

        # --- rewards: the flat cfg already tunes most terms; add Unitree extras ---
        # Stronger flat_orientation (stock G1 = -1.0; Unitree uses -5.0)
        self.rewards.flat_orientation_l2.weight = -5.0
        # lin_vel_z penalty (flat cfg has -0.2; Unitree uses -2.0)
        self.rewards.lin_vel_z_l2.weight = -2.0
        # ang_vel_xy rate penalty
        self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        # Alive bonus
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        # Tighter joint-position-limits penalty (stock was -1.0)
        self.rewards.dof_pos_limits.weight = -5.0


@configclass
class G1UnitreeStyleEnvCfg_PLAY(G1UnitreeStyleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeHeightmapObservationsCfg(G1UnitreeObservationsCfg):
    """Unitree-style asymmetric obs + 17x11 ego-centric heightmap (187 dims) on both groups."""

    @configclass
    class PolicyCfg(G1UnitreeObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    @configclass
    class CriticCfg(G1UnitreeObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1UnitreeStyleHardEnvCfg(G1UnitreeStyleEnvCfg):
    """Hard terrain + sprint-range commands, blind (same obs spec as flat).

    Inherits flat-trained policy's obs spec (no heightmap) so the checkpoint
    from `g1_unitree_style_diffusion` can resume here. Rough terrain generator
    includes pyramid stairs (up + inverted = pits), slopes, random rough, boxes.
    Terrain-level curriculum lifts robots from easy → hard as they succeed.
    """

    def __post_init__(self):
        super().__post_init__()
        # Re-enable generator-based terrain (parent G1FlatEnvCfg set it to plane)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # Put back the terrain-level curriculum stripped by G1FlatEnvCfg
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # Moderate sprint — reachable from 1-m/s flat baseline (softer than initial −1..+3 try)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1UnitreeStyleHardEnvCfg_PLAY(G1UnitreeStyleHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Small terrain grid for play
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapEnvCfg(G1UnitreeStyleHardEnvCfg):
    """Hard terrain + 17x11 ego-centric heightmap (187 dims) on actor AND critic.

    Modern Rudin-2021 recipe with Agarwal-2023-style noise augmentation:
    Unoise ±0.1m + clip ±1m on the actor's heightmap. Critic sees the clean
    ground truth. Sensor: RayCasterCfg grid 0.1m, 1.6m x 1.0m, mounted on torso.

    CANNOT resume flat ckpt — obs dim changed. Fresh train from scratch.
    """

    observations: G1UnitreeHeightmapObservationsCfg = G1UnitreeHeightmapObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Restore the height_scanner sensor that G1FlatEnvCfg stripped.
        # Pattern matches IsaacLab stock G1RoughEnvCfg: grid on torso, offset Z=20 casting down.
        from isaaclab.sensors import RayCasterCfg, patterns  # local import to keep top clean
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class G1UnitreeStyleHardMapV2EnvCfg(G1UnitreeStyleHardMapEnvCfg):
    """HardMap + anti-stall reward shaping. Resume-compatible (obs dim unchanged).

    Fixes the survive-stand local optimum found in v1:
    - `alive` weight 0.15 → 0.02 (survival bonus no longer dominates tracking)
    - `track_lin_vel_xy_exp` weight x2 (sharper pull toward commanded speed)
    - `stalled` penalty = −1.0 * shortage^2 (direct signal against freezing
      when a command is issued; exp-based tracking is too flat at large err)

    NOTE v2 at these weights destabilized: value_loss exploded, base_contact
    spiked 9→78%. Use v3 cfg instead.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.alive.weight = 0.02
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=-1.0,
            params={"min_cmd": 0.1},
        )


@configclass
class G1UnitreeStyleHardMapV3EnvCfg(G1UnitreeStyleHardMapEnvCfg):
    """V3: softer anti-stall. Keeps rewards away from NaN-precursor regime.

    v2 had stalled=-1, track=2.0 → policy raced forward, fell at 78%, value
    loss hit 115 and climbing. v3 dials back:
    - stalled weight −1.0 → −0.3 (still breaks standing lock-in, less whip)
    - track_lin_vel_xy_exp weight x1.5 (was x2.0)
    - alive stays 0.02 (keep survive-stand attractor dead)
    Combined with runner LR 1e-3→5e-4 and max_grad_norm 1.0→0.5 for stability.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.alive.weight = 0.02
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=-0.3,
            params={"min_cmd": 0.1},
        )


@configclass
class G1UnitreeStyleHardMapV5EnvCfg(G1UnitreeStyleHardMapEnvCfg):
    """V5: heightmap + v1-style alive-bonus base + DELAYED stalled penalty.

    Cold-start lesson from v3/v4 fresh-train attempts: stalled penalty applied
    from step 0 punishes the policy before it can even stand → entropy blows up,
    base_contact stuck at 98%.

    V5 fix: keep v1 rewards (alive=0.15) for the first 1000 iters so policy
    learns to stand + walk. Then at iter 1000, curriculum flips stalled weight
    from 0 to -0.3 to break the survive-stand local optimum.
    num_steps_per_env = 24 → iter 1000 = 24000 common_step_counter.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=0.0,  # disabled until curriculum fires at iter 1000
            params={"min_cmd": 0.1},
        )
        self.curriculum.stalled_curriculum = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "stalled", "weight": -0.15, "num_steps": 24_000},
        )


@configclass
class G1UnitreeStyleHardMapV6EnvCfg(G1UnitreeStyleHardMapEnvCfg):
    """Minimal anti-stall on top of the proven HardMap cfg.

    HardMap base (model_1499 reward -1.36, model_1700 "looks fine") is
    preserved unchanged EXCEPT one term: feet_air_time weight 0.25 → 1.0.
    This is the known SOTA anti-stall signal (Booster Gym, Humanoid-Gym,
    IsaacLab stock H1). Rewards stepping motion directly; zero reward for
    standing still. No other reward change, no hparam change, no σ tweak.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 1.0


@configclass
class G1UnitreeStyleHardMapV7EnvCfg(G1UnitreeStyleHardMapV6EnvCfg):
    """V7: V6 base with `alive` gated to stationary commands only.

    Root cause of V6 stall bimodality: `alive=+0.15/step` collected unconditionally,
    while `feet_air_time` is gated by `|cmd|>0.1` → zero when stalling under walk
    command. Net per-step reward for standing under walk cmd ≈ +0.17, making
    stall locally optimal on flat terrain. Fix: reward `alive` only when
    `|cmd|<0.1` (commanded to stand). When commanded to walk, passive survival
    earns nothing — policy must track or step to gain reward.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.alive = RewTerm(
            func=alive_when_stationary,
            weight=0.15,
            params={"max_cmd": 0.1},
        )


@configclass
class G1UnitreeStyleHardMapV7EnvCfg_PLAY(G1UnitreeStyleHardMapV7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV8EnvCfg(G1UnitreeStyleHardMapV7EnvCfg):
    """V8: resume V7 policy onto a parkour terrain mix — gaps, stepping
    stones, pits, stairs, slopes, waves, boxes, obstacles. Curriculum promotes
    envs that walk far; demotes envs that fall off early."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = PARKOUR_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class G1ParkourV8EnvCfg_PLAY(G1ParkourV8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV12EnvCfg(G1ParkourV8EnvCfg):
    """V12: parkour + mixed-level curriculum init. Starts robots distributed
    across levels so fresh resume doesn't rebuild curriculum from zero."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = None


@configclass
class G1ParkourV12EnvCfg_PLAY(G1ParkourV12EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ProprioPrivilegedObservationsCfg:
    """V13 novel obs: actor = proprio-only hist=10 (deployable on G1 w/o depth);
    critic = proprio + heightmap + base_lin_vel + friction hist=10 (privileged).

    Diffusion K-step reverse denoising serves as implicit recurrent inference
    over hidden terrain state — bypasses explicit LSTM/heightmap in actor.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV13EnvCfg(G1ParkourV8EnvCfg):
    """V13 NOVEL: proprio-only actor + privileged critic + upgraded rewards.

    Paper contribution: Asymmetric Diffusion-PPO with Proprio-Only Deploy.
    - Actor (deployable): 10-step proprio history, NO heightmap. G1 can deploy
      without depth camera / LiDAR. K=4 denoising = implicit terrain inference.
    - Critic (training-only): proprio + heightmap + base_lin_vel, hist=10.
    - Reward upgrade (Booster/Castro/HugWBC consensus): terrain-aware base
      height (-10), lin_vel_z (-2), torso body_lin_acc (-1e-4), stronger
      action_rate (-0.01 -> -0.02 per HugWBC), ang_vel_xy (-0.05).

    Fresh train — obs dim changes from V10's 1325 -> 780 actor, 2680 critic.
    """

    observations: G1ProprioPrivilegedObservationsCfg = G1ProprioPrivilegedObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Terrain-aware base height (primary fix for crouch / low-CoM attractor).
        # Safe version: handles inf ray_hits on pits/gaps, clips err at +/-0.5m.
        self.rewards.base_height = RewTerm(
            func=base_height_safe_l2,
            weight=-10.0,
            params={
                "target_height": 0.74,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "max_err": 0.5,
            },
        )
        # Suppress vertical bouncing (stock G1 zeroes this).
        self.rewards.lin_vel_z_l2.weight = -2.0
        # Torso acceleration smoothness (Booster term).
        self.rewards.body_lin_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
        # Stronger action-rate (HugWBC uses -0.02 vs stock -0.01).
        self.rewards.action_rate_l2.weight = -0.02
        # Ang-vel-xy rate (suppresses pitch/roll oscillation on slopes).
        if not hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)


@configclass
class G1ProprioNoisyHeightmapObservationsCfg:
    """V13.5 obs: actor = proprio + NOISED heightmap hist=10 (models real
    onboard elevation-map sim-to-real gap); critic = proprio + CLEAN heightmap
    + base_lin_vel hist=10 (privileged).

    Novelty vs V13: actor gets heightmap but corrupted (8cm Gaussian + 15%
    dropout + inf-safe). Forces policy to learn noise-robust terrain reading
    that transfers zero-shot to onboard D435i-derived or Livox-derived
    heightmaps with their native per-cell uncertainty.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=noisy_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
                "noise_std": 0.08,
                "dropout_p": 0.15,
                "dropout_fill": 1.0,
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV13_5EnvCfg(G1ParkourV13EnvCfg):
    """V13.5: V13 + noised heightmap in actor for sim-to-real elevation maps.

    Same rewards as V13 (base_height safe, lin_vel_z -2, body_lin_acc, etc).
    Only change: actor obs adds heightmap with 8cm Gaussian noise + 15%
    per-ray dropout. Training resumes from V13 final checkpoint so the
    proprio-only backbone is preserved and only the new height_scan
    channel needs to be integrated.
    """

    observations: G1ProprioNoisyHeightmapObservationsCfg = G1ProprioNoisyHeightmapObservationsCfg()


@configclass
class G1ParkourV13_5EnvCfg_PLAY(G1ParkourV13_5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)


@configclass
class G1ParkourV13EnvCfg_PLAY(G1ParkourV13EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# =============================================================================
# V14 — DINOv3 ConvNeXt-tiny depth features (simple + fast, no cross-attn).
# Mirror-matches Hiking-in-the-Wild camera hardware: stock Unitree G1 D435i
# mounted on head_link, 87°x58° FOV, depth range 0.28-4m, ~60Hz.
# Key difference vs Hiking: our 64x36 depth -> frozen pretrained ConvNeXt
# -> 768-dim pooled vector, concatenated to proprio history. Hiking uses a
# lightweight trained-from-scratch CNN; we use frozen DINOv3 features for
# generalization, trading compute for richer semantics.
# =============================================================================


@configclass
class G1ProprioDinoDepthObservationsCfg:
    """V14 obs: actor = proprio + DINOv3 depth feat hist=10 (deployable w/
    stock D435i); critic = proprio + heightmap + base_lin_vel hist=10
    (privileged, same as V13 for apples-to-apples reward comparison)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        depth_feat = ObsTerm(
            func=dinov3_depth_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV14EnvCfg(G1ParkourV13EnvCfg):
    """V14: V13 + DINOv3 ConvNeXt-tiny depth features in actor obs.

    Scene: adds TiledCamera mimicking factory-integrated Intel D435i on G1
    head_link (FOV 87°x58°, clipping 0.28-4m, 64x36 depth at 50Hz policy
    rate). Camera is rigid (G1 has 0-DoF head).
    Observations: actor gets frozen ConvNeXt-tiny 768-dim pooled depth feat
    concatenated to proprio (hist=10). Critic keeps privileged heightmap.
    Rewards: unchanged from V13 (terrain-aware base_height safe, lin_vel_z,
    body_lin_acc, action_rate, ang_vel_xy).

    Num envs drops 2048 -> 1024 due to camera-rendering compute budget.
    """

    observations: G1ProprioDinoDepthObservationsCfg = G1ProprioDinoDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Soften V13 reward deltas — V13 fresh-train hit stand-still local
        # optimum (action_rate accumulated -1.12/ep dominated track_lin_vel
        # 0.20). Dial back to V10-era magnitudes.
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.base_height.weight = -5.0
        # Intel D435i depth camera mounted on G1 head_link (stock factory
        # configuration, same as Hiking-in-the-Wild — no hardware mod).
        # FOV: 87° H x 58° V (D435i spec ±3°).
        #   HFOV with aperture=20.955mm, focal=11.04mm:
        #     2*atan(20.955/(2*11.04)) = 87.0° ✓
        #   VFOV at 16:9 aspect (64x36 render): ~56° (within ±3° spec).
        # Mount: stock G1 has head rigidly attached to torso (0 DoF head),
        # camera faces straight forward (no pitch) at ~6cm forward offset
        # from head_link origin.
        # Range: 0.28-4m (D435i sharp depth range).
        # Update rate: 50Hz = policy rate (real D435i runs 60Hz, close).
        # Ray-cast depth camera: no RTX render, no DLSS/annotator pipeline.
        # Much cheaper (no shaders) and gives clean ground-truth depth against
        # the terrain mesh. Noise + dropout is injected in the encoder to
        # simulate D435i sensor imperfections.
        self.scene.depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            mesh_prim_paths=["/World/ground"],
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.06, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["distance_to_image_plane"],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=11.04,
                horizontal_aperture=20.955,
                width=64,
                height=36,
            ),
            max_distance=4.0,
            depth_clipping_behavior="max",
            update_period=0.02,
        )


@configclass
class G1ParkourV14EnvCfg_PLAY(G1ParkourV14EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ------------------------------------------------------------------
# V14.1 — tiny end-to-end CNN on depth, split into its own obs group.
# ------------------------------------------------------------------


@configclass
class G1V14CNNObsCfg:
    """V14.1 obs: actor = proprio+heightmap (hist=10, group 'policy') + raw depth
    frame (hist=1, group 'policy_depth'). Depth is a single 18x36 frame fed to a
    tiny CNN inside the actor — PPO gradient flows through the conv weights.
    Group name 'policy' kept for G1FlatEnvCfg parent compatibility (it pokes
    `self.observations.policy.height_scan` in __post_init__).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class PolicyDepthCfg(ObsGroup):
        depth_feat = ObsTerm(
            func=dinov3_depth_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    policy_depth: PolicyDepthCfg = PolicyDepthCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV14CNNEnvCfg(G1ParkourV14EnvCfg):
    """V14.1: V14 scene (raycast D435i depth cam) + split obs groups so a tiny
    depth CNN can train end-to-end inside the actor."""

    observations: G1V14CNNObsCfg = G1V14CNNObsCfg()


@configclass
class G1ParkourV14CNNEnvCfg_PLAY(G1ParkourV14CNNEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.observations.policy_depth.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV9EnvCfg(G1ParkourV8EnvCfg):
    """V9: V8 parkour + harder perturbations. More frequent pushes, wider
    velocity impulses; trains robustness once V8 has learned the terrain."""

    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = (5.0, 8.0)
            self.events.push_robot.params["velocity_range"] = {
                "x": (-0.8, 0.8), "y": (-0.8, 0.8),
            }


@configclass
class G1ParkourV9EnvCfg_PLAY(G1ParkourV9EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapV6EnvCfg_PLAY(G1UnitreeStyleHardMapV6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapV5EnvCfg_PLAY(G1UnitreeStyleHardMapV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapV4EnvCfg(G1UnitreeStyleHardMapV3EnvCfg):
    """V4 YOLO: same rewards as v3, much harder commands + perturbations.

    Wider cmd range (sprint up to 3 m/s, strong lateral, big yaw) and more
    frequent pushes. V3 rewards unchanged — already proven to move without
    crashing. Difficulty boost comes from the task, not reward tuning.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- wider than v3 but still within reach of model_1700's training distribution ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        # --- moderate pushes every 8-15s (not 3-6s; v4-YOLO-v1 at that rate killed value loss) ---
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )


@configclass
class G1UnitreeStyleHardMapV4EnvCfg_PLAY(G1UnitreeStyleHardMapV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapV3EnvCfg_PLAY(G1UnitreeStyleHardMapV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapV2EnvCfg_PLAY(G1UnitreeStyleHardMapV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeStyleHardMapEnvCfg_PLAY(G1UnitreeStyleHardMapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1UnitreeMLPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MLP PPO, asymmetric actor-critic. Baseline recipe from Unitree repo."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_unitree_style_mlp"
    empirical_normalization = False

    # Asymmetric: critic gets extra base_lin_vel obs group
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = _make_critic()
    algorithm = RslRlPpoAlgorithmCfg(**_SHARED_ALGO)


@configclass
class G1UnitreeDiffusionPPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """DDPM actor on the same asymmetric Unitree recipe. Apples-to-apples vs MLP."""

    experiment_name = "g1_unitree_style_diffusion"

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.diffusion_model:DiffusionModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )


@configclass
class G1UnitreeDiffusionHardPPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """Continue diffusion training from flat-trained ckpt on rough terrain + sprint."""

    experiment_name = "g1_unitree_style_diffusion_hard"
    max_iterations = 2000


@configclass
class G1UnitreeDiffusionHardMapPPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """Diffusion + heightmap exteroception on hard terrain. Fresh train (obs dim changed)."""

    experiment_name = "g1_unitree_style_diffusion_hardmap"
    max_iterations = 1500


@configclass
class G1UnitreeDiffusionHardMapV2PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """Resume hardmap ckpt with anti-stall rewards. Separate exp dir preserves v1."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v2"
    max_iterations = 800


_SAFER_ALGO = dict(_SHARED_ALGO)
_SAFER_ALGO["learning_rate"] = 5.0e-4  # half the default 1e-3
_SAFER_ALGO["max_grad_norm"] = 0.5  # half the default 1.0
_SAFER_ALGO["entropy_coef"] = 0.005  # was 0.008 — less entropy blow-up


@configclass
class G1UnitreeDiffusionHardMapV3PPORunnerCfg(G1UnitreeDiffusionHardMapV2PPORunnerCfg):
    """V3 resume: softer reward + tighter LR/grad clip to avoid v2 divergence."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v3"
    max_iterations = 800
    algorithm = RslRlPpoAlgorithmCfg(**_SAFER_ALGO)


@configclass
class G1UnitreeDiffusionHardMapV4PPORunnerCfg(G1UnitreeDiffusionHardMapV3PPORunnerCfg):
    """V4 YOLO: long training with harder task + same stable PPO hparams."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v4"
    max_iterations = 2000


@configclass
class G1UnitreeDiffusionHardMapV5PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V5 fresh train: delayed anti-stall via reward curriculum. Standard LR/grad."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v5"
    max_iterations = 2000


@configclass
class G1UnitreeMLPHardMapV6PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V6 MLP runner — stock hparams on the minimal-change HardMap cfg."""

    experiment_name = "g1_unitree_style_mlp_hardmap_v6"
    max_iterations = 1500


@configclass
class G1UnitreeDiffusionHardMapV6PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V6 diffusion runner — stock hparams on the minimal-change HardMap cfg."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v6"
    max_iterations = 1500


@configclass
class G1UnitreeDiffusionHardMapV7PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V7 diffusion runner — resumes from v6 model_1300 with conditional alive."""

    experiment_name = "g1_unitree_style_diffusion_hardmap_v7"
    max_iterations = 500
    resume = True
    load_run = "resume_from_v6_1300"
    load_checkpoint = "model_1300.pt"


@configclass
class G1ParkourV8PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V8 parkour runner — resumes V7 model_1799 onto parkour terrain mix."""

    experiment_name = "g1_unitree_style_diffusion_parkour_v8"
    max_iterations = 3000
    resume = True
    load_run = "resume_from_v7_1799"
    load_checkpoint = "model_1799.pt"


@configclass
class G1ParkourV9PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V9 parkour runner — resumes V8 4798 onto parkour + heavier pushes."""

    experiment_name = "g1_unitree_style_diffusion_parkour_v9"
    max_iterations = 3000
    resume = True
    load_run = "resume_from_v8_4798"
    load_checkpoint = "model_4798.pt"


@configclass
class G1ParkourV10PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V10 parkour consolidator — resumes V9 7797 with *default* (light) pushes.

    V9 got base_contact 93% because heavy pushes + parkour = constant falls.
    V10 drops push back to default (±0.5 m/s, 10-15s) to let terrain mastery
    consolidate with the V9 policy's sharp velocity tracking intact.
    """

    experiment_name = "g1_unitree_style_diffusion_parkour_v10"
    max_iterations = 2000
    resume = True
    load_run = "resume_from_v9_7797"
    load_checkpoint = "model_7797.pt"


@configclass
class G1ParkourV11PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V11: further consolidate V10 at new peak terrain_level 3.50.
    Same env as V10 (parkour + default pushes). 1500 iters."""

    experiment_name = "g1_unitree_style_diffusion_parkour_v11"
    max_iterations = 1500
    resume = True
    load_run = "resume_from_v10_9796"
    load_checkpoint = "model_9796.pt"


@configclass
class G1ParkourV12PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V12 runner — resumes V10/9796 with mixed-level curriculum init."""

    experiment_name = "g1_unitree_style_diffusion_parkour_v12"
    max_iterations = 1500
    resume = True
    load_run = "resume_from_v10_9796"
    load_checkpoint = "model_9796.pt"


@configclass
class G1ParkourV13PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V13 runner — fresh train: proprio-only actor, privileged critic.

    Novel method. Obs dim changes vs V10 so no resume. 3000 iters at 2048 envs
    (~2.5h on RTX 5070 Ti). Experiment name isolates checkpoints.
    """

    experiment_name = "g1_parkour_v13_asymmetric_diffusion"
    max_iterations = 3000
    resume = False


@configclass
class G1ParkourV13_5PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V13.5 runner — fresh train: proprio + noised heightmap actor.

    Actor input dim changes vs V13 (+1870 from heightmap), so no weight
    transfer. 1500 iters at 2048 envs (~75 min). Separate experiment name
    to preserve V13 ckpt for ablation comparison.
    """

    experiment_name = "g1_parkour_v13_5_noisy_heightmap"
    max_iterations = 1500
    resume = False


@configclass
class G1ParkourV14PPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V14 runner — fresh train: proprio + DINOv3 depth features actor.

    Actor input dim 8460 (78x10 proprio + 768x10 depth), fresh train.
    3000 iters at 1024 envs (~3h with camera+ConvNeXt overhead).
    """

    experiment_name = "g1_parkour_v14_dinov3_convnext_depth"
    max_iterations = 3000
    resume = False


@configclass
class RslRlDiffusionActorCNNCfg(RslRlMLPModelCfg):
    """Actor cfg for DiffusionModel with a tiny depth CNN head.

    depth_encoder_cfg keys:
      - group:   obs-group name that carries the flat depth frame (e.g. "policy_depth")
      - shape:   per-frame (C, H, W) — currently (1, 18, 36) for D435i center crop
      - out_dim: CNN feature dim appended to the flat non-depth latent (default 64)
    """

    depth_encoder_cfg: dict | None = None


@configclass
class G1ParkourV14CNNPPORunnerCfg(G1UnitreeDiffusionPPORunnerCfg):
    """V14.1: tiny 2-conv CNN on depth, trained end-to-end inside actor.

    Actor input: non-depth (proprio+heightmap hist=10 = ~2830) + depth_cnn(64).
    Critic: full priv obs hist=10.
    """

    experiment_name = "g1_parkour_v14_1_cnn_depth"
    max_iterations = 3000
    resume = False

    obs_groups = {
        "actor": ["policy", "policy_depth"],
        "critic": ["critic"],
    }

    actor = RslRlDiffusionActorCNNCfg(
        class_name="diff_loco.diffusion_model:DiffusionModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
        depth_encoder_cfg={
            "group": "policy_depth",
            "shape": (1, 18, 36),
            "out_dim": 64,
        },
    )


# =============================================================================
# V15: small MLP + cross-attention heightmap encoder.
# Per-term history=5 on proprio only; cmd / base_lin_vel / height have no history.
# Term order in PolicyCfg MUST match v15_model.py EXPECTED_OBS_DIM layout:
#   base_ang_vel(3 h5) | projected_gravity(3 h5) | joint_pos(29 h5) |
#   joint_vel(29 h5) | last_action(29 h5) | velocity_commands(3) |
#   base_lin_vel(3) | height_scan(187) = 658 total.
# =============================================================================


@configclass
class G1V15ObsCfg:
    """V15 obs: one flat 'policy' group, per-term history only on proprio."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action, history_length=5, flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Same term order and histories as PolicyCfg -> identical 658-dim layout.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV15EnvCfg(G1ParkourV13EnvCfg):
    """V15: V13 parkour terrain + V13 rewards + attention-based V15 actor.

    Actor input: 658-dim flat (proprio hist=5 + cmd + base_lin_vel + height 17x11).
    Critic: identical shape, separate weights.
    """

    observations: G1V15ObsCfg = G1V15ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # G1FlatEnvCfg.__post_init__ up the chain nukes observations.policy.height_scan = None.
        # HardMapEnvCfg restores the scene.height_scanner sensor but NOT the obs term.
        # Restore by copying a fresh term from a pristine PolicyCfg instance.
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        import isaaclab.envs.mdp as mdp
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )


@configclass
class G1ParkourV15EnvCfg_PLAY(G1ParkourV15EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# =============================================================================
# V16: Flow-Matching DiT locomotion policy (GR00T-style).
# Inputs: height(17x11) + depth(48x48) + proprio hist=5 + command + base_lin_vel.
# Output: 8-action chunk via 4-step Euler ODE; execute chunk[:,0] per env.step.
# =============================================================================


@configclass
class G1V16ObsCfg:
    """V16 obs: height + raw 48x48 depth + proprio hist=5 + cmd + base_lin_vel."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        depth_48 = ObsTerm(
            func=__import__("diff_loco.v16_depth", fromlist=["raw_depth_48"]).raw_depth_48,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": True,
                "out_h": 32, "out_w": 32,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        depth_48 = ObsTerm(
            func=__import__("diff_loco.v16_depth", fromlist=["raw_depth_48"]).raw_depth_48,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": False,
                "out_h": 32, "out_w": 32,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV16EnvCfg(G1ParkourV13EnvCfg):
    """V16: V13 parkour terrain + V13 rewards + flow-matching DiT actor.

    Actor input: height + depth + proprio hist=5 + cmd + base_lin_vel.
    Output: 8-step action chunk via 4-step Euler ODE; execute chunk[:,0] per env step.
    """

    observations: G1V16ObsCfg = G1V16ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Mount head-link depth camera (matches V14; 48x48 render at 50Hz policy rate).
        self.scene.depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            mesh_prim_paths=["/World/ground"],
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.06, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
            ),
            data_types=["distance_to_image_plane"],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=11.04, horizontal_aperture=20.955, width=32, height=32,
            ),
            max_distance=4.0,
            depth_clipping_behavior="max",
            update_period=0.02,
        )
        # Re-add height_scan on policy if chain nukes it (same fix as V15).
        from isaaclab.managers import SceneEntityCfg as _Sec
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as _U
        from isaaclab.managers import ObservationTermCfg as _Obs
        import isaaclab.envs.mdp as _mdp
        self.observations.policy.height_scan = _Obs(
            func=_mdp.height_scan,
            params={"sensor_cfg": _Sec("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=_U(n_min=-0.1, n_max=0.1),
        )


@configclass
class G1ParkourV16EnvCfg_PLAY(G1ParkourV16EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# =============================================================================
# V18: heightmap-only transformer (no depth camera).
# =============================================================================
@configclass
class G1V18ObsCfg:
    """V18 obs: heightmap + proprio hist=5 + cmd + base_lin_vel. NO depth."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV18EnvCfg(G1ParkourV13EnvCfg):
    """V18: V13 parkour terrain + V13 rewards + height-only transformer actor (no depth)."""

    observations: G1V18ObsCfg = G1V18ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Parent chain nukes observations.policy.height_scan = None; restore it.
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
        import isaaclab.envs.mdp as mdp
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )


@configclass
class G1ParkourV18EnvCfg_PLAY(G1ParkourV18EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV18PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V18: heightmap-only single-pass transformer actor."""

    experiment_name = "g1_parkour_v18_heightonly"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v18_heightonly:V18HeightOnlyModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v18_heightonly:V18HeightOnlyModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


# =============================================================================
# V19: pure ViT (all 25 heightmap tokens, CLS readout) + reward fix.
# =============================================================================
@configclass
class G1ParkourV19EnvCfg(G1ParkourV13EnvCfg):
    """V19: V13 rewards + forward-only cmd + stalled penalty + feet_air_time boost.

    Reward fix kills the stand-still local optimum V18 got stuck in:
    - lin_vel_x forced to (0.5, 1.5): no zero/negative commands during training,
      so exp(-|v-v*|^2) cannot reward standing still.
    - track_lin_vel_xy_exp weight x1.5 (sharper pull toward commanded speed).
    - stalled_penalty -0.3: quadratic shortage penalty when command not tracked.
    - feet_air_time boost to 1.0: rewards stepping motion directly.
    """

    observations: G1V18ObsCfg = G1V18ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Restore height_scan (parent chain nukes it).
        from isaaclab.managers import SceneEntityCfg as _SEC
        from isaaclab.managers import ObservationTermCfg as _ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as _Unoise
        import isaaclab.envs.mdp as _mdp
        self.observations.policy.height_scan = _ObsTerm(
            func=_mdp.height_scan,
            params={"sensor_cfg": _SEC("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=_Unoise(n_min=-0.1, n_max=0.1),
        )
        # Reward fix: force forward, punish stalls, boost tracking + stepping.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=-0.3,
            params={"min_cmd": 0.1},
        )


@configclass
class G1ParkourV19EnvCfg_PLAY(G1ParkourV19EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV19PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V19: pure ViT, all 25 heightmap tokens, CLS readout, out_proj gain=0.5."""

    experiment_name = "g1_parkour_v19_vit"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v19_vit:V19ViTModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v19_vit:V19ViTModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


# =============================================================================
# V20: V19 ViT + depth camera + cross-attention readout (no recon loss).
# =============================================================================
@configclass
class G1V20ObsCfg:
    """V20 obs: proprio hist=5 + cmd + base_lin_vel + height(187) + depth(256)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        depth_48 = ObsTerm(
            func=__import__("diff_loco.v16_depth", fromlist=["raw_depth_48"]).raw_depth_48,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": True,
                "out_h": 48, "out_w": 48,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        depth_48 = ObsTerm(
            func=__import__("diff_loco.v16_depth", fromlist=["raw_depth_48"]).raw_depth_48,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "add_noise": False,
                "out_h": 48, "out_w": 48,
            },
        )
        # Privileged (sim-only): append 15 dims for value function. Actor never sees these.
        priv_feet_force_z = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_force_z"]).feet_contact_force_z,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_feet_contact = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_binary"]).feet_contact_binary,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_base_lin_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_lin_acc_b"]).base_lin_acc_b,
        )
        priv_base_ang_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_ang_acc_b"]).base_ang_acc_b,
        )
        priv_external_force = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["external_force_b"]).external_force_b,
        )
        priv_friction = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["friction_coef"]).friction_coef,
        )
        priv_root_height = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["root_height_w"]).root_height_w,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV20EnvCfg(G1ParkourV13EnvCfg):
    """V20: V13 parkour terrain + V19 reward fix + heightmap + 16x16 depth camera."""

    observations: G1V20ObsCfg = G1V20ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Head-link raycast depth camera, 48x48 at 50Hz policy rate
        # (patchified to 16 tokens via Conv2d k=12 s=12 in V20 model).
        self.scene.depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            mesh_prim_paths=["/World/ground"],
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.06, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
            ),
            data_types=["distance_to_image_plane"],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=11.04, horizontal_aperture=20.955, width=48, height=48,
            ),
            max_distance=4.0,
            depth_clipping_behavior="max",
            update_period=0.02,
        )
        # V20 heightmap: match V19 exactly — 1.6m x 1.0m at 0.1m -> 17x11 = 187 cells, centered.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=[1.6, 1.0],
        )
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))
        # Restore height_scan (parent chain nukes it).
        from isaaclab.managers import SceneEntityCfg as _SEC
        from isaaclab.managers import ObservationTermCfg as _ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as _Unoise
        import isaaclab.envs.mdp as _mdp
        self.observations.policy.height_scan = _ObsTerm(
            func=_mdp.height_scan,
            params={"sensor_cfg": _SEC("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=_Unoise(n_min=-0.1, n_max=0.1),
        )
        # V19 reward fix: forward-only cmd, stalled penalty, tracking/step boost.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=-0.3,
            params={"min_cmd": 0.1},
        )


@configclass
class G1ParkourV20EnvCfg_PLAY(G1ParkourV20EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV20PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V20: V19 ViT backbone + depth encoder + cross-attn CLS readout."""

    experiment_name = "g1_parkour_v20_vit_depth"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v20_vit:V20ViTModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v20_vit:V20ViTModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV21EnvCfg(G1ParkourV20EnvCfg):
    """V21 env: identical to V20 (187 heightmap + 48x48 depth). Model differs."""


@configclass
class G1ParkourV21EnvCfg_PLAY(G1ParkourV20EnvCfg_PLAY):
    pass


# =============================================================================
# V22: V19 reward + V20 heightmap encoder + loco cross-attn. NO DEPTH.
# =============================================================================
@configclass
class G1V22ObsCfg:
    """V22 obs: proprio hist=5 + cmd + base_lin_vel + height(187). No depth."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        priv_feet_force_z = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_force_z"]).feet_contact_force_z,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_feet_contact = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_binary"]).feet_contact_binary,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_base_lin_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_lin_acc_b"]).base_lin_acc_b,
        )
        priv_base_ang_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_ang_acc_b"]).base_ang_acc_b,
        )
        priv_external_force = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["external_force_b"]).external_force_b,
        )
        priv_friction = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["friction_coef"]).friction_coef,
        )
        priv_root_height = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["root_height_w"]).root_height_w,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV22EnvCfg(G1ParkourV13EnvCfg):
    """V22: V13 parkour terrain + V19 reward fix + heightmap only (no depth)."""

    observations: G1V22ObsCfg = G1V22ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # V19-matched heightmap: 1.6m x 1.0m at 0.1m -> 17x11 = 187 cells, centered.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=[1.6, 1.0],
        )
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))
        from isaaclab.managers import SceneEntityCfg as _SEC
        from isaaclab.managers import ObservationTermCfg as _ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as _Unoise
        import isaaclab.envs.mdp as _mdp
        self.observations.policy.height_scan = _ObsTerm(
            func=_mdp.height_scan,
            params={"sensor_cfg": _SEC("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=_Unoise(n_min=-0.1, n_max=0.1),
        )
        # V19 reward fix: forward-only cmd, stalled penalty, tracking/step boost.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.stalled = RewTerm(
            func=stalled_penalty,
            weight=-0.3,
            params={"min_cmd": 0.1},
        )


@configclass
class G1ParkourV22EnvCfg_PLAY(G1ParkourV22EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV22PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V22: heightmap-only ViT + loco cross-attn (V20 minus depth)."""

    experiment_name = "g1_parkour_v22_heightattn"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v22_heightattn:V22HeightAttnModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v22_heightattn:V22HeightAttnModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


# =============================================================================
# V23: V22 arch + ByteDance-style forward-biased 21x11 heightmap + 15x30 terrain
# =============================================================================
PARKOUR_TERRAINS_V23_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.15), noise_step=0.02, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.05, 0.28), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.05, 0.28), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.07, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.07, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.08, grid_width=0.45, grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.06, amplitude_range=(0.05, 0.2), num_waves=5, border_width=0.25
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.08, obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.0), obstacle_height_range=(0.05, 0.2),
            num_obstacles=40, platform_width=2.0, border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.08, stone_height_max=0.0,
            stone_width_range=(0.3, 0.5), stone_distance_range=(0.05, 0.35),
            holes_depth=-1.0, platform_width=2.0, border_width=0.25,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05, gap_width_range=(0.1, 0.4), platform_width=2.0,
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05, pit_depth_range=(0.1, 0.6), platform_width=2.0, double_pit=False,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.05, rail_thickness_range=(0.02, 0.06),
            rail_height_range=(0.05, 0.15), platform_width=2.0,
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.05, num_bars=5, bar_width_range=(0.5, 1.0),
            bar_height_range=(0.05, 0.20), platform_width=2.0,
        ),
        "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.06, ring_width_range=(0.4, 1.0),
            ring_height_range=(0.05, 0.30), ring_thickness=0.1, platform_width=2.0,
        ),
    },
)
"""V23 mega-chaos: 15x30=450 patches, 14 terrain types (adds rails/star/floating_ring)."""


@configclass
class G1V23ObsCfg:
    """V23 obs: proprio hist=5 + cmd + base_lin_vel + height(231). No depth."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5, flatten_history_dim=True)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5, flatten_history_dim=True)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=5, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5, flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )
        priv_feet_force_z = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_force_z"]).feet_contact_force_z,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_feet_contact = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["feet_contact_binary"]).feet_contact_binary,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        priv_base_lin_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_lin_acc_b"]).base_lin_acc_b,
        )
        priv_base_ang_acc = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["base_ang_acc_b"]).base_ang_acc_b,
        )
        priv_external_force = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["external_force_b"]).external_force_b,
        )
        priv_friction = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["friction_coef"]).friction_coef,
        )
        priv_root_height = ObsTerm(
            func=__import__("diff_loco.priv_obs", fromlist=["root_height_w"]).root_height_w,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1ParkourV23EnvCfg(G1ParkourV22EnvCfg):
    """V23: mega-chaos terrain + forward-biased 21x11 heightmap."""

    observations: G1V23ObsCfg = G1V23ObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # V13 terrain (fast, proven) — mega-chaos was 15x slower per-iter.
        # ByteDance-style forward-biased heightmap: 2.0m x 1.0m at 0.1m = 21x11 = 231 cells.
        # Offset +0.5m forward -> covers -0.5m behind to +1.5m ahead.
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=[2.0, 1.0],
        )
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0))
        # Bump cmd vel to G1 realistic top speed (~2 m/s).
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.0)


@configclass
class G1ParkourV23EnvCfg_PLAY(G1ParkourV23EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1ParkourV23PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V23: ByteDance-style heightmap + mega-chaos terrain."""

    experiment_name = "g1_parkour_v23_bytedance"
    max_iterations = 1_000_000  # effectively unlimited; stopped manually
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v23_heightattn:V23HeightAttnModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v23_heightattn:V23HeightAttnModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV24PurePPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V24: pure-token transformer, no Conv2d patchify, no cross-attn.

    Same V23 env (21x11 ByteDance heightmap). Every heightmap cell = 1 token
    via Linear(1 -> d_model). One shared self-attn encoder over all tokens.
    """

    experiment_name = "g1_parkour_v24_pure_token"
    max_iterations = 1_000_000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v24_pure_token:V24PureTokenModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v24_pure_token:V24PureTokenModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV23MLPBaselinePPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """Vanilla MLP PPO baseline on the V23 parkour env (same obs, same rewards, same terrain).

    Apples-to-apples comparison: replaces the V23 cross-attn transformer actor/critic with
    a standard [512,256,128] elu MLP. Asymmetric critic uses privileged `critic` obs group.
    """

    experiment_name = "g1_parkour_v23_mlp_baseline"
    max_iterations = 1_000_000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )


@configclass
class G1ParkourV21PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V21: single unified transformer over [height, depth, proprio, state] tokens."""

    experiment_name = "g1_parkour_v21_unified"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v21_unified:V21UnifiedModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v21_unified:V21UnifiedModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV16PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V16 flow-matching DiT policy (4 ODE steps, chunk=8, d_model=128, 4 DiT layers)."""

    experiment_name = "g1_parkour_v16_flow"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v16_flow:V16FlowModel",
        hidden_dims=[256, 128],  # unused for actor (DiT); kept for cfg parity
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v16_flow:V16FlowModel",
        hidden_dims=[256, 128],  # critic MLP head over pooled context
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV17PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V17: single-pass transformer actor (d=128, 3 decoder blocks, 4 action queries)."""

    experiment_name = "g1_parkour_v17_transformer"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v17_transformer:V17TransformerModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v17_transformer:V17TransformerModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


@configclass
class G1ParkourV15PPORunnerCfg(G1UnitreeMLPPPORunnerCfg):
    """V15: small MLP [512,256,128] + cross-attn heightmap encoder. ~600K params.

    Uses V15Model for BOTH actor (29-dim tanh*0.25) and critic (1-dim raw).
    obs_groups: both actor and critic see 'policy' group (658-dim identical layout).
    """

    experiment_name = "g1_parkour_v15_attn"
    max_iterations = 3000
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v15_model:V15Model",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v15_model:V15Model",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


# =============================================================================
# V26 MegaLoco — long single-GPU run. V22 obs + upgraded transformer + progressive curriculum.
# =============================================================================
@configclass
class G1V26MegaEnvCfg(G1ParkourV22EnvCfg):
    """V26 MegaLoco env: V22 parkour base + progressive robustness curriculum.

    Staged toughening via CurrTerm:
      iter   0 :  base V22 (no pushes, V22 reward shaping)
      iter 600 :  mild pushes (±0.5 m/s), mass rand enabled at startup
      iter 1500:  full pushes (±1.0 m/s), stalled penalty bumped
      iter 2500:  add ang_vel_xy rate penalty, tighten action_rate

    Mass/friction randomization is startup-only in IsaacLab, so it can't be
    curriculum'd — we enable it from t=0 with moderate range.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-1.5, 3.0),
                "operation": "add",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 14.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )

        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Progressive reward curriculum. num_steps_per_env=24 -> iter N = N*24 steps.
        # Stalled penalty off at start (fresh actor explores), strong at iter 800.
        self.curriculum.v26_stalled = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "stalled", "weight": -0.5, "num_steps": 19_200},
        )
        # Tighten action smoothness once gait is established (iter 1500).
        self.curriculum.v26_action_rate = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate_l2", "weight": -0.02, "num_steps": 36_000},
        )
        # Boost feet-air-time reward at iter 2500 for higher-cadence gait.
        self.curriculum.v26_air_time = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "feet_air_time", "weight": 1.5, "num_steps": 60_000},
        )


@configclass
class G1V26MegaEnvCfg_PLAY(G1V26MegaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1V26MegaMLPPPORunnerCfg(G1ParkourV22PPORunnerCfg):
    """V26 MLP baseline: same env, same obs, plain MLP actor+critic. Speed comparison."""

    experiment_name = "g1_v26_megaloco_mlp_speed"
    max_iterations = 100
    save_interval = 50
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = _make_critic()


@configclass
class G1V26MegaPPORunnerCfg(G1ParkourV22PPORunnerCfg):
    """V26: v26_megaloco transformer actor/critic, 4000 iters, bigger model."""

    experiment_name = "g1_v26_megaloco"
    max_iterations = 4000
    save_interval = 100
    resume = False

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        class_name="diff_loco.v26_megaloco:V26MegaLocoModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        distribution_cfg=_make_gaussian_cfg(),
    )
    critic = RslRlMLPModelCfg(
        class_name="diff_loco.v26_megaloco:V26MegaLocoModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        distribution_cfg=None,
    )


def register_tasks() -> None:
    """Register G1 rough locomotion variants: diffusion actor + MLP baseline."""
    from diff_loco.v28_parkour import register_v28_parkour
    register_v28_parkour()
    from diff_loco.v28b_parkour import register_v28b_parkour
    register_v28b_parkour()
    from diff_loco.v28c_parkour import register_v28c_parkour
    register_v28c_parkour()
    from diff_loco.v28d_parkour import register_v28d_parkour
    register_v28d_parkour()
    from diff_loco.v28e_bundle import register_v28e_parkour
    register_v28e_parkour()
    from diff_loco.v28f_parkour import register_v28f_parkour
    register_v28f_parkour()
    from diff_loco.v28g_parkour import register_v28g_parkour
    register_v28g_parkour()
    from diff_loco.v28h_parkour import register_v28h_parkour
    register_v28h_parkour()
    from diff_loco.v28n_clean import register_v28n_parkour
    register_v28n_parkour()
    from diff_loco.v28o_parkour import register_v28o_parkour
    register_v28o_parkour()
    from diff_loco.v28p_parkour import register_v28p_parkour
    register_v28p_parkour()
    from diff_loco.v28q_parkour import register_v28q_parkour
    register_v28q_parkour()
    from diff_loco.v28r_parkour import register_v28r_parkour
    register_v28r_parkour()
    from diff_loco.v28t_parkour import register_v28t_parkour
    register_v28t_parkour()
    from diff_loco.v28u_parkour import register_v28u_parkour
    register_v28u_parkour()
    from diff_loco.v29_parkour import register_v29_parkour
    register_v29_parkour()
    from diff_loco.v30_clean.v0_flat import register_v30_v0
    register_v30_v0()
    from diff_loco.v30_clean.v1_slopes import register_v30_v1
    register_v30_v1()
    from diff_loco.v30_clean.v2_stairs import register_v30_v2
    register_v30_v2()
    from diff_loco.v30_clean.v3_parkour import register_v30_v3
    register_v30_v3()
    from diff_loco.v30_clean.v4_baseheight import register_v30_v4
    register_v30_v4()
    from diff_loco.v30_clean.v4b_baseheight_full import register_v30_v4b
    register_v30_v4b()
    from diff_loco.v30_parkour import register_v30_parkour
    register_v30_parkour()
    from diff_loco.v30_full import register_v30_full
    register_v30_full()
    from diff_loco.v30_parkour_moe import register_v30_parkour_moe
    register_v30_parkour_moe()
    from diff_loco.v31_parkour import register_v31_parkour
    register_v31_parkour()
    from diff_loco.v30a_parkour import register_v30a_parkour
    register_v30a_parkour()
    from diff_loco.v30b_parkour import register_v30b_parkour
    register_v30b_parkour()
    from diff_loco.g1_stock_parkour_terrain import register_g1_stock_parkour_terrain
    register_g1_stock_parkour_terrain()
    from diff_loco.v30c_parkour import register_v30c_parkour
    register_v30c_parkour()
    from diff_loco.v30d_parkour import register_v30d_parkour
    register_v30d_parkour()
    from diff_loco.v30e_parkour import register_v30e_parkour
    register_v30e_parkour()
    from diff_loco.v30f_parkour import register_v30f_parkour
    register_v30f_parkour()
    from diff_loco.v30g_parkour import register_v30g_parkour
    register_v30g_parkour()
    from diff_loco.v30h_parkour import register_v30h_parkour
    register_v30h_parkour()
    from diff_loco.v30i_parkour import register_v30i_parkour
    register_v30i_parkour()
    from diff_loco.v40_parkour import register_v40_parkour
    register_v40_parkour()
    from diff_loco.vbasev1_parkour import register_vbasev1_parkour
    register_vbasev1_parkour()
    from diff_loco.vbasev2_parkour import register_vbasev2_parkour
    register_vbasev2_parkour()
    from diff_loco.vbasev3_parkour import register_vbasev3_parkour
    register_vbasev3_parkour()
    from diff_loco.v50_parkour import register_v50_parkour
    register_v50_parkour()
    from diff_loco.v51_parkour import register_v51_parkour
    register_v51_parkour()
    from diff_loco.v52_transformer import register_v52_parkour
    register_v52_parkour()
    from diff_loco.v53_parkour import register_v53_parkour
    register_v53_parkour()
    from diff_loco.v54_parkour import register_v54_parkour
    register_v54_parkour()
    from diff_loco.v55_parkour import register_v55_parkour
    register_v55_parkour()
    if "Isaac-Velocity-Rough-G1-Diffusion-v0" in gym.envs.registry:
        return

    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{G1RoughEnvCfg.__module__}:G1RoughEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{G1RoughEnvCfg_PLAY.__module__}:G1RoughEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-MLPBaseline-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{G1RoughEnvCfg.__module__}:G1RoughEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughMLPBaselinePPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-Robust-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRobustEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRobustPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-Robust-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRobustEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRobustPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-Icy-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughIcyEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionIcyPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-Icy-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughIcyEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionIcyPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RobustV2-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRobustV2EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRobustV2PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RobustV2-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRobustV2EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRobustV2PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-TinyTF-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughTinyTFEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughTinyTFPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-TinyTF-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughTinyTFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughTinyTFPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Mega-G1-V26-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1V26MegaEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1V26MegaPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Mega-G1-V26-MLP-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1V26MegaEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1V26MegaMLPPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Mega-G1-V26-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1V26MegaEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1V26MegaPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RealObs-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRealObsEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRealObsPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RealObs-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRealObsEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRealObsPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RealObsLidar-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRealObsLidarEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRealObsLidarPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Rough-G1-Diffusion-RealObsLidar-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1RoughRealObsLidarEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1RoughDiffusionRealObsLidarPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeMLPPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeMLPPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-Hard-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-Hard-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMap-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMap-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV2-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV2EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV2PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV2-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV2EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV2PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV3-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV3EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV3PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV3-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV3EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV3PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV4-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV4EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV4PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV4-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV4EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV4PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV5-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV5EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV5PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV5-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV5EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV5PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV6-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV6EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV6PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV6-MLP-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV6EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeMLPHardMapV6PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV6-MLP-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV6EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeMLPHardMapV6PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV6-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV6EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV6PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV7-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV7EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV7PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-UnitreeStyle-HardMapV7-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1UnitreeStyleHardMapV7EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1UnitreeDiffusionHardMapV7PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V8-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV8PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V8-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV8PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V9-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV9EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV9PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V9-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV9EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV9PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V10-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV10PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V10-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV10PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V11-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV11PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V11-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV8EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV11PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V12-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV12EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV12PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V12-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV12EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV12PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V13-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV13EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV13PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V13-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV13EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV13PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V14-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV14EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV14PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V14-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV14EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV14PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V14-CNN-Diffusion-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV14CNNEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV14CNNPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V14-CNN-Diffusion-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV14CNNEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV14CNNPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V15-Attn-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV15EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV15PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V15-Attn-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV15EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV15PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V16-Flow-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV16EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV16PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V16-Flow-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV16EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV16PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V17-Transformer-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV16EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV17PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V17-Transformer-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV16EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV17PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V18-HeightOnly-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV18EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV18PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V18-HeightOnly-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV18EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV18PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V19-ViT-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV19EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV19PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V19-ViT-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV19EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV19PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V20-ViTDepth-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV20EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV20PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V20-ViTDepth-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV20EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV20PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V21-Unified-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV21EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV21PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V21-Unified-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV21EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV21PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V22-HeightAttn-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV22EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV22PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V22-HeightAttn-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV22EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV22PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V23-ByteDance-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV23PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V23-ByteDance-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV23PPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V23-MLP-Baseline-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV23MLPBaselinePPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V23-MLP-Baseline-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV23MLPBaselinePPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V24-Pure-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV24PurePPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-G1-Parkour-V24-Pure-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.tasks:G1ParkourV23EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.tasks:G1ParkourV24PurePPORunnerCfg",
        },
    )
