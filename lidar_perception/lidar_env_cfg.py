"""G1 rough env with MID-360 lidar + dense GT downward raycaster for offline dataset collection.

Subclasses the stock G1RoughEnvCfg and adds two RayCaster sensors:

  * `lidar_scanner` — Livox MID-360 proxy, 360° H × asymmetric V (-7°, +52°), 5 m clip,
    10 Hz update, 40 channels × 500 az = 20000 rays. Mounted on `torso_link` with a
    0.3 m Z offset (head height; the real G1 head is rigidly fixed above the torso).
    `ray_alignment="base"` → rays inherit full torso roll/pitch/yaw.

  * `gt_scanner` — dense 64×64 downward grid covering 3.2 m × 3.2 m at 5 cm resolution.
    Mounted on `torso_link`, `ray_alignment="yaw"` (body-yaw frame, level w.r.t. world)
    so we get an occlusion-free local heightmap. Used as the reconstruction target.

Other tweaks vs. parent cfg:
  * commands resample every 3-5 s (instead of 10 s).
  * lin_vel ranges widened to (-0.5, 1.0)×(-0.5, 0.5) m/s.
  * obs corruption disabled (cleaner data; the policy still walks fine).
  * push_robot / external force / domain rand events disabled.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg


@configclass
class G1RoughLidarCollectCfg(G1RoughEnvCfg):
    """G1 rough env with lidar + GT downward raycaster, tuned for dataset collection."""

    def __post_init__(self):
        super().__post_init__()

        # widen velocity command ranges + faster resampling for trajectory diversity
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # cleaner data for the perception target
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # MID-360 lidar — head-mounted on G1.
        # NOTE: in the IsaacLab G1 USD, all upper-body link frames (pelvis, torso_link,
        # head_link, logo_link, pelvis_contour_link) share the SAME kinematic origin
        # at the pelvis (z = 1.45 in standing pose). The visual head mesh is offset
        # WITHIN the head_link prim, but the link frame itself is at pelvis.
        # → mount on head_link with an explicit 0.65 m Z offset to put the lidar at
        #   real-world G1 head height.
        self.scene.lidar_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.65)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=40,
                # Real G1 mounts the MID-360 upside-down → physical V-FOV is
                # −52° (DOWN) to +7° (UP) on the robot, not the stock
                # −7° to +52°. Confirmed by G1 localization repos.
                vertical_fov_range=(-52.0, 7.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=0.72,  # ≈ 500 az bins
            ),
            max_distance=30.0,
            update_period=0.0,
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )

        # dense downward GT scanner: 64×64 over 3.2 m × 3.2 m, body-yaw aligned
        self.scene.gt_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.05,
                size=(3.15, 3.15),  # produces 64×64 with step 0.05 m
                direction=(0.0, 0.0, -1.0),
            ),
            max_distance=3.5,
            update_period=0.0,  # every sim step
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )


# register a gym id so play-style scripts can `gym.make(...)`
gym.register(
    id="Isaac-Velocity-Rough-G1-LidarCollect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lidar_env_cfg:G1RoughLidarCollectCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)


def _make_directed_cfg(vx_const: float, episode_s: float = 12.0):
    """Build a cfg subclass with constant commanded vx, zero vy / yaw, no resampling."""

    @configclass
    class _DirCfg(G1RoughLidarCollectCfg):
        def __post_init__(self):
            super().__post_init__()
            self.episode_length_s = episode_s
            self.commands.base_velocity.resampling_time_range = (episode_s, episode_s)
            # FORCE constant commanded velocity (range collapsed to a single value)
            self.commands.base_velocity.ranges.lin_vel_x = (vx_const, vx_const)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
            # parent cfg stochastically zeros some envs to make robot stand —
            # disable so commanded velocity actually drives every env.
            self.commands.base_velocity.rel_standing_envs = 0.0
            # heading-target mode would also override yaw; force linear-velocity mode
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

    return _DirCfg


@configclass
class G1FwdLidarCollectCfg(_make_directed_cfg(+1.0)):
    """Forward only: vx = +1.0 m/s sustained for the whole episode."""


@configclass
class G1BwdLidarCollectCfg(_make_directed_cfg(-1.0)):
    """Backward only: vx = -1.0 m/s sustained for the whole episode."""


gym.register(
    id="Isaac-Velocity-Rough-G1-LidarFwd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lidar_env_cfg:G1FwdLidarCollectCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)


@configclass
class G1DiverseLidarCollectCfg(G1RoughLidarCollectCfg):
    """Diverse motion: variable forward speed + small yaw + small lateral.
    Resamples within an episode so a single trajectory contains different commands.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # resample command every 4-6 s so each 20s episode contains 3-5 commands
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        # mostly forward; allow a little backward (G1 rough policy mostly stable forward)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # don't randomly stand still (we want continuous motion + history coverage)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if hasattr(self.commands.base_velocity, "heading_command"):
            self.commands.base_velocity.heading_command = False


gym.register(
    id="Isaac-Velocity-Rough-G1-LidarDiverse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lidar_env_cfg:G1DiverseLidarCollectCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)
gym.register(
    id="Isaac-Velocity-Rough-G1-LidarBwd-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lidar_env_cfg:G1BwdLidarCollectCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
    },
)
