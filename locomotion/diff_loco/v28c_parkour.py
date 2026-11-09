"""V28c G1 Parkour — reward rebalance atop V28b for asymmetric-gait fix.

Built on `v28b_parkour.G1V28bParkourEnvCfg`: keeps asymmetric critic, bigger
nets, 15k iter teacher. Only rewards + one curriculum change.

Changes vs V28b (motivation in notes below):

  1. `feet_air_time.weight` 1.0 → 0.25   (revert to IsaacLab stock).
     V28/V28b bumped this to 1.0 and curriculum-ramped to 1.5 by iter 48k.
     `feet_air_time_positive_biped` rewards per-foot single-stance time
     capped at 0.4s. With weight 1.5 the swing-foot reward dominates every
     other term → policy finds local optimum of one leg hoisted super high
     while other leg does tiny stance steps (asymmetric "pull-up" gait
     observed in V28 play videos). Stock 0.25 keeps the gait cue without
     drowning velocity tracking.

  2. Drop `v28_air_time` curriculum entirely (no 1.5 bump).

  3. `dof_torques_l2.weight` -2.0e-6 → -5.0e-6   (2.5× bump).
     Recent humanoid work (Unitree rl_lab H1, Booster-T1 2025, 15-min sim-to-
     real 2512.01996) does NOT use explicit `dof_power` / τ·q̇ penalty —
     that is an older ANYmal idiom. Plain torque-L2 remains the standard
     energy proxy; bumping it is the biological substitute for smoother
     joint use without the τ·q̇ pathology of penalizing desired motion.

  4. Add `feet_contact_forces` penalty (threshold=400N, weight=-1e-4).
     From Unitree rl_lab H1 reward set. Penalizes excessive ground-impact
     forces — complements feet_slide (lateral) with a vertical-impact term.
     Discourages slamming the foot, which should also reduce the yank
     that comes from asymmetric leg swings.

Unchanged from V28b:
  - Asymmetric actor/critic obs (actor=priv, critic=priv+foot wrench).
  - MLP sizes (actor 768/512/256/128, critic 1024/512/256/128).
  - 15k iterations, adaptive KL, entropy_coef 0.005.
  - Action scale 0.30, residual default.
  - DR: mass perturb, CoM perturb, friction 0.3-1.2, push every 6-12s.
  - Terrain: V28_PARKOUR_TERRAINS_CFG (shared).

Deferred (intentional):
  - Symmetry augmentation: still disabled (half-mirror bug — needs full obs
    mirror before re-enable). Once fixed, will be #1 anti-asymmetric-gait
    intervention; reward rebalance here is the fallback that works without
    touching the PPO loss.
  - Action-delay DR (mentioned as V28c TODO in v28b_parkour.py header):
    left to V28d; scope kept tight for clean attribution of reward changes.

Ref for reward choices:
  - Unitree rl_lab H1 (deepwiki.com/unitreerobotics/unitree_rl_lab/6.2-h1-humanoid)
  - Booster-Gym T1 (2025) Table II
  - 15-min sim-to-real (arxiv 2512.01996)
  - Gait-Conditioned G1 RL (arxiv 2505.20619)
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.v28b_parkour import (
    G1V28bObservationsCfg,
    G1V28bParkourEnvCfg,
    G1V28bParkourEnvCfg_PLAY,
    G1V28bParkourStudentDistillRunnerCfg,
    G1V28bParkourTeacherPPORunnerCfg,
)


# ============================================================================
# V28c env cfg
# ============================================================================
@configclass
class G1V28cParkourEnvCfg(G1V28bParkourEnvCfg):
    """V28c: V28b with rewards rebalanced to fix asymmetric gait."""

    observations: G1V28bObservationsCfg = G1V28bObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- reward rebalance ---

        # (1) feet_air_time: revert to stock 0.25 (V28b had it at 1.0).
        self.rewards.feet_air_time.weight = 0.25

        # (3) dof_torques_l2: 2.5× bump — primary energy proxy.
        self.rewards.dof_torques_l2.weight = -5.0e-6

        # (4) feet_contact_forces: new penalty, 400N threshold.
        self.rewards.feet_contact_forces = RewTerm(
            func=mdp.contact_forces,
            weight=-1.0e-4,
            params={
                "threshold": 400.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # (2) drop the 1.5 air_time curriculum ramp (set to None to disable).
        if hasattr(self.curriculum, "v28_air_time"):
            self.curriculum.v28_air_time = None


@configclass
class G1V28cParkourEnvCfg_PLAY(G1V28cParkourEnvCfg):
    """Play cfg: inherit V28c rewards, apply V28b play overrides."""

    def __post_init__(self):
        super().__post_init__()
        # Re-apply V28b play overrides (can't inherit from PLAY + env together).
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
        # Video cam: (8,8,6) still flickered on seed 42 — terrain blocks between
        # cam and pelvis occluded pelvis → near-plane clip → blue sky frames.
        # Pull farther out + higher eye for steeper angle that clears obstacles.
        self.viewer = ViewerCfg(
            eye=(14.0, 14.0, 10.0),
            lookat=(0.0, 0.0, 0.8),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
        )


# ============================================================================
# Runner configs — same as V28b, just new experiment_name.
# ============================================================================
@configclass
class G1V28cParkourTeacherPPORunnerCfg(G1V28bParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28c_parkour_teacher"


@configclass
class G1V28cParkourStudentDistillRunnerCfg(G1V28bParkourStudentDistillRunnerCfg):
    experiment_name = "g1_v28c_parkour_student"


# ============================================================================
# Gym task registration
# ============================================================================
def register_v28c_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28c-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28c-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28c-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28c-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28c-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28c_parkour:G1V28cParkourStudentDistillRunnerCfg",
        },
    )
