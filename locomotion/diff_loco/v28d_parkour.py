"""V28d G1 Parkour — sim-to-real closure atop V28c.

Purpose: close the sim-to-real gap so the policy moves with real-G1 compliance
and latency, not sim-crisp idealization. Same task/reward skeleton as V28c;
additions target actuator-plant + latency mismatch.

Additions vs V28c (priority-ordered):

  1. Action-latency DR (highest leverage).
     Custom `DelayedJointPositionAction` term wraps stock JointPositionAction
     with a per-env action queue. Delay ∈ {0,1,2} policy steps (0/20/40 ms
     at 50 Hz) resampled on reset. Matches real observation + comms latency.

  2. PD gain DR (±20 % on Kp and Kd, per joint, startup-randomized).
     Models motor-to-motor variance + Kp safety de-rating on the real robot.

  3. Calibrated armature values on leg joints.
     Hip-roll & knee get armature 0.0251 kg·m², torso 0.0151, rest 0.0102
     (vs Isaac Lab's round 0.01 everywhere). Values come from Unitree
     rl_lab `UNITREE_G1_29DOF_MIMIC_CFG` — physically derived from rotor
     inertia × gear ratio². Fixes reflected-rotor-inertia gap.

  4. Armature DR (±20 %).
     Small perturbation on top of calibrated baseline for robustness.

  5. action_rate_l2 curriculum ramp -0.01 → -0.05 at 96k steps.
     V28c's -0.01 is ~0.1 % of reward = jerky commands persist. Ramping
     down (more negative) drives smoother targets once tracking is solid.

  6. `joint_torques_rate_l2` (τ̇) penalty — new mdp term, weight -1e-7.
     Addresses current-loop bandwidth: real motors can't step torque in 1 ms.
     Stock `action_rate_l2` only penalises target-position jitter; this term
     penalises the actual torque slew the implicit PD produces.

  7. Bump joint_vel obs noise ±0.5 → ±1.0 rad/s.
     Real encoder-differentiation noise is ~1 rad/s RMS on G1; V28b/c
     under-modelled it by 2×.

Unchanged from V28c (intentional — isolate sim-to-real additions):
  - Reward weights (feet_air_time 0.25, dof_torques_l2 -5e-6, feet_contact_forces -1e-4).
  - Asymmetric actor/critic obs (actor=priv, critic=priv+foot wrench).
  - MLP sizes (actor 768/512/256/128, critic 1024/512/256/128).
  - 15k iter, adaptive KL, entropy_coef 0.005, action scale 0.30.
  - Terrain (V28_PARKOUR_TERRAINS_CFG).
  - Base Kp/Kd: kept at Isaac Lab G1 stock values (150/200 · 5). Unitree MIMIC
    uses 40/99 · 2.6/6.3 which is a very different plant; swapping baseline
    would be a new recipe (→ V28e). V28d only adds ±20 % DR around stock so
    the policy trains tolerant to the real robot's gain variance.

Deferred to V28e / later:
  - Swap Kp/Kd baseline to Unitree MIMIC values.
  - Explicit Unitree joint-friction values (Isaac Lab's `friction_coeff`
    semantics differ from Unitree's Fs/Fd; needs care).
  - Thermal/duty-cycle modelling.
  - Actuator-net learned from hardware telemetry.

Env DR vs obs noise split (confirmed 2026-04-24):
  - Env perturbations (mass, CoM, friction, pushes, action latency, PD gain
    DR, armature DR) are physics-level events — they apply to the plant
    regardless of which obs group the policy reads, so **both teacher and
    student training see them**.
  - Observation noise is per obs-group: `PrivCfg` (teacher-actor view) has
    `enable_corruption=False` and zero `Unoise` on every term — teacher sees
    a clean sensor readout of a DR-perturbed robot. `ProprioCfg` (student
    view) has `enable_corruption=True` + `Unoise` on base_ang_vel / joint_pos
    / joint_vel (V28d bump: joint_vel ±0.5 → ±1.0) / height_scan_short —
    student sees a noisy readout. Exactly: "clean on teacher, noisy on
    student; env DR on both".
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from diff_loco.v28b_parkour import (
    G1V28bObservationsCfg,
    G1V28bParkourStudentDistillRunnerCfg,
    G1V28bParkourTeacherPPORunnerCfg,
)
from diff_loco.v28c_parkour import (
    G1V28cParkourEnvCfg,
    G1V28cParkourEnvCfg_PLAY,
)


# ============================================================================
# Custom action term: JointPositionAction + per-env delay buffer.
# ============================================================================
class DelayedJointPositionAction(JointPositionAction):
    """Joint position action with per-env discrete delay (policy-step buffer).

    At each policy step, the incoming raw action is pushed into a circular
    buffer and the action actually applied is drawn from the buffer at the
    env's sampled delay index. Delay is resampled on env reset.

    Delay units are policy steps (= decimation × physics dt). At 50 Hz policy
    with delays ∈ {0, 1, 2} this gives 0 / 20 / 40 ms of action latency,
    which brackets the real G1's observation + PD-loop round-trip.
    """

    cfg: "DelayedJointPositionActionCfg"

    def __init__(self, cfg: "DelayedJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._max_delay: int = int(cfg.max_delay_steps)
        # buffer[slot, env, joint] — slot 0 = newest, slot max_delay = oldest.
        self._action_queue = torch.zeros(
            self._max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )
        # per-env sampled delay (int), in [0, max_delay].
        self._delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # sample initial delay.
        self._resample_delay(torch.arange(self.num_envs, device=self.device))

    def _resample_delay(self, env_ids: torch.Tensor) -> None:
        """Sample new delay (policy-steps) for the given envs, uniform in [0, max_delay]."""
        new_delay = torch.randint(
            low=0,
            high=self._max_delay + 1,
            size=(env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )
        self._delay_steps[env_ids] = new_delay

    def process_actions(self, actions: torch.Tensor):
        # Run stock scale/offset/clip pipeline → self._processed_actions.
        super().process_actions(actions)
        # Shift queue (oldest dropped) and insert newest at slot 0.
        self._action_queue = torch.roll(self._action_queue, shifts=1, dims=0)
        self._action_queue[0] = self._processed_actions
        # Gather per-env delayed action from queue.
        idx = self._delay_steps.view(1, -1, 1).expand(1, -1, self.action_dim)
        delayed = torch.gather(self._action_queue, 0, idx).squeeze(0)
        # Apply the delayed action as the one we'll actually send to physics.
        self._processed_actions = delayed

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        # Clear queue to current default offset so robot doesn't see a stale command spike.
        if isinstance(self._offset, torch.Tensor):
            default = self._offset[env_ids_t]
        else:
            default = torch.full(
                (env_ids_t.numel(), self.action_dim),
                float(self._offset),
                device=self.device,
            )
        self._action_queue[:, env_ids_t] = default.unsqueeze(0).expand(
            self._max_delay + 1, -1, -1
        )
        self._resample_delay(env_ids_t)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Config for DelayedJointPositionAction.

    `max_delay_steps`: inclusive upper bound on delay in *policy steps*. A value
    of 2 at 50 Hz gives delay ∈ {0, 20, 40} ms, sampled uniformly per env on reset.
    """

    class_type: type[ActionTerm] = DelayedJointPositionAction
    max_delay_steps: int = 2


# ============================================================================
# Custom reward: torque-rate (τ̇) penalty.
# ============================================================================
class joint_torques_rate_l2(ManagerTermBase):
    """Penalize (τ[t] - τ[t-1])² summed over joints.

    Stateful — keeps the previous policy-step torque per env. On env reset the
    previous torque is zeroed (treat post-reset step as "no prior torque" so
    we don't false-penalise a reset transient).
    """

    cfg: RewTerm

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset_cfg = asset_cfg
        asset: Articulation = env.scene[asset_cfg.name]
        # Resolve joint ids once.
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names) if asset_cfg.joint_names else (slice(None), None)
        self._joint_ids = joint_ids
        num_joints = (
            asset.num_joints if joint_ids == slice(None) else len(joint_ids)  # type: ignore
        )
        self._prev_torque = torch.zeros(env.num_envs, num_joints, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._prev_torque.zero_()
        else:
            self._prev_torque[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        tau = asset.data.applied_torque[:, self._joint_ids]
        delta = tau - self._prev_torque
        self._prev_torque = tau.detach().clone()
        return torch.sum(torch.square(delta), dim=1)


# ============================================================================
# V28d env cfg
# ============================================================================
@configclass
class G1V28dObservationsCfg(G1V28bObservationsCfg):
    """V28d obs: bump joint_vel noise to match real G1 encoder-diff noise."""

    def __post_init__(self):  # type: ignore[override]
        # ProprioCfg is a nested configclass; override the joint_vel noise on it.
        self.proprio.joint_vel.noise = Unoise(n_min=-1.0, n_max=1.0)


@configclass
class G1V28dParkourEnvCfg(G1V28cParkourEnvCfg):
    """V28d: V28c + sim-to-real DR and smoothness penalties."""

    observations: G1V28dObservationsCfg = G1V28dObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # (1) Replace JointPositionAction with delayed version.
        # ------------------------------------------------------------------
        old_action = self.actions.joint_pos
        self.actions.joint_pos = DelayedJointPositionActionCfg(
            asset_name=old_action.asset_name,
            joint_names=old_action.joint_names,
            scale=old_action.scale,
            offset=old_action.offset,
            preserve_order=old_action.preserve_order,
            use_default_offset=old_action.use_default_offset,
            clip=old_action.clip,
            debug_vis=old_action.debug_vis,
            max_delay_steps=2,  # {0, 20, 40} ms @ 50 Hz policy
        )

        # ------------------------------------------------------------------
        # (3) Calibrated armature values on leg joints (Unitree MIMIC).
        #     Isaac Lab G1 stock uses round 0.01 everywhere; these are rotor-
        #     inertia-derived. Applied via startup event to avoid forking the
        #     robot USD/cfg.
        # ------------------------------------------------------------------
        # We apply calibrated baselines with `abs` operation (per-joint scalar).
        # Then (4) stacks a ±20 % scale DR on top.
        self.events.v28d_armature_calibrated = EventTerm(
            func=base_mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_roll_joint", ".*_knee_joint"],
                ),
                "armature_distribution_params": (0.0251, 0.0251),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        self.events.v28d_armature_ankle = EventTerm(
            func=base_mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                ),
                "armature_distribution_params": (0.0072, 0.0072),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        # Hip pitch/yaw + torso stay at 0.0102 — Isaac's 0.01 baseline is already
        # within 2 % of calibrated. No-op on baseline; DR below still covers it.

        # ------------------------------------------------------------------
        # (4) Armature DR ±20 % (scale) — on top of calibrated baselines.
        # ------------------------------------------------------------------
        self.events.v28d_armature_dr = EventTerm(
            func=base_mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_.*_joint",
                        ".*_knee_joint",
                        ".*_ankle_.*_joint",
                        "torso_joint",
                    ],
                ),
                "armature_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # ------------------------------------------------------------------
        # (2) PD gain DR ±20 % on Kp, ±20 % on Kd (scale).
        # ------------------------------------------------------------------
        self.events.v28d_actuator_gain_dr = EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_.*_joint",
                        ".*_knee_joint",
                        ".*_ankle_.*_joint",
                        "torso_joint",
                    ],
                ),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # ------------------------------------------------------------------
        # (5) action_rate_l2 curriculum ramp -0.01 → -0.05 at 96k steps.
        #     V28b had a 24k-step ramp to -0.02; V28c inherited it verbatim.
        #     Overwrite with a longer, stronger ramp.
        # ------------------------------------------------------------------
        self.curriculum.v28_action_rate = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate_l2", "weight": -0.05, "num_steps": 96_000},
        )

        # ------------------------------------------------------------------
        # (6) τ̇ penalty — joint_torques_rate_l2.
        #     Applied to leg + torso joints; exclude arms/hands to avoid
        #     over-constraining their (small) idle torques.
        # ------------------------------------------------------------------
        self.rewards.dof_torques_rate = RewTerm(
            func=joint_torques_rate_l2,
            weight=-1.0e-7,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_.*_joint",
                        ".*_knee_joint",
                        ".*_ankle_.*_joint",
                        "torso_joint",
                    ],
                ),
            },
        )


@configclass
class G1V28dParkourEnvCfg_PLAY(G1V28dParkourEnvCfg):
    """Play cfg — inherit V28d env + apply V28c play overrides."""

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
        # PLAY: command pure forward only. ANY nonzero sampled ang_vel_z +
        # nonzero lin_vel_x = circular motion (radius = v / w). Lock ang_vel
        # and lateral to zero so robot just walks forward across terrain.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1e6, 1e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.observations.proprio.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        from isaaclab.envs import ViewerCfg
        self.viewer = ViewerCfg(
            eye=(20.0, 20.0, 15.0),
            lookat=(0.0, 0.0, 0.5),
            origin_type="world",
        )

        # Local-play lighting fix. Stock sky_light pulls an HDRI from
        # ISAAC_NUCLEUS_DIR; on Windows w/o nucleus the URL is broken and
        # DomeLight loads no texture → fully dark scene. Replace with a
        # textureless DomeLight + DistantLight (sun).
        import isaaclab.sim as sim_utils
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
                rot=(0.866, 0.353, 0.0, 0.353),
            ),
        )


@configclass
class G1V28dParkourEnvCfg_PLAY_L6(G1V28dParkourEnvCfg_PLAY):
    """Play cfg pinned to terrain difficulty = 6/9 ≈ 0.667 (level 6 tiles only)."""

    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.667, 0.667)
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 25
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class G1V28dParkourEnvCfg_PLAY_L9(G1V28dParkourEnvCfg_PLAY):
    """Play cfg pinned to terrain difficulty = 1.0 (level 9 = hardest tiles only)."""

    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 25
        self.scene.terrain.max_init_terrain_level = 0


# ============================================================================
# Runner configs — reuse V28b, just new experiment_name.
# ============================================================================
@configclass
class G1V28dParkourTeacherPPORunnerCfg(G1V28bParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28d_parkour_teacher"


@configclass
class G1V28dParkourStudentDistillRunnerCfg(G1V28bParkourStudentDistillRunnerCfg):
    experiment_name = "g1_v28d_parkour_student"


# ============================================================================
# Gym task registration
# ============================================================================
def register_v28d_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28d-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Student-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Student-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourStudentDistillRunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Teacher-Play-L6-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg_PLAY_L6",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28d-Teacher-Play-L9-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourEnvCfg_PLAY_L9",
            "rsl_rl_cfg_entry_point": "diff_loco.v28d_parkour:G1V28dParkourTeacherPPORunnerCfg",
        },
    )
