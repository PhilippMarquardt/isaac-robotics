"""V31 stateful reward terms — terrain-conditioned constrained objective.

Implements the priority subset of the V31 redesign spec:

  - terrain_progress_ratchet      best-so-far progress + stagnation penalty
  - support_relative_base_height_l2  pelvis vs support plane (last-valid buffer)
  - terrain_swing_clearance       hinge penalty for low swing-foot toe height
  - track_lin_vel_xy_terrain      velocity tracking weighted by terrain type

Deferred to V31.x (require heavier refactor or new sim primitives):
  - foothold_overlap (foot polygon sampling against terrain)
  - edge_impact_l2 (terrain edge detection + per-point impact)
  - flat-patch target commands (heightmap forward arc query)
  - multi-head critic (rsl_rl change)
  - terrain-normal posture
  - symmetry augmentation
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


# ============================================================================
# 1. Ratchet progress reward
# ============================================================================
class terrain_progress_ratchet(ManagerTermBase):
    """Best-so-far radial progress with stagnation penalty.

      r_t = max(0, b_t - b_{t-1}) / dt - rho * 1[no new best for T steps]

    where b_t = max_{tau <= t} ||xy(t) - xy(spawn)||.

    Velocity-free supervision for gap/stones/inv_stairs traversal — the policy
    is rewarded for any net advance toward a far landing, and lightly punished
    for stalling. No regression penalty (eta=0) by default; over a gap or step
    the robot legitimately drops back briefly during flight.
    """
    cfg: RewTerm

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.zeros(env.num_envs, device=env.device)
        self._steps_since_best = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._stagnation_steps: int = int(cfg.params.get("stagnation_steps", 50))
        self._stagnation_pen: float = float(cfg.params.get("stagnation_pen", 0.05))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._best.zero_()
            self._steps_since_best.zero_()
        else:
            self._best[env_ids] = 0.0
            self._steps_since_best[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        stagnation_steps: int = 50,
        stagnation_pen: float = 0.05,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        s = torch.norm(
            asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1
        )
        # Cap delta at 2.0*dt → max 2 m/s contribution. Prevents runaway from
        # physics teleports / curriculum spawn shifts.
        delta = torch.clamp(s - self._best, min=0.0, max=2.0 * env.step_dt)
        new_best = delta > 0
        self._steps_since_best = torch.where(
            new_best,
            torch.zeros_like(self._steps_since_best),
            self._steps_since_best + 1,
        )
        self._best = torch.maximum(self._best, s)
        stagnant = (self._steps_since_best > self._stagnation_steps).float()
        return delta / env.step_dt - self._stagnation_pen * stagnant


# ============================================================================
# 2. Support-relative base height
# ============================================================================
class support_relative_base_height_l2(ManagerTermBase):
    """Pelvis height vs local support plane (heightmap-median, last-valid buffer).

    Replaces world-z base_height which fights stairs/slopes/platforms.

    Implementation: read full height_scanner ray hits, take mean of finite z.
    During gap flight (all hits invalid), reuse last valid value.
    Clipped to ±0.5m to avoid explosion if both buffers are stale.
    """
    cfg: RewTerm

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_valid = torch.zeros(env.num_envs, device=env.device)
        self._target: float = float(cfg.params.get("target_height", 0.74))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._last_valid.zero_()
        else:
            self._last_valid[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_height: float = 0.74,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: RayCaster = env.scene[sensor_cfg.name]
        z_pelvis = asset.data.root_pos_w[:, 2]

        hits_z = sensor.data.ray_hits_w[..., 2]  # (N, P)
        valid = torch.isfinite(hits_z)
        valid_count = valid.sum(dim=1).clamp(min=1)
        z_mean = torch.where(
            valid, hits_z, torch.zeros_like(hits_z)
        ).sum(dim=1) / valid_count
        any_valid = valid.any(dim=1)
        z_support = torch.where(any_valid, z_mean, self._last_valid)
        # Update buffer where we got a valid read.
        self._last_valid = torch.where(any_valid, z_support, self._last_valid)

        # Clip ±1.0 (was ±0.5): collapsed robot at floor (rel ≈ 0) gives
        # err = -0.74 → squared 0.55, weighted -1.5 = -0.83/step. Forgiving
        # ±0.5 made max penalty -0.375/step, dwarfed by ratchet +0.5/step.
        err = torch.clamp(
            (z_pelvis - z_support) - target_height, min=-1.0, max=1.0
        )
        return torch.square(err)


# ============================================================================
# 3. Swing-foot clearance hinge
# ============================================================================
def terrain_swing_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    height_target: float = 0.08,
) -> torch.Tensor:
    """Hinge penalty: max(0, h_target - z_toe)^2 when the foot is in swing.

    Replaces feet_air_time as the foot-raising signal. Air-time rewards
    long airborne phases (distorts gait); this only penalises *low* swing
    over the support plane, so the policy is free to set its own cadence.
    """
    from isaaclab.sensors import ContactSensor
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # Per-foot contact mask via current normal forces
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]  # (N, B, 3)
    in_contact = torch.norm(forces, dim=-1) > 1.0  # (N, B)

    # Foot z position in world frame
    z_foot = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, B)

    # Support plane proxy: lowest in-contact foot z (assumes at least one foot down).
    # If none in contact, use mean foot z (defensive — mid-step).
    z_support = torch.where(
        in_contact.any(dim=1, keepdim=True),
        torch.where(in_contact, z_foot, torch.full_like(z_foot, float("inf"))).min(dim=1, keepdim=True).values,
        z_foot.mean(dim=1, keepdim=True),
    )
    rel_z = z_foot - z_support
    swing = ~in_contact
    deficit = torch.clamp(height_target - rel_z, min=0.0)
    pen = torch.square(deficit) * swing.float()
    return pen.sum(dim=1)
