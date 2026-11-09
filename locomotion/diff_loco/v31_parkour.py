"""V31 — terrain-conditioned constrained reward + skill-based curriculum.

Built atop V30-parkour. Replaces the V30 reward shape with a redesign aimed at
the chronic gaps + stairs_inv plateau:

  - **Ratchet progress** (+4.0): velocity-free dense supervision via best-so-far
    distance. Replaces velocity tracking as the dominant gap/stones signal.
  - **Support-relative base height** (-1.5): pelvis vs heightmap-median support
    plane (last-valid buffer for gap flight). Replaces world-z base_height
    which fights stairs/slopes.
  - **Swing clearance** (-0.5): hinge penalty on low swing-foot toe height.
    Replaces feet_air_time (which encouraged asymmetric pull-up gait).
  - **alive 0.15 → 0.03**: kill the fence-sit subsidy. Survival now has to
    earn its keep via progress reward.
  - **feet_air_time 1.0 → 0.05**: kept at near-zero per Humanoid Parkour
    Learning's finding (airtime conflicts with terrain locomotion).
  - **termination_penalty -50** (V30 was -100/-50): high enough to discourage
    falls, low enough to not block gap attempts.
  - **rel_standing_envs 0.02 → 0.0** on parkour terrain: standing not a parkour
    skill — train it separately if needed.
  - **Larger forward height scan** (2.4 × 1.6 m, was 1.6 × 1.0): teacher needs
    visibility past the gap lip to plan landings.
  - **Skill-based curriculum** (terrain_levels_success_v3): per-terrain success
    predicate replaces radial-distance-only promotion.

Deferred to V31.x (require non-trivial new infra):
  - Flat-patch target commands
  - Foothold-overlap reward (foot polygon × terrain raycast)
  - Edge-impact penalty
  - Multi-head critic (sparse vs dense advantage)
  - Symmetry augmentation (full mirror)
  - Two-stage gap dynamics curriculum
"""
from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from diff_loco.v28_curriculums import terrain_level_per_type_logger
from diff_loco.v30_parkour import (
    G1V30ParkourEnvCfg,
    G1V30ParkourEnvCfg_PLAY,
    G1V30ParkourTeacherPPORunnerCfg,
)
from diff_loco.v31_curriculums import terrain_levels_success_v3
from diff_loco.v31_rewards import (
    support_relative_base_height_l2,
    terrain_progress_ratchet,
    terrain_swing_clearance,
)


_V31_TERRAIN_TYPES = [
    "flat", "pyramid_stairs", "pyramid_stairs_inv", "boxes",
    "gaps", "stepping_stones", "slope_up", "slope_down",
]


def _apply_v31_overrides(self) -> None:
    """Shared env overrides for train + play cfgs."""
    # --- Reward redesign ---
    # alive: 0.03 → 0.10 (was 0.15 V30). Give a real standing gradient so the
    # policy has a positive signal even when it can't move yet.
    if hasattr(self.rewards, "alive"):
        self.rewards.alive.weight = 0.10
    # feet_air_time: near-zero per Humanoid Parkour Learning
    self.rewards.feet_air_time.weight = 0.05
    # termination: -50 was too weak — a 0.5m lurch earns +100 ratchet (4*v),
    # so net +50 per fall trip → policy loves lurch+faceplant. Bump to -150
    # to make any fall a hard net loss.
    self.rewards.termination_penalty.weight = -150.0
    # velocity tracking: keep dominant on flat/slope, but lower than V28 default
    # (V31 ratchet replaces it as primary parkour signal)
    self.rewards.track_lin_vel_xy_exp.weight = 0.5
    self.rewards.track_ang_vel_z_exp.weight = 0.2

    # Replace V30 base_height_safe_l2 (world-z) with support-relative.
    if hasattr(self.rewards, "base_height_l2"):
        delattr(self.rewards, "base_height_l2")
    self.rewards.support_height = RewTerm(
        func=support_relative_base_height_l2,
        weight=-1.5,
        params={"target_height": 0.74},
    )

    # New: ratchet progress (primary parkour signal)
    # v3: 4.0 → 1.5. v2 economics: ratchet 1.18/step × 210 ep_len = +248 vs
    # termination -150 = net +98 per lurch-fall episode. PPO converged to
    # lurch-fall (98% base_contact, vel tracking 0.006). v3 cut: 1.18 × 0.375
    # = 0.44/step × 210 = +93, now < 150 termination → fall is hard net loss.
    self.rewards.ratchet = RewTerm(
        func=terrain_progress_ratchet,
        weight=1.5,
        params={"stagnation_steps": 50, "stagnation_pen": 0.05},
    )

    # New: swing clearance hinge (replaces airtime as foot-raising signal)
    self.rewards.swing_clearance = RewTerm(
        func=terrain_swing_clearance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "height_target": 0.08,
        },
    )

    # --- Commands: standing not a parkour skill, kill it on parkour terrain.
    self.commands.base_velocity.rel_standing_envs = 0.0

    # --- Larger forward height scan for gap planning visibility.
    # Stock V28b set 1.6 × 1.0; bump to 2.4 × 1.6 (teacher only — student keeps
    # the short scanner unchanged).
    self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=0.1, size=(2.4, 1.6)
    )

    # --- Skill-based curriculum (replace V28o's hysteresis_v2)
    self.curriculum.terrain_levels = CurrTerm(
        func=terrain_levels_success_v3,
        params={
            "fail_threshold": 3,
            "demote_bar_floor_m": 1.0,
            "warmup_steps": 3_200,
            "warmup_terrain_idxs": (0, 6, 7),
            "stairs_height_threshold_m": 0.3,
            "gap_promote_bar_m": 2.0,
            "stairs_inv_promote_bar_m": 2.0,
            "stones_promote_bar_m": 2.0,
        },
    )

    # Re-attach per-type level loggers (CurrTerm replacement above wipes them
    # from the manager's order). Keep the same names so tfevents continuity.
    for _i, _name in enumerate(_V31_TERRAIN_TYPES):
        setattr(
            self.curriculum,
            f"terrain_level_{_name}",
            CurrTerm(
                func=terrain_level_per_type_logger,
                params={"type_idx": _i},
            ),
        )


@configclass
class G1V31ParkourEnvCfg(G1V30ParkourEnvCfg):
    """V31 train env: V30 + reward redesign + skill curriculum."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v31_overrides(self)


@configclass
class G1V31ParkourEnvCfg_PLAY(G1V30ParkourEnvCfg_PLAY):
    """V31 play env."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v31_overrides(self)


@configclass
class G1V31ParkourTeacherPPORunnerCfg(G1V30ParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v31_parkour_teacher"


def register_v31_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V31-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V31-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v31_parkour:G1V31ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v31_parkour:G1V31ParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V31-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v31_parkour:G1V31ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v31_parkour:G1V31ParkourTeacherPPORunnerCfg",
        },
    )
