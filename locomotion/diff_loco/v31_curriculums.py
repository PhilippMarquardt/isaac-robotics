"""V31 curriculum — terrain_levels_success_v3.

Skill-based promotion (was distance-based in v2). Per-terrain success predicate:

  - flat / slope_up / slope_down : distance > size_half (4 m)
  - pyramid_stairs               : distance > size_half AND z_pelvis - z_spawn > 0.3
  - pyramid_stairs_inv           : distance > 2 m AND z_pelvis - z_spawn < -0.3
  - boxes                        : distance > size_half
  - gaps                         : distance > 2 m AND non-fall episode end
  - stepping_stones              : distance > 2 m AND non-fall episode end

ALL promotions also gated on `~termination_manager.terminated[env_ids]` —
no promotion if episode ended via base_contact / bad_orientation. Without
this gate the policy learns "ballistic lurch + faceplant" (V31 v0 bug:
99% base_contact, terrain_lvl 4.85 at iter 268).

Demote: K=3 consecutive failed-promote episodes (hysteresis kept from v2)
        OR pure stagnation (no progress past floor in episode).
Warmup: same as v2 (3200 steps on flat + slopes).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from diff_loco.v28_curriculums import _build_col_to_type

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Terrain index in V28_PARKOUR_TERRAINS_CFG.sub_terrains dict order:
#   0 flat, 1 pyramid_stairs, 2 pyramid_stairs_inv, 3 boxes,
#   4 gaps, 5 stepping_stones, 6 slope_up, 7 slope_down
_FLAT = 0
_STAIRS = 1
_STAIRS_INV = 2
_BOXES = 3
_GAPS = 4
_STONES = 5
_SLOPE_UP = 6
_SLOPE_DN = 7


def terrain_levels_success_v3(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fail_threshold: int = 3,
    demote_bar_floor_m: float = 1.0,
    warmup_steps: int = 3_200,
    warmup_terrain_idxs: tuple = (_FLAT, _SLOPE_UP, _SLOPE_DN),
    stairs_height_threshold_m: float = 0.3,
    gap_promote_bar_m: float = 2.0,
    stairs_inv_promote_bar_m: float = 2.0,
    stones_promote_bar_m: float = 2.0,
) -> torch.Tensor:
    """V31 curriculum: per-terrain skill predicates + warmup + hysteresis."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if not hasattr(env, "_v28_col_to_type"):
        env._v28_col_to_type = _build_col_to_type(env)
    if not hasattr(env, "_v28_orig_terrain_types"):
        env._v28_orig_terrain_types = terrain.terrain_types.clone()

    # Warmup: temporarily route resetting envs onto warmup tiles only.
    in_warmup = warmup_steps > 0 and env.common_step_counter < warmup_steps
    if in_warmup:
        col_to_type = env._v28_col_to_type
        warmup_mask = torch.zeros_like(col_to_type, dtype=torch.bool)
        for t in warmup_terrain_idxs:
            warmup_mask |= col_to_type == int(t)
        warmup_cols = torch.where(warmup_mask)[0]
        if warmup_cols.numel() > 0:
            picks = warmup_cols[torch.randint(
                0, warmup_cols.numel(), (len(env_ids),), device=warmup_cols.device
            )]
            terrain.terrain_types[env_ids] = picks
    else:
        terrain.terrain_types[env_ids] = env._v28_orig_terrain_types[env_ids]

    env_type = env._v28_col_to_type[terrain.terrain_types[env_ids]]
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    z_now = asset.data.root_pos_w[env_ids, 2]
    z_spawn = env.scene.env_origins[env_ids, 2]
    dz = z_now - z_spawn

    size_half = terrain.cfg.terrain_generator.size[0] / 2

    # Episode-end gating: only promote on non-fall termination (timeout).
    # termination_manager.terminated = True if a fall/contact term fired;
    # promoting on a fallen lurch teaches the policy "ballistic dive == win".
    fell = env.termination_manager.terminated[env_ids]
    no_fall = ~fell

    # Per-terrain success predicate (skill-based promotion). Gated on no-fall.
    promote = torch.zeros_like(distance, dtype=torch.bool)
    flat_like = (env_type == _FLAT) | (env_type == _SLOPE_UP) | (env_type == _SLOPE_DN) | (env_type == _BOXES)
    promote = promote | (flat_like & (distance > size_half))
    promote = promote | ((env_type == _STAIRS) & (distance > size_half) & (dz > stairs_height_threshold_m))
    promote = promote | ((env_type == _STAIRS_INV) & (distance > stairs_inv_promote_bar_m) & (dz < -stairs_height_threshold_m))
    promote = promote | ((env_type == _GAPS) & (distance > gap_promote_bar_m))
    promote = promote | ((env_type == _STONES) & (distance > stones_promote_bar_m))
    promote = promote & no_fall

    # Demote bar with floor — kills standing-spin blackholes.
    cmd_norm = torch.norm(command[env_ids, :2], dim=1)
    demote_bar = cmd_norm * env.max_episode_length_s * 0.125
    demote_bar = torch.clamp(demote_bar, min=demote_bar_floor_m)
    demote = (distance < demote_bar) & (~promote)

    # Hysteresis K=3.
    if not hasattr(env, "_v31_fail_count") or env._v31_fail_count.shape[0] != env.num_envs:
        env._v31_fail_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    counter = env._v31_fail_count[env_ids]
    counter = torch.where(demote, counter + 1, torch.zeros_like(counter))
    counter = torch.where(promote, torch.zeros_like(counter), counter)
    env._v31_fail_count[env_ids] = counter

    do_demote = counter >= fail_threshold
    env._v31_fail_count[env_ids] = torch.where(
        do_demote, torch.zeros_like(counter), counter
    )

    terrain.terrain_levels[env_ids] += promote.long() - do_demote.long()
    terrain.terrain_levels[env_ids] = torch.clip(
        terrain.terrain_levels[env_ids], 0, terrain.max_terrain_level - 1
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_levels.float())
