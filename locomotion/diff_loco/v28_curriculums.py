"""V28 curriculum helpers.

Two functions:

1. ``terrain_level_per_type_logger`` — read-only, used by V28e + V28f.
   Returns the mean ``terrain_levels`` over envs whose ``terrain_types`` index
   matches ``type_idx``. No mutation. Registered once per terrain type so the
   curriculum manager logs each as ``Curriculum/<term_name>`` in tfevents.
   Tells us whether a global mean of 5.7 means "everything at 5-6 uniformly"
   or "stairs at 9, gaps at 2".

2. ``terrain_levels_hysteresis`` — V28f only. Same distance-based promotion as
   stock ``terrain_levels_vel``, but demote requires ``fail_threshold``
   consecutive failed episodes (per env). Stops single falls at level 6 from
   throwing away mastered levels.

Both functions are usable as ``CurriculumTermCfg.func``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _build_col_to_type(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Replicate IsaacLab's TerrainGenerator column→sub-terrain mapping.

    Mirrors ``terrain_generator.py:243-247``: each column gets the sub-terrain
    whose cumulative proportion bucket it falls into. We can't read the
    generator's ``sub_indices`` directly because it's a local in the build
    method, so we rebuild it from the ``sub_terrains`` cfg dict.
    """
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


def terrain_level_per_type_logger(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    type_idx: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return mean ``terrain_levels`` for envs whose column maps to ``type_idx``.

    ``terrain.terrain_types`` is the column index (0..num_cols-1), not the
    sub-terrain index — many columns map to the same sub-terrain via the
    generator's proportion-based allocation. We compute that col→type map
    once and cache it on the env. Read-only; returns 0.0 if no envs match.
    """
    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(env, "_v28_col_to_type"):
        env._v28_col_to_type = _build_col_to_type(env)
    env_type = env._v28_col_to_type[terrain.terrain_types]
    mask = env_type == type_idx
    if not bool(mask.any()):
        return torch.tensor(0.0, device=terrain.terrain_levels.device)
    return terrain.terrain_levels[mask].float().mean()


def terrain_levels_hysteresis(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fail_threshold: int = 3,
    gap_terrain_idx: int = 4,
    gap_promote_bar_m: float = 2.0,
    demote_speed_frac: float = 0.125,
) -> torch.Tensor:
    """Hysteresis curriculum + per-terrain promote bar + softer demote bar.

    Three changes vs stock ``terrain_levels_vel``:
      1. Demotion needs ``fail_threshold`` consecutive failures (hysteresis).
      2. Promotion bar lowered to ``gap_promote_bar_m`` for gap tiles —
         spawn-locked otherwise (robot must clear platform+gap from cold
         start, can't reach 4m without gap-stepping skill).
      3. Demote bar = ``cmd * episode_len * demote_speed_frac`` (default
         0.125 = avg 0.125 m/s on a 1 m/s cmd, vs stock 0.5 = 0.5 m/s).
         Stock bar of 0.5 means a slow-cmd env had to walk 10m on an 8m
         tile to NOT demote — practically impossible on rough terrain, so
         every env auto-demoted regardless of skill.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if not hasattr(env, "_v28_col_to_type"):
        env._v28_col_to_type = _build_col_to_type(env)
    env_type = env._v28_col_to_type[terrain.terrain_types[env_ids]]

    # standard distance-based criteria
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    size_half = terrain.cfg.terrain_generator.size[0] / 2
    promote_bar = torch.where(
        env_type == gap_terrain_idx,
        torch.full_like(distance, gap_promote_bar_m),
        torch.full_like(distance, size_half),
    )
    move_up_raw = distance > promote_bar
    move_down_raw = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * demote_speed_frac
    move_down_raw = move_down_raw & (~move_up_raw)

    # lazy-init per-env failure counter
    if not hasattr(env, "_terrain_fail_count") or env._terrain_fail_count.shape[0] != env.num_envs:
        env._terrain_fail_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # update counter for resetting envs only
    counter = env._terrain_fail_count[env_ids]
    counter = torch.where(move_down_raw, counter + 1, torch.zeros_like(counter))
    counter = torch.where(move_up_raw, torch.zeros_like(counter), counter)
    env._terrain_fail_count[env_ids] = counter

    # demote only when threshold reached
    move_down = counter >= fail_threshold
    # reset counter once demotion fires so we don't demote twice in a row
    env._terrain_fail_count[env_ids] = torch.where(move_down, torch.zeros_like(counter), counter)

    # clip-not-wrap: stock TerrainImporter wraps to random level on hitting max,
    # which pulls stationary distribution to mid-level (~6.0 for max=10). Clip
    # at max-1 so mastered envs stick at top instead of resetting downward.
    terrain.terrain_levels[env_ids] += move_up_raw.long() - move_down.long()
    terrain.terrain_levels[env_ids] = torch.clip(
        terrain.terrain_levels[env_ids], 0, terrain.max_terrain_level - 1
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_hysteresis_v2(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fail_threshold: int = 3,
    gap_terrain_idx: int = 4,
    gap_promote_bar_m: float = 2.0,
    stairs_inv_terrain_idx: int = 2,
    stairs_inv_promote_bar_m: float = 2.0,
    demote_speed_frac: float = 0.125,
    demote_bar_floor_m: float = 1.0,
    warmup_steps: int = 0,
    warmup_terrain_idxs: tuple = (0, 6, 7),
) -> torch.Tensor:
    """V28o curriculum: hysteresis + spawn-locked fixes + warmup.

    Three additions vs ``terrain_levels_hysteresis``:
      1. ``stairs_inv_promote_bar_m`` mirrors gap fix — inverted pyramid stairs
         are spawn-locked (robot spawns on top platform, must traverse pit).
      2. ``demote_bar_floor_m`` floors the demote bar so pure-yaw / near-zero
         lin_vel cmds still demote if env makes no progress (kills the
         standing-spin blackhole).
      3. ``warmup_steps``: for the first N common steps, force resetting
         envs onto warmup terrain types (default: flat, slope_up, slope_down).
         Lets policy bootstrap basic walking before harder tiles unlock.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if not hasattr(env, "_v28_col_to_type"):
        env._v28_col_to_type = _build_col_to_type(env)

    # Cache the original terrain_types ONCE so we can restore after warmup.
    # Without this, warmup permanently rewrites assignment and stairs/gaps
    # are never seen post-warmup.
    if not hasattr(env, "_v28_orig_terrain_types"):
        env._v28_orig_terrain_types = terrain.terrain_types.clone()

    # Warmup: temporarily reassign resetting envs to warmup tiles.
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
        # Post-warmup: restore the original column assignment for resetting envs.
        # Resets reroll the column choice each time, so over a few episodes all
        # envs get back to their natural distribution.
        terrain.terrain_types[env_ids] = env._v28_orig_terrain_types[env_ids]

    env_type = env._v28_col_to_type[terrain.terrain_types[env_ids]]

    # Per-type promote bar: gap + stairs_inv get lower bar (spawn-locked).
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    size_half = terrain.cfg.terrain_generator.size[0] / 2
    promote_bar = torch.full_like(distance, size_half)
    promote_bar = torch.where(
        env_type == gap_terrain_idx,
        torch.full_like(distance, gap_promote_bar_m),
        promote_bar,
    )
    promote_bar = torch.where(
        env_type == stairs_inv_terrain_idx,
        torch.full_like(distance, stairs_inv_promote_bar_m),
        promote_bar,
    )
    move_up_raw = distance > promote_bar

    # Demote bar with floor — kills yaw-only / standing blackholes.
    cmd_norm = torch.norm(command[env_ids, :2], dim=1)
    demote_bar = cmd_norm * env.max_episode_length_s * demote_speed_frac
    demote_bar = torch.clamp(demote_bar, min=demote_bar_floor_m)
    move_down_raw = distance < demote_bar
    move_down_raw = move_down_raw & (~move_up_raw)

    # Hysteresis counter
    if not hasattr(env, "_terrain_fail_count") or env._terrain_fail_count.shape[0] != env.num_envs:
        env._terrain_fail_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    counter = env._terrain_fail_count[env_ids]
    counter = torch.where(move_down_raw, counter + 1, torch.zeros_like(counter))
    counter = torch.where(move_up_raw, torch.zeros_like(counter), counter)
    env._terrain_fail_count[env_ids] = counter

    move_down = counter >= fail_threshold
    env._terrain_fail_count[env_ids] = torch.where(move_down, torch.zeros_like(counter), counter)

    terrain.terrain_levels[env_ids] += move_up_raw.long() - move_down.long()
    terrain.terrain_levels[env_ids] = torch.clip(
        terrain.terrain_levels[env_ids], 0, terrain.max_terrain_level - 1
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_levels.float())
