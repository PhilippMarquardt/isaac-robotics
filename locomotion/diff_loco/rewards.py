"""Custom reward functions for diff_loco."""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def base_height_safe_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    max_err: float = 0.5,
) -> torch.Tensor:
    """Terrain-aware base-height L2 penalty, robust to gaps/pits.

    Stock base_height_l2 uses mean(ray_hits_w[..., 2]); on parkour terrains
    with pits/gaps some rays return +/-inf, making the mean inf and nuking
    training. This version replaces non-finite hits with the median of the
    finite hits (per env), falling back to 0.0 if an env has no finite hits.
    Error is clipped at +/-max_err so spikes on cliffs don't explode value loss.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]  # (N, R)
    finite_mask = torch.isfinite(hits_z)
    # per-env median of finite hits; if none finite, use 0.0
    safe_hits = torch.where(finite_mask, hits_z, torch.zeros_like(hits_z))
    any_finite = finite_mask.any(dim=1)
    # mean of finite hits (masked)
    count = finite_mask.sum(dim=1).clamp(min=1).float()
    mean_hit = safe_hits.sum(dim=1) / count
    mean_hit = torch.where(any_finite, mean_hit, torch.zeros_like(mean_hit))
    adjusted_target = target_height + mean_hit
    err = asset.data.root_pos_w[:, 2] - adjusted_target
    err = torch.clamp(err, -max_err, max_err)
    return err * err


def alive_when_stationary(
    env: ManagerBasedRLEnv,
    max_cmd: float = 0.1,
) -> torch.Tensor:
    """is_alive, but only when the velocity command is near zero.

    Kills the flat-terrain stand-still local optimum: passive survival reward
    is only collected when commanded to stand. When commanded to move, the
    policy must earn reward by tracking / stepping.
    """
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_mag = torch.norm(cmd, dim=-1)
    is_stationary_cmd = (cmd_mag < max_cmd).float()
    not_terminated = (~env.termination_manager.terminated).float()
    return not_terminated * is_stationary_cmd


def walking_yaw_l2(
    env: ManagerBasedRLEnv,
    min_lin_cmd: float = 0.3,
    max_yaw_cmd: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on |ang_vel_z|^2 when forward-commanded with no yaw command.

    Kills the gap-bail circling optimum: policy farms survival by spinning
    in place when it sees a hard obstacle (gap, stones), avoiding -200
    base_contact. Active only when |cmd_lin_xy| > min_lin_cmd AND
    |cmd_yaw| < max_yaw_cmd, so it does NOT punish commanded turns or
    natural yaw correction during stair climb.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")
    cmd_lin = torch.norm(cmd[:, :2], dim=-1)
    cmd_yaw = cmd[:, 2].abs()
    walking_mask = ((cmd_lin > min_lin_cmd) & (cmd_yaw < max_yaw_cmd)).float()
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    return walking_mask * (yaw_rate * yaw_rate)


def stalled_penalty(
    env: ManagerBasedRLEnv,
    min_cmd: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on velocity shortfall when a nonzero command is issued.

    shortage = max(|cmd_xy| - |vel_xy|, 0); returns shortage^2 gated by
    (|cmd_xy| > min_cmd). Crushes the survive-and-stand local optimum that PPO
    hits when the tracking-exp reward is too flat around zero velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    cmd_mag = torch.norm(cmd, dim=-1)
    vel_mag = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    shortage = torch.clamp(cmd_mag - vel_mag, min=0.0)
    is_commanded = (cmd_mag > min_cmd).float()
    return (shortage ** 2) * is_commanded
