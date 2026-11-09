"""Privileged observations for asymmetric PPO critic.

Sim-only ground truth info that helps the value function estimate returns
more accurately without leaking into the deployable actor.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse


def feet_contact_force_z(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """(N, B) per-foot vertical ground-reaction force (tanh-squashed)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return torch.tanh(forces * 0.001)


def feet_contact_binary(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """(N, B) per-foot contact bit."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    mag = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return (mag > threshold).float()


def base_lin_acc_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 3) root linear acceleration in body frame (tanh-squashed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_lin_acc_w[:, 0]
    acc_b = quat_rotate_inverse(asset.data.root_quat_w, acc_w)
    return torch.tanh(acc_b * 0.05)


def base_ang_acc_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 3) root angular acceleration in body frame (tanh-squashed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_ang_acc_w[:, 0]
    acc_b = quat_rotate_inverse(asset.data.root_quat_w, acc_w)
    return torch.tanh(acc_b * 0.05)


def external_force_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 3) applied external perturbation force on root, body frame, tanh-squashed."""
    asset: Articulation = env.scene[asset_cfg.name]
    f = getattr(asset, "_external_force_b", None)
    if f is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return torch.tanh(f[:, 0].to(env.device) * 0.01)


def friction_coef(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 1) mean static friction across bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    try:
        mp = asset.root_physx_view.get_material_properties()
        out = mp[..., 0].mean(dim=-1, keepdim=True)
    except Exception:
        out = torch.ones((env.num_envs, 1))
    return out.to(env.device)


def root_height_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(N, 1) root z in world."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]
