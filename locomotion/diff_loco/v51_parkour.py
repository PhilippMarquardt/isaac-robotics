"""V51 — V50 stock-G1 + ONE change: 5-frame proprioception history obs.

V50/VBaseV1 plateau ~5.7 mean lvl on parkour terrain mix under upstream
``terrain_levels_vel`` (promote 4m, demote cmd*ep*0.5). Per-terrain split:
flat 5.66, slope_up 5.72, slope_dn 6.30, p_stairs 5.90, boxes 5.87,
**stones 5.47**, **gaps 5.79**, **stairs_inv 5.76**. Hard terrains
(stones/gaps/stairs_inv) lag — they need temporal foot-placement
planning the single-step obs cannot provide.

Curriculum is NOT touched (was tempting to lower bars but that just
relabels the meter — V30-plain "lvl 8.45" was bar-relaxation, not
skill).

Single change vs V50: stack 5 steps of joint_pos, joint_vel, last_action
in the policy obs group via ``history_length=5``. Heightmap stays
single-step (history would explode obs dim). Commands and base
linear/angular velocity stay single-step (already smoothed by IMU).

Hypothesis: temporal proprioception lets policy plan foot trajectories
for stones (precise landing) / gaps (pre-jump prep) / stairs_inv
(descent timing). On easy terrains effect should be small.

Obs growth: per-step joint dims = 23+23+23 = 69; +4 extra steps =
+276 dim. Total policy obs grows from ~200 to ~480.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from diff_loco.v50_parkour import (
    G1V50ParkourEnvCfg,
    G1V50ParkourEnvCfg_PLAY,
    G1V50ParkourPPORunnerCfg,
    V50ObservationsCfg,
)


_HIST = 5


@configclass
class V51ObservationsCfg(V50ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=_HIST,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=_HIST,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=_HIST,
            flatten_history_dim=True,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1V51ParkourEnvCfg(G1V50ParkourEnvCfg):
    observations: V51ObservationsCfg = V51ObservationsCfg()


@configclass
class G1V51ParkourEnvCfg_PLAY(G1V50ParkourEnvCfg_PLAY):
    observations: V51ObservationsCfg = V51ObservationsCfg()


@configclass
class G1V51ParkourPPORunnerCfg(G1V50ParkourPPORunnerCfg):
    experiment_name = "g1_v51_parkour"


def register_v51_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V51-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V51-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v51_parkour:G1V51ParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v51_parkour:G1V51ParkourPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V51-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v51_parkour:G1V51ParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v51_parkour:G1V51ParkourPPORunnerCfg",
        },
    )
