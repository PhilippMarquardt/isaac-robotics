"""V28u G1 Parkour — V28o base + literature-standard avoid-death fix.

V28o was the last good run before the V28p/q/r/s/t series of narrow fixes.
Only problem: avoid-death / fence-sit attractor on hard terrain.

Literature recipe for parkour reward (research 2026-04-25):

  | Stack            | alive | termination | track_sigma | only_pos |
  |------------------|-------|-------------|-------------|----------|
  | Extreme Parkour  | none  | none        | 0.2         | true     |
  | Booster Gym T1   | 0.25  | none        | 0.25        | true     |
  | Unitree rl_gym   | 0.15  | -0.0 (off)  | 0.25        | -        |
  | ANYmal Parkour   | none  | -200        | (pos-track) | -        |
  | IsaacLab default | none  | -200        | 0.5         | -        |
  | V28o (broken)    | +0.15 | -100        | 0.5         | -        |

Modern parkour recipe (Extreme Parkour, Booster Gym, Unitree):
  drop alive + drop termination penalty + tight sigma (0.2-0.25)

V28o fence-sit math at cmd=(1,0,0), v=(0,0):
  track_lin_vel_xy_exp (std=0.5):  exp(-1/0.25) × 2.0 = +0.036/step
  alive:                                              = +0.15/step
  per 1000-step episode:                              +186
  termination on fall:                                -100
  net standing-on-flat (no fall): +186  (fence-sit wins outright)
  net standing-on-edge (no fall): +186  (avoid-death attractor)
  net trying-and-falling:         track_until_fall - 100 ≈ -80
  net trying-and-clearing:        ≈ +1800

V28u three-change fix vs V28o (literature-standard):

  1. Drop alive bonus (Extreme Parkour, ANYmal Parkour, IsaacLab default).
  2. Drop termination penalty (Extreme Parkour, Booster Gym, Unitree).
     Falling still costs naturally via lin_vel_z_l2, ang_vel_xy_l2,
     dof_torques_l2, action_rate_l2, dof_pos_limits — no need for -100.
  3. Tighten track_lin_vel_xy_exp std 0.5 -> 0.25 (Booster Gym, Unitree).
     std=0.25: exp(-1/0.0625) = 1e-7 at standing, exp(-0.25/0.0625) = 0.018
     at half-speed, exp(-0.0625/0.0625) = 0.37 at v=0.75, 1.0 at perfect.

V28u math at cmd=(1,0,0):
  standing-on-flat:    ~0/step × 1000  =  ~0   (fence-sit dies)
  trying-and-falling:  +track_partial - 0  = +50 to +200 (no termination)
  trying-and-clearing: ~2/step × 1000 = +2000

Trying-and-falling > standing > trying-and-clearing. Avoid-death structurally
killed: any forward motion, even ending in fall, beats standing still.

Everything else V28o unchanged: hysteresis curriculum (K=3, demote_floor=1m,
warmup_steps=3200, gap+stairs_inv promote bar 2.0m), per-terrain logger,
rel_standing_envs=0.02, V28d sim2real DR, V28e PPO knobs, plain MLP actor
[768,512,256,128].

Trained from scratch — reward changes the loss landscape, no resume.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from diff_loco.v28o_parkour import (
    G1V28oParkourEnvCfg,
    G1V28oParkourEnvCfg_PLAY,
    G1V28oParkourTeacherPPORunnerCfg,
)


@configclass
class G1V28uParkourEnvCfg(G1V28oParkourEnvCfg):
    """V28o + drop alive + tighten track std → kill fence-sit attractor."""

    def __post_init__(self):
        super().__post_init__()

        # --- 1. Drop alive bonus. Extreme Parkour / ANYmal Parkour / IsaacLab
        # default all run without it. V28b added +0.15 which combined with
        # loose track std created fence-sit attractor. ---
        if hasattr(self.rewards, "alive") and self.rewards.alive is not None:
            self.rewards.alive = None

        # --- 2. Drop termination penalty. Extreme Parkour, Booster Gym,
        # Unitree rl_gym all run with termination=0. Falls still cost via
        # lin_vel_z, ang_vel_xy, dof_torques, action_rate, dof_pos_limits —
        # natural penalties suffice. -100 created fence-sit avoid-death corner. ---
        if hasattr(self.rewards, "termination_penalty") and self.rewards.termination_penalty is not None:
            self.rewards.termination_penalty = None

        # --- 3. Tighten track_lin_vel_xy_exp std 0.5 -> 0.25 (Booster Gym /
        # Unitree rl_gym standard for parkour). Standing reward collapses
        # from 0.018 to 1e-7 (×2 weight = 2e-7/step). ---
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25


@configclass
class G1V28uParkourEnvCfg_PLAY(G1V28oParkourEnvCfg_PLAY):
    """Inherit V28o play overrides; reward changes apply via env cfg."""

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.rewards, "alive") and self.rewards.alive is not None:
            self.rewards.alive = None
        if hasattr(self.rewards, "termination_penalty") and self.rewards.termination_penalty is not None:
            self.rewards.termination_penalty = None
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25


@configclass
class G1V28uParkourTeacherPPORunnerCfg(G1V28oParkourTeacherPPORunnerCfg):
    experiment_name = "g1_v28u_parkour_teacher"


def register_v28u_parkour() -> None:
    import gymnasium as gym
    if "Isaac-Velocity-Parkour-G1-V28u-Teacher-v0" in gym.envs.registry:
        return
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28u-Teacher-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28u_parkour:G1V28uParkourEnvCfg",
            "rsl_rl_cfg_entry_point": "diff_loco.v28u_parkour:G1V28uParkourTeacherPPORunnerCfg",
        },
    )
    gym.register(
        id="Isaac-Velocity-Parkour-G1-V28u-Teacher-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "diff_loco.v28u_parkour:G1V28uParkourEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "diff_loco.v28u_parkour:G1V28uParkourTeacherPPORunnerCfg",
        },
    )
