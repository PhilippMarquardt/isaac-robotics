"""WASD-controlled single-robot playback for diff_loco checkpoints.

Key bindings:
    W / S          : forward / backward (vx)
    A / D          : strafe left / right (vy)
    Q / E          : yaw left / right (wz)
    Arrow keys     : same as WASD
    SHIFT          : 2x speed boost while held
    SPACE          : zero command (stand)
    R              : reset robot
    ESC            : quit

Usage (Windows, env_isaaclab):
    python -m diff_loco.play_keyboard \\
        --task Isaac-Velocity-Parkour-G1-V54-Play-v0 \\
        --checkpoint logs/rsl_rl/g1_v54_parkour/<run>/model_last.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy  # noqa: F401
import numpy.random  # noqa: F401
import numpy.linalg  # noqa: F401
import numpy.core  # noqa: F401
import torch  # noqa: F401

from isaaclab.app import AppLauncher

_ISAAC_SCRIPT_DIR = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab")),
    "scripts", "reinforcement_learning", "rsl_rl",
)
sys.path.insert(0, _ISAAC_SCRIPT_DIR)
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="WASD-steer a diff_loco checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--vx_max", type=float, default=1.5, help="forward/back speed cap (m/s)")
parser.add_argument("--vy_max", type=float, default=0.8, help="strafe speed cap (m/s)")
parser.add_argument("--wz_max", type=float, default=1.5, help="yaw rate cap (rad/s)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402
import weakref  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402
import omni  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
except ImportError:
    def handle_deprecated_rsl_rl_cfg(cfg, _ver):
        return cfg

import isaaclab_tasks  # noqa: E402, F401
from diff_loco.tasks import register_tasks  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

register_tasks()
installed_version = metadata.version("rsl-rl-lib")


class WasdTeleop:
    """Press-and-hold WASD/arrow keys to set body-frame velocity command."""

    def __init__(self, vx_max: float, vy_max: float, wz_max: float, device: str):
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.wz_max = wz_max
        self.device = device
        # axis state — accumulates +1/-1 from key holds
        self._axis = np.zeros(3, dtype=np.float32)  # [vx, vy, wz] in normalized units
        self._boost = False
        self._reset_pressed = False
        self._quit = False

        # carb input subscription
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

        # key → (axis_index, sign)  — vy is positive-left (body frame)
        self._press_map = {
            "W": (0, +1.0), "UP": (0, +1.0),
            "S": (0, -1.0), "DOWN": (0, -1.0),
            "A": (1, +1.0), "LEFT": (1, +1.0),
            "D": (1, -1.0), "RIGHT": (1, -1.0),
            "Q": (2, +1.0),
            "E": (2, -1.0),
        }

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def _on_event(self, event, *args, **kwargs):
        inp = event.input
        name = getattr(inp, "name", None) or str(inp)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in self._press_map:
                idx, sign = self._press_map[name]
                self._axis[idx] += sign
            elif name in ("LEFT_SHIFT", "RIGHT_SHIFT"):
                self._boost = True
            elif name == "SPACE":
                self._axis[:] = 0.0
            elif name == "R":
                self._reset_pressed = True
            elif name == "ESCAPE":
                self._quit = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if name in self._press_map:
                idx, sign = self._press_map[name]
                self._axis[idx] -= sign
            elif name in ("LEFT_SHIFT", "RIGHT_SHIFT"):
                self._boost = False
        return True

    def command(self) -> torch.Tensor:
        a = np.clip(self._axis, -1.0, 1.0)
        scale = 2.0 if self._boost else 1.0
        cmd = np.array([
            a[0] * self.vx_max * scale,
            a[1] * self.vy_max * scale,
            a[2] * self.wz_max * scale,
        ], dtype=np.float32)
        return torch.from_numpy(cmd).to(self.device)

    def pop_reset(self) -> bool:
        v = self._reset_pressed
        self._reset_pressed = False
        return v


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # force single robot
    env_cfg.scene.num_envs = 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disable command resampling: keep the manager but make ranges tiny so any
    # accidental resample produces ~zero, then we overwrite vel_command_b each step.
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        bv = env_cfg.commands.base_velocity
        bv.resampling_time_range = (1.0e9, 1.0e9)
        bv.rel_standing_envs = 0.0
        if hasattr(bv, "rel_heading_envs"):
            bv.rel_heading_envs = 0.0
        if hasattr(bv, "heading_command"):
            bv.heading_command = False

    # shrink the world: 1x1 terrain grid (still has a few sub-terrains depending on cfg)
    try:
        terr = env_cfg.scene.terrain
        if getattr(terr, "terrain_generator", None) is not None:
            tg = terr.terrain_generator
            tg.num_rows = 1
            tg.num_cols = 1
            tg.curriculum = False
            # pick the easiest sub-terrain (level 0) by collapsing difficulty range
            if hasattr(tg, "difficulty_range"):
                tg.difficulty_range = (0.0, 0.0)
    except AttributeError:
        pass

    # third-person follow camera locked to the robot
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-3.0, -3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    # locate checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pretrained checkpoint available."); return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    _loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in _loaded and "actor_state_dict" not in _loaded:
        _msd = _loaded["model_state_dict"]
        actor_sd, critic_sd = {}, {}
        for k, v in _msd.items():
            if k.startswith("actor."):
                actor_sd["mlp." + k[len("actor."):]] = v
            elif k.startswith("critic."):
                critic_sd["mlp." + k[len("critic."):]] = v
            elif k == "std":
                actor_sd["distribution.std_param"] = v
        converted = {
            "actor_state_dict": actor_sd,
            "critic_state_dict": critic_sd,
            "optimizer_state_dict": {},
            "iter": _loaded.get("iter", 0),
            "infos": _loaded.get("infos"),
        }
        new_path = resume_path.replace(".pt", "_converted.pt")
        torch.save(converted, new_path)
        runner.load(new_path, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False}, strict=False)
    else:
        runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # monkey-patch the velocity command term so it stops auto-resampling/updating
    cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")
    if cmd_term is not None:
        cmd_term._resample_command = lambda env_ids: None
        cmd_term._update_command = lambda: None

    teleop = WasdTeleop(args_cli.vx_max, args_cli.vy_max, args_cli.wz_max, str(env.unwrapped.device))

    print("=" * 70)
    print(" WASD STEER — focus the Isaac Sim window then use:")
    print("   W/S forward/back   A/D strafe   Q/E yaw   SHIFT boost")
    print("   SPACE stand   R reset   ESC quit")
    print(f"   caps: vx=±{args_cli.vx_max:.1f}  vy=±{args_cli.vy_max:.1f}  wz=±{args_cli.wz_max:.1f}")
    print("=" * 70)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running() and not teleop._quit:
        start_time = time.time()

        if cmd_term is not None:
            cmd_term.vel_command_b[0] = teleop.command()

        if teleop.pop_reset():
            obs, _ = env.reset()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
