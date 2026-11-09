"""Zero-action viewer for diff_loco tasks. Useful to inspect a terrain
visually without needing a trained checkpoint.

Usage:
    python -m diff_loco.zero_agent --task Isaac-Velocity-Parkour-G1-V53-Play-v0 --num_envs 4
"""
from __future__ import annotations

import argparse

import numpy  # noqa: F401  (eager-import numpy first; see diff_loco.play header)
import numpy.random  # noqa: F401
import numpy.linalg  # noqa: F401
import numpy.core  # noqa: F401
import torch  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action viewer for diff_loco envs.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import diff_loco.tasks as _diff_tasks  # noqa: E402
_diff_tasks.register_tasks()

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] obs space: {env.observation_space}")
    print(f"[INFO] act space: {env.action_space}")
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
