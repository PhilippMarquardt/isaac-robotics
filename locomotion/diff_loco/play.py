"""Play a trained diff_loco checkpoint in the IsaacLab GUI.

Thin copy of IsaacLab's scripts/reinforcement_learning/rsl_rl/play.py with
`register_tasks()` injected after AppLauncher so Hydra can resolve the task.

Usage:
    python -m diff_loco.play --task Isaac-Velocity-Rough-G1-Diffusion-Play-v0 \
        --num_envs 16
"""
from __future__ import annotations

import argparse
import os
import sys

# Import torch + numpy BEFORE AppLauncher. Two Windows-specific issues:
#   1. Isaac Sim loads GPU DLLs that shadow torch's CUDA runtime → WinError 1114 on c10.dll.
#   2. Isaac Sim prepends its bundled numpy to sys.path; humanoid_rl's numpy package then
#      lazy-imports .random from that bundled copy, causing dtype-size ABI mismatch.
# Eager-import numpy submodules so they bind to the env's numpy before sys.path is polluted.
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

parser = argparse.ArgumentParser(description="Play an RSL-RL diff_loco checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
except ImportError:
    def handle_deprecated_rsl_rl_cfg(cfg, _ver):
        return cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from diff_loco.tasks import register_tasks  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

register_tasks()

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pretrained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # Auto-convert old rsl_rl (ActorCritic single-module) pretrained to new schema.
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
        print(f"[INFO]: Converted legacy checkpoint -> {new_path}")
        runner.load(new_path, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False}, strict=False)
    else:
        runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        try:
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        except NotImplementedError:
            print("[INFO]: jit export not implemented for this policy; skipping")
        try:
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        except NotImplementedError:
            print("[INFO]: onnx export not implemented for this policy; skipping")
    else:
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
