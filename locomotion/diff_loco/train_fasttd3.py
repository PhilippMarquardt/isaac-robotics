"""Launch FastTD3 on IsaacLab G1 locomotion env.

Usage:
    python -m diff_loco.train_fasttd3 \\
        --task Isaac-Velocity-Mega-G1-V26-v0 \\
        --num_envs 4096 \\
        --max_env_steps 120000 \\
        --headless

Not tied to RSL-RL runner — uses raw IsaacLab env.step / env.reset directly.
Saves ckpts + scalar log to logs/fasttd3/<task>/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI (parse BEFORE AppLauncher so --headless etc. work)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train FastTD3 on IsaacLab G1 locomotion.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_env_steps", type=int, default=120_000, help="Total env-steps (rollout iters).")
parser.add_argument("--warmup_env_steps", type=int, default=25)
parser.add_argument("--utd", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=32768)
parser.add_argument("--buffer_size", type=int, default=1_000_000)
parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
parser.add_argument("--n_atoms", type=int, default=101)
parser.add_argument("--v_min", type=float, default=-500.0)
parser.add_argument("--v_max", type=float, default=500.0)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--actor_lr", type=float, default=3e-4)
parser.add_argument("--critic_lr", type=float, default=3e-4)
parser.add_argument("--exploration_noise", type=float, default=0.1)
parser.add_argument("--policy_noise", type=float, default=0.2)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Heavy imports after AppLauncher
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from diff_loco.tasks import register_tasks  # noqa: E402
from diff_loco.fasttd3 import FastTD3, FastTD3Cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

register_tasks()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Obs extraction — support dict obs ({"policy","critic"}) or flat tensor
# -----------------------------------------------------------------------------
def _extract_obs(obs_raw) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (actor_obs, critic_obs). If env only has "policy", use it for both."""
    if isinstance(obs_raw, dict):
        pol = obs_raw["policy"]
        cri = obs_raw.get("critic", pol)
        return pol, cri
    return obs_raw, obs_raw


def _flatten_obs(o: torch.Tensor) -> torch.Tensor:
    if o.dim() == 2:
        return o
    return o.reshape(o.shape[0], -1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):  # noqa: C901
    # CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    device = args_cli.device if args_cli.device is not None else "cuda:0"
    task = args_cli.task
    log_root = os.path.abspath(os.path.join("logs", "fasttd3", task))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, stamp)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(task, cfg=env_cfg, render_mode=None)

    obs_raw, _ = env.reset()
    actor_obs, critic_obs = _extract_obs(obs_raw)
    actor_obs = _flatten_obs(actor_obs)
    critic_obs = _flatten_obs(critic_obs)
    obs_dim = actor_obs.shape[-1]
    crit_dim = critic_obs.shape[-1]
    act_dim = int(np.prod(env.action_space.shape[1:]))
    # IsaacLab actions are typically [-1, 1] scaled; use action_scale=1.0
    print(f"[FastTD3] obs_dim={obs_dim} crit_dim={crit_dim} act_dim={act_dim} num_envs={args_cli.num_envs}")

    cfg = FastTD3Cfg(
        obs_dim=obs_dim,
        crit_dim=crit_dim,
        act_dim=act_dim,
        hidden=tuple(args_cli.hidden),
        n_atoms=args_cli.n_atoms,
        v_min=args_cli.v_min,
        v_max=args_cli.v_max,
        gamma=args_cli.gamma,
        tau=args_cli.tau,
        actor_lr=args_cli.actor_lr,
        critic_lr=args_cli.critic_lr,
        policy_noise=args_cli.policy_noise,
        exploration_noise=args_cli.exploration_noise,
        policy_delay=args_cli.policy_delay,
        buffer_size=args_cli.buffer_size,
        batch_size=args_cli.batch_size,
        utd=args_cli.utd,
        warmup_steps=args_cli.warmup_env_steps,
        action_scale=1.0,
        device=device,
    )
    agent = FastTD3(cfg)

    start_step = 0
    if args_cli.resume is not None and os.path.exists(args_cli.resume):
        sd = torch.load(args_cli.resume, map_location=device)
        agent.load_state_dict(sd["agent"])
        start_step = sd.get("env_step", 0)
        print(f"[FastTD3] resumed from {args_cli.resume} at env_step={start_step}")

    # save config
    with open(os.path.join(log_dir, "cfg.json"), "w") as f:
        json.dump({k: str(v) for k, v in vars(args_cli).items()}, f, indent=2)

    best_rew = -math.inf
    rew_window = []  # running episode reward means
    ep_ret = torch.zeros(args_cli.num_envs, device=device)
    metrics_log = []
    log_path = os.path.join(log_dir, "log.jsonl")
    log_f = open(log_path, "w", buffering=1)

    # graceful shutdown
    shutdown = {"flag": False}

    def _handle(sig, frame):
        shutdown["flag"] = True
        print("\n[FastTD3] shutdown requested, saving ckpt...")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    t0 = time.time()
    step_i = start_step
    while step_i < args_cli.max_env_steps and not shutdown["flag"]:
        # ---- act ----
        if step_i < args_cli.warmup_env_steps:
            act = torch.empty(args_cli.num_envs, act_dim, device=device).uniform_(-1.0, 1.0)
        else:
            act = agent.act(actor_obs, noise=True)

        # ---- step env ----
        act_np = act  # IsaacLab accepts torch tensors
        obs_raw, rew, term, trunc, info = env.step(act_np)
        next_actor, next_critic = _extract_obs(obs_raw)
        next_actor = _flatten_obs(next_actor)
        next_critic = _flatten_obs(next_critic)
        done = (term | trunc).float() if isinstance(term, torch.Tensor) else torch.tensor(term | trunc, device=device).float()
        rew_t = rew.float() if isinstance(rew, torch.Tensor) else torch.tensor(rew, device=device).float()

        # ---- store ----
        agent.buffer.add_batch(
            obs=actor_obs,
            crit=critic_obs,
            act=act,
            rew=rew_t,
            next_obs=next_actor,
            next_crit=next_critic,
            done=done,
        )

        # ---- episode-return tracking ----
        ep_ret += rew_t
        if done.any():
            finished = ep_ret[done.bool()]
            for v in finished.cpu().tolist():
                rew_window.append(v)
            if len(rew_window) > 200:
                rew_window = rew_window[-200:]
            ep_ret[done.bool()] = 0.0

        actor_obs = next_actor
        critic_obs = next_critic

        # ---- learn ----
        if step_i >= args_cli.warmup_env_steps and agent.buffer.size >= args_cli.batch_size:
            for _ in range(args_cli.utd):
                batch = agent.buffer.sample(args_cli.batch_size)
                agent.update(batch)

        # ---- log ----
        if step_i % args_cli.log_interval == 0:
            dt = time.time() - t0
            mean_rew = float(np.mean(rew_window)) if rew_window else float("nan")
            row = {
                "env_step": step_i,
                "time": round(dt, 1),
                "mean_ep_return": mean_rew,
                "buffer": agent.buffer.size,
                "sps": round(args_cli.num_envs * (step_i - start_step) / max(dt, 1e-3), 1),
            }
            msg = f"step={step_i:6d}  ep_ret={mean_rew:7.2f}  buf={agent.buffer.size:>7d}  sps={row['sps']:8.0f}  t={dt:7.1f}s"
            print(msg, flush=True)
            log_f.write(json.dumps(row) + "\n")
            metrics_log.append(row)

        # ---- save ----
        if step_i > 0 and step_i % args_cli.save_interval == 0:
            ckpt = {"agent": agent.state_dict(), "env_step": step_i, "cfg": cfg.__dict__}
            torch.save(ckpt, os.path.join(log_dir, "model_last.pt"))
            if rew_window and np.mean(rew_window) > best_rew:
                best_rew = float(np.mean(rew_window))
                torch.save(ckpt, os.path.join(log_dir, "model_best.pt"))

        step_i += 1

    # final save
    ckpt = {"agent": agent.state_dict(), "env_step": step_i, "cfg": cfg.__dict__}
    torch.save(ckpt, os.path.join(log_dir, "model_last.pt"))
    log_f.close()
    env.close()
    print(f"[FastTD3] done. log_dir={log_dir}  total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
