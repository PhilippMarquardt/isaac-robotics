"""FastTD3: off-policy twin-delayed actor-critic with distributional (C51) critic.

Minimal from-scratch impl for IsaacLab parallel-env humanoid locomotion.
Paper: https://arxiv.org/abs/2505.22642 (Seo et al., May 2025).

Core differences from vanilla TD3:
  1. Distributional critic (C51 with 101 atoms) — stabilizes Q-learning off-policy.
  2. LayerNorm throughout actor and critic — prevents Q-value explosion.
  3. Large batch (32k) + low UTD (1-2) — compute-efficient on single GPU.
  4. Works with 4096+ parallel envs — replay buffer fills in ~250 env-steps.

Not implemented (not needed for humanoid loco, keep impl minimal):
  - SALE latent encoders (for pixels)
  - prioritized replay
  - n-step returns
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# MLP building block
# -----------------------------------------------------------------------------
def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, layernorm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [in_dim, *hidden]
    for i in range(len(hidden)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Actor — deterministic tanh policy
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim, layernorm=True)
        self.action_scale = action_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_scale * torch.tanh(self.net(obs))


# -----------------------------------------------------------------------------
# Distributional critic — C51 head
# -----------------------------------------------------------------------------
class DistCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
        n_atoms: int = 101,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.net = _mlp(obs_dim + act_dim, hidden, n_atoms, layernorm=True)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Return logits [B, n_atoms]."""
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class TwinDistCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int], n_atoms: int) -> None:
        super().__init__()
        self.q1 = DistCritic(obs_dim, act_dim, hidden, n_atoms)
        self.q2 = DistCritic(obs_dim, act_dim, hidden, n_atoms)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, act), self.q2(obs, act)


# -----------------------------------------------------------------------------
# Replay buffer — on-GPU circular tensor store
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        crit_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.crit = torch.zeros((capacity, crit_dim), dtype=torch.float32, device=device)
        self.act = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.rew = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_crit = torch.zeros((capacity, crit_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.idx = 0
        self.size = 0

    def add_batch(
        self,
        obs: torch.Tensor,
        crit: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        next_crit: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        n = obs.shape[0]
        end = self.idx + n
        if end <= self.capacity:
            sl = slice(self.idx, end)
            self.obs[sl] = obs
            self.crit[sl] = crit
            self.act[sl] = act
            self.rew[sl] = rew
            self.next_obs[sl] = next_obs
            self.next_crit[sl] = next_crit
            self.done[sl] = done
            self.idx = end % self.capacity
        else:
            # wrap
            first = self.capacity - self.idx
            self.obs[self.idx :] = obs[:first]
            self.crit[self.idx :] = crit[:first]
            self.act[self.idx :] = act[:first]
            self.rew[self.idx :] = rew[:first]
            self.next_obs[self.idx :] = next_obs[:first]
            self.next_crit[self.idx :] = next_crit[:first]
            self.done[self.idx :] = done[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.crit[:rest] = crit[first:]
            self.act[:rest] = act[first:]
            self.rew[:rest] = rew[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.next_crit[:rest] = next_crit[first:]
            self.done[:rest] = done[first:]
            self.idx = rest
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        ix = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(
            obs=self.obs[ix],
            crit=self.crit[ix],
            act=self.act[ix],
            rew=self.rew[ix],
            next_obs=self.next_obs[ix],
            next_crit=self.next_crit[ix],
            done=self.done[ix],
        )


# -----------------------------------------------------------------------------
# FastTD3 trainer
# -----------------------------------------------------------------------------
@dataclass
class FastTD3Cfg:
    obs_dim: int = 0
    crit_dim: int = 0
    act_dim: int = 0
    hidden: tuple[int, ...] = (512, 512, 512)
    n_atoms: int = 101
    v_min: float = -500.0
    v_max: float = 500.0
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2          # target policy smoothing noise (std)
    noise_clip: float = 0.5
    exploration_noise: float = 0.1     # rollout exploration noise (std)
    policy_delay: int = 2              # actor update frequency (critic updates per actor update)
    buffer_size: int = 1_000_000
    batch_size: int = 32768
    utd: int = 2                       # updates per env step
    warmup_steps: int = 25             # env-steps of random actions before learning (×num_envs transitions)
    action_scale: float = 1.0
    device: str = "cuda:0"


class FastTD3:
    def __init__(self, cfg: FastTD3Cfg) -> None:
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.device = dev

        self.actor = Actor(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.action_scale).to(dev)
        self.actor_targ = Actor(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.action_scale).to(dev)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        for p in self.actor_targ.parameters():
            p.requires_grad_(False)

        self.critic = TwinDistCritic(cfg.crit_dim, cfg.act_dim, cfg.hidden, cfg.n_atoms).to(dev)
        self.critic_targ = TwinDistCritic(cfg.crit_dim, cfg.act_dim, cfg.hidden, cfg.n_atoms).to(dev)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # C51 atoms
        self.atoms = torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms, device=dev)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)

        self.buffer = ReplayBuffer(
            cfg.buffer_size, cfg.obs_dim, cfg.crit_dim, cfg.act_dim, dev
        )
        self.update_count = 0

    # -- rollout helpers ------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, noise: bool = True) -> torch.Tensor:
        a = self.actor(obs)
        if noise:
            a = a + torch.randn_like(a) * self.cfg.exploration_noise * self.cfg.action_scale
        return a.clamp(-self.cfg.action_scale, self.cfg.action_scale)

    # -- projection for C51 ---------------------------------------------------
    @torch.no_grad()
    def _project_target(self, rew: torch.Tensor, done: torch.Tensor, next_probs: torch.Tensor) -> torch.Tensor:
        """Project target distribution onto fixed atoms (C51 Bellman update)."""
        B = rew.shape[0]
        # projected atom positions: r + γ(1-done) z
        not_done = 1.0 - done
        tz = rew.unsqueeze(-1) + self.cfg.gamma * not_done.unsqueeze(-1) * self.atoms.unsqueeze(0)
        tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
        b = (tz - self.cfg.v_min) / self.delta_z  # [B, n_atoms]
        lo = b.floor().long()
        hi = b.ceil().long()
        # handle equal case (lo==hi): push hi up so mass stays (when lo==hi==atom)
        lo_eq_hi = (lo == hi)
        hi = torch.where(lo_eq_hi & (lo < self.cfg.n_atoms - 1), hi + 1, hi)
        lo = torch.where(lo_eq_hi & (lo == self.cfg.n_atoms - 1), lo - 1, lo)
        m = torch.zeros_like(next_probs)
        # distribute next_probs mass to lo/hi
        m.scatter_add_(1, lo, next_probs * (hi.float() - b))
        m.scatter_add_(1, hi, next_probs * (b - lo.float()))
        return m

    # -- learn step -----------------------------------------------------------
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"]
        crit = batch["crit"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        next_crit = batch["next_crit"]
        done = batch["done"]

        # --- critic update ---
        with torch.no_grad():
            next_a = self.actor_targ(next_obs)
            noise = (torch.randn_like(next_a) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (next_a + noise * cfg.action_scale).clamp(-cfg.action_scale, cfg.action_scale)

            logits1, logits2 = self.critic_targ(next_crit, next_a)
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            # Mean-of-expectations, take min via expectation (standard TD3-distributional)
            q1_mean = (probs1 * self.atoms).sum(-1)
            q2_mean = (probs2 * self.atoms).sum(-1)
            use_q1 = (q1_mean <= q2_mean).unsqueeze(-1)
            next_probs = torch.where(use_q1, probs1, probs2)

            target_probs = self._project_target(rew, done, next_probs)

        logits1, logits2 = self.critic(crit, act)
        log_probs1 = F.log_softmax(logits1, dim=-1)
        log_probs2 = F.log_softmax(logits2, dim=-1)
        critic_loss = -(target_probs * log_probs1).sum(-1).mean() - (target_probs * log_probs2).sum(-1).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40.0)
        self.critic_opt.step()

        out = {"critic_loss": critic_loss.item()}

        # --- actor update (delayed) ---
        self.update_count += 1
        if self.update_count % cfg.policy_delay == 0:
            a_pred = self.actor(obs)
            logits_a, _ = self.critic(crit, a_pred)
            probs_a = F.softmax(logits_a, dim=-1)
            q_a = (probs_a * self.atoms).sum(-1)
            actor_loss = -q_a.mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40.0)
            self.actor_opt.step()

            # soft target update
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(p.data, alpha=cfg.tau)
                for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(p.data, alpha=cfg.tau)

            out["actor_loss"] = actor_loss.item()
            out["q_mean"] = q_a.mean().item()

        return out

    # -- checkpoint -----------------------------------------------------------
    def state_dict(self) -> dict:
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            actor_targ=self.actor_targ.state_dict(),
            critic_targ=self.critic_targ.state_dict(),
            actor_opt=self.actor_opt.state_dict(),
            critic_opt=self.critic_opt.state_dict(),
            update_count=self.update_count,
        )

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_targ.load_state_dict(sd["actor_targ"])
        self.critic_targ.load_state_dict(sd["critic_targ"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.update_count = sd["update_count"]
