"""Diffusion-policy PPO for G1 rough locomotion.

Do NOT eagerly import tasks here — the task cfg pulls isaaclab.utils which
requires the Isaac Sim app (pxr) to be up. Call `register_tasks()` manually
after AppLauncher finishes (see diff_loco/train.py).
"""
