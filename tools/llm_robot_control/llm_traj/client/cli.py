"""One-shot CLI wrapper around SimClient so each tool call is a single Bash invocation.

Truncates the episode log on `reset`. Every call appends a row to
`llm_traj/logs/episode_live.jsonl` with (cmd, args, result).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from llm_traj.client.tools import SimClient

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "episode_live.jsonl")
LOG_PATH = os.path.abspath(LOG_PATH)


def _log(row: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("reset"); p.add_argument("--seed", type=int, default=None); p.add_argument("--task", default=None)
    sub.add_parser("get_state")
    p = sub.add_parser("move_ee_to")
    p.add_argument("pos", help='JSON list, e.g. "[0.5, 0.0, 0.2]"')
    p.add_argument("quat", help='JSON list wxyz, e.g. "[0,1,0,0]"')
    p.add_argument("--gripper", default=None, choices=[None, "open", "close"])
    p.add_argument("--hand", default=None, choices=[None, "left", "right"])
    p.add_argument("--tol_pos", type=float, default=0.01)
    p.add_argument("--tol_rot", type=float, default=0.3)
    p.add_argument("--max_steps", type=int, default=240)
    p = sub.add_parser("set_gripper"); p.add_argument("state", choices=["open", "close"]); p.add_argument("--hold_steps", type=int, default=20); p.add_argument("--hand", default=None)
    sub.add_parser("is_grasped")
    p = sub.add_parser("check_lifted"); p.add_argument("--min_lift", type=float, default=0.10)
    p = sub.add_parser("walk_to")
    p.add_argument("x", type=float); p.add_argument("y", type=float)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--tol_pos", type=float, default=0.15)
    p.add_argument("--max_steps", type=int, default=900)

    args = ap.parse_args()
    if args.cmd == "reset":
        # fresh episode log
        try:
            os.remove(LOG_PATH)
        except FileNotFoundError:
            pass

    c = SimClient()
    try:
        if args.cmd == "reset":
            r = c.reset(seed=args.seed, task=args.task)
        elif args.cmd == "get_state":
            r = c.get_state()
        elif args.cmd == "move_ee_to":
            r = c.move_ee_to(json.loads(args.pos), json.loads(args.quat),
                             gripper=args.gripper, tol_pos=args.tol_pos, tol_rot=args.tol_rot,
                             max_steps=args.max_steps, hand=args.hand)
        elif args.cmd == "set_gripper":
            r = c.set_gripper(args.state, hold_steps=args.hold_steps, hand=args.hand)
        elif args.cmd == "is_grasped":
            r = c.is_grasped()
        elif args.cmd == "check_lifted":
            r = c.check_lifted(min_lift=args.min_lift)
        elif args.cmd == "walk_to":
            r = c.walk_to(args.x, args.y, yaw=args.yaw, tol_pos=args.tol_pos, max_steps=args.max_steps)
        else:
            raise SystemExit(f"unknown cmd {args.cmd}")
    finally:
        c.close()

    _log({"t": time.time(), "cmd": args.cmd, "args": {k: v for k, v in vars(args).items() if k != "cmd"},
          "result": r})
    # compact summary to stdout
    keep = ("arrived", "pos_err", "rot_err", "lifted", "grasped", "cube_height", "delta_from_rest",
            "gripper_opening", "left_gripper_opening", "ee_pos", "ee_quat_wxyz",
            "left_ee_pos", "left_ee_quat_wxyz", "active_hand",
            "cube_pos", "cube_quat_wxyz", "cube_rest_z_world", "task", "steps_used")
    print(json.dumps({k: v for k, v in r.items() if k in keep}, indent=2))


if __name__ == "__main__":
    main()
