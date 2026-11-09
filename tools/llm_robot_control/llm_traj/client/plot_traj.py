"""Read llm_traj/logs/episode_live.jsonl and plot the EE trajectory.

Four panels:
  (a) 3D trajectory with cube marker + start/end markers
  (b) Top-down XY
  (c) Z over tool-call index (waypoint timeline)
  (d) Table of commanded vs actual pos_err per move
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "episode_live.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "..", "logs", "episode_live.png")


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows = load(LOG)
    waypoints = []            # commanded positions from move_ee_to
    actual_positions = []     # EE pos after each call
    gripper_events = []       # (idx, state)
    cube_pos = None
    cube_rest_z = None

    for i, r in enumerate(rows):
        result = r.get("result", {})
        if cube_pos is None and "cube_pos" in result:
            cube_pos = result["cube_pos"]
        if cube_rest_z is None and "cube_rest_z_world" in result:
            cube_rest_z = result["cube_rest_z_world"]
        if r["cmd"] == "move_ee_to":
            waypoints.append({
                "idx": i,
                "target": json.loads(r["args"]["pos"]),
                "actual": result.get("ee_pos"),
                "pos_err": result.get("pos_err"),
                "arrived": result.get("arrived"),
            })
        if "ee_pos" in result:
            actual_positions.append(result["ee_pos"])
        if r["cmd"] == "set_gripper":
            gripper_events.append((i, r["args"]["state"]))

    fig = plt.figure(figsize=(14, 10))

    # 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    if actual_positions:
        xs = [p[0] for p in actual_positions]
        ys = [p[1] for p in actual_positions]
        zs = [p[2] for p in actual_positions]
        ax.plot(xs, ys, zs, "-o", color="#1f77b4", label="EE actual", markersize=4)
    if waypoints:
        tx = [w["target"][0] for w in waypoints]
        ty = [w["target"][1] for w in waypoints]
        tz = [w["target"][2] for w in waypoints]
        ax.plot(tx, ty, tz, "x--", color="#d62728", label="LLM target", markersize=10)
    if cube_pos is not None:
        ax.scatter([cube_pos[0]], [cube_pos[1]], [cube_pos[2]], c="red", s=150, marker="s", label="cube")
    ax.set_xlabel("x (m, robot frame)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D trajectory — LLM targets vs EE actual")
    ax.legend(fontsize=8)

    # Top-down
    ax2 = fig.add_subplot(2, 2, 2)
    if actual_positions:
        ax2.plot([p[0] for p in actual_positions], [p[1] for p in actual_positions],
                 "-o", color="#1f77b4", label="EE actual", markersize=4)
    if waypoints:
        ax2.plot([w["target"][0] for w in waypoints], [w["target"][1] for w in waypoints],
                 "x--", color="#d62728", label="LLM target", markersize=10)
    if cube_pos is not None:
        ax2.scatter([cube_pos[0]], [cube_pos[1]], c="red", s=150, marker="s", label="cube")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.set_title("Top-down XY"); ax2.axis("equal"); ax2.grid(True); ax2.legend(fontsize=8)

    # Z over call index
    ax3 = fig.add_subplot(2, 2, 3)
    if actual_positions:
        ax3.plot(range(len(actual_positions)), [p[2] for p in actual_positions],
                 "-o", color="#1f77b4", label="EE z")
    if waypoints:
        ax3.plot([w["idx"] for w in waypoints], [w["target"][2] for w in waypoints],
                 "x--", color="#d62728", label="target z")
    if cube_pos is not None:
        ax3.axhline(cube_pos[2], color="red", linestyle=":", label=f"cube z={cube_pos[2]:.3f}")
    for idx, state in gripper_events:
        ax3.axvline(idx, color="green" if state == "close" else "orange",
                    linestyle="--", alpha=0.5, label=f"gripper {state} @ {idx}")
    ax3.set_xlabel("tool call idx"); ax3.set_ylabel("z (m)")
    ax3.set_title("Z vs tool call"); ax3.grid(True); ax3.legend(fontsize=8)

    # Table
    ax4 = fig.add_subplot(2, 2, 4); ax4.axis("off")
    lines = ["Episode tool calls:\n"]
    for i, r in enumerate(rows):
        args = r.get("args", {})
        result = r.get("result", {})
        if r["cmd"] == "move_ee_to":
            lines.append(f"{i:02d}  move_ee_to pos={args['pos']}  gripper={args.get('gripper')}"
                         f"  arrived={result.get('arrived')}  err={result.get('pos_err', 0):.3f}")
        elif r["cmd"] == "set_gripper":
            lines.append(f"{i:02d}  set_gripper {args['state']}  opening={result.get('gripper_opening', 0):.3f}")
        elif r["cmd"] == "reset":
            lines.append(f"{i:02d}  reset seed={args.get('seed')}  cube={result.get('cube_pos')}")
        elif r["cmd"] == "check_lifted":
            lines.append(f"{i:02d}  check_lifted lifted={result.get('lifted')}  Δz={result.get('delta_from_rest', 0):.3f}")
        elif r["cmd"] == "is_grasped":
            lines.append(f"{i:02d}  is_grasped grasped={result.get('grasped')}")
        else:
            lines.append(f"{i:02d}  {r['cmd']}")
    ax4.text(0.01, 0.98, "\n".join(lines), family="monospace", fontsize=8, va="top")

    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
