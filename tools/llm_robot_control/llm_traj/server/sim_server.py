"""Isaac Lab sim process. Holds one Franka IK-Abs Lift env and serves JSON over TCP.

Boot once, reuse for many episodes. Every command is a blocking request/response.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9876)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument(
    "--step_budget",
    type=int,
    default=240,
    help="Max sim steps per move_ee_to call before giving up.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import socket
import traceback

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  — registers envs
from isaaclab_tasks.utils import parse_env_cfg

from llm_traj.server.protocol import recv_message, send_message

# ---------------------------------------------------------------------------
# env setup
# ---------------------------------------------------------------------------

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
# Long horizon so the LLM has a step budget per call. Default is 5s; bump to 60s.
env_cfg.episode_length_s = 60.0
env = gym.make(args_cli.task, cfg=env_cfg)
unwrapped = env.unwrapped
device = unwrapped.device

OPEN, CLOSE = 1.0, -1.0  # BinaryJointPositionActionCfg sign convention


def _ee_pose_robot_frame() -> tuple[list[float], list[float]]:
    """EE pose in robot base frame (same frame the IK action expects)."""
    ee_frame = unwrapped.scene["ee_frame"]
    robot = unwrapped.scene["robot"]
    ee_pos_w = ee_frame.data.target_pos_w[0, 0]
    ee_quat_w = ee_frame.data.target_quat_w[0, 0]
    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]
    from isaaclab.utils.math import subtract_frame_transforms

    pos_b, quat_b = subtract_frame_transforms(
        root_pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
        ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0),
    )
    return pos_b[0].cpu().tolist(), quat_b[0].cpu().tolist()


def _cube_pose_robot_frame() -> tuple[list[float], list[float]]:
    obj = unwrapped.scene["object"]
    robot = unwrapped.scene["robot"]
    from isaaclab.utils.math import subtract_frame_transforms

    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1],
        obj.data.root_pos_w[0:1], obj.data.root_quat_w[0:1],
    )
    return pos_b[0].cpu().tolist(), quat_b[0].cpu().tolist()


def _cube_height_world() -> float:
    return float(unwrapped.scene["object"].data.root_pos_w[0, 2].item())


def _gripper_opening() -> float:
    """Sum of the two finger joint positions (approx opening width in m)."""
    robot = unwrapped.scene["robot"]
    ids, _ = robot.find_joints(["panda_finger.*"])
    return float(robot.data.joint_pos[0, ids].sum().item())


def _current_state(last_success: bool | None = None) -> dict:
    ee_pos, ee_quat = _ee_pose_robot_frame()
    cube_pos, cube_quat = _cube_pose_robot_frame()
    return {
        "ee_pos": ee_pos,
        "ee_quat_wxyz": ee_quat,
        "cube_pos": cube_pos,
        "cube_quat_wxyz": cube_quat,
        "cube_height_world": _cube_height_world(),
        "gripper_opening": _gripper_opening(),
        "sim_time": float(unwrapped.sim.current_time),
        "last_move_success": last_success,
    }


def _step_with_action(target_pos, target_quat, gripper_cmd: float, n_steps: int) -> None:
    action = torch.zeros((1, 8), device=device)
    action[0, 0:3] = torch.tensor(target_pos, device=device)
    action[0, 3:7] = torch.tensor(target_quat, device=device)
    action[0, 7] = gripper_cmd
    for _ in range(n_steps):
        with torch.inference_mode():
            env.step(action)


def _ee_error(target_pos, target_quat) -> tuple[float, float]:
    cur_pos, cur_quat = _ee_pose_robot_frame()
    pos_err = sum((a - b) ** 2 for a, b in zip(cur_pos, target_pos)) ** 0.5
    # quat cosine similarity; handle sign ambiguity
    dot = abs(sum(a * b for a, b in zip(cur_quat, target_quat)))
    rot_err = 2.0 * (1.0 - min(dot, 1.0))  # small-angle proxy
    return pos_err, rot_err


# ---------------------------------------------------------------------------
# command handlers
# ---------------------------------------------------------------------------

_last_gripper_cmd = OPEN
_cube_rest_z_world: float = 0.0  # set on reset, used by check_lifted


def handle_reset(payload: dict) -> dict:
    global _last_gripper_cmd, _cube_rest_z_world
    _last_gripper_cmd = OPEN
    seed = payload.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    with torch.inference_mode():
        env.reset(seed=seed)
    # hold current pose + open gripper for 2 steps to settle observations
    ee_pos, ee_quat = _ee_pose_robot_frame()
    _step_with_action(ee_pos, ee_quat, OPEN, n_steps=2)
    _cube_rest_z_world = _cube_height_world()
    state = _current_state()
    state["task"] = payload.get("task", "pick up the cube and lift it 20 cm above the table")
    state["cube_rest_z_world"] = _cube_rest_z_world
    return {"status": "ok", **state}


def handle_get_state(_payload: dict) -> dict:
    return {"status": "ok", **_current_state()}


def handle_move_ee_to(payload: dict) -> dict:
    """Drive the EE toward (pos, quat) by holding that target through the IK-Abs
    action until convergence or step budget is exhausted. Gripper command is held
    at its last-commanded state unless the caller overrides it.
    """
    global _last_gripper_cmd
    pos = payload["pos"]
    quat = payload["quat_wxyz"]
    gripper = payload.get("gripper")
    if gripper is not None:
        _last_gripper_cmd = OPEN if gripper == "open" else CLOSE
    tol_pos = float(payload.get("tol_pos", 0.01))
    tol_rot = float(payload.get("tol_rot", 0.05))
    budget = int(payload.get("max_steps", args_cli.step_budget))

    steps_done = 0
    per_chunk = 6
    while steps_done < budget:
        chunk = min(per_chunk, budget - steps_done)
        _step_with_action(pos, quat, _last_gripper_cmd, chunk)
        steps_done += chunk
        pos_err, rot_err = _ee_error(pos, quat)
        if pos_err < tol_pos and rot_err < tol_rot:
            break
    pos_err, rot_err = _ee_error(pos, quat)
    success = pos_err < tol_pos and rot_err < tol_rot
    return {
        "status": "ok",
        "arrived": success,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "steps_used": steps_done,
        **_current_state(last_success=success),
    }


def handle_set_gripper(payload: dict) -> dict:
    """Toggle the gripper and hold current EE pose while fingers move."""
    global _last_gripper_cmd
    _last_gripper_cmd = OPEN if payload["state"] == "open" else CLOSE
    ee_pos, ee_quat = _ee_pose_robot_frame()
    _step_with_action(ee_pos, ee_quat, _last_gripper_cmd, n_steps=int(payload.get("hold_steps", 20)))
    return {"status": "ok", **_current_state()}


def handle_is_grasped(_payload: dict) -> dict:
    """Heuristic: cube lifted > 3cm above rest AND gripper not fully closed."""
    cube_z = _cube_height_world()
    opening = _gripper_opening()
    # opening is sum of two finger joints; a grasped ~4cm cube keeps fingers apart ~3-4cm.
    grasped = (cube_z - _cube_rest_z_world > 0.03) and (opening > 0.005)
    return {"status": "ok", "grasped": grasped, "cube_height": cube_z,
            "delta_from_rest": cube_z - _cube_rest_z_world, "opening": opening}


def handle_check_lifted(payload: dict) -> dict:
    """Success if cube lifted at least `min_lift` m above its rest height at reset."""
    min_lift = float(payload.get("min_lift", 0.10))
    cube_z = _cube_height_world()
    delta = cube_z - _cube_rest_z_world
    return {"status": "ok", "lifted": delta >= min_lift, "cube_height": cube_z,
            "delta_from_rest": delta, "rest_z": _cube_rest_z_world}


HANDLERS = {
    "reset": handle_reset,
    "get_state": handle_get_state,
    "move_ee_to": handle_move_ee_to,
    "set_gripper": handle_set_gripper,
    "is_grasped": handle_is_grasped,
    "check_lifted": handle_check_lifted,
}


# ---------------------------------------------------------------------------
# socket loop
# ---------------------------------------------------------------------------

def serve() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args_cli.host, args_cli.port))
    srv.listen(1)
    print(f"[sim_server] listening on {args_cli.host}:{args_cli.port}", flush=True)
    # Initial reset so the env has a valid state before first client connects.
    with torch.inference_mode():
        env.reset()
    while simulation_app.is_running():
        print("[sim_server] waiting for client…", flush=True)
        conn, addr = srv.accept()
        print(f"[sim_server] client {addr} connected", flush=True)
        with conn:
            try:
                while True:
                    try:
                        msg = recv_message(conn)
                    except ConnectionError:
                        break
                    cmd = msg.get("cmd")
                    if cmd == "shutdown":
                        send_message(conn, {"status": "ok", "bye": True})
                        srv.close()
                        env.close()
                        simulation_app.close()
                        return
                    handler = HANDLERS.get(cmd)
                    if handler is None:
                        send_message(conn, {"status": "error", "reason": f"unknown cmd: {cmd}"})
                        continue
                    try:
                        resp = handler(msg)
                    except Exception as e:
                        resp = {"status": "error", "reason": f"{type(e).__name__}: {e}",
                                "tb": traceback.format_exc()}
                    send_message(conn, resp)
            except Exception:
                traceback.print_exc()
        print("[sim_server] client disconnected", flush=True)


if __name__ == "__main__":
    try:
        serve()
    finally:
        simulation_app.close()
