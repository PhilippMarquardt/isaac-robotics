"""Isaac Lab sim process for Unitree G1 locomanipulation.

Loads `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`, which bundles:
  * a pretrained RL locomotion policy driving hips/knees/ankles,
  * a pink IK action for the upper body + hands.

The env's action tensor is 32D = 28D upper-body IK + 4D loco cmd [vx, vy, wz, h].
Same TCP/JSON protocol as sim_server.py, plus a new `walk_to` command.
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9876)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--step_budget", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import socket
import traceback

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from llm_traj.server.protocol import recv_message, send_message

# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
env_cfg.episode_length_s = 300.0
env = gym.make(args_cli.task, cfg=env_cfg)
unwrapped = env.unwrapped
device = unwrapped.device

# Action layout (upper 28, lower 4)
# upper: [left_pos(3), left_quat(4), right_pos(3), right_quat(4), hands(14)]
# lower: [vx, vy, wz, hip_height]
ACTION_DIM = 32
HIP_HEIGHT = 0.72

# Hand joint order from G1_UPPER_BODY_IK_ACTION_CFG.hand_joint_names
RIGHT_HAND_IDX = [3, 4, 5, 9, 10, 11, 13]
LEFT_HAND_IDX = [0, 1, 2, 6, 7, 8, 12]
CLOSE_VALS = [0.8, 0.8, 0.3, 0.8, 0.8, 0.8, 0.8]  # index0, mid0, thumb0, index1, mid1, thumb1, thumb2


def _env_origin() -> torch.Tensor:
    return unwrapped.scene.env_origins[0]


def _body_pose(link: str) -> tuple[list[float], list[float]]:
    robot = unwrapped.scene["robot"]
    idx = robot.data.body_names.index(link)
    pos = (robot.data.body_pos_w[0, idx] - _env_origin()).cpu().tolist()
    quat = robot.data.body_quat_w[0, idx].cpu().tolist()
    return pos, quat


def _root_pose() -> tuple[list[float], list[float]]:
    robot = unwrapped.scene["robot"]
    pos = (robot.data.root_pos_w[0] - _env_origin()).cpu().tolist()
    quat = robot.data.root_quat_w[0].cpu().tolist()
    return pos, quat


def _yaw_from_quat(q: list[float]) -> float:
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrist_pose(side: str) -> tuple[list[float], list[float]]:
    return _body_pose(f"{side}_wrist_yaw_link")


def _object_pose() -> tuple[list[float], list[float]]:
    obj = unwrapped.scene["object"]
    pos = (obj.data.root_pos_w[0] - _env_origin()).cpu().tolist()
    quat = obj.data.root_quat_w[0].cpu().tolist()
    return pos, quat


def _object_height_world() -> float:
    return float(unwrapped.scene["object"].data.root_pos_w[0, 2].item())


def _hand_opening(side: str) -> float:
    robot = unwrapped.scene["robot"]
    ids, _ = robot.find_joints([f"{side}_hand_index_0_joint", f"{side}_hand_middle_0_joint"])
    return float(robot.data.joint_pos[0, ids].sum().item())


# ---------------------------------------------------------------------------
# action builder + global "last command" state
# ---------------------------------------------------------------------------

_active_hand = "right"
_park_left = {"pos": None, "quat": None}
_park_right = {"pos": None, "quat": None}
_last_right_target = {"pos": None, "quat": None}
_last_left_target = {"pos": None, "quat": None}
_right_close = False
_left_close = False
_object_rest_z_world = 0.0
_last_loco = [0.0, 0.0, 0.0, HIP_HEIGHT]


def _build_action() -> torch.Tensor:
    hands = [0.0] * 14
    if _right_close:
        for i, v in zip(RIGHT_HAND_IDX, CLOSE_VALS):
            hands[i] = v
    if _left_close:
        for i, v in zip(LEFT_HAND_IDX, CLOSE_VALS):
            hands[i] = v
    upper = (
        _last_left_target["pos"] + _last_left_target["quat"]
        + _last_right_target["pos"] + _last_right_target["quat"]
        + hands
    )
    action = upper + _last_loco
    assert len(action) == ACTION_DIM, f"expected {ACTION_DIM}, got {len(action)}"
    return torch.tensor([action], dtype=torch.float32, device=device)


def _step_hold(n: int) -> None:
    for _ in range(n):
        action = _build_action()
        with torch.inference_mode():
            env.step(action)


def _ee_error(target_pos, target_quat, side: str) -> tuple[float, float]:
    cur_pos, cur_quat = _wrist_pose(side)
    pos_err = sum((a - b) ** 2 for a, b in zip(cur_pos, target_pos)) ** 0.5
    dot = abs(sum(a * b for a, b in zip(cur_quat, target_quat)))
    rot_err = 2.0 * (1.0 - min(dot, 1.0))
    return pos_err, rot_err


def _current_state(last_success: bool | None = None) -> dict:
    right_pos, right_quat = _wrist_pose("right")
    left_pos, left_quat = _wrist_pose("left")
    obj_pos, obj_quat = _object_pose()
    root_pos, root_quat = _root_pose()
    return {
        "ee_pos": right_pos,
        "ee_quat_wxyz": right_quat,
        "left_ee_pos": left_pos,
        "left_ee_quat_wxyz": left_quat,
        "base_pos": root_pos,
        "base_quat_wxyz": root_quat,
        "base_yaw": _yaw_from_quat(root_quat),
        "cube_pos": obj_pos,
        "cube_quat_wxyz": obj_quat,
        "cube_height_world": _object_height_world(),
        "gripper_opening": _hand_opening("right"),
        "active_hand": _active_hand,
        "sim_time": float(unwrapped.sim.current_time),
        "last_move_success": last_success,
    }


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def handle_reset(payload: dict) -> dict:
    global _park_left, _park_right, _last_left_target, _last_right_target
    global _right_close, _left_close, _object_rest_z_world, _active_hand, _last_loco
    _right_close = False
    _left_close = False
    _active_hand = "right"
    _last_loco = [0.0, 0.0, 0.0, HIP_HEIGHT]
    seed = payload.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    with torch.inference_mode():
        env.reset(seed=seed)
    rp, rq = _wrist_pose("right")
    lp, lq = _wrist_pose("left")
    _park_left = {"pos": lp, "quat": lq}
    _park_right = {"pos": rp, "quat": rq}
    _last_left_target = dict(_park_left)
    _last_right_target = dict(_park_right)
    # let the loco policy settle at zero-velocity stance
    _step_hold(10)
    _object_rest_z_world = _object_height_world()
    state = _current_state()
    state["task"] = payload.get(
        "task",
        "walk to the table, grasp the steering wheel with right hand, lift it 15 cm.",
    )
    state["cube_rest_z_world"] = _object_rest_z_world
    return {"status": "ok", **state}


def handle_get_state(_payload: dict) -> dict:
    return {"status": "ok", **_current_state()}


def handle_move_ee_to(payload: dict) -> dict:
    global _last_right_target, _last_left_target, _right_close, _left_close
    global _active_hand, _last_loco
    pos = payload["pos"]
    quat = payload["quat_wxyz"]
    hand = payload.get("hand") or _active_hand
    _active_hand = hand
    gripper = payload.get("gripper")
    if gripper is not None:
        close = gripper == "close"
        if hand == "right":
            _right_close = close
        else:
            _left_close = close
    tol_pos = float(payload.get("tol_pos", 0.03))
    tol_rot = float(payload.get("tol_rot", 0.3))
    budget = int(payload.get("max_steps", args_cli.step_budget))

    if hand == "right":
        _last_right_target = {"pos": pos, "quat": quat}
    else:
        _last_left_target = {"pos": pos, "quat": quat}
    # hold loco at standstill during manip
    _last_loco = [0.0, 0.0, 0.0, HIP_HEIGHT]

    steps = 0
    per_chunk = 6
    while steps < budget:
        chunk = min(per_chunk, budget - steps)
        _step_hold(chunk)
        steps += chunk
        pos_err, rot_err = _ee_error(pos, quat, hand)
        if pos_err < tol_pos and rot_err < tol_rot:
            break
    pos_err, rot_err = _ee_error(pos, quat, hand)
    success = pos_err < tol_pos and rot_err < tol_rot
    return {
        "status": "ok", "arrived": success, "pos_err": pos_err, "rot_err": rot_err,
        "steps_used": steps, **_current_state(last_success=success),
    }


def handle_set_gripper(payload: dict) -> dict:
    global _right_close, _left_close, _last_loco
    hand = payload.get("hand") or _active_hand
    close = payload["state"] == "close"
    if hand == "right":
        _right_close = close
    else:
        _left_close = close
    _last_loco = [0.0, 0.0, 0.0, HIP_HEIGHT]
    _step_hold(int(payload.get("hold_steps", 40)))
    return {"status": "ok", **_current_state()}


def handle_walk_to(payload: dict) -> dict:
    """Walk the robot to world-frame (x, y, yaw) using the pretrained loco policy.

    The server runs a P controller: at each step it measures base pose,
    computes body-frame velocity commands to close the gap, and passes them to
    the loco action. The upper body holds its last-commanded pose.
    """
    global _last_loco
    target_x = float(payload["x"])
    target_y = float(payload["y"])
    target_yaw = float(payload.get("yaw", 0.0))
    tol_pos = float(payload.get("tol_pos", 0.15))
    tol_yaw = float(payload.get("tol_yaw", 0.2))
    v_max = float(payload.get("v_max", 0.5))
    w_max = float(payload.get("w_max", 0.8))
    budget = int(payload.get("max_steps", 900))

    steps = 0
    pos_err = yaw_err = float("inf")
    while steps < budget:
        root_pos, root_quat = _root_pose()
        yaw = _yaw_from_quat(root_quat)
        dx_w = target_x - root_pos[0]
        dy_w = target_y - root_pos[1]
        pos_err = (dx_w * dx_w + dy_w * dy_w) ** 0.5
        yaw_err = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if pos_err < tol_pos and abs(yaw_err) < tol_yaw:
            break
        # world → body frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        vx_b = c * dx_w - s * dy_w
        vy_b = s * dx_w + c * dy_w
        # P gains + cap
        kp_v = 1.0
        vx = max(-v_max, min(v_max, kp_v * vx_b))
        vy = max(-v_max, min(v_max, kp_v * vy_b))
        wz = max(-w_max, min(w_max, 1.2 * yaw_err))
        _last_loco = [vx, vy, wz, HIP_HEIGHT]
        _step_hold(20)
        steps += 20
    # stop
    _last_loco = [0.0, 0.0, 0.0, HIP_HEIGHT]
    _step_hold(8)
    arrived = pos_err < tol_pos and abs(yaw_err) < tol_yaw
    return {
        "status": "ok", "arrived": arrived, "pos_err": pos_err, "yaw_err": yaw_err,
        "steps_used": steps, **_current_state(last_success=arrived),
    }


def handle_is_grasped(_payload: dict) -> dict:
    cube_z = _object_height_world()
    delta = cube_z - _object_rest_z_world
    grasped = (delta > 0.03) and _right_close
    return {"status": "ok", "grasped": grasped, "cube_height": cube_z,
            "delta_from_rest": delta, "opening": _hand_opening("right")}


def handle_check_lifted(payload: dict) -> dict:
    min_lift = float(payload.get("min_lift", 0.10))
    cube_z = _object_height_world()
    delta = cube_z - _object_rest_z_world
    return {"status": "ok", "lifted": delta >= min_lift, "cube_height": cube_z,
            "delta_from_rest": delta, "rest_z": _object_rest_z_world}


HANDLERS = {
    "reset": handle_reset,
    "get_state": handle_get_state,
    "move_ee_to": handle_move_ee_to,
    "set_gripper": handle_set_gripper,
    "walk_to": handle_walk_to,
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
    print(f"[sim_server_g1_loco] listening on {args_cli.host}:{args_cli.port}", flush=True)
    with torch.inference_mode():
        env.reset()
    while simulation_app.is_running():
        print("[sim_server_g1_loco] waiting for client…", flush=True)
        conn, addr = srv.accept()
        print(f"[sim_server_g1_loco] client {addr} connected", flush=True)
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
        print("[sim_server_g1_loco] client disconnected", flush=True)


if __name__ == "__main__":
    try:
        serve()
    finally:
        simulation_app.close()
