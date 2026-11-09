"""Isaac Lab sim process for the Unitree G1 humanoid, fixed-base upper-body IK.

Same TCP / JSON protocol as sim_server.py so the existing client tools work
unchanged. The env loaded is `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`:
the G1's pelvis is welded to the world, upper body driven by pink IK, and a
steering wheel sits on a packing table as the grasp object.

Action layout (28D):
  [left_pos(3), left_quat_wxyz(4),
   right_pos(3), right_quat_wxyz(4),
   hand_joints(14)]
Poses are in env-origin frame; the action term converts to pelvis frame.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9876)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
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
# pick_place is blacklisted by isaaclab_tasks auto-import; force register.
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from llm_traj.server.protocol import recv_message, send_message

# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
env_cfg.episode_length_s = 120.0

# Swap steering wheel → small cube so the G1 tri-hand can actually grasp it.
import isaaclab.sim as _sim_utils
from isaaclab.assets import RigidObjectCfg as _RigidObjectCfg
env_cfg.scene.object = _RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=_RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.21, 0.80], rot=[1, 0, 0, 0]),
    spawn=_sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.15),
        rigid_props=_sim_utils.RigidBodyPropertiesCfg(),
        mass_props=_sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=_sim_utils.CollisionPropertiesCfg(),
        visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1), metallic=0.1),
    ),
)

env = gym.make(args_cli.task, cfg=env_cfg)
unwrapped = env.unwrapped
device = unwrapped.device

# Hand joint order from G1_UPPER_BODY_IK_ACTION_CFG.hand_joint_names
#  0 left_hand_index_0      7 left_hand_middle_1
#  1 left_hand_middle_0     8 left_hand_thumb_1
#  2 left_hand_thumb_0      9 right_hand_index_1
#  3 right_hand_index_0    10 right_hand_middle_1
#  4 right_hand_middle_0   11 right_hand_thumb_1
#  5 right_hand_thumb_0    12 left_hand_thumb_2
#  6 left_hand_index_1     13 right_hand_thumb_2
RIGHT_HAND_IDX = [3, 4, 5, 9, 10, 11, 13]
LEFT_HAND_IDX = [0, 1, 2, 6, 7, 8, 12]
# Closed values per finger-joint (rough grasp config; thumb abducts inward)
CLOSE_VALS = {
    "index_0": 0.8, "middle_0": 0.8, "thumb_0": 0.3,
    "index_1": 0.8, "middle_1": 0.8, "thumb_1": 0.8, "thumb_2": 0.8,
}
CLOSE_RIGHT = [CLOSE_VALS[k] for k in ("index_0", "middle_0", "thumb_0",
                                       "index_1", "middle_1", "thumb_1", "thumb_2")]
# Left hand curl joints have mirrored (negative) limits on G1. Flip signs for
# index_0, middle_0, index_1, middle_1. Thumb joints are near-symmetric.
CLOSE_LEFT = [-CLOSE_VALS["index_0"], -CLOSE_VALS["middle_0"], -CLOSE_VALS["thumb_0"],
              -CLOSE_VALS["index_1"], -CLOSE_VALS["middle_1"], CLOSE_VALS["thumb_1"],
              CLOSE_VALS["thumb_2"]]


def _env_origin() -> torch.Tensor:
    return unwrapped.scene.env_origins[0]


def _pelvis_pose() -> tuple[list[float], list[float]]:
    robot = unwrapped.scene["robot"]
    idx = robot.data.body_names.index("pelvis")
    pos = (robot.data.body_pos_w[0, idx] - _env_origin()).cpu().tolist()
    quat = robot.data.body_quat_w[0, idx].cpu().tolist()
    return pos, quat


def _wrist_pose(side: str) -> tuple[list[float], list[float]]:
    link = f"{side}_wrist_yaw_link"
    robot = unwrapped.scene["robot"]
    idx = robot.data.body_names.index(link)
    pos = (robot.data.body_pos_w[0, idx] - _env_origin()).cpu().tolist()
    quat = robot.data.body_quat_w[0, idx].cpu().tolist()
    return pos, quat


def _object_pose() -> tuple[list[float], list[float]]:
    obj = unwrapped.scene["object"]
    pos = (obj.data.root_pos_w[0] - _env_origin()).cpu().tolist()
    quat = obj.data.root_quat_w[0].cpu().tolist()
    return pos, quat


def _object_height_world() -> float:
    return float(unwrapped.scene["object"].data.root_pos_w[0, 2].item())


def _hand_opening(side: str) -> float:
    """Sum of finger proximal/distal joints on one hand (larger = more open)."""
    robot = unwrapped.scene["robot"]
    ids, _ = robot.find_joints([f"{side}_hand_index_0_joint", f"{side}_hand_middle_0_joint"])
    # Inverse: opening ~= -sum(flexion). Just report sum for monitoring.
    return float(robot.data.joint_pos[0, ids].sum().item())


# ---------------------------------------------------------------------------
# action builder
# ---------------------------------------------------------------------------

_active_hand = "right"                 # which hand move_ee_to drives
_park_left = {"pos": None, "quat": None}
_park_right = {"pos": None, "quat": None}
_last_right_target = {"pos": None, "quat": None}
_last_left_target = {"pos": None, "quat": None}
_right_close = False
_left_close = False
_object_rest_z_world: float = 0.0


def _build_action(left_pos, left_quat, right_pos, right_quat,
                  right_close: bool, left_close: bool) -> torch.Tensor:
    hands = [0.0] * 14
    if right_close:
        for i, v in zip(RIGHT_HAND_IDX, CLOSE_RIGHT):
            hands[i] = v
    if left_close:
        for i, v in zip(LEFT_HAND_IDX, CLOSE_LEFT):
            hands[i] = v
    payload = [*left_pos, *left_quat, *right_pos, *right_quat, *hands]
    return torch.tensor([payload], dtype=torch.float32, device=device)


def _step_hold(n: int) -> None:
    action = _build_action(
        _last_left_target["pos"], _last_left_target["quat"],
        _last_right_target["pos"], _last_right_target["quat"],
        _right_close, _left_close,
    )
    for _ in range(n):
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
    pelvis_pos, pelvis_quat = _pelvis_pose()
    return {
        # Primary EE: right wrist. Keep key names identical to Franka server so
        # the client CLI/plot code doesn't need to branch.
        "ee_pos": right_pos,
        "ee_quat_wxyz": right_quat,
        "left_ee_pos": left_pos,
        "left_ee_quat_wxyz": left_quat,
        "pelvis_pos": pelvis_pos,
        "pelvis_quat_wxyz": pelvis_quat,
        "cube_pos": obj_pos,              # reuse "cube" key; it's actually the steering wheel
        "cube_quat_wxyz": obj_quat,
        "cube_height_world": _object_height_world(),
        "gripper_opening": _hand_opening("right"),
        "left_gripper_opening": _hand_opening("left"),
        "active_hand": _active_hand,
        "sim_time": float(unwrapped.sim.current_time),
        "last_move_success": last_success,
    }


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def handle_reset(payload: dict) -> dict:
    global _park_left, _park_right, _last_left_target, _last_right_target
    global _right_close, _left_close, _object_rest_z_world, _active_hand
    _right_close = False
    _left_close = False
    _active_hand = "right"
    seed = payload.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    with torch.inference_mode():
        env.reset(seed=seed)
    # park both wrists at their post-reset poses
    rp, rq = _wrist_pose("right")
    lp, lq = _wrist_pose("left")
    _park_left = {"pos": lp, "quat": lq}
    _park_right = {"pos": rp, "quat": rq}
    _last_left_target = dict(_park_left)
    _last_right_target = dict(_park_right)
    _step_hold(4)
    _object_rest_z_world = _object_height_world()
    state = _current_state()
    state["task"] = payload.get("task", "grasp the steering wheel and lift it 15 cm")
    state["cube_rest_z_world"] = _object_rest_z_world
    return {"status": "ok", **state}


def handle_get_state(_payload: dict) -> dict:
    return {"status": "ok", **_current_state()}


def handle_move_ee_to(payload: dict) -> dict:
    """Drive the selected wrist toward (pos, quat). Other wrist parks at its
    last commanded target. Hand close/open state is held unless overridden.
    """
    global _last_right_target, _last_left_target, _right_close, _left_close, _active_hand
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
    tol_pos = float(payload.get("tol_pos", 0.02))
    tol_rot = float(payload.get("tol_rot", 0.15))
    budget = int(payload.get("max_steps", args_cli.step_budget))

    if hand == "right":
        _last_right_target = {"pos": pos, "quat": quat}
    else:
        _last_left_target = {"pos": pos, "quat": quat}

    steps_done = 0
    per_chunk = 6
    while steps_done < budget:
        chunk = min(per_chunk, budget - steps_done)
        _step_hold(chunk)
        steps_done += chunk
        pos_err, rot_err = _ee_error(pos, quat, hand)
        if pos_err < tol_pos and rot_err < tol_rot:
            break
    pos_err, rot_err = _ee_error(pos, quat, hand)
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
    """Open/close the active (or specified) hand. Both wrists hold pose."""
    global _right_close, _left_close
    hand = payload.get("hand") or _active_hand
    close = payload["state"] == "close"
    if hand == "right":
        _right_close = close
    else:
        _left_close = close
    _step_hold(int(payload.get("hold_steps", 30)))
    return {"status": "ok", **_current_state()}


def handle_debug_hands(_payload: dict) -> dict:
    """Dump all 14 hand joint positions + the last action tensor for diagnosis."""
    robot = unwrapped.scene["robot"]
    names = ["left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
             "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
             "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
             "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
             "left_hand_thumb_2_joint", "right_hand_thumb_2_joint"]
    out = {}
    for n in names:
        ids, _ = robot.find_joints([n])
        if ids:
            out[n] = {"id": int(ids[0]),
                      "pos": float(robot.data.joint_pos[0, ids[0]].item()),
                      "target": float(robot.data.joint_pos_target[0, ids[0]].item()),
                      "limit_low": float(robot.data.joint_pos_limits[0, ids[0], 0].item()),
                      "limit_high": float(robot.data.joint_pos_limits[0, ids[0], 1].item())}
        else:
            out[n] = None
    return {"status": "ok", "joints": out,
            "all_joint_names": list(robot.data.joint_names),
            "right_close": _right_close, "left_close": _left_close,
            "active_hand": _active_hand}


def handle_raw_hand(payload: dict) -> dict:
    """Directly set specific hand joints by name and step. Bypasses pink IK action."""
    robot = unwrapped.scene["robot"]
    targets = payload["targets"]  # {joint_name: value}
    hold = int(payload.get("hold_steps", 40))
    ids = []
    vals = []
    for n, v in targets.items():
        jids, _ = robot.find_joints([n])
        if jids:
            ids.append(jids[0])
            vals.append(float(v))
    if ids:
        import torch as _t
        t = _t.tensor([vals], dtype=_t.float32, device=device)
        for _ in range(hold):
            robot.set_joint_position_target(t, joint_ids=ids)
            with torch.inference_mode():
                env.step(_build_action(
                    _last_left_target["pos"], _last_left_target["quat"],
                    _last_right_target["pos"], _last_right_target["quat"],
                    _right_close, _left_close,
                ))
    out = {}
    for n in targets:
        jids, _ = robot.find_joints([n])
        if jids:
            out[n] = float(robot.data.joint_pos[0, jids[0]].item())
    return {"status": "ok", "after": out}


def handle_is_grasped(_payload: dict) -> dict:
    """Cube lifted > 3 cm from rest AND right hand commanded closed."""
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
    "is_grasped": handle_is_grasped,
    "check_lifted": handle_check_lifted,
    "debug_hands": handle_debug_hands,
    "raw_hand": handle_raw_hand,
}


# ---------------------------------------------------------------------------
# socket loop (identical to Franka server)
# ---------------------------------------------------------------------------

def serve() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args_cli.host, args_cli.port))
    srv.listen(1)
    print(f"[sim_server_g1] listening on {args_cli.host}:{args_cli.port}", flush=True)
    with torch.inference_mode():
        env.reset()
    while simulation_app.is_running():
        print("[sim_server_g1] waiting for client…", flush=True)
        conn, addr = srv.accept()
        print(f"[sim_server_g1] client {addr} connected", flush=True)
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
        print("[sim_server_g1] client disconnected", flush=True)


if __name__ == "__main__":
    try:
        serve()
    finally:
        simulation_app.close()
