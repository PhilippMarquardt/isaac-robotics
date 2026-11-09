"""MuJoCo trajectory viz for GR00T pillow-pickup eval.

The GPU machine runs the policy on held-out videos (eval/gen_trajectories.py), dumps JSON:
  episodes: [{frames: [{t, robot_q_obs(36), hand_state(12), pred_chunk(H,48), gt_chunk(H,48)}]}]
Chunk layout: [0:36]=robot_q (root q[0:7] zeros in pred, not model-predicted),
              [36:48]=hand_cmd. For FK we take body DOFs [7:36] and reuse obs root7.
Per frame: set qpos from obs, FK to wrist xyz, draw pred + gt future-wrist paths.

Joint order in dataset (BODY_JOINT_NAMES) != MuJoCo g1.xml joint order, so we
build a name-indexed map once and remap each frame.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Ego head-cam approx (dataset cam_0 = robot head cam). Offset and pitch
# calibrated against first-frame pillow framing in dataset episode 0.
HEAD_LOCAL_OFFSET = np.array([0.08, 0.0, 0.50])   # forward, lateral, up from torso_link
HEAD_CAM_PITCH_DEG = -55.0
HEAD_CAM_FOVY_DEG = 75.0

# The scene includes the official Unitree G1 29-DoF + Inspire hand MJCF (g1_29dof_inspire_DFQ.xml
# + meshes/), which must be placed next to it in sim_viz/assets/ (not redistributed here).
SCENE_XML = str(Path(__file__).resolve().parent / "assets" / "scene_pillow_inspire_official.xml")

# Matches dataset robot_q_desired / robot_q_current ordering which follows
# Unitree G1 SDK convention: per leg [hip_pitch, hip_roll, hip_yaw, knee,
# ankle_pitch, ankle_roll] contiguous, same as the MuJoCo menagerie g1.xml
# joint declaration order. Earlier revision scrambled hip_yaw first + ankles
# after both legs, which mapped knee commands onto hip_yaw and caused leg
# thrashing. This order is load-bearing — do not reorder without updating
# pack_action_chunk/slice_state too.
BODY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# Dex3 hand layout in g1_with_hands.xml — only thumb/middle/index articulate (3
# fingers per side, 7 joints per side). Pinky/ring are rigid features of the
# palm mesh, no joints. Inspire→Dex3 mapping drops dims 0,1 (pinky/ring).
HAND_JOINT_NAMES_L = [
    "left_hand_thumb_0_joint",   # thumb rotation (axis Y)
    "left_hand_thumb_1_joint",   # thumb bend root
    "left_hand_thumb_2_joint",   # thumb bend tip
    "left_hand_middle_0_joint",  # middle curl root
    "left_hand_middle_1_joint",  # middle curl tip
    "left_hand_index_0_joint",   # index curl root
    "left_hand_index_1_joint",   # index curl tip
]
HAND_JOINT_NAMES_R = [n.replace("left", "right", 1) for n in HAND_JOINT_NAMES_L]

# Inspire 5-finger layout in g1_29dof_inspire.xml — 12 joints per side, all 5
# fingers articulated. Order here follows DFS body-tree order inside the XML
# (thumb 4 joints, then index/middle/ring/pinky each 2 joints).
HAND_JOINT_NAMES_INSPIRE_L = [
    "L_thumb_proximal_yaw_joint",   # thumb rotation
    "L_thumb_proximal_pitch_joint", # thumb bend root
    "L_thumb_intermediate_joint",   # thumb bend mid
    "L_thumb_distal_joint",         # thumb bend tip
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
]
HAND_JOINT_NAMES_INSPIRE_R = [n.replace("L_", "R_", 1) for n in HAND_JOINT_NAMES_INSPIRE_L]


def build_joint_qpos_map(m: mujoco.MjModel) -> np.ndarray:
    """BODY_JOINT_NAMES[i] -> qpos index. Root freejoint occupies qpos[0:7]."""
    idx = np.zeros(len(BODY_JOINT_NAMES), dtype=np.int32)
    for i, name in enumerate(BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name!r} not in model")
        idx[i] = m.jnt_qposadr[jid]
    return idx


def build_hand_qpos_map(m: mujoco.MjModel):
    """Detect hand variant. Returns tuple (variant, lmap, rmap) or None.

    variant == "inspire": 12 joints/side, 5 fingers articulate (full Inspire URDF).
    variant == "dex3":    7 joints/side, 3 fingers articulate (G1 menagerie Dex3).
    """
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINT_NAMES_INSPIRE_L[0]) >= 0:
        lmap = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in HAND_JOINT_NAMES_INSPIRE_L], dtype=np.int32)
        rmap = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in HAND_JOINT_NAMES_INSPIRE_R], dtype=np.int32)
        return "inspire", lmap, rmap
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINT_NAMES_L[0]) >= 0:
        lmap = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in HAND_JOINT_NAMES_L], dtype=np.int32)
        rmap = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in HAND_JOINT_NAMES_R], dtype=np.int32)
        return "dex3", lmap, rmap
    return None


def inspire6_to_inspire_urdf(u: np.ndarray, side: str) -> np.ndarray:
    """Expand 6-dim Inspire input [pinky, ring, middle, index, thumb_bend, thumb_rot]
    to the 12 joint angles of the Inspire URDF hand.

    Inspire convention: 0.0 = fully CLOSED, 1.0 = fully OPEN (per avp_teleoperate
    robot_hand_inspire.py:119 — "0.0: fully closed  1.0: fully open").
    URDF joints positive = closed, so invert u before scaling.

    Right-hand URDF flips joint axes so positive angles close regardless of side.
    """
    u = np.clip(u, 0.0, 1.0).astype(np.float64)
    u_pinky, u_ring, u_middle, u_index, u_tb, u_tr = u
    c_pinky  = 1.0 - u_pinky
    c_ring   = 1.0 - u_ring
    c_middle = 1.0 - u_middle
    c_index  = 1.0 - u_index
    c_tb     = 1.0 - u_tb
    c_tr     = 1.0 - u_tr
    return np.array([
        1.3 * c_tr,      # thumb_proximal_yaw
        0.5 * c_tb,      # thumb_proximal_pitch
        0.8 * c_tb,      # thumb_intermediate
        1.2 * c_tb,      # thumb_distal
        1.7 * c_index,   # index_proximal
        1.7 * c_index,   # index_intermediate
        1.7 * c_middle,  # middle_proximal
        1.7 * c_middle,  # middle_intermediate
        1.7 * c_ring,    # ring_proximal
        1.7 * c_ring,    # ring_intermediate
        1.7 * c_pinky,   # pinky_proximal
        1.7 * c_pinky,   # pinky_intermediate
    ], dtype=np.float64)


def inspire6_to_dex3(u: np.ndarray, side: str) -> np.ndarray:
    """Map Inspire 6-dim finger vector (normalized [0,1], 1=closed) to 7 Dex3 joint angles.

    Inspire layout: [pinky, ring, middle, index, thumb_bend, thumb_rotation].
    Dex3 has joints only for thumb/middle/index — pinky/ring dims silently dropped.
    Left hand closes in the negative direction (range [-lim, 0]); right hand closes positive.
    """
    u_middle = float(np.clip(u[2], 0.0, 1.0))
    u_index  = float(np.clip(u[3], 0.0, 1.0))
    u_tb     = float(np.clip(u[4], 0.0, 1.0))
    u_tr     = float(np.clip(u[5], 0.0, 1.0))
    s = -1.0 if side == "L" else 1.0
    return np.array([
        s * 1.0472  * u_tr,         # thumb_0 abduction (L +/-, R flip)
        s * 1.0472  * u_tb,         # thumb_1 bend root
        s * 1.74533 * u_tb,         # thumb_2 bend tip (coupled to bend)
        s * 1.5708  * u_middle,     # middle_0
        s * 1.74533 * u_middle,     # middle_1 (coupled)
        s * 1.5708  * u_index,      # index_0
        s * 1.74533 * u_index,      # index_1 (coupled)
    ], dtype=np.float64)


def set_robot_qpos(d: mujoco.MjData, root7: np.ndarray, body29: np.ndarray, qmap: np.ndarray):
    """Write root pose + 29 body joints into qpos. root7=[x,y,z,qw,qx,qy,qz]."""
    d.qpos[0:7] = root7
    for i, qi in enumerate(qmap):
        d.qpos[qi] = body29[i]


def set_hand_qpos(d: mujoco.MjData, hand12: np.ndarray, hmap):
    """Write Inspire 12-dim (L6+R6) into hand qpos. Dispatches on variant from hmap.

    hmap = (variant, lmap, rmap) from build_hand_qpos_map, or None."""
    if hmap is None:
        return
    variant, lmap, rmap = hmap
    if variant == "inspire":
        ql = inspire6_to_inspire_urdf(hand12[0:6],  side="L")
        qr = inspire6_to_inspire_urdf(hand12[6:12], side="R")
    else:
        ql = inspire6_to_dex3(hand12[0:6],  side="L")
        qr = inspire6_to_dex3(hand12[6:12], side="R")
    for i, qi in enumerate(lmap):
        d.qpos[qi] = ql[i]
    for i, qi in enumerate(rmap):
        d.qpos[qi] = qr[i]


def fk_wrist(m, d, root7, body29, qmap, body_ids):
    set_robot_qpos(d, root7, body29, qmap)
    mujoco.mj_kinematics(m, d)
    return np.stack([d.xpos[b].copy() for b in body_ids])  # (K, 3)


def head_camera(m: mujoco.MjModel, d: mujoco.MjData, torso_bid: int) -> mujoco.MjvCamera:
    """Free cam pinned to torso_link + HEAD_LOCAL_OFFSET, pitched down by HEAD_CAM_PITCH_DEG."""
    R = d.xmat[torso_bid].reshape(3, 3)
    head_w = d.xpos[torso_bid] + R @ HEAD_LOCAL_OFFSET
    fwd_w = R @ np.array([1.0, 0.0, 0.0])
    p = math.radians(HEAD_CAM_PITCH_DEG)
    c, s = math.cos(p), math.sin(p)
    fwd_w = np.array([fwd_w[0] * c, fwd_w[1] * c, s])
    az = math.degrees(math.atan2(fwd_w[1], fwd_w[0]))
    el = math.degrees(math.asin(fwd_w[2] / np.linalg.norm(fwd_w)))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.01
    cam.lookat[:] = head_w + fwd_w * 0.01
    cam.azimuth = az
    cam.elevation = el
    return cam


def render_headcam(m, d, torso_bid, width=960, height=720) -> np.ndarray:
    m.vis.global_.fovy = HEAD_CAM_FOVY_DEG
    r = mujoco.Renderer(m, height=height, width=width)
    r.update_scene(d, camera=head_camera(m, d, torso_bid))
    img = r.render()
    r.close()
    return img


def add_trace_spheres(scene: mujoco.MjvScene, pts: np.ndarray, rgba, radius=0.008):
    for p in pts:
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius]), p.astype(np.float64),
            np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, default=None, help="JSON trajectory file from eval/gen_trajectories.py")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    wrist_ids = np.array([
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    ])

    # Load trajectory or fall back to idle pose demo
    if args.traj and Path(args.traj).exists():
        data = json.loads(Path(args.traj).read_text())
        frames = data["episodes"][args.episode]["frames"]
        print(f"loaded {len(frames)} frames from {args.traj}")
    else:
        print("no --traj given; idle-pose viewer only")
        frames = None

    dt = 1.0 / args.fps
    with mujoco.viewer.launch_passive(m, d) as viewer:
        idx = 0
        t0 = time.time()
        while viewer.is_running():
            if frames is not None:
                fr = frames[idx % len(frames)]
                root7 = np.array(fr["robot_q_obs"][0:7], dtype=np.float64)
                body29 = np.array(fr["robot_q_obs"][7:36], dtype=np.float64)
                set_robot_qpos(d, root7, body29, qmap)
                mujoco.mj_forward(m, d)

                # Build pred + gt wrist traces via FK rollout. Chunks are (H, 48);
                # model doesn't predict root, so reuse obs root7 for FK.
                pred = np.array(fr["pred_chunk"], dtype=np.float64)  # (H, 48)
                gt = np.array(fr["gt_chunk"], dtype=np.float64)
                pred_trace = [fk_wrist(m, d, root7, p[7:36], qmap, wrist_ids) for p in pred]
                gt_trace = [fk_wrist(m, d, root7, p[7:36], qmap, wrist_ids) for p in gt]
                # Restore current pose
                set_robot_qpos(d, root7, body29, qmap)
                mujoco.mj_forward(m, d)

                viewer.user_scn.ngeom = 0
                # pred = red, gt = green; stack wrist points (K=2) across horizon
                pred_pts = np.concatenate(pred_trace, axis=0)
                gt_pts = np.concatenate(gt_trace, axis=0)
                add_trace_spheres(viewer.user_scn, pred_pts, rgba=(0.9, 0.2, 0.2, 0.9))
                add_trace_spheres(viewer.user_scn, gt_pts, rgba=(0.2, 0.9, 0.2, 0.6), radius=0.006)
                idx += 1
            else:
                mujoco.mj_forward(m, d)
            viewer.sync()
            elapsed = time.time() - t0
            sleep = (idx * dt) - elapsed if frames is not None else dt
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
