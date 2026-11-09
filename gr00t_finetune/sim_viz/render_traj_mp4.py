"""Headless MP4 render of GR00T trajectory JSON on MuJoCo G1 scene.

Replays each frame's obs qpos, draws pred+gt future-wrist paths,
renders 3rd-person view + head-cam view side-by-side to MP4.

Usage:
  python render_traj_mp4.py --traj eval_out.json --episode 0 --out traj_ep0.mp4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from viz_mujoco import (
    BODY_JOINT_NAMES,
    HEAD_CAM_FOVY_DEG,
    HEAD_CAM_PITCH_DEG,
    HEAD_LOCAL_OFFSET,
    SCENE_XML,
    add_trace_spheres,
    build_joint_qpos_map,
    fk_wrist,
    set_robot_qpos,
)


def third_person_camera(m) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.8
    cam.lookat[:] = [0.0, 0.4, 0.8]
    cam.azimuth = 100.0
    cam.elevation = -15.0
    return cam


def head_camera(m, d, torso_bid):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    data = json.loads(Path(args.traj).read_text())
    frames = data["episodes"][args.episode]["frames"]
    print(f"loaded {len(frames)} frames from ep{args.episode}")

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    wrist_ids = np.array([
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    ])
    torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    m.vis.global_.fovy = HEAD_CAM_FOVY_DEG
    r_tp = mujoco.Renderer(m, height=args.h, width=args.w)
    r_hd = mujoco.Renderer(m, height=args.h, width=args.w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.w * 2, args.h))

    # Accumulator: when drawing traces, reuse Renderer._scene.user_scn-less injection
    for i, fr in enumerate(frames):
        root7 = np.array(fr["robot_q_obs"][0:7], dtype=np.float64)
        body29 = np.array(fr["robot_q_obs"][7:36], dtype=np.float64)
        set_robot_qpos(d, root7, body29, qmap)
        mujoco.mj_forward(m, d)

        pred = np.array(fr["pred_chunk"], dtype=np.float64)
        gt = np.array(fr["gt_chunk"], dtype=np.float64)
        pred_trace = np.concatenate(
            [fk_wrist(m, d, root7, p[7:36], qmap, wrist_ids) for p in pred], axis=0
        )
        gt_trace = np.concatenate(
            [fk_wrist(m, d, root7, p[7:36], qmap, wrist_ids) for p in gt], axis=0
        )
        # Restore current obs pose
        set_robot_qpos(d, root7, body29, qmap)
        mujoco.mj_forward(m, d)

        # Third-person render with traces
        r_tp.update_scene(d, camera=third_person_camera(m))
        add_trace_spheres(r_tp.scene, pred_trace, rgba=(0.9, 0.2, 0.2, 0.9))
        add_trace_spheres(r_tp.scene, gt_trace, rgba=(0.2, 0.9, 0.2, 0.6), radius=0.006)
        tp = r_tp.render()

        # Head-cam render
        r_hd.update_scene(d, camera=head_camera(m, d, torso_bid))
        hd = r_hd.render()

        both = np.concatenate([tp, hd], axis=1)
        bgr = cv2.cvtColor(both, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"t={fr['t']}  red=pred  green=gt", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(bgr)
        if i % 5 == 0:
            print(f"  frame {i}/{len(frames)}")

    r_tp.close()
    r_hd.close()
    writer.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
