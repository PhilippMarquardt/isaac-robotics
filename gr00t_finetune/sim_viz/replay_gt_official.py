"""Replay dataset GT (robot_q_current + hand_state) through the new official MJCF.

No policy. Just verifies the official G1+Inspire MJCF animates dataset trajectories
correctly. Side-by-side: dataset ego video | sim third-person.
"""
from __future__ import annotations

import argparse
import os
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

from viz_mujoco import (
    HEAD_CAM_FOVY_DEG,
    HEAD_CAM_PITCH_DEG,
    HEAD_LOCAL_OFFSET,
    SCENE_XML,
    build_hand_qpos_map,
    build_joint_qpos_map,
    set_hand_qpos,
    set_robot_qpos,
)

SAMPLES = Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")) / "_dataset_samples"


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"cv2 frame {idx}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def third_person_cam():
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance = 2.4
    c.lookat[:] = [0.0, 0.4, 0.8]
    c.azimuth = 110.0
    c.elevation = -12.0
    return c


def head_cam(m, d, torso_bid):
    R = d.xmat[torso_bid].reshape(3, 3)
    head_w = d.xpos[torso_bid] + R @ HEAD_LOCAL_OFFSET
    fwd_w = R @ np.array([1.0, 0.0, 0.0])
    p = math.radians(HEAD_CAM_PITCH_DEG)
    c, s = math.cos(p), math.sin(p)
    fwd_w = np.array([fwd_w[0] * c, fwd_w[1] * c, s])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.01
    cam.lookat[:] = head_w + fwd_w * 0.01
    cam.azimuth = math.degrees(math.atan2(fwd_w[1], fwd_w[0]))
    cam.elevation = math.degrees(math.asin(fwd_w[2] / np.linalg.norm(fwd_w)))
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    pq = SAMPLES / f"episode_{a.episode:06d}.parquet"
    mp4 = SAMPLES / f"ep{a.episode:03d}_cam0.mp4"
    df = pd.read_parquet(pq)
    print(f"[gt] ep{a.episode} {len(df)} frames  scene={Path(SCENE_XML).name}")

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    hmap = build_hand_qpos_map(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    m.vis.global_.fovy = HEAD_CAM_FOVY_DEG
    r_tp = mujoco.Renderer(m, height=a.h, width=a.w)
    r_hd = mujoco.Renderer(m, height=a.h, width=a.w)

    out = a.out or str(SAMPLES / f"replay_official_ep{a.episode}.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.w * 3, a.h))
    cap = cv2.VideoCapture(str(mp4))

    ts = list(range(0, len(df), a.stride))[: a.max]
    for t in ts:
        row = df.iloc[t]
        rq = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        hs = np.asarray(row["observation.state.hand_state"], dtype=np.float64)

        set_robot_qpos(d, rq[0:7], rq[7:36], qmap)
        set_hand_qpos(d, hs[0:12], hmap)
        mujoco.mj_forward(m, d)

        real = cv2.resize(read_frame(cap, t), (a.w, a.h))

        r_tp.update_scene(d, camera=third_person_cam())
        tp = r_tp.render()
        r_hd.update_scene(d, camera=head_cam(m, d, torso))
        hd = r_hd.render()

        three = np.concatenate([real, tp, hd], axis=1)
        bgr = cv2.cvtColor(three, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "DATASET", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bgr, "SIM 3RD", (a.w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bgr, "SIM HEAD", (a.w * 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bgr, f"t={t}", (10, a.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(bgr)

    cap.release()
    writer.release()
    r_tp.close()
    r_hd.close()
    print(f"[gt] wrote {out}")


if __name__ == "__main__":
    main()
