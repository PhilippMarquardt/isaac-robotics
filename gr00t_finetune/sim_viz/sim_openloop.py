"""Open-loop policy rollout for MP4 viz.

- Feed REAL dataset video frames to GR00T (policy sees its training distribution).
- Apply predicted actions to MuJoCo robot.
- Render 3-panel video: [dataset frame | 3rd-person sim | head-cam sim].

Because obs is the real video, predictions are in-distribution and should
produce coherent reaching motion toward the pillow, not the random thrashing
seen when the VLM was fed MuJoCo's cartoon scene.
"""
from __future__ import annotations

import argparse
import os
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
import torch

from viz_mujoco import (
    HEAD_CAM_FOVY_DEG,
    HEAD_CAM_PITCH_DEG,
    HEAD_LOCAL_OFFSET,
    SCENE_XML,
    build_joint_qpos_map,
    build_hand_qpos_map,
    set_robot_qpos,
    set_hand_qpos,
)
from sim_run_policy import slice_state, pack_action_chunk, head_camera, third_person_camera

CKPT = str(Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")) / "ckpt")
SAMPLES = Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")) / "_dataset_samples"
TASK = "Pick up the pillow and put it on the sofa."


def read_video_frame(mp4: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cv2 frame {idx}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--action_apply", type=int, default=8, help="num actions applied per chunk")
    ap.add_argument("--max_frames", type=int, default=30)
    ap.add_argument("--out_mp4", type=str, default=str(SAMPLES / "sim_openloop.mp4"))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    pq = SAMPLES / f"episode_{args.episode:06d}.parquet"
    mp4 = SAMPLES / f"ep{args.episode:03d}_cam0.mp4"
    df = pd.read_parquet(pq)
    n = len(df)
    print(f"[ol] ep{args.episode} {n} frames")

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    hmap = build_hand_qpos_map(m)
    torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    m.vis.global_.fovy = HEAD_CAM_FOVY_DEG
    r_tp = mujoco.Renderer(m, height=args.height, width=args.width)
    r_hd = mujoco.Renderer(m, height=args.height, width=args.width)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"[ol] loading policy")
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=CKPT,
        device="cuda:0",
        strict=False,
    )
    print("[ol] policy ready")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (args.width * 3, args.height))

    ts = list(range(0, n, args.stride))[: args.max_frames]
    for t in ts:
        row = df.iloc[t]
        rq = np.asarray(row["observation.state.robot_q_current"], dtype=np.float32)
        hs = np.asarray(row["observation.state.hand_state"], dtype=np.float32)

        # Seed MuJoCo robot from real dataset state at this t
        set_robot_qpos(d, rq[0:7].astype(np.float64), rq[7:36].astype(np.float64), qmap)
        mujoco.mj_forward(m, d)

        ego_real = read_video_frame(mp4, t)

        obs = {
            "video": {"ego_view": ego_real[None, None]},
            "state": slice_state(rq, hs),
            "language": {"annotation.human.task_description": [[TASK]]},
        }
        t0 = time.time()
        with torch.no_grad():
            action, _ = policy.get_action(obs)
        body_chunk, hand_chunk = pack_action_chunk(action)
        infer = time.time() - t0
        print(f"[ol] t={t} infer={infer:.2f}s arm_R[0]={body_chunk[0, 22:29].round(2).tolist()}")

        # Resize real cam to height matching sim panels
        real_resized = cv2.resize(ego_real, (args.width, args.height))

        for k in range(min(args.action_apply, body_chunk.shape[0])):
            set_robot_qpos(d, d.qpos[0:7].copy(), body_chunk[k].astype(np.float64), qmap)
            set_hand_qpos(d, hand_chunk[k].astype(np.float64), hmap)
            mujoco.mj_forward(m, d)

            r_tp.update_scene(d, camera=third_person_camera())
            tp = r_tp.render()
            r_hd.update_scene(d, camera=head_camera(m, d, torso_bid))
            hd = r_hd.render()

            three = np.concatenate([real_resized, tp, hd], axis=1)
            bgr = cv2.cvtColor(three, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"DATASET FRAME", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, f"SIM 3RD PERSON", (args.width + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, f"SIM HEAD CAM", (args.width * 2 + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, f"t={t} k={k}", (10, args.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(bgr)

    writer.release()
    r_tp.close()
    r_hd.close()
    print(f"[ol] wrote {args.out_mp4}")


if __name__ == "__main__":
    main()
