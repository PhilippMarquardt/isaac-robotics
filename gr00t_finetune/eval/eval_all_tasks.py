"""Closed-loop policy eval on all 4 trained tasks.

For each task (val episode):
  - Read dataset ego MP4 + parquet
  - Seed sim from row 0 robot_q_current / hand_state
  - Loop: feed dataset ego frame → policy → apply absolute action[0:K] in sim → render
  - Write dataset | sim side-by-side MP4 per task, then concat all into one.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
import torch

REPO = os.environ.get("GR00T_REPO", "Isaac-GR00T")  # path to an Isaac-GR00T checkout
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GROOT_EVAL = os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")  # ckpts + pulled val episodes
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_viz"))

from viz_mujoco import (
    SCENE_XML,
    build_hand_qpos_map,
    build_joint_qpos_map,
    set_hand_qpos,
    set_robot_qpos,
)

CKPT = Path(GROOT_EVAL) / "ckpt_alltasks"
OTHER = Path(GROOT_EVAL) / "_samples_other"

TASKS = [
    ("Pickup Pillow",
     "Pick up the pillow and put it on the sofa.",
     OTHER / "G1_WB_Dex5_Pickup_Pillow.parquet",
     OTHER / "G1_WB_Dex5_Pickup_Pillow.mp4"),
    ("Collect Clothes",
     "Collect the clothes on the floor.",
     OTHER / "G1_WB_Dex5_Collect_Clothes.parquet",
     OTHER / "G1_WB_Dex5_Collect_Clothes.mp4"),
    ("Put Clothes in Washing Machine",
     "Put the clothes into the washing machine.",
     OTHER / "G1_WB_Dex5_Put_Clothes_into_Washing_Machine.parquet",
     OTHER / "G1_WB_Dex5_Put_Clothes_into_Washing_Machine.mp4"),
    ("Put Drinks in Fridge",
     "Put the drinks into the fridge.",
     OTHER / "G1_WBT_Inspire_Put_Drinks_Into_Fridge.parquet",
     OTHER / "G1_WBT_Inspire_Put_Drinks_Into_Fridge.mp4"),
]

OUT_MP4 = OTHER / "eval_alltasks_closedloop.mp4"
W, H = 540, 540
FPS = 15
ACTION_STRIDE = 4           # apply this many of each action chunk's steps
FRAME_SKIP = 3              # consume this many dataset frames per write
STEPS_PER_TASK = 60         # inference calls per task


def third_person(pelvis):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [pelvis[0], pelvis[1] + 0.3, pelvis[2] + 0.1]
    cam.distance = 2.6
    cam.azimuth = 100.0
    cam.elevation = -15.0
    return cam


def slice_state(rq, hs):
    return {
        "left_leg":   rq[7:13][None, None],
        "right_leg":  rq[13:19][None, None],
        "waist":      rq[19:22][None, None],
        "left_arm":   rq[22:29][None, None],
        "right_arm":  rq[29:36][None, None],
        "left_hand":  hs[0:6][None, None],
        "right_hand": hs[6:12][None, None],
    }


def pack_action_chunk(action):
    def u(x):
        a = np.asarray(x)
        return a[0] if a.ndim == 3 else a
    ll = u(action["left_leg"]); rl = u(action["right_leg"]); w = u(action["waist"])
    la = u(action["left_arm"]); ra = u(action["right_arm"])
    lh = u(action["left_hand"]); rh = u(action["right_hand"])
    H = ll.shape[0]
    body = np.zeros((H, 29), dtype=np.float32)
    body[:, 0:6]   = ll
    body[:, 6:12]  = rl
    body[:, 12:15] = w
    body[:, 15:22] = la
    body[:, 22:29] = ra
    hand = np.zeros((H, 12), dtype=np.float32)
    hand[:, 0:6]  = lh
    hand[:, 6:12] = rh
    return body, hand


def main():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    hmap = build_hand_qpos_map(m)
    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    stats = json.load(open(CKPT / "experiment_cfg" / "dataset_statistics.json"))
    A = stats["new_embodiment"]["action"]
    body_min = np.concatenate([A["left_leg"]["min"], A["right_leg"]["min"], A["waist"]["min"],
                               A["left_arm"]["min"], A["right_arm"]["min"]]).astype(np.float64)
    body_max = np.concatenate([A["left_leg"]["max"], A["right_leg"]["max"], A["waist"]["max"],
                               A["left_arm"]["max"], A["right_arm"]["max"]]).astype(np.float64)
    hand_min = np.concatenate([A["left_hand"]["min"], A["right_hand"]["min"]]).astype(np.float64)
    hand_max = np.concatenate([A["left_hand"]["max"], A["right_hand"]["max"]]).astype(np.float64)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"[eval] loading policy from {CKPT}")
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(CKPT),
        device="cuda:0",
    )
    print("[eval] policy ready")

    r_tp = mujoco.Renderer(m, height=H, width=W)
    writer = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))

    for label, prompt, pq_path, mp4_path in TASKS:
        print(f"\n=== {label} ===")
        df = pd.read_parquet(pq_path)
        cap = cv2.VideoCapture(str(mp4_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seed sim from row 0
        row0 = df.iloc[0]
        rq0 = np.asarray(row0["observation.state.robot_q_current"], dtype=np.float64)
        hs0 = np.asarray(row0["observation.state.hand_state"], dtype=np.float64)
        set_robot_qpos(d, rq0[0:7], rq0[7:36], qmap)
        set_hand_qpos(d, hs0[0:12], hmap)
        mujoco.mj_forward(m, d)
        hs = hs0.astype(np.float32)
        print(f"[{label}] seeded from row 0, pelvis={d.xpos[pelvis_bid].round(2)}")

        def read_frame():
            ok, bgr = cap.read()
            if not ok:
                return None
            for _ in range(FRAME_SKIP - 1):
                cap.grab()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (W, H))

        cur_ego = read_frame()
        step_infer = 0
        while step_infer < STEPS_PER_TASK and cur_ego is not None:
            cur_rq = np.zeros(36, dtype=np.float32)
            cur_rq[0:7] = d.qpos[0:7]
            for i, qi in enumerate(qmap):
                cur_rq[7 + i] = d.qpos[qi]
            obs = {
                "video": {"ego_view": cur_ego[None, None]},
                "state": slice_state(cur_rq, hs),
                "language": {"annotation.human.task_description": [[prompt]]},
            }
            t0 = time.time()
            with torch.no_grad():
                action, _ = policy.get_action(obs)
            body_chunk, hand_chunk = pack_action_chunk(action)
            dt = time.time() - t0
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"  infer={step_infer}/{STEPS_PER_TASK} frame={pos}/{n_frames} dt={dt:.2f}s")

            for k in range(min(ACTION_STRIDE, body_chunk.shape[0])):
                target_body = np.clip(body_chunk[k].astype(np.float64), body_min, body_max)
                target_hand = np.clip(hand_chunk[k].astype(np.float64), hand_min, hand_max)
                set_robot_qpos(d, d.qpos[0:7].copy(), target_body, qmap)
                set_hand_qpos(d, target_hand, hmap)
                hs = target_hand.astype(np.float32)
                mujoco.mj_forward(m, d)

                pelvis = d.xpos[pelvis_bid].copy()
                r_tp.update_scene(d, camera=third_person(pelvis))
                tp = r_tp.render()

                both = np.concatenate([cur_ego, tp], axis=1)
                out = cv2.cvtColor(both, cv2.COLOR_RGB2BGR)
                cv2.putText(out, "DATASET EGO (fed to policy)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(out, "SIM (policy response)", (W + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(out, label, (10, H - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f"prompt: {prompt}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                writer.write(out)

                nxt = read_frame()
                if nxt is None:
                    break
                cur_ego = nxt

            step_infer += 1

        cap.release()

    writer.release()
    r_tp.close()
    print(f"\n[eval] wrote {OUT_MP4}")


if __name__ == "__main__":
    main()
