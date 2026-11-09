"""3 val episodes per task → open-loop inference → save traj → replay in sim.

Output: one MP4 per task (3 episodes stitched) + one MP4 for OOD task.
Each frame: dataset ego (left) | sim policy replay (right).
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
VAL_DIR = OTHER / "val_episodes"
TRAJ_DIR = OTHER / "rollouts_3ep"
TRAJ_DIR.mkdir(exist_ok=True)
OUT_DIR = OTHER / "eval_per_task"
OUT_DIR.mkdir(exist_ok=True)

TASKS = [
    ("pillow",     "Pick up the pillow and put it on the sofa."),
    ("clothes",    "Collect the clothes on the floor."),
    ("washing",    "Put the clothes into the washing machine."),
    ("fridge",     "Put the drinks into the fridge."),
    ("basket_OOD", "Put the clothes into the basket."),   # OOD
]

ACTION_STRIDE = 4
FRAME_SKIP = 4
W, H = 540, 540
FPS = 15


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


def rollout_episode(policy, prompt, pq_path, mp4_path, bounds):
    body_min, body_max, hand_min, hand_max = bounds
    df = pd.read_parquet(pq_path)
    cap = cv2.VideoCapture(str(mp4_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_par = len(df)
    n = min(n_vid, n_par)

    pred_body, pred_hand, gt_root7 = [], [], []
    frame_idx = 0
    hs = np.asarray(df.iloc[0]["observation.state.hand_state"], dtype=np.float32)

    while frame_idx < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ego = cv2.resize(rgb, (W, H))

        row = df.iloc[frame_idx]
        rq = np.asarray(row["observation.state.robot_q_current"], dtype=np.float32)
        hs = np.asarray(row["observation.state.hand_state"], dtype=np.float32)
        root7 = rq[0:7].astype(np.float64)

        obs = {
            "video": {"ego_view": ego[None, None]},
            "state": slice_state(rq, hs),
            "language": {"annotation.human.task_description": [[prompt]]},
        }
        with torch.no_grad():
            action, _ = policy.get_action(obs)
        body_chunk, hand_chunk = pack_action_chunk(action)

        K = min(ACTION_STRIDE, body_chunk.shape[0])
        for k in range(K):
            pred_body.append(np.clip(body_chunk[k].astype(np.float64), body_min, body_max))
            pred_hand.append(np.clip(hand_chunk[k].astype(np.float64), hand_min, hand_max))
            gt_root7.append(root7)

        frame_idx += FRAME_SKIP

    cap.release()
    return (np.stack(pred_body), np.stack(pred_hand), np.stack(gt_root7))


def replay_episode(m, d, qmap, hmap, pelvis_bid, r_tp, writer,
                   label, ep_tag, prompt, pred_body, pred_hand, gt_root7, mp4_path):
    cap = cv2.VideoCapture(str(mp4_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = len(pred_body)

    for t in range(T):
        vid_frame = min(int(t * FRAME_SKIP / ACTION_STRIDE), n_vid - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame)
        ok, bgr = cap.read()
        if not ok:
            dataset_vis = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            dataset_vis = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H))

        set_robot_qpos(d, gt_root7[t], pred_body[t], qmap)
        set_hand_qpos(d, pred_hand[t], hmap)
        mujoco.mj_forward(m, d)

        pelvis = d.xpos[pelvis_bid].copy()
        r_tp.update_scene(d, camera=third_person(pelvis))
        sim_vis = r_tp.render()

        both = np.concatenate([dataset_vis, sim_vis], axis=1)
        out = cv2.cvtColor(both, cv2.COLOR_RGB2BGR)
        cv2.putText(out, "DATASET EGO (policy input)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, "SIM (predicted trajectory)", (W + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"{label}  {ep_tag}", (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(out, f"prompt: {prompt}", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, f"t={t}/{T}", (W + 10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        writer.write(out)

    cap.release()


def main():
    stats = json.load(open(CKPT / "experiment_cfg" / "dataset_statistics.json"))
    A = stats["new_embodiment"]["action"]
    body_min = np.concatenate([A["left_leg"]["min"], A["right_leg"]["min"], A["waist"]["min"],
                               A["left_arm"]["min"], A["right_arm"]["min"]]).astype(np.float64)
    body_max = np.concatenate([A["left_leg"]["max"], A["right_leg"]["max"], A["waist"]["max"],
                               A["left_arm"]["max"], A["right_arm"]["max"]]).astype(np.float64)
    hand_min = np.concatenate([A["left_hand"]["min"], A["right_hand"]["min"]]).astype(np.float64)
    hand_max = np.concatenate([A["left_hand"]["max"], A["right_hand"]["max"]]).astype(np.float64)
    bounds = (body_min, body_max, hand_min, hand_max)

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"[eval] loading policy from {CKPT}")
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(CKPT),
        device="cuda:0",
    )
    print("[eval] policy ready")

    # Phase 1: rollouts
    rollouts = {}  # label -> list of (ep_tag, traj, mp4_path)
    for label, prompt in TASKS:
        rollouts[label] = []
        task_dir = VAL_DIR / label
        eps = sorted(task_dir.glob("ep*.parquet"))
        print(f"\n=== rollout: {label}  ({len(eps)} episodes) ===")
        for pq_path in eps:
            ep_tag = pq_path.stem  # ep000 ...
            mp4_path = pq_path.with_suffix(".mp4")
            t0 = time.time()
            pb, ph, gt = rollout_episode(policy, prompt, pq_path, mp4_path, bounds)
            print(f"  {ep_tag}: T={len(pb)}  {time.time()-t0:.1f}s")
            np.savez_compressed(TRAJ_DIR / f"{label}_{ep_tag}.npz",
                                pred_body=pb, pred_hand=ph, gt_root7=gt, prompt=prompt)
            rollouts[label].append((ep_tag, pb, ph, gt, mp4_path))

    del policy
    torch.cuda.empty_cache()

    # Phase 2: per-task replay MP4
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    hmap = build_hand_qpos_map(m)
    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    r_tp = mujoco.Renderer(m, height=H, width=W)

    for label, prompt in TASKS:
        out_path = OUT_DIR / f"{label}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
        print(f"\n[replay] {label} -> {out_path.name}")
        for ep_tag, pb, ph, gt, mp4 in rollouts[label]:
            print(f"  {ep_tag}  T={len(pb)}")
            replay_episode(m, d, qmap, hmap, pelvis_bid, r_tp, writer,
                           label, ep_tag, prompt, pb, ph, gt, mp4)
        writer.release()

    r_tp.close()
    print("\n[eval] all done.")
    for label, _ in TASKS:
        print(f"  {OUT_DIR / (label + '.mp4')}")


if __name__ == "__main__":
    main()
