"""Closed-loop eval: seed sim from row 0, then sim-state is fed back to policy.

Per episode:
  - Seed floating base + joints from dataset row 0
  - At each step t: read dataset ego frame → build obs with SIM's current joints as state
    → policy.get_action() → apply chunk[0:K] to sim
  - Root pose pinned to seed (NOT updated from GT) — no free locomotion cheat
  - Accumulate per-step action MSE vs GT action.robot_q_desired for quantitative signal

Output: per-task MP4 showing dataset ego | closed-loop sim + MSE overlay.
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
OUT_DIR = OTHER / "eval_closedloop"
OUT_DIR.mkdir(exist_ok=True)

TASKS = [
    ("pillow",     "Pick up the pillow and put it on the sofa."),
    ("clothes",    "Collect the clothes on the floor."),
    ("washing",    "Put the clothes into the washing machine."),
    ("fridge",     "Put the drinks into the fridge."),
    ("basket_OOD", "Put the clothes into the basket."),
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


def run_episode_closedloop(m, d, qmap, hmap, pelvis_bid, r_tp, writer,
                           policy, label, ep_tag, prompt,
                           pq_path, mp4_path, bounds):
    body_min, body_max, hand_min, hand_max = bounds
    df = pd.read_parquet(pq_path)
    cap = cv2.VideoCapture(str(mp4_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(len(df), n_vid)

    # Seed — GT at t=0, pinned for remainder of episode
    seed = df.iloc[0]
    rq0 = np.asarray(seed["observation.state.robot_q_current"], dtype=np.float64)
    hs0 = np.asarray(seed["observation.state.hand_state"], dtype=np.float64)
    root7 = rq0[0:7].copy()  # pinned
    set_robot_qpos(d, root7, rq0[7:36], qmap)
    set_hand_qpos(d, hs0[0:12], hmap)
    mujoco.mj_forward(m, d)
    hs = hs0.astype(np.float32)

    mse_pred_vs_gt_action = []   # learning signal
    mse_pred_vs_gt_state  = []   # identity baseline (pred vs current state)
    frame_idx = 0
    stride_count = 0

    while frame_idx < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ego = cv2.resize(rgb, (W, H))

        # build obs with SIM's current state (not dataset's)
        sim_rq = np.zeros(36, dtype=np.float32)
        sim_rq[0:7] = d.qpos[0:7]
        for i, qi in enumerate(qmap):
            sim_rq[7 + i] = d.qpos[qi]

        obs = {
            "video": {"ego_view": ego[None, None]},
            "state": slice_state(sim_rq, hs),
            "language": {"annotation.human.task_description": [[prompt]]},
        }
        with torch.no_grad():
            action, _ = policy.get_action(obs)
        body_chunk, hand_chunk = pack_action_chunk(action)

        # MSE vs GT
        row = df.iloc[frame_idx]
        gt_act = np.asarray(row["action.robot_q_desired"], dtype=np.float64)  # 36
        gt_body_act = gt_act[7:36]                                             # 29
        pred_body0 = body_chunk[0].astype(np.float64)
        cur_body_state = sim_rq[7:36].astype(np.float64)
        mse_pred_vs_gt_action.append(float(np.mean((pred_body0 - gt_body_act) ** 2)))
        mse_pred_vs_gt_state.append(float(np.mean((cur_body_state - gt_body_act) ** 2)))

        K = min(ACTION_STRIDE, body_chunk.shape[0])
        for k in range(K):
            target_body = np.clip(body_chunk[k].astype(np.float64), body_min, body_max)
            target_hand = np.clip(hand_chunk[k].astype(np.float64), hand_min, hand_max)
            set_robot_qpos(d, root7, target_body, qmap)    # root pinned
            set_hand_qpos(d, target_hand, hmap)
            hs = target_hand.astype(np.float32)
            mujoco.mj_forward(m, d)

            pelvis = d.xpos[pelvis_bid].copy()
            r_tp.update_scene(d, camera=third_person(pelvis))
            sim_vis = r_tp.render()

            # match video frame that corresponds to this write index
            vid_frame = min(frame_idx + k, n_vid - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame)
            okf, bgrf = cap.read()
            ego_vis = cv2.resize(cv2.cvtColor(bgrf, cv2.COLOR_BGR2RGB), (W, H)) if okf else ego

            both = np.concatenate([ego_vis, sim_vis], axis=1)
            out = cv2.cvtColor(both, cv2.COLOR_RGB2BGR)
            cv2.putText(out, "DATASET EGO (policy input)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(out, "SIM closed-loop from seed", (W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(out, f"{label}  {ep_tag}", (10, H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            mse_avg_pred = float(np.mean(mse_pred_vs_gt_action))
            mse_avg_id   = float(np.mean(mse_pred_vs_gt_state))
            cv2.putText(out,
                        f"MSE(pred,GT)={mse_avg_pred:.3f}  MSE(state,GT)={mse_avg_id:.3f}",
                        (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(out, f"prompt: {prompt}", (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(out, f"t={stride_count}", (W + 10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            writer.write(out)
            stride_count += 1

        frame_idx += FRAME_SKIP

    cap.release()
    return {
        "pred_vs_gt_action_mean": float(np.mean(mse_pred_vs_gt_action)),
        "state_vs_gt_action_mean": float(np.mean(mse_pred_vs_gt_state)),
        "n_infers": len(mse_pred_vs_gt_action),
    }


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
    print(f"[cl] loading policy from {CKPT}")
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(CKPT),
        device="cuda:0",
    )
    print("[cl] policy ready")

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    qmap = build_joint_qpos_map(m)
    hmap = build_hand_qpos_map(m)
    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    r_tp = mujoco.Renderer(m, height=H, width=W)

    summary = {}
    for label, prompt in TASKS:
        task_dir = VAL_DIR / label
        eps = sorted(task_dir.glob("ep*.parquet"))
        out_path = OUT_DIR / f"{label}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
        task_stats = []
        for pq_path in eps:
            ep_tag = pq_path.stem
            mp4_path = pq_path.with_suffix(".mp4")
            t0 = time.time()
            s = run_episode_closedloop(m, d, qmap, hmap, pelvis_bid, r_tp, writer,
                                       policy, label, ep_tag, prompt,
                                       pq_path, mp4_path, bounds)
            print(f"[{label}/{ep_tag}] MSE(pred,GT)={s['pred_vs_gt_action_mean']:.4f}  "
                  f"MSE(state,GT)={s['state_vs_gt_action_mean']:.4f}  "
                  f"n={s['n_infers']}  {time.time()-t0:.1f}s")
            task_stats.append(s)
        writer.release()
        summary[label] = task_stats
        print(f"  wrote {out_path}")

    r_tp.close()
    json.dump(summary, open(OUT_DIR / "mse_summary.json", "w"), indent=2)
    print(f"\n[cl] summary -> {OUT_DIR / 'mse_summary.json'}")

    # aggregate
    print("\n==== AGGREGATE ====")
    print(f"{'task':<15} {'pred_vs_GT':>12} {'state_vs_GT':>14}  (lower pred = real learning)")
    for label, _ in TASKS:
        s = summary[label]
        pred = np.mean([x["pred_vs_gt_action_mean"] for x in s])
        iden = np.mean([x["state_vs_gt_action_mean"] for x in s])
        print(f"{label:<15} {pred:>12.4f} {iden:>14.4f}  ratio={pred/iden:.3f}")


if __name__ == "__main__":
    main()
