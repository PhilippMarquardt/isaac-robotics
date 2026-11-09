"""GPU-side: run trained GR00T policy on holdout episodes, dump JSON for MuJoCo viz.

Per frame: {t, robot_q_obs(36), hand_state(12), pred_action_chunk(40,48), gt_action_chunk(40,48)}.
Actions concat order matches dataset: 36-dim robot_q_desired + 12-dim hand_cmd = 48.

Runs inside the Isaac-GR00T venv on the training machine, e.g.:
  cd /workspace/dreamdojo/Isaac-GR00T && source .venv/bin/activate && python gen_trajectories.py --episodes 0 1 2 --out /workspace/eval_out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

CKPT = "/workspace/runs/gr00t_pillow/g1_pillow_v1/checkpoint-10000"
DATA = "/workspace/data/pillow/G1_WB_Dex5_Pickup_Pillow"
TASK = "Pick up the pillow and put it on the sofa."
STRIDE = 15  # ~0.5 s at 30 Hz, keep JSON small


def slice_state(rq: np.ndarray, hs: np.ndarray) -> dict:
    return {
        "left_leg":   rq[7:13][None],
        "right_leg":  rq[13:19][None],
        "waist":      rq[19:22][None],
        "left_arm":   rq[22:29][None],
        "right_arm":  rq[29:36][None],
        "left_hand":  hs[0:6][None],
        "right_hand": hs[6:12][None],
    }


def build_state_from_row(row) -> tuple[np.ndarray, np.ndarray]:
    rq = np.asarray(row["observation.state.robot_q_current"], dtype=np.float32)
    hs = np.asarray(row["observation.state.hand_state"], dtype=np.float32)
    return rq, hs


def read_video_frame(mp4: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cv2 failed frame {frame_idx} in {mp4}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stack_gt_chunk(df: pd.DataFrame, t: int, horizon: int) -> np.ndarray:
    """Return (H, 48) = concat(robot_q_desired[36], hand_cmd[12]) over future H frames (clipped)."""
    H = horizon
    n = len(df)
    rows = [df.iloc[min(t + k, n - 1)] for k in range(H)]
    rq = np.stack([np.asarray(r["action.robot_q_desired"], dtype=np.float32) for r in rows])
    hc = np.stack([np.asarray(r["action.hand_cmd"], dtype=np.float32) for r in rows])
    return np.concatenate([rq, hc], axis=1)


def pack_action_chunk(action: dict) -> np.ndarray:
    """Model output (dict of 7 keys) → (H, 48). Dataset order: robot_q[7..36] + hand[12].
    Dataset q[0:7] = root pose (not predicted by 29-DOF body actions); fill with zeros."""
    H = np.asarray(action["left_leg"]).shape[0]
    q = np.zeros((H, 36), dtype=np.float32)
    q[:, 7:13]  = np.asarray(action["left_leg"])
    q[:, 13:19] = np.asarray(action["right_leg"])
    q[:, 19:22] = np.asarray(action["waist"])
    q[:, 22:29] = np.asarray(action["left_arm"])
    q[:, 29:36] = np.asarray(action["right_arm"])
    h = np.zeros((H, 12), dtype=np.float32)
    h[:, 0:6]  = np.asarray(action["left_hand"])
    h[:, 6:12] = np.asarray(action["right_hand"])
    return np.concatenate([q, h], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=str, default="/workspace/eval_out.json")
    ap.add_argument("--stride", type=int, default=STRIDE)
    args = ap.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=CKPT,
        device="cuda:0",
        strict=False,
    )
    print(f"[gen] policy loaded from {CKPT}")

    out = {"task": TASK, "stride": args.stride, "episodes": []}
    for ep in args.episodes:
        pq = Path(DATA) / f"data/chunk-000/episode_{ep:06d}.parquet"
        mp4 = Path(DATA) / f"videos/chunk-000/observation.images.cam_0/episode_{ep:06d}.mp4"
        df = pd.read_parquet(pq)
        print(f"[gen] ep{ep}: {len(df)} frames")
        frames = []
        for t in range(0, len(df), args.stride):
            row = df.iloc[t]
            rq, hs = build_state_from_row(row)
            img = read_video_frame(mp4, t)
            obs = {
                "video": {"ego_view": img[None]},
                "state": slice_state(rq, hs),
                "language": {"annotation.human.task_description": [TASK]},
            }
            with torch.no_grad():
                action, _ = policy.get_action(obs)
            pred = pack_action_chunk(action)           # (H, 48)
            gt = stack_gt_chunk(df, t, pred.shape[0])  # (H, 48)
            frames.append({
                "t": int(t),
                "robot_q_obs": rq.tolist(),
                "hand_state": hs.tolist(),
                "pred_chunk": pred.tolist(),
                "gt_chunk": gt.tolist(),
            })
            print(f"  t={t} pred[0,22:29]={pred[0,22:29].round(3).tolist()} gt[0,22:29]={gt[0,22:29].round(3).tolist()}")
        out["episodes"].append({"episode_idx": ep, "n_total": len(df), "frames": frames})

    Path(args.out).write_text(json.dumps(out))
    print(f"[gen] wrote {args.out}  size_bytes={Path(args.out).stat().st_size}")


if __name__ == "__main__":
    main()
