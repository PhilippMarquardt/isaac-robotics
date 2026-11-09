"""Pure open-loop prediction MSE — no sim, no integration.

For each dataset frame: feed GT state + GT video → predict action → compare to GT action.
If model actually learned, MSE should be tiny (<< 0.01). If MSE is high here,
training itself failed or there's a feature-mismatch between train and eval.

Also per-joint breakdown so we can see WHICH joints are off.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = os.environ.get("GR00T_REPO", "Isaac-GR00T")  # path to an Isaac-GR00T checkout
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GROOT_EVAL = Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir"))
CKPT = GROOT_EVAL / "ckpt_alltasks"
VAL_DIR = GROOT_EVAL / "_samples_other" / "val_episodes_real"
OUT_DIR = GROOT_EVAL / "_samples_other" / "eval_pure_real"
OUT_DIR.mkdir(exist_ok=True)

TASKS = [
    ("pillow",     "Pick up the pillow and put it on the sofa."),
    ("clothes",    "Collect the clothes on the floor."),
    ("washing",    "Put the clothes into the washing machine."),
    ("fridge",     "Put the drinks into the fridge."),
    ("basket_OOD", "Put the clothes into the basket."),
]

W, H = 540, 540
FRAME_SKIP = 4

GROUPS = [
    ("left_leg",   7, 13),
    ("right_leg", 13, 19),
    ("waist",     19, 22),
    ("left_arm",  22, 29),
    ("right_arm", 29, 36),
]


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


def pack_pred(action):
    def u(x):
        a = np.asarray(x)
        return a[0] if a.ndim == 3 else a
    ll = u(action["left_leg"]); rl = u(action["right_leg"]); w = u(action["waist"])
    la = u(action["left_arm"]); ra = u(action["right_arm"])
    lh = u(action["left_hand"]); rh = u(action["right_hand"])
    body = np.concatenate([ll[0], rl[0], w[0], la[0], ra[0]])   # (29,)
    hand = np.concatenate([lh[0], rh[0]])                        # (12,)
    return body, hand


def eval_episode(policy, prompt, pq_path, mp4_path):
    df = pd.read_parquet(pq_path)
    cap = cv2.VideoCapture(str(mp4_path))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(len(df), n_vid)

    body_sq_err = []       # per step (29,)
    hand_sq_err = []       # per step (12,)
    pred_body_all = []
    gt_body_all = []
    state_body_all = []    # for identity baseline

    frame_idx = 0
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
        gt_act = np.asarray(row["action.robot_q_desired"], dtype=np.float64)

        obs = {
            "video": {"ego_view": ego[None, None]},
            "state": slice_state(rq, hs),
            "language": {"annotation.human.task_description": [[prompt]]},
        }
        with torch.no_grad():
            action, _ = policy.get_action(obs)
        pb, ph = pack_pred(action)

        gt_body = gt_act[7:36]
        body_sq_err.append((pb.astype(np.float64) - gt_body) ** 2)
        pred_body_all.append(pb.astype(np.float64))
        gt_body_all.append(gt_body)
        state_body_all.append(rq[7:36].astype(np.float64))

        frame_idx += FRAME_SKIP

    cap.release()

    body_sq_err = np.stack(body_sq_err)        # (T, 29)
    pred_body_all = np.stack(pred_body_all)    # (T, 29)
    gt_body_all = np.stack(gt_body_all)
    state_body_all = np.stack(state_body_all)

    per_joint = body_sq_err.mean(axis=0)         # (29,)
    overall = float(per_joint.mean())
    identity = float(np.mean((state_body_all - gt_body_all) ** 2))

    group = {}
    for name, lo, hi in GROUPS:
        group[name] = float(per_joint[lo - 7:hi - 7].mean())

    return {
        "n": int(body_sq_err.shape[0]),
        "overall_mse": overall,
        "identity_mse": identity,
        "group_mse": group,
        "per_joint": per_joint.tolist(),
    }


def main():
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"[pure] loading policy from {CKPT}")
    policy = Gr00tPolicy(
        embodiment_tag="new_embodiment",
        model_path=str(CKPT),
        device="cuda:0",
    )
    print("[pure] policy ready")

    summary = {}
    for label, prompt in TASKS:
        eps = sorted((VAL_DIR / label).glob("ep*.parquet"))
        task_out = []
        for pq in eps:
            mp4 = pq.with_suffix(".mp4")
            t0 = time.time()
            s = eval_episode(policy, prompt, pq, mp4)
            gm = s["group_mse"]
            print(f"[{label}/{pq.stem}] n={s['n']}  MSE={s['overall_mse']:.4f}  "
                  f"identity={s['identity_mse']:.4f}  "
                  f"ll={gm['left_leg']:.3f} rl={gm['right_leg']:.3f} "
                  f"w={gm['waist']:.3f} la={gm['left_arm']:.3f} ra={gm['right_arm']:.3f}  "
                  f"{time.time()-t0:.1f}s")
            task_out.append(s)
        summary[label] = task_out

    json.dump(summary, open(OUT_DIR / "pure_mse_summary.json", "w"), indent=2)

    print("\n==== AGGREGATE (pure open-loop prediction, 1 chunk step vs GT action) ====")
    print(f"{'task':<15} {'MSE':>8} {'identity':>9} {'ratio':>6}  {'ll':>6} {'rl':>6} {'w':>6} {'la':>6} {'ra':>6}")
    for label, _ in TASKS:
        s = summary[label]
        mse = np.mean([x["overall_mse"] for x in s])
        ide = np.mean([x["identity_mse"] for x in s])
        gm = {k: np.mean([x["group_mse"][k] for x in s]) for k in ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]}
        print(f"{label:<15} {mse:>8.4f} {ide:>9.4f} {mse/ide:>6.3f}  "
              f"{gm['left_leg']:>6.3f} {gm['right_leg']:>6.3f} {gm['waist']:>6.3f} "
              f"{gm['left_arm']:>6.3f} {gm['right_arm']:>6.3f}")


if __name__ == "__main__":
    main()
