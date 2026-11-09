"""Pull 3 val episodes per task from HF v3 LeRobot shards.

For each HF dataset, download file-000 of data/ + meta/episodes/ + videos/<cam>/. Use
meta/episodes to compute per-episode frame ranges, then slice the parquet by episode_index
and cut the mp4 using ffmpeg/cv2 to per-episode clips.

Output layout: _samples_other/val_episodes/<task>/ep<idx>.{parquet,mp4}
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")) / "_samples_other" / "val_episodes"
OUT.mkdir(exist_ok=True, parents=True)

# (local_label, HF repo, subdir inside repo (or ""), camera dir, episode indices to pull)
TASKS = [
    ("pillow",  "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
     "G1_WB_Dex5_Pickup_Pillow", "observation.images.cam_0", [0, 1, 2]),
    ("clothes", "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
     "G1_WB_Dex5_Collect_Clothes", "observation.images.cam_0", [0, 1, 2]),
    ("washing", "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
     "G1_WB_Dex5_Put_Clothes_into_Washing_Machine", "observation.images.cam_0", [0, 1, 2]),
    ("fridge",  "G1_WBT_Inspire_Put_Drinks_Into_Fridge",
     "", "observation.images.head_stereo_left", [0, 1, 2]),
    # OOD — task not in training set
    ("basket_OOD", "G1_WBT_Inspire_Put_Clothes_Into_Basket",
     "", "observation.images.head_stereo_left", [0, 1, 2]),
]


def pull_task(label, repo, subdir, cam, ep_indices):
    print(f"\n=== {label}  ({repo}) ===")
    task_out = OUT / label
    task_out.mkdir(exist_ok=True)

    prefix = f"{subdir}/" if subdir else ""
    files = {
        "data": f"{prefix}data/chunk-000/file-000.parquet",
        "episodes": f"{prefix}meta/episodes/chunk-000/file-000.parquet",
        "video": f"{prefix}videos/{cam}/chunk-000/file-000.mp4",
        "tasks": f"{prefix}meta/tasks.parquet",
    }
    cached = {}
    for key, path in files.items():
        print(f"  [dl] {path}")
        p = hf_hub_download(f"unitreerobotics/{repo}", path, repo_type="dataset")
        cached[key] = Path(p)

    episodes = pd.read_parquet(cached["episodes"])
    data = pd.read_parquet(cached["data"])
    tasks = pd.read_parquet(cached["tasks"])
    print(f"  episodes table cols: {list(episodes.columns)[:10]}")
    print(f"  num episodes in file-000: {len(episodes)}")
    print(f"  tasks: {tasks.to_dict(orient='records')[:5]}")

    # Episode table in v3 has columns like: episode_index, length, tasks, data/chunk_index,
    # data/file_index, videos/<cam>/chunk_index, videos/<cam>/file_index,
    # videos/<cam>/from_timestamp, videos/<cam>/to_timestamp, dataset_from_index, dataset_to_index
    vid_from_col = None
    vid_to_col = None
    for c in episodes.columns:
        if c.endswith("from_timestamp") and cam.split(".")[-1] in c:
            vid_from_col = c
        if c.endswith("to_timestamp") and cam.split(".")[-1] in c:
            vid_to_col = c
    if vid_from_col is None:
        cand_from = [c for c in episodes.columns if c.endswith("from_timestamp")]
        cand_to = [c for c in episodes.columns if c.endswith("to_timestamp")]
        vid_from_col = cand_from[0] if cand_from else None
        vid_to_col = cand_to[0] if cand_to else None
    print(f"  video range cols: {vid_from_col} / {vid_to_col}")

    # Get ordered unique episode_indices in this file
    if "episode_index" in episodes.columns:
        ep_all = sorted(episodes["episode_index"].unique().tolist())
    else:
        ep_all = list(range(len(episodes)))
    chosen = [ep_all[i] for i in ep_indices if i < len(ep_all)]
    print(f"  chosen episodes: {chosen}")

    for ep in chosen:
        if "episode_index" in episodes.columns:
            ep_row = episodes[episodes["episode_index"] == ep].iloc[0]
        else:
            ep_row = episodes.iloc[ep]

        # slice data rows
        if "episode_index" in data.columns:
            sub = data[data["episode_index"] == ep].reset_index(drop=True)
        else:
            # use from/to dataset indices
            lo = int(ep_row.get("dataset_from_index", 0))
            hi = int(ep_row.get("dataset_to_index", len(data)))
            sub = data.iloc[lo:hi].reset_index(drop=True)
        print(f"    ep{ep}: {len(sub)} rows")

        pq_path = task_out / f"ep{ep:03d}.parquet"
        sub.to_parquet(pq_path)

        # cut mp4 by timestamps (v3 stores from_timestamp/to_timestamp per ep)
        t0 = float(ep_row[vid_from_col]) if vid_from_col else 0.0
        t1 = float(ep_row[vid_to_col]) if vid_to_col else None
        mp4_path = task_out / f"ep{ep:03d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{t0:.3f}",
               "-i", str(cached["video"])]
        if t1 is not None:
            cmd += ["-to", f"{t1:.3f}"]
        cmd += ["-c", "copy", str(mp4_path)]
        print(f"    cut mp4 [{t0:.2f}-{t1 if t1 else '?':.2f}] -> {mp4_path.name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ffmpeg failed, retrying with re-encode")
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-ss", f"{t0:.3f}", "-i", str(cached["video"])]
            if t1 is not None:
                cmd += ["-to", f"{t1:.3f}"]
            cmd += ["-c:v", "libx264", "-preset", "fast", str(mp4_path)]
            subprocess.run(cmd, check=True)


def main():
    for label, repo, subdir, cam, eps in TASKS:
        try:
            pull_task(label, repo, subdir, cam, eps)
        except Exception as e:
            print(f"!! {label} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
