"""Pull TRUE held-out val episodes using seed=42 80/20 split_datasets.py indices.

Previous pull_val_episodes.py used ep_indices [0,1,2] which leaked into train for 3 of 4 tasks.
This script uses the reproduced val indices from reproduce_split.py.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(os.environ.get("GR00T_EVAL_DIR", "gr00t_eval_workdir")) / "_samples_other" / "val_episodes_real"
OUT.mkdir(exist_ok=True, parents=True)

TASKS = [
    ("pillow",     "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
     "G1_WB_Dex5_Pickup_Pillow", "observation.images.cam_0", [8, 12, 14]),
    ("clothes",    "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
     "G1_WB_Dex5_Collect_Clothes", "observation.images.cam_0", [0, 8, 9]),
    ("washing",    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
     "G1_WB_Dex5_Put_Clothes_into_Washing_Machine", "observation.images.cam_0", [4, 6, 15]),
    ("fridge",     "G1_WBT_Inspire_Put_Drinks_Into_Fridge",
     "", "observation.images.head_stereo_left", [3, 4, 11]),
    ("basket_OOD", "G1_WBT_Inspire_Put_Clothes_Into_Basket",
     "", "observation.images.head_stereo_left", [0, 1, 2]),  # task not trained -> all held out
]


def pull_episode(label, repo, subdir, cam, ep_idx, task_out):
    """Pull single episode by episode_index. Auto-discover which chunk/file it's in."""
    prefix = f"{subdir}/" if subdir else ""

    # Download episodes meta first (single file-000 assumed; fallback: walk chunks)
    # For v3 large datasets episodes are sharded across chunk-000/file-000..N
    # We query all chunk-000 files until we find the ep.
    found_ep_row = None
    file_tried = 0
    while found_ep_row is None and file_tried < 10:
        try:
            ep_meta_path = hf_hub_download(
                f"unitreerobotics/{repo}",
                f"{prefix}meta/episodes/chunk-000/file-{file_tried:03d}.parquet",
                repo_type="dataset",
            )
        except Exception:
            break
        ep_df = pd.read_parquet(ep_meta_path)
        hit = ep_df[ep_df["episode_index"] == ep_idx]
        if len(hit) > 0:
            found_ep_row = hit.iloc[0]
            ep_meta_file = file_tried
            break
        file_tried += 1

    if found_ep_row is None:
        print(f"  [skip] ep{ep_idx}: not found in first 10 meta files")
        return

    # Get data file index + video file index
    data_chunk = int(found_ep_row.get("data/chunk_index", 0))
    data_file = int(found_ep_row.get("data/file_index", 0))
    vid_chunk = int(found_ep_row.get(f"videos/{cam}/chunk_index", 0))
    vid_file = int(found_ep_row.get(f"videos/{cam}/file_index", 0))

    # Download the right data parquet + video
    data_path = hf_hub_download(
        f"unitreerobotics/{repo}",
        f"{prefix}data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet",
        repo_type="dataset",
    )
    vid_path = hf_hub_download(
        f"unitreerobotics/{repo}",
        f"{prefix}videos/{cam}/chunk-{vid_chunk:03d}/file-{vid_file:03d}.mp4",
        repo_type="dataset",
    )

    data = pd.read_parquet(data_path)
    if "episode_index" in data.columns:
        sub = data[data["episode_index"] == ep_idx].reset_index(drop=True)
    else:
        lo = int(found_ep_row["dataset_from_index"])
        hi = int(found_ep_row["dataset_to_index"])
        sub = data.iloc[lo:hi].reset_index(drop=True)
    print(f"    ep{ep_idx}: {len(sub)} rows (meta file {ep_meta_file}, data file {data_file})")

    pq_out = task_out / f"ep{ep_idx:03d}.parquet"
    sub.to_parquet(pq_out)

    # Find video from/to timestamps
    vid_from_col = next((c for c in ep_df.columns
                         if c.endswith("from_timestamp") and cam.split(".")[-1] in c), None)
    vid_to_col = next((c for c in ep_df.columns
                       if c.endswith("to_timestamp") and cam.split(".")[-1] in c), None)
    if vid_from_col is None:
        vid_from_col = next((c for c in ep_df.columns if c.endswith("from_timestamp")), None)
        vid_to_col = next((c for c in ep_df.columns if c.endswith("to_timestamp")), None)
    t0 = float(found_ep_row[vid_from_col]) if vid_from_col else 0.0
    t1 = float(found_ep_row[vid_to_col]) if vid_to_col else None

    mp4_out = task_out / f"ep{ep_idx:03d}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{t0:.3f}", "-i", str(vid_path)]
    if t1 is not None:
        cmd += ["-to", f"{t1:.3f}"]
    cmd += ["-c", "copy", str(mp4_out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{t0:.3f}", "-i", str(vid_path)]
        if t1 is not None:
            cmd += ["-to", f"{t1:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "fast", str(mp4_out)]
        subprocess.run(cmd, check=True)


def main():
    for label, repo, subdir, cam, eps in TASKS:
        print(f"\n=== {label} ({repo}) — eps {eps} ===")
        task_out = OUT / label
        task_out.mkdir(exist_ok=True)
        for ep in eps:
            try:
                pull_episode(label, repo, subdir, cam, ep, task_out)
            except Exception as e:
                print(f"    !! ep{ep} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
