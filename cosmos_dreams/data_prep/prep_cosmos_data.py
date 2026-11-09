"""Convert Unitree G1 WBT v3 LeRobot datasets → Cosmos-Predict2 flat training layout.

Input: HF v3 LeRobot datasets (`unitreerobotics/G1_WBT_Inspire_*`)
Output:
  <out>/
    metadata.csv                # rows: filename,prompt
    videos/<task>_ep<idx>.mp4   # 432x768 @ 16fps, exactly 93 frames
    metas/                      # (empty — auto-filled by T5 script)
    t5_xxl/                     # (empty — auto-filled by T5 script)

Per episode: take first 93 frames @ 16fps after resampling + center-crop/resize to 432x768.
Short episodes are skipped. Long episodes optionally sliding-windowed.

Run on the GPU machine after downloading the source data.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

TASKS = {
    "pillow":  ("G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
                "G1_WB_Dex5_Pickup_Pillow", "observation.images.cam_0",
                "Pick up the pillow and put it on the sofa."),
    "clothes": ("G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
                "G1_WB_Dex5_Collect_Clothes", "observation.images.cam_0",
                "Collect the clothes on the floor."),
    "washing": ("G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
                "G1_WB_Dex5_Put_Clothes_into_Washing_Machine", "observation.images.cam_0",
                "Put the clothes into the washing machine."),
    "fridge":  ("G1_WBT_Inspire_Put_Drinks_Into_Fridge", "",
                "observation.images.head_stereo_left",
                "Put the drinks into the fridge."),
}

COSMOS_W, COSMOS_H = 768, 432
COSMOS_FPS = 16
COSMOS_FRAMES = 93  # ~5.8s
CLIP_SECONDS = COSMOS_FRAMES / COSMOS_FPS  # 5.8125


def load_meta(repo, subdir):
    prefix = f"{subdir}/" if subdir else ""
    # Episodes may be across multiple meta files — walk until we've covered all
    eps = []
    idx = 0
    while True:
        try:
            p = hf_hub_download(f"unitreerobotics/{repo}",
                                f"{prefix}meta/episodes/chunk-000/file-{idx:03d}.parquet",
                                repo_type="dataset")
            df = pd.read_parquet(p)
            eps.append(df)
            idx += 1
        except Exception:
            break
    return pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()


def cut_clip(src_mp4, t0, out_mp4, slide_start_s=0.0):
    """Cut exactly 93 frames at 16fps (5.8s) starting at (t0 + slide_start_s),
    resize + center-crop to 432x768, re-encode H264."""
    start = float(t0) + float(slide_start_s)
    vf = (
        f"fps={COSMOS_FPS},"
        f"scale={COSMOS_W}:-2:force_original_aspect_ratio=increase,"
        f"crop={COSMOS_W}:{COSMOS_H}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src_mp4),
        "-frames:v", str(COSMOS_FRAMES),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ffmpeg FAIL: {r.stderr[:200]}")
        return False
    return True


def process_task(label, out_root: Path, windows_per_ep: int, max_eps: int | None):
    repo, subdir, cam, prompt = TASKS[label]
    print(f"\n=== {label} ({repo}) ===")
    prefix = f"{subdir}/" if subdir else ""
    eps_df = load_meta(repo, subdir)
    if eps_df.empty:
        print(f"  no episodes found")
        return []
    print(f"  {len(eps_df)} episodes in meta")

    vid_from_col = next((c for c in eps_df.columns
                         if c.endswith("from_timestamp") and cam.split(".")[-1] in c), None)
    vid_to_col = next((c for c in eps_df.columns
                       if c.endswith("to_timestamp") and cam.split(".")[-1] in c), None)
    if vid_from_col is None:
        vid_from_col = next((c for c in eps_df.columns if c.endswith("from_timestamp")), None)
        vid_to_col = next((c for c in eps_df.columns if c.endswith("to_timestamp")), None)

    vid_chunk_col = next((c for c in eps_df.columns if "chunk_index" in c and cam.split(".")[-1] in c), None)
    vid_file_col = next((c for c in eps_df.columns if "file_index" in c and cam.split(".")[-1] in c), None)

    rows = []
    # Cache source video files (one download per (chunk,file))
    vid_cache = {}
    for row_i, row in eps_df.iterrows():
        if max_eps and row_i >= max_eps:
            break
        ep_idx = int(row["episode_index"]) if "episode_index" in eps_df.columns else row_i
        t0 = float(row[vid_from_col]) if vid_from_col else 0.0
        t1 = float(row[vid_to_col]) if vid_to_col else t0 + CLIP_SECONDS
        length_s = t1 - t0

        if length_s < CLIP_SECONDS:
            continue  # too short

        vid_chunk = int(row[vid_chunk_col]) if vid_chunk_col else 0
        vid_file = int(row[vid_file_col]) if vid_file_col else 0
        key = (vid_chunk, vid_file)
        if key not in vid_cache:
            try:
                vid_cache[key] = hf_hub_download(
                    f"unitreerobotics/{repo}",
                    f"{prefix}videos/{cam}/chunk-{vid_chunk:03d}/file-{vid_file:03d}.mp4",
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"    !! video download fail ep{ep_idx}: {e}")
                continue
        src_mp4 = vid_cache[key]

        # Generate up to `windows_per_ep` clips per episode
        max_windows = max(1, int(length_s // CLIP_SECONDS))
        n_win = min(windows_per_ep, max_windows)
        for w in range(n_win):
            slide = w * CLIP_SECONDS
            if slide + CLIP_SECONDS > length_s:
                break
            clip_id = f"{label}_ep{ep_idx:05d}_w{w}"
            out_mp4 = out_root / "videos" / f"{clip_id}.mp4"
            if out_mp4.exists():
                rows.append((f"{clip_id}.mp4", prompt))
                continue
            if cut_clip(src_mp4, t0, out_mp4, slide_start_s=slide):
                rows.append((f"{clip_id}.mp4", prompt))
        if row_i % 50 == 0:
            print(f"  ep{ep_idx}: {len(rows)} clips so far")

    print(f"  {label}: {len(rows)} clips emitted")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output dir (e.g., datasets/benchmark_train/g1_wbt)")
    p.add_argument("--tasks", nargs="+", default=["pillow", "clothes"])
    p.add_argument("--windows-per-ep", type=int, default=1,
                   help="Max clips per episode (sliding-window). 1 = first 93 frames only.")
    p.add_argument("--max-eps", type=int, default=None, help="Cap episodes per task (debug)")
    a = p.parse_args()

    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "metas").mkdir(exist_ok=True)
    (out / "t5_xxl").mkdir(exist_ok=True)

    all_rows = []
    for label in a.tasks:
        all_rows.extend(process_task(label, out, a.windows_per_ep, a.max_eps))

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_filename", "prompt"])
        w.writerows(all_rows)

    print(f"\n[done] {len(all_rows)} clips -> {out / 'metadata.csv'}")
    print(f"  next: run cosmos-predict2 T5 embed script to fill metas/ + t5_xxl/")


if __name__ == "__main__":
    main()
