"""Convert local G1 v2.1 LeRobot data -> Cosmos-Predict2 flat layout.

Input (local):
  <root>/meta/episodes.jsonl
  <root>/videos/chunk-000/observation.images.cam_0/episode_XXXXXX.mp4
Output:
  <out>/metadata.csv   video_filename,prompt
  <out>/videos/*.mp4   432x768 @ 16fps, 93 frames
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

TASKS = {
    "pillow": (
        "/workspace/data/pillow/G1_WB_Dex5_Pickup_Pillow",
        "observation.images.cam_0",
        "Pick up the pillow and put it on the sofa.",
    ),
    "clothes": (
        "/workspace/data/wbt_new/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/G1_WB_Dex5_Collect_Clothes",
        "observation.images.cam_0",
        "Collect the clothes on the floor.",
    ),
    "washing": (
        "/workspace/data/wbt_new/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly/G1_WB_Dex5_Put_Clothes_into_Washing_Machine",
        "observation.images.cam_0",
        "Put the clothes into the washing machine.",
    ),
    "fridge": (
        "/workspace/data/wbt_new/G1_WBT_Inspire_Put_Drinks_Into_Fridge",
        "observation.images.head_stereo_left",
        "Put the drinks into the fridge.",
    ),
}

COSMOS_W, COSMOS_H = 768, 432
COSMOS_FPS = 16
COSMOS_FRAMES = 93


def cut_clip(src, out_mp4):
    vf = (
        f"fps={COSMOS_FPS},"
        f"scale={COSMOS_W}:-2:force_original_aspect_ratio=increase,"
        f"crop={COSMOS_W}:{COSMOS_H}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-frames:v", str(COSMOS_FRAMES),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg FAIL {out_mp4.name}: {r.stderr[:200]}")
        return False
    return True


def process_task(label, out_root, max_eps):
    root, cam, prompt = TASKS[label]
    root = Path(root)
    ep_jsonl = root / "meta" / "episodes.jsonl"
    if not ep_jsonl.exists():
        print(f"[{label}] {ep_jsonl} missing")
        return []
    print(f"=== {label} ({root.name}) ===")
    rows = []
    n = 0
    with open(ep_jsonl) as f:
        for line in f:
            if max_eps and n >= max_eps:
                break
            rec = json.loads(line)
            ep_idx = int(rec["episode_index"])
            chunk = ep_idx // 1000
            src_mp4 = root / "videos" / f"chunk-{chunk:03d}" / cam / f"episode_{ep_idx:06d}.mp4"
            if not src_mp4.exists():
                continue
            # need at least 93 frames at 16fps = 5.8s. Source 30fps, episode length frames.
            length_s = rec.get("length", 0) / 30.0
            if length_s < COSMOS_FRAMES / COSMOS_FPS:
                continue
            clip_id = f"{label}_ep{ep_idx:05d}"
            out_mp4 = out_root / "videos" / f"{clip_id}.mp4"
            if out_mp4.exists():
                rows.append((f"{clip_id}.mp4", prompt))
                n += 1
                continue
            if cut_clip(src_mp4, out_mp4):
                rows.append((f"{clip_id}.mp4", prompt))
            n += 1
            if n % 50 == 0:
                print(f"  ep{ep_idx} total {len(rows)}")
    print(f"  {label}: {len(rows)} clips")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="+", default=["pillow", "clothes"])
    p.add_argument("--max-eps", type=int, default=None)
    a = p.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "metas").mkdir(exist_ok=True)
    (out / "t5_xxl").mkdir(exist_ok=True)
    all_rows = []
    for label in a.tasks:
        all_rows.extend(process_task(label, out, a.max_eps))
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_filename", "prompt"])
        w.writerows(all_rows)
    print(f"[done] {len(all_rows)} clips -> {out/'metadata.csv'}")


if __name__ == "__main__":
    main()
