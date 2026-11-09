"""Split existing cosmos-predict2 g1_wbt clips + metadata into train/val dirs
for cosmos-predict2.5 layout (metas/*.txt + videos/*.mp4).

Source:  /root/data/g1_wbt_4task/{videos,metadata.csv}
Output:
  /root/cosmos-predict2.5/datasets/g1_wbt/train/{videos,metas}  (pillow+clothes+washing)
  /root/cosmos-predict2.5/datasets/g1_wbt/val/{videos,metas}    (fridge)

Caption source: metadata.csv line format "video_filename,prompt"
"""
import csv
import os
import shutil
import sys
from pathlib import Path

VAL_TASK = "fridge"
PROMPT_PREFIX = "The humanoid robot is performing a task. "


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/data/g1_wbt_4task", help="dir with metadata.csv + videos/")
    ap.add_argument("--dst", default="/root/cosmos-predict2.5/datasets/g1_wbt")
    args = ap.parse_args()
    global SRC, DST
    SRC, DST = Path(args.src), Path(args.dst)

    meta = SRC / "metadata.csv"
    with open(meta) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    print(f"total clips: {len(rows)}")

    for split in ("train", "val"):
        (DST / split / "videos").mkdir(parents=True, exist_ok=True)
        (DST / split / "metas").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for video_filename, prompt in rows:
        clip_id = video_filename.replace(".mp4", "")
        task = clip_id.split("_ep")[0]
        split = "val" if task == VAL_TASK else "train"
        src_mp4 = SRC / "videos" / video_filename
        if not src_mp4.exists():
            continue
        dst_mp4 = DST / split / "videos" / video_filename
        if not dst_mp4.exists():
            # hardlink to save disk
            try:
                os.link(src_mp4, dst_mp4)
            except OSError:
                shutil.copy(src_mp4, dst_mp4)
        dst_txt = DST / split / "metas" / f"{clip_id}.txt"
        with open(dst_txt, "w") as fp:
            fp.write(PROMPT_PREFIX + prompt.strip('"'))
        counts[split] += 1

    print(f"train: {counts['train']}")
    print(f"val:   {counts['val']}")


if __name__ == "__main__":
    main()
