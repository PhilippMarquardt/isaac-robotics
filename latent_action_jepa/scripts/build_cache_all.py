"""Build a unified vjepa2_clips cache from all mp4 files in a directory."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.vjepa2_clips import build_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films_dir", default="/root/jepa_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame_lo", type=int, default=200)
    ap.add_argument("--frame_hi", type=int, default=3600)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.films_dir, "*.mp4")))
    names = [os.path.splitext(os.path.basename(p))[0].lower() for p in paths]
    print(f"found {len(paths)} films")
    for p, n in zip(paths, names):
        print(f"  {n}: {p}")
    build_cache(paths, names, args.out, frame_lo=args.frame_lo, frame_hi=args.frame_hi, size=args.size)


if __name__ == "__main__":
    main()
