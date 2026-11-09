"""Combine multiple chunked .pt datasets into a single output dir.

Skips chunks that fail to load. Renumbers output chunks contiguously.

Usage:
    python combine_datasets.py \\
        --inputs /workspace/lidar_dataset_v1 /workspace/lidar_dataset_v2 \\
        --output /workspace/lidar_dataset_combined
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--copy", action="store_true",
                    help="Copy chunks (default: hardlink). Use copy if crossing filesystems.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_idx = 0
    skipped = 0
    total_records = 0

    for src_dir in args.inputs:
        chunks = sorted(glob.glob(os.path.join(src_dir, "chunk_*.pt")))
        for c in chunks:
            try:
                d = torch.load(c, map_location="cpu", weights_only=True)
                F = int(d["episode_id"].shape[0])
            except Exception as e:
                print(f"[SKIP] {c}: {e.__class__.__name__}", flush=True)
                skipped += 1
                continue

            dst = os.path.join(args.output, f"chunk_{out_idx:05d}.pt")
            if args.copy:
                shutil.copy2(c, dst)
            else:
                try:
                    os.link(c, dst)
                except OSError:
                    shutil.copy2(c, dst)
            print(f"[ADD] {dst} <- {c}  F={F}", flush=True)
            out_idx += 1
            total_records += F

    print(f"[DONE] {out_idx} chunks, {total_records} records, {skipped} skipped → {args.output}", flush=True)


if __name__ == "__main__":
    main()
