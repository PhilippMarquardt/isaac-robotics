#!/usr/bin/env python3
"""Make 80/20 per-dataset train/val splits by writing filtered meta dirs + symlinked data/videos.

Reads each source lerobot v2.1 dataset, picks 80% episodes (deterministic seed) into train,
20% into val. Output layout:
  <out_root>/train/<name>/  -> meta filtered, data+videos symlinked to src
  <out_root>/val/<name>/    -> same
Episode indices preserved (not re-indexed) so per-episode parquet/video paths still resolve.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
import random


def load_episodes(src: Path) -> list[dict]:
    ep_file = src / "meta" / "episodes.jsonl"
    return [json.loads(l) for l in ep_file.read_text().splitlines() if l.strip()]


def load_stats(src: Path) -> dict[int, str]:
    """Return dict[episode_index -> raw jsonl line]."""
    f = src / "meta" / "episodes_stats.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["episode_index"]] = line
    return out


def write_split(src: Path, dst: Path, episodes: list[dict], stats: dict[int, str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # symlink data + videos (read-only, saves disk)
    for sub in ("data", "videos"):
        link = dst / sub
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        os.symlink(src.resolve() / sub, link)

    meta = dst / "meta"
    meta.mkdir(exist_ok=True)

    # copy tasks.jsonl + modality.json (if exists) as-is
    for name in ("tasks.jsonl", "modality.json", "stats.json"):
        p = src / "meta" / name
        if p.exists():
            shutil.copy2(p, meta / name)

    # write filtered episodes.jsonl
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n")

    # write filtered episodes_stats.jsonl
    if stats:
        lines = [stats[e["episode_index"]] for e in episodes if e["episode_index"] in stats]
        (meta / "episodes_stats.jsonl").write_text("\n".join(lines) + "\n")

    # patch info.json: update total_episodes + total_frames
    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(e["length"] for e in episodes)
    (meta / "info.json").write_text(json.dumps(info, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True, help="Source dataset paths (v2.1 roots)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    out = Path(a.out_root)

    summary = []
    for src_str in a.sources:
        src = Path(src_str)
        name = src.name
        eps = load_episodes(src)
        stats = load_stats(src)
        idxs = list(range(len(eps)))
        rng.shuffle(idxs)
        n_val = max(1, int(len(eps) * a.val_ratio))
        val_set = set(idxs[:n_val])
        train_eps = [eps[i] for i in range(len(eps)) if i not in val_set]
        val_eps = [eps[i] for i in range(len(eps)) if i in val_set]

        write_split(src, out / "train" / name, train_eps, stats)
        write_split(src, out / "val" / name, val_eps, stats)

        tot_tr = sum(e["length"] for e in train_eps)
        tot_va = sum(e["length"] for e in val_eps)
        summary.append(f"{name}: train {len(train_eps)} eps / {tot_tr} frames, val {len(val_eps)} / {tot_va}")

    print("\n".join(summary))
    (out / "split_summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
