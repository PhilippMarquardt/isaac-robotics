"""Reproduce the 80/20 split_datasets.py shuffle with seed=42 for the 4 trained tasks.

Gets episode count per task from HF meta, then applies the same rng.shuffle logic
split_datasets.py used to discover which episode_indices landed in val.
"""
from __future__ import annotations

import random
import pandas as pd
from huggingface_hub import hf_hub_download

# Order must match conf.yaml dataset_paths order (shared rng is stateful!)
TASKS = [
    ("pillow",  "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
     "G1_WB_Dex5_Pickup_Pillow"),
    ("clothes", "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
     "G1_WB_Dex5_Collect_Clothes"),
    ("washing", "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
     "G1_WB_Dex5_Put_Clothes_into_Washing_Machine"),
    ("fridge",  "G1_WBT_Inspire_Put_Drinks_Into_Fridge", ""),
]


def get_episode_count(repo, subdir):
    prefix = f"{subdir}/" if subdir else ""
    # HF v3 stores episodes in possibly multiple chunk files — need total
    # Easiest: info.json has total_episodes
    try:
        info_path = hf_hub_download(f"unitreerobotics/{repo}", f"{prefix}meta/info.json",
                                    repo_type="dataset")
        import json
        info = json.load(open(info_path))
        return int(info["total_episodes"])
    except Exception as e:
        print(f"  info.json fail: {e}")
        # fallback: aggregate all meta/episodes/chunk-*/file-*.parquet
        raise


def main():
    rng = random.Random(42)
    results = {}
    for label, repo, subdir in TASKS:
        print(f"\n=== {label} ({repo}) ===")
        n_eps = get_episode_count(repo, subdir)
        print(f"  total episodes: {n_eps}")

        idxs = list(range(n_eps))
        rng.shuffle(idxs)  # <-- MUST match split_datasets.py exactly
        n_val = max(1, int(n_eps * 0.2))
        val_set = sorted(idxs[:n_val])
        train_set = sorted(idxs[n_val:])

        # Pick first 3 val episodes that are in file-000 (low indices more likely there)
        val_first_3 = val_set[:3]
        print(f"  val set size: {len(val_set)}")
        print(f"  first 3 val episodes: {val_first_3}")
        print(f"  are [0,1,2] in val?  0:{0 in val_set}  1:{1 in val_set}  2:{2 in val_set}")

        results[label] = {
            "n_total": n_eps,
            "n_val": len(val_set),
            "first_3_val_eps": val_first_3,
            "ep0_in_val": 0 in val_set,
            "ep1_in_val": 1 in val_set,
            "ep2_in_val": 2 in val_set,
        }

    print("\n=== SUMMARY ===")
    for label, r in results.items():
        print(f"{label:<10}  val eps [0,1,2] leaked into train? "
              f"{'LEAK' if any([not r['ep0_in_val'], not r['ep1_in_val'], not r['ep2_in_val']]) else 'clean'}"
              f"  — use val ids: {r['first_3_val_eps']}")


if __name__ == "__main__":
    main()
