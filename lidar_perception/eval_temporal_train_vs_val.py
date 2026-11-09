"""Eval a temporal UNet checkpoint on TRAIN and VAL splits using identical code.

Reports the same metrics as `train_temporal_unet.val_metrics` but on each split.
Lets you read the train/val gap directly from the same numbers used in training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_temporal import TemporalLidarDataset, temporal_collate
from model_temporal_unet import TemporalUNet, N_INPUT_CHANNELS
from train_temporal_unet import split_entries_by_trajectory, val_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_fwd_long")
    ap.add_argument("--n_batches", type=int, default=32)
    args = ap.parse_args()

    h = json.load(open(os.path.join(args.run_dir, "hparams.json")))
    full = TemporalLidarDataset(args.dataset_dir, tiles=None,
                                 history_n=h["history_n"], history_stride=h["history_stride"],
                                 max_lidar_pts=20000, max_gt_pts=200000)
    train_idx, val_idx = split_entries_by_trajectory(full.entries, val_frac=0.2, seed=h["seed"])
    train_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    train_ds.__dict__.update(full.__dict__)
    train_ds.entries = [full.entries[i] for i in train_idx]
    val_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    val_ds.__dict__.update(full.__dict__)
    val_ds.entries = [full.entries[i] for i in val_idx]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=h["batch_size"],
        shuffle=True, num_workers=2, persistent_workers=True, collate_fn=temporal_collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=h["batch_size"],
        shuffle=True, num_workers=2, persistent_workers=True, collate_fn=temporal_collate)

    device = torch.device("cuda")
    model = TemporalUNet(in_c=N_INPUT_CHANNELS, base_c=h["base_c"]).to(device)
    ckpts = sorted([f for f in os.listdir(args.run_dir) if f.startswith("ckpt_")])
    if not ckpts:
        raise RuntimeError(f"no ckpts in {args.run_dir}")
    ck_path = os.path.join(args.run_dir, ckpts[-1])
    state = torch.load(ck_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[INFO] {args.run_dir} ckpt={ckpts[-1]}", flush=True)

    print(f"[INFO] computing TRAIN metrics over {args.n_batches} batches", flush=True)
    train_m = val_metrics(model, train_loader, device, args.n_batches,
                           h["pos_weight"], h["dice_weight"])
    print(f"[INFO] computing VAL metrics over {args.n_batches} batches", flush=True)
    val_m = val_metrics(model, val_loader, device, args.n_batches,
                         h["pos_weight"], h["dice_weight"])

    keys = ["iou_full", "iou_reach", "iou_full_tol1", "iou_reach_tol1",
            "blind_recall", "blind_precision", "blind_f1",
            "reach_precision", "reach_recall", "reach_f1", "full_pred_pos_rate"]
    print()
    print(f"{'metric':<22} {'train':>8} {'val':>8} {'gap':>8}")
    for k in keys:
        t, v = train_m.get(k, float("nan")), val_m.get(k, float("nan"))
        gap = t - v if (t == t and v == v) else float("nan")
        print(f"  {k:<20} {t:>8.4f} {v:>8.4f} {gap:>+8.4f}")


if __name__ == "__main__":
    main()
