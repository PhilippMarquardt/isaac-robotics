"""Train the temporal GRU lidar reconstruction model.

Usage:
    python train_temporal.py --dataset_dir /root/lidar_dataset_fwd \
        --out_dir /workspace/isaac-robotics/evals/lidar_recon/temporal_gru \
        --steps 1500 --batch_size 2 --history_n 10 --history_stride 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_temporal import TemporalLidarDataset, temporal_collate
from model_temporal import (TemporalGRUModel, voxelize_history_to_dense, gt_to_dense,
                             GRID, VOXEL_SIZE, HALF)


def voxel_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float,
              region: torch.Tensor | None = None, tolerant: int = 0):
    pm = pred_logits.sigmoid() > threshold
    tm = target > 0.5
    if tolerant > 0:
        k = tolerant * 2 + 1
        tgt_dil = F.max_pool3d(tm.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        pm_dil = F.max_pool3d(pm.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        if region is not None:
            pm = pm & region; tm = tm & region
            tgt_dil = tgt_dil & region; pm_dil = pm_dil & region
        tp = (pm & tgt_dil).sum().float() + (tm & pm_dil).sum().float()
        denom = pm.sum().float() + tm.sum().float()
        if denom == 0:
            return float("nan")
        return float(tp / (denom + 1e-9))
    if region is not None:
        pm = pm & region; tm = tm & region
    inter = (pm & tm).sum().float()
    union = (pm | tm).sum().float()
    if union == 0:
        return float("nan")
    return float(inter / union)


def step_loss_iou(model, batch, device, *, return_iou=False, iou_thresh=0.5,
                  mask_to_lidar=False, lidar_dilate=3, pos_weight: float = 3.0):
    pcl = batch["lidar_pcl"].to(device)            # (B, T, N, 3)
    mask = batch["lidar_mask"].to(device)          # (B, T, N)
    base_pos = batch["base_pos"].to(device)        # (B, T, 3)
    base_quat = batch["base_quat"].to(device)      # (B, T, 4)
    age_s = batch["frame_age_s"].to(device)        # (B, T)
    valid_history = batch["valid_history"].to(device)  # (B, T)
    gt_pcl = batch["gt_pcl"].to(device)
    gt_n = batch["gt_n"].to(device)
    cur_idx = int(batch["current_idx"][0].item())
    base_pos_t = base_pos[:, cur_idx]
    base_quat_t = base_quat[:, cur_idx]

    # voxelize all history into dense (B, T, 3, 64, 64, 64)
    x = voxelize_history_to_dense(pcl, mask, base_pos_t, base_quat_t, age_s, valid_history)
    target = gt_to_dense(gt_pcl, gt_n, base_pos_t, base_quat_t)  # (B, 1, 64, 64, 64)

    logits = model(x, valid_mask=valid_history)  # (B, 1, 64, 64, 64)

    # full-cube BCE loss (dense head can emit anywhere)
    pw = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)

    if mask_to_lidar:
        # reachable = current-frame lidar voxels dilated by k
        cur_lidar = x[:, cur_idx, 0:1]  # (B, 1, 64, 64, 64) occ channel of current slot
        k = lidar_dilate * 2 + 1
        reachable = F.max_pool3d(cur_lidar, kernel_size=k, stride=1, padding=lidar_dilate) > 0.5
        valid = reachable
        if valid.any():
            loss = bce(logits[valid], target[valid])
        else:
            loss = bce(logits, target)
    else:
        loss = bce(logits, target)

    if return_iou:
        with torch.no_grad():
            iou = voxel_iou(logits, target, iou_thresh)
        return loss, iou
    return loss


def val_iou(model, val_loader, device, n_batches: int = 8, mask_to_lidar=False,
            lidar_dilate=3, pos_weight: float = 3.0):
    model.eval()
    ious = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            _, iou = step_loss_iou(model, batch, device, return_iou=True,
                                   mask_to_lidar=mask_to_lidar, lidar_dilate=lidar_dilate,
                                   pos_weight=pos_weight)
            if iou == iou:
                ious.append(iou)
    model.train()
    return float(np.mean(ious)) if ious else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_fwd")
    ap.add_argument("--val_dataset_dir", default=None,
                    help="if provided use a separate dir for val; else split tiles 80/20")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--history_n", type=int, default=10)
    ap.add_argument("--history_stride", type=int, default=5)
    ap.add_argument("--base_c", type=int, default=32)
    ap.add_argument("--hidden_c", type=int, default=256)
    ap.add_argument("--gru_hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask_to_lidar", action="store_true")
    ap.add_argument("--lidar_dilate", type=int, default=3)
    ap.add_argument("--pos_weight", type=float, default=30.0,
                    help="BCE positive class weight (raise for sparse positives in 64³ cube)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # tile-based 80/20 split (same machinery as single-frame trainer would do)
    full_ds = TemporalLidarDataset(args.dataset_dir, tiles=None,
                                   history_n=args.history_n, history_stride=args.history_stride,
                                   max_lidar_pts=20000, max_gt_pts=200000)
    n = len(full_ds)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = set(perm[:n_val].tolist())
    train_entries = [full_ds.entries[i] for i in range(n) if i not in val_idx]
    val_entries   = [full_ds.entries[i] for i in range(n) if i in val_idx]
    train_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    train_ds.__dict__.update(full_ds.__dict__)
    train_ds.entries = train_entries
    val_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    val_ds.__dict__.update(full_ds.__dict__)
    val_ds.entries = val_entries
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=temporal_collate)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=temporal_collate)

    device = torch.device(args.device)
    model = TemporalGRUModel(in_c=3, base_c=args.base_c,
                              hidden_c=args.hidden_c, gru_hidden=args.gru_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] temporal GRU model params: {n_params/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    log_path = os.path.join(args.out_dir, "train.log")
    with open(log_path, "w") as f:
        f.write("step,loss,iou_train,iou_val,lr,elapsed_s\n")
    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)

    t0 = time.time()
    step = 0
    iterator = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); batch = next(iterator)
        opt.zero_grad()
        do_iou = (step + 1) % args.log_every == 0
        out = step_loss_iou(model, batch, device, return_iou=do_iou,
                            mask_to_lidar=args.mask_to_lidar, lidar_dilate=args.lidar_dilate,
                            pos_weight=args.pos_weight)
        if do_iou:
            loss, iou_train = out
        else:
            loss, iou_train = out, float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        iou_v = float("nan")
        if step % args.eval_every == 0:
            iou_v = val_iou(model, val_loader, device, args.eval_batches,
                            mask_to_lidar=args.mask_to_lidar, lidar_dilate=args.lidar_dilate,
                            pos_weight=args.pos_weight)
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            line = f"{step},{loss.item():.4f},{iou_train:.4f},{iou_v:.4f},{sched.get_last_lr()[0]:.6e},{elapsed:.1f}"
            print(f"[step {step:5d}/{args.steps}] loss={loss.item():.4f} iou_tr={iou_train:.3f} iou_val={iou_v:.3f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if step % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"ckpt_{step:06d}.pt")
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item()}, ck)
            print(f"[CKPT] {ck}", flush=True)

    print(f"[DONE] {args.steps} steps in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
