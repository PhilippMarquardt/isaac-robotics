"""Train temporal early-fusion 3D UNet for lidar reconstruction.

Improvements over `train_temporal.py`:
  - early-fusion at 64³ + UNet skip connections (model_temporal_unet.py)
  - val split groups by (env_id, episode_id) so trajectories don't leak
  - current_idx derived from tensor shape (no fragile per-batch indexing)
  - reports full-cube + reachable + ±1-tolerant + blind_recall + support metrics

Usage:
    python train_temporal_unet.py --dataset_dir /root/lidar_dataset_fwd \\
        --out_dir /workspace/.../temporal_unet --steps 800 --batch_size 2 \\
        --history_n 10 --history_stride 5 --pos_weight 30
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
from model_temporal_unet import (TemporalUNet, aggregate_history_to_dense,
                                  gt_to_dense, GRID, VOXEL_SIZE, HALF, N_INPUT_CHANNELS)
from augment import augment_lidar


def augment_temporal_lidar(pcl, mask, origin, quat_xyzw, **kwargs):
    """Apply per-frame augmentation across the (B, T) batch-time axis."""
    B, T, R, _ = pcl.shape
    pcl_a, mask_a, origin_a = augment_lidar(
        pcl.reshape(B * T, R, 3),
        mask.reshape(B * T, R),
        origin.reshape(B * T, 3),
        quat_xyzw.reshape(B * T, 4),
        **kwargs,
    )
    return (pcl_a.reshape(B, T, R, 3), mask_a.reshape(B, T, R), origin_a.reshape(B, T, 3))


# ---------------- IoU / metrics ----------------


def voxel_iou(pred_logits, target, threshold=0.5, region=None, tolerant=0):
    pm = pred_logits.sigmoid() > threshold
    tm = target > 0.5
    if tolerant > 0:
        k = tolerant * 2 + 1
        tdil = F.max_pool3d(tm.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        pdil = F.max_pool3d(pm.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        if region is not None:
            pm = pm & region; tm = tm & region
            tdil = tdil & region; pdil = pdil & region
        tp = (pm & tdil).sum().float() + (tm & pdil).sum().float()
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


def _region_stats(pred, tgt, region):
    """Return (TP, FP, FN, P_total) within `region`."""
    pred_r = pred & region
    tgt_r = tgt & region
    tp = (pred_r & tgt_r).sum().float()
    fp = (pred_r & ~tgt_r).sum().float()
    fn = (~pred_r & tgt_r).sum().float()
    p_total = pred_r.sum().float()
    return tp, fp, fn, p_total


def precision_recall_f1(pred_logits, target, region, threshold=0.5):
    pred = pred_logits.sigmoid() > threshold
    tgt = target > 0.5
    tp, fp, fn, p_total = _region_stats(pred, tgt, region)
    region_total = region.sum().float().clamp_min(1.0)
    prec = float(tp / (tp + fp + 1e-9)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn + 1e-9)) if (tp + fn) > 0 else float("nan")
    f1 = (2.0 * prec * recall / (prec + recall + 1e-9)) if (prec == prec and recall == recall and (prec + recall) > 0) else float("nan")
    pred_pos_rate = float(p_total / region_total)
    return prec, recall, f1, pred_pos_rate


def blind_recall(pred_logits, target, reachable, threshold=0.5):
    """Backwards-compat wrapper — recall on GT inside the blind region."""
    blind = ~reachable
    _, recall, _, _ = precision_recall_f1(pred_logits, target, blind, threshold)
    return recall


# ---------------- step / loss ----------------


def dice_bce_loss(logits, target, pos_weight=5.0, dice_weight=0.5):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    p = logits.sigmoid()
    inter = (p * target).sum()
    union = p.sum() + target.sum()
    dice = 1.0 - 2.0 * inter / (union + 1e-6)
    return bce * (1.0 - dice_weight) + dice * dice_weight


def step(model, batch, device, *, return_iou=False, pos_weight=5.0, dice_weight=0.5,
         mask_to_lidar=False, lidar_dilate=3, augment=False,
         return_extra=False, sanity: str = "none"):
    """
    sanity:
      "none"           target = GT (the real task)
      "copy_self"      target = current_occ (model must learn to copy input)
      "current_to_gt"  same as "none" but input zeros out history channels
                       (single-frame baseline of the same UNet)
    """
    pcl = batch["lidar_pcl"].to(device)
    msk = batch["lidar_mask"].to(device)
    base_pos = batch["base_pos"].to(device)
    base_quat = batch["base_quat"].to(device)
    lidar_origin = batch["lidar_origin"].to(device)
    lidar_quat = batch["lidar_quat"].to(device)
    age_s = batch["frame_age_s"].to(device)
    valid_history = batch["valid_history"].to(device)
    gt_pcl = batch["gt_pcl"].to(device)
    gt_n = batch["gt_n"].to(device)

    if augment:
        pcl, msk, lidar_origin = augment_temporal_lidar(pcl, msk, lidar_origin, lidar_quat)

    cur_idx = pcl.shape[1] - 1
    base_pos_t = base_pos[:, cur_idx]
    base_quat_t = base_quat[:, cur_idx]

    x = aggregate_history_to_dense(pcl, msk, base_pos_t, base_quat_t, age_s, valid_history)

    # Sanity modes
    if sanity == "current_to_gt":
        # zero out history channels — only current_occ remains
        x = x.clone()
        x[:, 1:] = 0.0
    if sanity == "copy_self":
        target = x[:, 0:1].clone()
    else:
        target = gt_to_dense(gt_pcl, gt_n, base_pos_t, base_quat_t)

    logits = model(x)

    if mask_to_lidar:
        # observed = current_occ OR any_history_occ (channels 0 + 1), then dilate
        observed = ((x[:, 0:1] > 0.5) | (x[:, 1:2] > 0.5)).float()
        k = lidar_dilate * 2 + 1
        reachable = F.max_pool3d(observed, kernel_size=k, stride=1, padding=lidar_dilate) > 0.5
        if reachable.any():
            loss_in = dice_bce_loss(logits[reachable], target[reachable], pos_weight, dice_weight)
        else:
            loss_in = dice_bce_loss(logits, target, pos_weight, dice_weight)
        # soft "default-negative" pressure outside the observed region — stops the
        # model from emitting positives in the unsupervised below-floor / behind-wall
        # voxels. Weight 0.05 keeps loss dominated by the real supervision inside.
        outside = ~reachable
        if outside.any():
            zero_target = torch.zeros_like(logits[outside])
            loss_out = F.binary_cross_entropy_with_logits(logits[outside], zero_target,
                                                           reduction="mean")
        else:
            loss_out = torch.zeros((), device=logits.device)
        loss = loss_in + 0.05 * loss_out
    else:
        loss = dice_bce_loss(logits, target, pos_weight, dice_weight)

    if return_iou or return_extra:
        with torch.no_grad():
            iou = voxel_iou(logits, target, 0.5)
            if return_extra:
                observed = ((x[:, 0:1] > 0.5) | (x[:, 1:2] > 0.5)).float()
                k = lidar_dilate * 2 + 1
                reach = F.max_pool3d(observed, kernel_size=k, stride=1, padding=lidar_dilate) > 0.5
                blind = ~reach
                iou_full   = voxel_iou(logits, target, 0.5)
                iou_reach  = voxel_iou(logits, target, 0.5, region=reach)
                iou_full_t = voxel_iou(logits, target, 0.5, tolerant=1)
                iou_reach_t = voxel_iou(logits, target, 0.5, region=reach, tolerant=1)
                full_p, full_r, full_f1, full_pr = precision_recall_f1(logits, target,
                    torch.ones_like(reach, dtype=torch.bool), 0.5)
                blind_p, blind_r, blind_f1, blind_pr = precision_recall_f1(logits, target, blind, 0.5)
                reach_p, reach_r, reach_f1, reach_pr = precision_recall_f1(logits, target, reach, 0.5)
                return loss, dict(
                    iou_full=iou_full, iou_reach=iou_reach,
                    iou_full_tol1=iou_full_t, iou_reach_tol1=iou_reach_t,
                    blind_recall=blind_r, blind_precision=blind_p, blind_f1=blind_f1,
                    blind_pred_pos_rate=blind_pr,
                    reach_precision=reach_p, reach_recall=reach_r, reach_f1=reach_f1,
                    full_pred_pos_rate=full_pr,
                )
        return loss, iou
    return loss


def val_metrics(model, val_loader, device, n_batches=8, pos_weight=5.0, dice_weight=0.5,
                sanity: str = "none"):
    model.eval()
    keys = ["iou_full", "iou_reach", "iou_full_tol1", "iou_reach_tol1",
            "blind_recall", "blind_precision", "blind_f1", "blind_pred_pos_rate",
            "reach_precision", "reach_recall", "reach_f1", "full_pred_pos_rate"]
    accum = {k: [] for k in keys}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            _, m = step(model, batch, device, return_extra=True, augment=False,
                        pos_weight=pos_weight, dice_weight=dice_weight, sanity=sanity)
            for k, v in m.items():
                if v == v:
                    accum[k].append(v)
    model.train()
    return {k: float(np.mean(v)) if v else float("nan") for k, v in accum.items()}


# ---------------- trajectory-leak-free split ----------------


def split_entries_by_trajectory(entries, val_frac=0.2, seed=0):
    """Group entries by (env_id, episode_id) and split GROUPS — no in-trajectory leak."""
    rng = np.random.default_rng(seed)
    by_traj: dict[tuple[int, int], list[int]] = {}
    for i, e in enumerate(entries):
        by_traj.setdefault((e.env_id, e.episode_id), []).append(i)
    keys = list(by_traj.keys())
    rng.shuffle(keys)
    n_val = max(1, int(val_frac * len(keys)))
    val_keys = set(keys[:n_val])
    train_idx, val_idx = [], []
    for k in by_traj:
        (val_idx if k in val_keys else train_idx).extend(by_traj[k])
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_fwd")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--history_n", type=int, default=10)
    ap.add_argument("--history_stride", type=int, default=5)
    ap.add_argument("--base_c", type=int, default=32)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--dice_weight", type=float, default=0.5,
                    help="weight of Dice in BCE+Dice combo loss (0 = pure BCE)")
    ap.add_argument("--augment", action="store_true",
                    help="enable lidar input augmentation (off by default)")
    ap.add_argument("--mask_to_lidar", action="store_true")
    ap.add_argument("--lidar_dilate", type=int, default=3)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="gradient-accumulation steps; effective batch = batch_size * grad_accum")
    ap.add_argument("--sanity", choices=["none", "copy_self", "current_to_gt"], default="none",
                    help="sanity training mode: copy_self (target=input), current_to_gt (zero history channels)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader worker processes (each gets its own chunk cache)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    full = TemporalLidarDataset(args.dataset_dir, tiles=None,
                                history_n=args.history_n, history_stride=args.history_stride,
                                max_lidar_pts=20000, max_gt_pts=200000)
    train_idx, val_idx = split_entries_by_trajectory(full.entries, val_frac=0.2, seed=args.seed)
    train_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    train_ds.__dict__.update(full.__dict__)
    train_ds.entries = [full.entries[i] for i in train_idx]
    val_ds = TemporalLidarDataset.__new__(TemporalLidarDataset)
    val_ds.__dict__.update(full.__dict__)
    val_ds.entries = [full.entries[i] for i in val_idx]
    print(f"[INFO] trajectory-split train={len(train_ds)} val={len(val_ds)} "
          f"(train-traj keys disjoint from val-traj keys)", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        collate_fn=temporal_collate)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        collate_fn=temporal_collate)

    device = torch.device(args.device)
    model = TemporalUNet(in_c=N_INPUT_CHANNELS, base_c=args.base_c).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] temporal UNet params: {n_params/1e6:.2f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    log_path = os.path.join(args.out_dir, "train.log")
    val_keys = ["iou_full", "iou_reach", "iou_full_tol1", "iou_reach_tol1",
                "blind_recall", "blind_precision", "blind_f1", "blind_pred_pos_rate",
                "reach_precision", "reach_recall", "reach_f1", "full_pred_pos_rate"]
    with open(log_path, "w") as f:
        cols = ["step", "loss", "iou_train"] + [f"val_{k}" for k in val_keys] + ["lr", "elapsed_s"]
        f.write(",".join(cols) + "\n")
    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2)

    t0 = time.time()
    step_i = 0
    iterator = iter(train_loader)
    while step_i < args.steps:
        opt.zero_grad()
        accum_loss = 0.0
        accum_iou = float("nan")
        do_iou = (step_i + 1) % args.log_every == 0
        for ga in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader); batch = next(iterator)
            out = step(model, batch, device,
                       return_iou=(do_iou and ga == args.grad_accum - 1),
                       pos_weight=args.pos_weight, dice_weight=args.dice_weight,
                       augment=args.augment, mask_to_lidar=args.mask_to_lidar,
                       lidar_dilate=args.lidar_dilate, sanity=args.sanity)
            if do_iou and ga == args.grad_accum - 1:
                loss, accum_iou = out
            else:
                loss = out
            (loss / args.grad_accum).backward()
            accum_loss += float(loss.item())
        accum_loss /= args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step_i += 1

        val = {k: float("nan") for k in val_keys}
        if step_i % args.eval_every == 0:
            val = val_metrics(model, val_loader, device, args.eval_batches,
                              args.pos_weight, args.dice_weight, sanity=args.sanity)
        if step_i % args.log_every == 0:
            elapsed = time.time() - t0
            row = [str(step_i), f"{accum_loss:.4f}", f"{accum_iou:.4f}"]
            for k in val_keys:
                row.append(f"{val[k]:.4f}")
            row.append(f"{sched.get_last_lr()[0]:.6e}")
            row.append(f"{elapsed:.1f}")
            line = ",".join(row)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            print(f"[step {step_i:5d}/{args.steps}] loss={accum_loss:.4f} "
                  f"iou_tr={accum_iou:.3f}  iou_full={val['iou_full']:.3f} "
                  f"iou_reach={val['iou_reach']:.3f} tol1={val['iou_full_tol1']:.3f} "
                  f"blind_F1={val['blind_f1']:.3f} blind_P={val['blind_precision']:.3f} "
                  f"blind_R={val['blind_recall']:.3f} reach_F1={val['reach_f1']:.3f} "
                  f"pred_rate={val['full_pred_pos_rate']:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} el={elapsed:.1f}s", flush=True)
        if step_i % args.save_every == 0:
            ck = os.path.join(args.out_dir, f"ckpt_{step_i:06d}.pt")
            torch.save({"step": step_i, "model": model.state_dict()}, ck)
            print(f"[CKPT] {ck}", flush=True)

    print(f"[DONE] {args.steps} steps in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
