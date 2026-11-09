"""Ablation eval for v6 trained models — runs IoU + chamfer + FPS + VRAM on val set.

Usage:
    python eval_v6.py --runs runs/hoeller_v6 runs/pointnet_v6 runs/ptv3_v6 \
        --dataset_dir /root/lidar_dataset_v6 --out RESULTS_v6.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import spconv.pytorch as spconv

from dataset import LidarFrameDataset, load_split
from train_v6 import (FullModel, make_encoder, train_step, sparse_to_dense_logits,
                       voxelize_pcl_local, world_to_local_batch, build_input_tensor,
                       quat_xyzw_to_R, GRID, VOXEL_SIZE, HALF, collate)


def voxel_iou(pred_dense: torch.Tensor, target_dense: torch.Tensor, threshold: float,
              region: torch.Tensor | None = None, tolerant: int = 0):
    """IoU on voxel grid. `tolerant`>0 dilates target by k voxels (loose match).
    pred_dense already carries fill=-20 for absent voxels (sigmoid≈0).
    """
    pred_mask = pred_dense.sigmoid() > threshold
    tgt_mask = target_dense > 0.5
    if tolerant > 0:
        k = tolerant * 2 + 1
        tgt_dil = F.max_pool3d(tgt_mask.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        # tolerant TP = pred ∩ dilated_tgt; tolerant union approximated by either side dilation
        pm_dil = F.max_pool3d(pred_mask.float(), kernel_size=k, stride=1, padding=tolerant) > 0.5
        if region is not None:
            pred_mask = pred_mask & region
            tgt_mask = tgt_mask & region
            tgt_dil = tgt_dil & region
            pm_dil = pm_dil & region
        # symmetric tolerant IoU: TP_pred = pred ∩ tgt_dil, TP_tgt = tgt ∩ pm_dil
        tp = (pred_mask & tgt_dil).sum().float() + (tgt_mask & pm_dil).sum().float()
        denom = pred_mask.sum().float() + tgt_mask.sum().float()
        if denom == 0:
            return float("nan")
        return float(tp / (denom + 1e-9))
    if region is not None:
        pred_mask = pred_mask & region
        tgt_mask = tgt_mask & region
    inter = (pred_mask & tgt_mask).sum().float()
    union = (pred_mask | tgt_mask).sum().float()
    if union == 0:
        return float("nan")
    return float(inter / union)


def blind_recall(pred_dense, target_dense, reachable, threshold: float):
    """Recall on GT voxels OUTSIDE the lidar-reachable mask (the blind cone).
    Measures whether model can hallucinate hidden ground.
    """
    blind = ~reachable
    tgt = (target_dense > 0.5) & blind
    if tgt.sum() == 0:
        return float("nan")
    pred = (pred_dense.sigmoid() > threshold) & blind
    tp = (pred & tgt).sum().float()
    return float(tp / tgt.sum().float())


def chamfer_l2(pred_pts: torch.Tensor, gt_pts: torch.Tensor, max_pts=4096):
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("nan")
    if pred_pts.shape[0] > max_pts:
        pred_pts = pred_pts[torch.randperm(pred_pts.shape[0])[:max_pts]]
    if gt_pts.shape[0] > max_pts:
        gt_pts = gt_pts[torch.randperm(gt_pts.shape[0])[:max_pts]]
    d = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
    return float(d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def run_one(run_dir: str, dataset_dir: str, n_eval: int, device: torch.device,
            mask_to_lidar: bool = False, lidar_dilate: int = 3) -> dict:
    hparams = json.load(open(os.path.join(run_dir, "hparams.json")))
    enc_name = hparams["encoder"]
    dense_head = bool(hparams.get("dense_head", False))
    encoder = make_encoder(enc_name)
    model = FullModel(encoder, dense_head=dense_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # find latest ckpt
    ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("ckpt_")])
    if not ckpts:
        return {"name": enc_name, "error": "no_ckpts"}
    ck = torch.load(os.path.join(run_dir, ckpts[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    _, val_tiles = load_split(os.path.join(dataset_dir, "split.json"))
    val_ds = LidarFrameDataset(dataset_dir, val_tiles)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)

    ious_full = {0.3: [], 0.5: [], 0.7: []}
    ious_reach = {0.3: [], 0.5: [], 0.7: []}
    ious_full_tol1 = {0.3: [], 0.5: [], 0.7: []}
    ious_reach_tol1 = {0.3: [], 0.5: [], 0.7: []}
    blind_recalls = {0.3: [], 0.5: [], 0.7: []}
    sup_full = []   # frac of GT covered by sparse output coords
    sup_reach = []  # frac of reachable GT covered by sparse output coords
    sup_blind = []  # frac of blind GT covered by sparse output coords
    chamfers = []
    fps_times = []
    n_done = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in val_loader:
            if n_done >= n_eval:
                break
            bs = batch["lidar_pcl"].shape[0]
            pcl = batch["lidar_pcl"].to(device)
            mask = batch["lidar_mask"].to(device)
            origin = batch["lidar_origin"].to(device)
            quat = batch["lidar_quat"].to(device)
            base = batch["base_pos"].to(device)
            base_quat = batch["base_quat"].to(device)
            gt_pcl = batch["gt_pcl"].to(device)
            gt_n = batch["gt_n"].to(device)

            pcl_local = world_to_local_batch(pcl, base, base_quat)
            coords_in_list, offsets_in_list = [], []
            lidar_dense = torch.zeros((bs, 1, GRID, GRID, GRID), device=device)
            for b in range(bs):
                c, o = voxelize_pcl_local(pcl_local[b], mask[b])
                coords_in_list.append(c)
                offsets_in_list.append(o)
                if c.shape[0] > 0:
                    lidar_dense[b, 0, c[:, 0], c[:, 1], c[:, 2]] = 1.0
            sp_in = build_input_tensor(coords_in_list, offsets_in_list, bs, device)
            # always compute reachable for blind-recall metric
            k = lidar_dilate * 2 + 1
            reachable_all = F.max_pool3d(lidar_dense, kernel_size=k, stride=1, padding=lidar_dilate) > 0.5
            t0 = time.time()
            outs, dense_logits = model(sp_in)
            torch.cuda.synchronize()
            dt = time.time() - t0
            fps_times.append(dt / bs)

            if dense_logits is not None:
                finest = dense_logits
                # dense head: support is the whole cube
                support = torch.ones((bs, 1, GRID, GRID, GRID), dtype=torch.bool, device=device)
            else:
                finest = sparse_to_dense_logits(outs[-1], fill=-20.0)
                # sparse decoder support — which voxels did the sparse decoder emit at all?
                support = torch.zeros((bs, 1, GRID, GRID, GRID), dtype=torch.bool, device=device)
                ind = outs[-1].indices.long()
                if ind.shape[0] > 0:
                    support[ind[:, 0], 0, ind[:, 1], ind[:, 2], ind[:, 3]] = True
            for b in range(bs):
                # build target dense
                target = torch.zeros(1, 1, GRID, GRID, GRID, device=device)
                n = int(gt_n[b].item())
                if n > 0:
                    gt_b = gt_pcl[b, :n]
                    R = quat_xyzw_to_R(base_quat[b])
                    gt_local = (R.T @ (gt_b - base[b]).T).T
                    in_box = ((gt_local > -HALF) & (gt_local < HALF)).all(-1)
                    gt_local = gt_local[in_box]
                    if gt_local.shape[0] > 0:
                        gi = ((gt_local + HALF) / VOXEL_SIZE).long().clamp_(0, GRID - 1)
                        target[0, 0, gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0
                pred = finest[b:b+1]
                reach_b = reachable_all[b:b+1]
                sup_b = support[b:b+1]
                gt_b = target > 0.5
                gt_total = gt_b.sum().float().clamp_min(1)
                gt_reach = (gt_b & reach_b).sum().float().clamp_min(1)
                gt_blind = (gt_b & ~reach_b).sum().float().clamp_min(1)
                sup_full.append(float((sup_b & gt_b).sum().float() / gt_total))
                sup_reach.append(float((sup_b & gt_b & reach_b).sum().float() / gt_reach))
                sup_blind.append(float((sup_b & gt_b & ~reach_b).sum().float() / gt_blind))
                for th in (0.3, 0.5, 0.7):
                    a = voxel_iou(pred, target, th)
                    b_ = voxel_iou(pred, target, th, region=reach_b)
                    c = voxel_iou(pred, target, th, tolerant=1)
                    d = voxel_iou(pred, target, th, region=reach_b, tolerant=1)
                    br = blind_recall(pred, target, reach_b, th)
                    if a == a:  ious_full[th].append(a)
                    if b_ == b_: ious_reach[th].append(b_)
                    if c == c:  ious_full_tol1[th].append(c)
                    if d == d:  ious_reach_tol1[th].append(d)
                    if br == br: blind_recalls[th].append(br)
                # chamfer in local coords
                pred_mask_b = (pred.sigmoid() > 0.5).squeeze()
                pred_idx = torch.nonzero(pred_mask_b, as_tuple=False)
                if pred_idx.shape[0] > 0:
                    pred_pts = (pred_idx.float() + 0.5) * VOXEL_SIZE - HALF
                    gt_idx = torch.nonzero(target.squeeze(), as_tuple=False)
                    gt_pts_local = (gt_idx.float() + 0.5) * VOXEL_SIZE - HALF
                    ch = chamfer_l2(pred_pts, gt_pts_local)
                    if ch == ch:
                        chamfers.append(ch)
                n_done += 1

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    fps = 1.0 / np.mean(fps_times) if fps_times else 0.0
    def _m(d, th): return float(np.mean(d[th])) if d[th] else float("nan")
    def _ml(lst): return float(np.mean(lst)) if lst else float("nan")
    return {
        "name": enc_name,
        "n_params_M": n_params / 1e6,
        "iou_full_t0.5":       _m(ious_full, 0.5),
        "iou_reach_t0.5":      _m(ious_reach, 0.5),
        "iou_full_tol1_t0.5":  _m(ious_full_tol1, 0.5),
        "iou_reach_tol1_t0.5": _m(ious_reach_tol1, 0.5),
        "blind_recall_t0.5":   _m(blind_recalls, 0.5),
        "support_recall_full":  _ml(sup_full),
        "support_recall_reach": _ml(sup_reach),
        "support_recall_blind": _ml(sup_blind),
        "iou_full_t0.3":       _m(ious_full, 0.3),
        "iou_full_t0.7":       _m(ious_full, 0.7),
        "chamfer_l2": float(np.mean(chamfers)) if chamfers else float("nan"),
        "fps": fps,
        "peak_vram_GB": peak_vram,
        "n_eval_frames": n_done,
        "ckpt": ckpts[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset_dir", default="/root/lidar_dataset_v6")
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--out", default="RESULTS_v6.md")
    ap.add_argument("--mask_to_lidar", action="store_true",
                    help="restrict IoU to lidar-reachable voxels (matches mask_to_lidar training)")
    ap.add_argument("--lidar_dilate", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    rows = []
    for r in args.runs:
        print(f"[INFO] eval {r}", flush=True)
        rows.append(run_one(r, args.dataset_dir, args.n_eval, device,
                            mask_to_lidar=args.mask_to_lidar,
                            lidar_dilate=args.lidar_dilate))

    # markdown table
    keys = ["name", "n_params_M",
            "iou_full_t0.5", "iou_reach_t0.5", "iou_full_tol1_t0.5", "iou_reach_tol1_t0.5",
            "blind_recall_t0.5",
            "support_recall_full", "support_recall_reach", "support_recall_blind",
            "iou_full_t0.3", "iou_full_t0.7",
            "chamfer_l2", "fps", "peak_vram_GB", "n_eval_frames"]
    with open(args.out, "w") as f:
        f.write("# Ablation results — v6\n\n")
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join("---" for _ in keys) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(
                f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r.get(k, "—"))
                for k in keys
            ) + " |\n")
    print(f"[SAVED] {args.out}", flush=True)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
