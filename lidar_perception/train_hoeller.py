"""Train Hoeller-2022 sparse 4D U-Net on chunked offline lidar trajectories.

Pipeline:

  1. Index every chunk file under `--dataset_dir`. Group frames into trajectories
     keyed by (chunk, env, episode).
  2. Sample a contiguous `rollout`-step window from each trajectory each iteration.
  3. For each step in the rollout:
       a. Voxelize the lidar pcl into a 64³ sparse tensor at 5 cm resolution
          (lidar-frame coords).
       b. If t > 0, transform the previous decoded occupancy back into the new
          lidar frame (using the relative pose between t-1 and t), detach, and
          concatenate as additional voxel features with `source_flag=1`.
       c. Voxelize the GT pcl as the supervision target.
  4. Forward the spconv U-Net, compute multi-resolution BCE-occupancy + L2 offset.
  5. AdamW + cosine LR.

Sanity-overfit (`--overfit N`): keep only first N trajectories. Use to confirm the
network drives loss → 0 before scaling.

The actual rollout + densify step is currently a TODO — voxelize, dataset index,
transforms, model factory, loss, and optimizer wiring are all in place. To complete
training, fill in the `_one_step` function: take a batch of (cur, prev, gt) per
trajectory step, build the SparseConvTensor with stack_batch, run model, densify
predictions to match the target voxel grid, call reconstruction_loss.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import torch

import hoeller_unet
from hoeller_unet import (
    BASE_C, GRID, HALF, IN_CHANNELS, NUM_LEVELS, VOXEL_SIZE,
    build_model, reconstruction_loss, stack_batch, voxelize_pcl_local,
)


# -------- transforms (world ↔ lidar frame) ---------------------------------------------

def quat_xyzw_to_mat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def world_to_local(points_w: torch.Tensor, origin_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    R = quat_xyzw_to_mat(quat_w)
    delta = points_w - origin_w.unsqueeze(-2)
    return torch.einsum("...ij,...nj->...ni", R.transpose(-2, -1), delta)


def local_to_world(points_l: torch.Tensor, origin_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    R = quat_xyzw_to_mat(quat_w)
    return torch.einsum("...ij,...nj->...ni", R, points_l) + origin_w.unsqueeze(-2)


# -------- dataset index ---------------------------------------------------------------


class LidarChunkIndex:
    """Lazy index over chunked .pt files. Groups frames into trajectories."""

    def __init__(self, dataset_dir: str):
        self.chunks = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
        if not self.chunks:
            raise FileNotFoundError(f"No .pt chunks in {dataset_dir}")
        self._meta = []
        for path in self.chunks:
            d = torch.load(path, map_location="cpu", weights_only=True)
            self._meta.append({
                "path": path,
                "F": int(d["episode_id"].shape[0]),
                "env_id": d["env_id"],
                "episode_id": d["episode_id"],
                "step_in_episode": d["step_in_episode"],
            })
        self._build_trajectories()

    def _build_trajectories(self):
        self.trajectories = []
        for ci, m in enumerate(self._meta):
            for env in m["env_id"].unique().tolist():
                env_mask = m["env_id"] == env
                eps_in_env = m["episode_id"][env_mask].unique().tolist()
                for ep in eps_in_env:
                    sel = env_mask & (m["episode_id"] == ep)
                    idxs = torch.nonzero(sel, as_tuple=True)[0].tolist()
                    if len(idxs) >= 2:
                        self.trajectories.append((ci, idxs))

    def __len__(self):
        return len(self.trajectories)

    def sample(self, traj_idx: int, rollout: int, downsample: int = 5) -> dict:
        """Sample a `rollout`-step window from trajectory `traj_idx`.

        `downsample` keeps every Nth step within the trajectory — useful because the
        dataset is recorded at 50 Hz but the real MID-360 / Hoeller setup runs at 10 Hz.
        """
        ci, idxs = self.trajectories[traj_idx]
        step = downsample
        max_start = max(0, len(idxs) - rollout * step)
        if max_start == 0:
            sel_idxs = idxs[:rollout * step:step]
            while len(sel_idxs) < rollout:
                sel_idxs.append(sel_idxs[-1])
        else:
            start = random.randint(0, max_start)
            sel_idxs = idxs[start:start + rollout * step:step]
            while len(sel_idxs) < rollout:
                sel_idxs.append(sel_idxs[-1])
        sel = torch.tensor(sel_idxs[:rollout])
        d = torch.load(self._meta[ci]["path"], map_location="cpu", weights_only=True)
        return {k: v[sel] for k, v in d.items() if isinstance(v, torch.Tensor)}


# -------- training loop --------------------------------------------------------------


def _voxelize_step(d: dict, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxelize the lidar pcl at step t in lidar frame."""
    pts_w = d["lidar_points_world"][t].to(device)  # (R, 3)
    mask = d["lidar_hit_mask"][t].to(device)  # (R,)
    origin_w = d["lidar_origin_world"][t].to(device)  # (3,)
    quat_w = d["lidar_quat_world"][t].to(device)  # (4,) xyzw
    pts_l = world_to_local(pts_w.unsqueeze(0), origin_w.unsqueeze(0), quat_w.unsqueeze(0)).squeeze(0)
    return voxelize_pcl_local(pts_l, mask)


def _voxelize_gt_step(d: dict, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxelize the GT pcl at step t in the lidar frame (so target aligns with input grid)."""
    pts_w = d["gt_points_world"][t].to(device)
    mask = d["gt_hit_mask"][t].to(device)
    origin_w = d["lidar_origin_world"][t].to(device)
    quat_w = d["lidar_quat_world"][t].to(device)
    pts_l = world_to_local(pts_w.unsqueeze(0), origin_w.unsqueeze(0), quat_w.unsqueeze(0)).squeeze(0)
    return voxelize_pcl_local(pts_l, mask)


def _rollout_step(
    model,
    batch_d_list: list[dict],
    rollout: int,
    device: torch.device,
    pos_weight: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Run a 12-step rollout for a batch of trajectories. Compute loss summed across
    timesteps and decoder levels. Prev-frame feedback is detached (no BPTT).

    `pos_weight` upweights the positive-class BCE term — without it the model
    converges to "predict empty everywhere" because most masked voxels are
    false-positives (lidar voxel ∉ GT). pos_weight=10 gives the rare GT-occupied
    voxels a fighting chance.
    """
    pw = torch.tensor([pos_weight], device=device)
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pw)

    total_loss = torch.zeros((), device=device)
    total_occ = torch.zeros((), device=device)
    n_levels = NUM_LEVELS
    n_voxels_per_grid = (GRID, GRID, GRID)

    # state per batch element: previous-step decoded occupancy at finest resolution
    # in lidar-frame voxel coordinates. Shape: list of (M_b, 4) sparse tensors —
    # we keep coords + offsets, source_flag=1, until we transform into the next frame.
    prev_dense: list[torch.Tensor | None] = [None] * len(batch_d_list)

    for t in range(rollout):
        coords_b, offsets_b, flags_b = [], [], []
        target_dense_levels = [
            torch.zeros((len(batch_d_list), 1, GRID >> lvl, GRID >> lvl, GRID >> lvl), device=device)
            for lvl in range(n_levels)
        ]

        for b, d in enumerate(batch_d_list):
            # current observation (lidar pcl voxelized into lidar frame)
            cc, oo = _voxelize_step(d, t, device)

            # prev observation: previously decoded dense occupancy at finest resolution
            # transformed into the current lidar frame. We approximate "dense" by
            # storing/retrieving as point cloud coordinates.
            if prev_dense[b] is not None:
                prev_pts_prevframe, prev_origin, prev_quat = prev_dense[b]
                # bring previous prediction's points back to world, then to current lidar frame
                cur_origin = d["lidar_origin_world"][t].to(device)
                cur_quat = d["lidar_quat_world"][t].to(device)
                pts_w = local_to_world(
                    prev_pts_prevframe.unsqueeze(0), prev_origin.unsqueeze(0), prev_quat.unsqueeze(0)
                ).squeeze(0)
                pts_l_cur = world_to_local(
                    pts_w.unsqueeze(0), cur_origin.unsqueeze(0), cur_quat.unsqueeze(0)
                ).squeeze(0)
                cc_prev, oo_prev = voxelize_pcl_local(
                    pts_l_cur, torch.ones(pts_l_cur.shape[0], dtype=torch.bool, device=device)
                )
                # merge cur + prev
                cc_all = torch.cat([cc, cc_prev], dim=0)
                oo_all = torch.cat([oo, oo_prev], dim=0)
                flags = torch.cat([
                    torch.zeros(cc.shape[0], 1, device=device),
                    torch.ones(cc_prev.shape[0], 1, device=device),
                ], dim=0)
            else:
                cc_all = cc
                oo_all = oo
                flags = torch.zeros(cc.shape[0], 1, device=device)

            coords_b.append(cc_all)
            offsets_b.append(oo_all)
            flags_b.append(flags)

            # GT target (voxelize GT pcl into the same lidar-frame grid)
            gc, _ = _voxelize_gt_step(d, t, device)
            for lvl in range(n_levels):
                gc_lvl = gc >> lvl
                in_box = ((gc_lvl >= 0) & (gc_lvl < (GRID >> lvl))).all(-1)
                gc_lvl = gc_lvl[in_box]
                if gc_lvl.shape[0] > 0:
                    target_dense_levels[lvl][b, 0, gc_lvl[:, 0].long(), gc_lvl[:, 1].long(), gc_lvl[:, 2].long()] = 1.0

        # build sparse input tensor for the batch
        batch_feats = []
        batch_coords = []
        for b, (cc, oo, fl) in enumerate(zip(coords_b, offsets_b, flags_b)):
            f = torch.cat([oo, fl], dim=-1)
            batch_feats.append(f)
            batch_coords.append(torch.cat([torch.full((cc.shape[0], 1), b, device=device, dtype=cc.dtype), cc], dim=-1))
        feats = torch.cat(batch_feats, dim=0).contiguous()
        coords = torch.cat(batch_coords, dim=0).contiguous()
        sp_in = hoeller_unet.spconv.SparseConvTensor(
            feats, coords, spatial_shape=list(n_voxels_per_grid), batch_size=len(batch_d_list)
        )

        # forward
        outs = model(sp_in)  # list of (occ_logits, offset) sparse tensors per level (coarse → fine)

        # Mask-aware BCE per level. Computing BCE on every voxel of the dense grid
        # makes the loss plateau near 0.693 because most voxels are empty in both
        # pred and GT (target=0, pred=sigmoid(0)=0.5 → BCE=0.693). Restricting to
        # union(pred, GT) lets gradients actually do something:
        #   - GT-occupied voxels: drive logits positive
        #   - pred-occupied voxels not in GT: drive logits negative (false positive)
        #   - voxels neither in pred nor GT: ignored (correct by default)
        loss_t = torch.zeros((), device=device)
        for i, (occ_sp, _off_sp) in enumerate(outs):
            tgt_lvl = (NUM_LEVELS - 1) - i
            occ_dense = occ_sp.dense()  # (B, 1, D, H, W)
            tgt = target_dense_levels[tgt_lvl]

            # build dense pred-occupancy mask from sparse indices
            B = occ_sp.batch_size
            spatial = occ_sp.spatial_shape  # list[int]
            pred_mask = torch.zeros((B, 1, *spatial), dtype=torch.bool, device=occ_dense.device)
            ind = occ_sp.indices.long()  # (M, 4) batch+xyz
            if ind.shape[0] > 0:
                pred_mask[ind[:, 0], 0, ind[:, 1], ind[:, 2], ind[:, 3]] = True
            valid_mask = pred_mask | (tgt > 0.5)
            if valid_mask.any():
                loss_t = loss_t + bce(occ_dense[valid_mask], tgt[valid_mask])

        total_loss = total_loss + loss_t
        total_occ = total_occ + loss_t.detach()

        # update prev_dense per batch using model's finest-resolution prediction (detached)
        finest_occ_sp = outs[-1][0]  # finest level
        finest_occ_dense = finest_occ_sp.dense().sigmoid().detach()  # (B, 1, 64, 64, 64)
        for b, d in enumerate(batch_d_list):
            occ_b = finest_occ_dense[b, 0]  # (64, 64, 64)
            mask = occ_b > 0.5
            idx = torch.nonzero(mask, as_tuple=False)  # (M, 3)
            if idx.shape[0] == 0:
                prev_dense[b] = None
                continue
            pts_l = (idx.float() + 0.5) * VOXEL_SIZE - HALF
            origin = d["lidar_origin_world"][t].to(device)
            quat = d["lidar_quat_world"][t].to(device)
            prev_dense[b] = (pts_l, origin, quat)

    return {"loss": total_loss / max(rollout, 1), "loss_occ": total_occ.detach() / max(rollout, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", default="/workspace/isaac-robotics/evals/lidar_recon/baseline_hoeller")
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=5,
                    help="50 Hz dataset → effective frame rate. 5 = 10 Hz (paper).")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] indexing {args.dataset_dir} ...", flush=True)
    idx = LidarChunkIndex(args.dataset_dir)
    print(f"[INFO] {len(idx)} trajectories, "
          f"{sum(m['F'] for m in idx._meta)} total frames", flush=True)
    if args.overfit > 0:
        idx.trajectories = idx.trajectories[:args.overfit]
        print(f"[INFO] overfit mode → {len(idx.trajectories)} trajectories", flush=True)

    device = torch.device(args.device)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params: {n_params/1e6:.2f}M  device: {device}", flush=True)

    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    log_path = os.path.join(args.out_dir, "train.log")
    with open(log_path, "w") as logf:
        logf.write("step,loss,lr,elapsed_s\n")
    for step in range(1, args.steps + 1):
        # sample a batch
        traj_idxs = random.choices(range(len(idx)), k=args.batch_size)
        batch_d_list = [idx.sample(ti, args.rollout, downsample=args.downsample) for ti in traj_idxs]

        opt.zero_grad()
        out = _rollout_step(model, batch_d_list, args.rollout, device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[step {step:5d}/{args.steps}] loss={out['loss'].item():.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  elapsed={elapsed:.1f}s",
                flush=True,
            )
            with open(log_path, "a") as logf:
                logf.write(f"{step},{out['loss'].item():.4f},{sched.get_last_lr()[0]:.6e},{elapsed:.1f}\n")

        if step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "loss": out["loss"].item(),
            }
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{step:06d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[CKPT] {ckpt_path}", flush=True)

    print(f"[DONE] {args.steps} steps in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
