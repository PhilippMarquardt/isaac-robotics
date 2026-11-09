"""Temporal LiDAR dataset — yields a sliding window of past frames for each target frame.

For target frame `t` (in some episode of some env), returns:

  - K+1 lidar frames: t-K*stride, t-(K-1)*stride, ..., t (current is last index)
  - K+1 base poses (world frame)
  - K+1 lidar mount poses (world frame)
  - K+1 commanded + actual velocities
  - K+1 frame-age values (seconds; 0 for current, negative for past)
  - K+1 valid_history mask (False where past frame doesn't exist within episode)
  - 1 GT cloud (only at time t)

History never crosses episode boundaries: samples earlier than the current
episode start get padded with the current frame and `valid_history=False`. This
matches deployment: cold-start has no history, model must learn to handle it.

All data is returned in WORLD frame; the training step is responsible for
transforming into the base frame at time t.

Splits: same env-stratified tile-id split as `dataset.py`.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import load_split  # reuse existing split loader


@dataclass
class TemporalIndexEntry:
    chunk_path: str
    chunk_frame_idx: int          # index inside the chunk
    env_id: int
    episode_id: int
    step_in_ep: int
    # for every history offset h in 0..K (h=0 is the oldest, K is current):
    # location of that frame; -1 chunk_frame_idx if the frame is unavailable.
    history_chunk: list[str]      # len K+1, paths
    history_chunk_idx: list[int]  # len K+1, frame index inside chunk (-1 if pad)
    history_dt_s: list[float]     # len K+1, frame_age (negative seconds)
    valid_history: list[bool]     # len K+1


class TemporalLidarDataset(Dataset):
    """Returns a windowed (K+1)-frame sample per target frame.

    Args:
        dataset_dir: directory with `chunk_*.pt` files
        tiles: set of tile_ids (str "x,y") used as filter (env-stratified split)
        history_n: K, number of PAST frames to include (sample length = K+1)
        history_stride: gap (in raw frames) between consecutive history slots
        max_lidar_pts: pad lidar to this length per frame
        max_gt_pts: pad GT to this length
    """

    def __init__(
        self,
        dataset_dir: str,
        tiles: set[str] | None = None,
        history_n: int = 10,
        history_stride: int = 5,
        max_lidar_pts: int = 20000,
        max_gt_pts: int = 200000,
        tile_size: float = 8.0,
    ):
        self.dataset_dir = dataset_dir
        self.history_n = history_n
        self.history_stride = history_stride
        self.max_lidar_pts = max_lidar_pts
        self.max_gt_pts = max_gt_pts

        chunk_paths = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.pt")))
        if not chunk_paths:
            raise RuntimeError(f"no chunk_*.pt found in {dataset_dir}")

        # Pass 1: build per-(env, episode) ordered frame lists across all chunks.
        # Also collect per-frame tile filter.
        frames_per_traj: dict[tuple[int, int], list[tuple[str, int, int, float]]] = {}
        # value: list of (chunk_path, chunk_idx, step_in_ep, sim_t)

        for cp in chunk_paths:
            d = torch.load(cp, map_location="cpu", weights_only=True)
            env_ids = d["env_id"].tolist()
            ep_ids = d["episode_id"].tolist()
            steps = d["step_in_episode"].tolist()
            ts = d["timestamp_sim"].tolist()
            base_xy = d["base_pos_world"][:, :2].numpy()

            tile_x = np.floor(base_xy[:, 0] / tile_size).astype(np.int64)
            tile_y = np.floor(base_xy[:, 1] / tile_size).astype(np.int64)
            tile_keys = [f"{tx},{ty}" for tx, ty in zip(tile_x, tile_y)]

            for i in range(len(env_ids)):
                key = (int(env_ids[i]), int(ep_ids[i]))
                frames_per_traj.setdefault(key, []).append(
                    (cp, i, int(steps[i]), float(ts[i]), tile_keys[i])
                )

        # sort each traj by step_in_episode
        for k in frames_per_traj:
            frames_per_traj[k].sort(key=lambda e: e[2])

        # Pass 2: for each traj, construct windowed entries
        self.entries: list[TemporalIndexEntry] = []
        for key, frames in frames_per_traj.items():
            for pos, (cp, ci, step, t_curr, tile_key) in enumerate(frames):
                if tiles is not None and tile_key not in tiles:
                    continue
                hist_chunk = []
                hist_idx = []
                hist_dt = []
                hist_valid = []
                # build K+1 slots: oldest first, current last
                for slot in range(history_n, -1, -1):
                    target_pos = pos - slot * history_stride
                    if target_pos < 0:
                        # pad with current frame (handler will mask via valid_history)
                        hist_chunk.append(cp)
                        hist_idx.append(ci)
                        hist_dt.append(0.0)
                        hist_valid.append(False)
                    else:
                        f_cp, f_ci, f_step, f_t, _ = frames[target_pos]
                        # ensure same episode (sorting by step inside same key)
                        hist_chunk.append(f_cp)
                        hist_idx.append(f_ci)
                        hist_dt.append(float(f_t - t_curr))
                        hist_valid.append(True)
                self.entries.append(TemporalIndexEntry(
                    chunk_path=cp, chunk_frame_idx=ci,
                    env_id=key[0], episode_id=key[1], step_in_ep=step,
                    history_chunk=hist_chunk,
                    history_chunk_idx=hist_idx,
                    history_dt_s=hist_dt,
                    valid_history=hist_valid,
                ))

        # cache loaded chunks (LRU-ish — keep all if dataset fits in RAM)
        self._chunk_cache: dict[str, dict] = {}
        self._chunk_lru: list[str] = []
        # 12 chunks × ~360 MB ≈ 4.3 GB — fine to keep all in RAM
        self._chunk_cache_max = max(16, len(chunk_paths))

    def _get_chunk(self, path: str) -> dict:
        if path in self._chunk_cache:
            self._chunk_lru.remove(path)
            self._chunk_lru.append(path)
            return self._chunk_cache[path]
        d = torch.load(path, map_location="cpu", weights_only=True)
        self._chunk_cache[path] = d
        self._chunk_lru.append(path)
        if len(self._chunk_lru) > self._chunk_cache_max:
            old = self._chunk_lru.pop(0)
            del self._chunk_cache[old]
        return d

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        K1 = self.history_n + 1

        lidar_pcl = torch.zeros(K1, self.max_lidar_pts, 3, dtype=torch.float32)
        lidar_mask = torch.zeros(K1, self.max_lidar_pts, dtype=torch.bool)
        lidar_origin = torch.zeros(K1, 3, dtype=torch.float32)
        lidar_quat = torch.zeros(K1, 4, dtype=torch.float32)
        lidar_quat[..., 3] = 1.0  # identity xyzw default
        base_pos = torch.zeros(K1, 3, dtype=torch.float32)
        base_quat = torch.zeros(K1, 4, dtype=torch.float32)
        base_quat[..., 3] = 1.0
        cmd_vel = torch.zeros(K1, 3, dtype=torch.float32)
        actual_vel = torch.zeros(K1, 3, dtype=torch.float32)
        frame_age_s = torch.tensor(e.history_dt_s, dtype=torch.float32)
        valid_history = torch.tensor(e.valid_history, dtype=torch.bool)

        for slot in range(K1):
            cp = e.history_chunk[slot]
            ci = e.history_chunk_idx[slot]
            d = self._get_chunk(cp)
            lid_pts = d["lidar_points_world"][ci]            # (Rmax, 3)
            lid_msk = d["lidar_hit_mask"][ci]                # (Rmax,)
            n = min(self.max_lidar_pts, lid_pts.shape[0])
            lidar_pcl[slot, :n] = lid_pts[:n].to(torch.float32)
            lidar_mask[slot, :n] = lid_msk[:n].bool()
            lidar_origin[slot] = d["lidar_origin_world"][ci].to(torch.float32)
            lidar_quat[slot] = d["lidar_quat_world"][ci].to(torch.float32)
            base_pos[slot] = d["base_pos_world"][ci].to(torch.float32)
            base_quat[slot] = d["base_quat_world"][ci].to(torch.float32)
            if "commanded_velocity" in d:
                cmd_vel[slot] = d["commanded_velocity"][ci].to(torch.float32)
            if "actual_base_velocity" in d:
                actual_vel[slot] = d["actual_base_velocity"][ci].to(torch.float32)

        # GT only at current frame (slot K)
        d_cur = self._get_chunk(e.chunk_path)
        gt_pts = d_cur["gt_points_world"][e.chunk_frame_idx]
        gt_n = int(d_cur["gt_n_points"][e.chunk_frame_idx].item())
        gt_pcl = torch.zeros(self.max_gt_pts, 3, dtype=torch.float32)
        gn = min(self.max_gt_pts, gt_n)
        if gn > 0:
            gt_pcl[:gn] = gt_pts[:gn].to(torch.float32)

        return {
            "lidar_pcl": lidar_pcl,                  # (K+1, N, 3)
            "lidar_mask": lidar_mask,                # (K+1, N)
            "lidar_origin": lidar_origin,            # (K+1, 3)
            "lidar_quat": lidar_quat,                # (K+1, 4)
            "base_pos": base_pos,                    # (K+1, 3)
            "base_quat": base_quat,                  # (K+1, 4)
            "frame_age_s": frame_age_s,              # (K+1,)
            "valid_history": valid_history,          # (K+1,)
            "commanded_velocity": cmd_vel,           # (K+1, 3)
            "actual_base_velocity": actual_vel,      # (K+1, 3)
            "gt_pcl": gt_pcl,                        # (M, 3)
            "gt_n": torch.tensor(gn, dtype=torch.long),
            "current_idx": torch.tensor(self.history_n, dtype=torch.long),
            "env_id": torch.tensor(e.env_id, dtype=torch.long),
            "episode_id": torch.tensor(e.episode_id, dtype=torch.long),
            "step_in_ep": torch.tensor(e.step_in_ep, dtype=torch.long),
        }


def temporal_collate(batch_list):
    keys = batch_list[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out
