"""Lidar dataset loader for v6 schema with env-stratified train/val split.

Dataset chunks contain:
  env_id, episode_id, step_in_episode, timestamp_sim
  lidar_points_world (F, 20000, 3), lidar_hit_mask (F, 20000)
  lidar_origin_world (F, 3), lidar_quat_world (F, 4)  # xyzw
  base_pos_world (F, 3), base_quat_world (F, 4)
  gt_points_world (F, MESH_GT_MAX_PTS, 3), gt_n_points (F,) int32

Env-stratified split: each terrain tile is one env. Tiles assigned to train OR val,
never both. The map (env_id → tile_id) is implicit since IsaacLab assigns env_origin
per tile during scene setup. Without offline access to that map, we approximate by
hashing the (lidar_origin world (x,y) bin) → bucket the (x,y) plane into a grid where
each grid cell = one terrain tile (8 m × 8 m by default). Same lidar location ⇒ same
tile ⇒ deterministic train/val assignment.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset

TILE_SIZE = 8.0  # IsaacLab ROUGH/diverse default sub-terrain size


def tile_id_from_world_xy(x: float, y: float, tile_size: float = TILE_SIZE) -> tuple[int, int]:
    return (int(x // tile_size), int(y // tile_size))


def build_split(dataset_dir: str, val_frac: float = 0.2, seed: int = 0,
                tile_size: float = TILE_SIZE) -> tuple[set, set]:
    """Scan all chunks, collect every (tile_x, tile_y) seen, split deterministically."""
    chunks = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.pt")))
    tiles: set = set()
    for c in chunks:
        d = torch.load(c, map_location="cpu", weights_only=True)
        bp = d["base_pos_world"].numpy()
        for i in range(bp.shape[0]):
            tiles.add(tile_id_from_world_xy(bp[i, 0], bp[i, 1], tile_size))
    tiles_list = sorted(tiles)
    rng = random.Random(seed)
    rng.shuffle(tiles_list)
    n_val = max(1, int(len(tiles_list) * val_frac))
    val_set = set(tiles_list[:n_val])
    train_set = set(tiles_list[n_val:])
    return train_set, val_set


def save_split(train_set: set, val_set: set, path: str):
    with open(path, "w") as f:
        json.dump({
            "train_tiles": [list(t) for t in sorted(train_set)],
            "val_tiles": [list(t) for t in sorted(val_set)],
            "tile_size": TILE_SIZE,
        }, f, indent=2)


def load_split(path: str) -> tuple[set, set]:
    with open(path) as f:
        d = json.load(f)
    return ({tuple(t) for t in d["train_tiles"]},
            {tuple(t) for t in d["val_tiles"]})


class LidarFrameDataset(Dataset):
    """One frame per item. Filters by env-stratified tile set.

    On __getitem__: returns dict of (lidar_pcl, lidar_mask, lidar_origin, lidar_quat,
    base_pos, base_quat, gt_pcl, gt_n) all torch tensors.
    """

    def __init__(self, dataset_dir: str, tiles: set, tile_size: float = TILE_SIZE):
        self.dataset_dir = dataset_dir
        self.tiles = tiles
        self.tile_size = tile_size
        self.chunks = sorted(glob.glob(os.path.join(dataset_dir, "chunk_*.pt")))
        # cache per-chunk per-frame tile membership
        self.index: list[tuple[int, int]] = []  # (chunk_idx, frame_idx_in_chunk)
        for ci, c in enumerate(self.chunks):
            d = torch.load(c, map_location="cpu", weights_only=True)
            bp = d["base_pos_world"]
            for fi in range(bp.shape[0]):
                t = tile_id_from_world_xy(float(bp[fi, 0]), float(bp[fi, 1]), tile_size)
                if t in tiles:
                    self.index.append((ci, fi))
        self._cache_idx = -1
        self._cache: dict | None = None

    def __len__(self):
        return len(self.index)

    def _load(self, ci: int):
        if self._cache_idx != ci:
            self._cache = torch.load(self.chunks[ci], map_location="cpu", weights_only=True)
            self._cache_idx = ci
        return self._cache

    def __getitem__(self, idx: int):
        ci, fi = self.index[idx]
        d = self._load(ci)
        return {
            "lidar_pcl": d["lidar_points_world"][fi],
            "lidar_mask": d["lidar_hit_mask"][fi],
            "lidar_origin": d["lidar_origin_world"][fi],
            "lidar_quat": d["lidar_quat_world"][fi],
            "base_pos": d["base_pos_world"][fi],
            "base_quat": d["base_quat_world"][fi],
            "gt_pcl": d["gt_points_world"][fi],
            "gt_n": d["gt_n_points"][fi],
        }
