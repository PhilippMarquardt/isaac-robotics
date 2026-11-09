"""BridgeData V2 (scripted subset) clip sampler for v6.

Layout on disk:
    /root/bridge/scripted_raw/<task_date>/<session>/raw/traj_group0/trajN/
        images0/im_0.jpg .. im_49.jpg     (5 Hz, 640x480)
        obs_dict.pkl                      (full_state [T,7], ...)
        policy_out.pkl                    (list of T-1 dicts with 'actions' (7,))

Returned batch (all torch):
    frames   [B, T, H, W, 3]  float in [-1, 1]
    actions  [B, S, 7]        mean GT action over each future tubelet (EVAL ONLY)
    flows    [B, S, gh, gw, 2] Farneback flow per future tubelet, token grid
    scene_id [B]              int id of the <task_date> folder (leakage probe)
"""
from __future__ import annotations

import glob
import os
import pickle
import random

import cv2
import numpy as np
import torch


def build_bridge_manifest(root: str, min_frames: int = 20) -> list[dict]:
    """One entry per trajectory. Scene id = top-level task_date folder."""
    # recursive: handles both scripted (<task>/<sess>/raw/...) and teleop
    # (<env>/<task>/<sess>/raw/...) depths.
    traj_dirs = sorted(glob.glob(os.path.join(root, "**", "raw", "traj_group*", "traj*"),
                                 recursive=True))
    scenes = {}
    manifest = []
    for td in traj_dirs:
        img_dir = os.path.join(td, "images0")
        if not os.path.isdir(img_dir):
            continue
        n = len([f for f in os.listdir(img_dir) if f.startswith("im_")])
        if n < min_frames:
            continue
        scene_name = os.path.relpath(td, root).split(os.sep)[0]
        if scene_name not in scenes:
            scenes[scene_name] = len(scenes)
        manifest.append({
            "traj_dir": td,
            "n_frames": n,
            "scene_id": scenes[scene_name],
            "scene_name": scene_name,
        })
    return manifest


def _load_frame(traj_dir: str, idx: int, img_size: int) -> np.ndarray:
    p = os.path.join(traj_dir, "images0", f"im_{idx}.jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = img_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (nh - img_size) // 2, (nw - img_size) // 2
    return img[y0:y0 + img_size, x0:x0 + img_size]


def _tubelet_flow(frames_u8: np.ndarray, k: int, n_ctx_frames: int,
                  tubelet: int, grid: int) -> np.ndarray:
    """Farneback flow across future tubelet k, downsampled to (grid, grid, 2).

    frames_u8: [T, H, W, 3] uint8. Flow computed between the two frames of the
    tubelet (adjacent-frame motion), in units of pixels at the input resolution.
    """
    i0 = n_ctx_frames + k * tubelet
    i1 = i0 + tubelet - 1
    g0 = cv2.cvtColor(frames_u8[i0], cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(frames_u8[i1], cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )  # [H, W, 2]
    return cv2.resize(flow, (grid, grid), interpolation=cv2.INTER_AREA)


def _traj_actions(traj_dir: str):
    with open(os.path.join(traj_dir, "policy_out.pkl"), "rb") as f:
        pol = pickle.load(f)
    return np.stack([np.asarray(p["actions"], dtype=np.float64) for p in pol])  # [T-1, 7]


def sample_bridge_batch(manifest: list[dict], bs: int, num_frames: int,
                        img_size: int, n_ctx_frames: int, tubelet: int,
                        grid: int, with_flow: bool = True,
                        frame_stride: int = 1,
                        rng: random.Random | None = None):
    """frame_stride s: sample raw frames a, a+s, a+2s, ... so one model frame
    spans s/5 seconds and one tubelet spans 2s/5 seconds."""
    rng = rng or random
    S = (num_frames - n_ctx_frames) // tubelet
    span = (num_frames - 1) * frame_stride + 1
    frames_out = np.zeros((bs, num_frames, img_size, img_size, 3), dtype=np.uint8)
    actions_out = np.zeros((bs, S, 7), dtype=np.float32)
    flows_out = np.zeros((bs, S, grid, grid, 2), dtype=np.float32)
    scene_out = np.zeros((bs,), dtype=np.int64)

    i = 0
    tries = 0
    while i < bs and tries < bs * 8:
        tries += 1
        spec = rng.choice(manifest)
        max_start = spec["n_frames"] - span - 1
        if max_start < 0:
            continue
        a = rng.randint(0, max_start)
        try:
            clip = np.stack([_load_frame(spec["traj_dir"], a + j * frame_stride, img_size)
                             for j in range(num_frames)])
            acts = _traj_actions(spec["traj_dir"])  # [T-1, 7]
        except Exception:
            continue
        frames_out[i] = clip
        for k in range(S):
            # raw-frame window covered by future tubelet k
            r0 = a + (n_ctx_frames + k * tubelet) * frame_stride - 1
            r1 = min(r0 + tubelet * frame_stride, len(acts))
            if r0 < len(acts) and r0 >= 0:
                actions_out[i, k] = acts[max(r0, 0):r1].mean(axis=0)
            if with_flow:
                flows_out[i, k] = _tubelet_flow(clip, k, n_ctx_frames, tubelet, grid)
        scene_out[i] = spec["scene_id"]
        i += 1
    if i < bs:
        raise RuntimeError(f"only sampled {i}/{bs}")
    frames = torch.from_numpy(frames_out).float() / 127.5 - 1.0
    return (frames, torch.from_numpy(actions_out),
            torch.from_numpy(flows_out), torch.from_numpy(scene_out))
