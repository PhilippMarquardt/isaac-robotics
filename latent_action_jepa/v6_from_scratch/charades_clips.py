"""On-the-fly Charades clip sampler.

Charades_v1_480 ships ~10 k mp4 files. Decoding the lot to a RAM cache is
infeasible (~1 TB), so we keep only per-video metadata + decode each clip
lazily with PyAV when the trainer asks for it.

Public API (mirrors the v2/v4 cache API the trainer expects):

    index = build_charades_index(video_dir, min_clip_frames=64)
            -> list[dict{"path", "n_frames", "fps"}]

    sample_clip_batch(index, video_dir, bs, num_frames, frame_stride,
                      img_size, augment=True)
            -> torch.FloatTensor [bs, T, H, W, 3] in [-1, 1]

The trainer used to load a single big `cache.pt` containing all video frames;
this drop-in replacement just stores `cache["index"]` (the manifest) and
decodes per-clip on demand.
"""
from __future__ import annotations

import os
import random
import subprocess

import av
import numpy as np
import torch


def _probe_video(path: str) -> tuple[int, float] | None:
    """Return (n_frames, fps). Reads container metadata only — fast."""
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 24.0
            # frames count: prefer stream.frames; fall back to duration*fps.
            n = stream.frames or 0
            if n <= 0 and stream.duration is not None and stream.time_base is not None:
                dur_s = float(stream.duration * stream.time_base)
                n = int(dur_s * fps)
            return n, fps
    except Exception:
        return None


def build_charades_index(video_dir: str, min_clip_frames: int = 64,
                         max_videos: int | None = None) -> list[dict]:
    files = sorted(
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(".mp4")
    )
    if max_videos:
        files = files[:max_videos]
    index = []
    for p in files:
        probed = _probe_video(p)
        if probed is None:
            continue
        n, fps = probed
        if n < min_clip_frames:
            continue
        index.append({"path": p, "n_frames": int(n), "fps": float(fps)})
    return index


def _decode_clip(path: str, start_frame: int, num_frames: int,
                 frame_stride: int, img_size: int) -> np.ndarray:
    """Decode a [num_frames, H, W, 3] uint8 array starting at `start_frame`.

    Frames are returned in (T, H, W, 3) uint8 — caller normalises.
    `frame_stride` skips frames between samples (stride=2 = every other frame).
    """
    target_frame_idxs = set(start_frame + i * frame_stride for i in range(num_frames))
    last_needed = max(target_frame_idxs)
    out = [None] * num_frames

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek to roughly the right place — PyAV seek uses stream time base.
        if stream.time_base and stream.average_rate:
            seek_pts = int(start_frame / float(stream.average_rate)
                           / float(stream.time_base))
            try:
                container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
            except av.AVError:
                pass

        target_to_slot = {start_frame + i * frame_stride: i for i in range(num_frames)}
        cur = 0
        for frame in container.decode(stream):
            # frame_count is the *decoded* index since seek; use frame.pts to
            # recover absolute frame number.
            abs_idx = (int(frame.pts * float(stream.time_base) * float(stream.average_rate))
                       if frame.pts is not None else cur)
            cur += 1
            if abs_idx in target_to_slot:
                arr = frame.to_ndarray(format="rgb24")  # [H, W, 3]
                out[target_to_slot[abs_idx]] = arr
            if abs_idx > last_needed + 4:
                break

    # If any slot still None (rare PyAV seek miss), retry from start.
    if any(s is None for s in out):
        return _decode_clip_strict(path, start_frame, num_frames, frame_stride, img_size)

    # Resize + center-crop to img_size.
    return _frames_to_array(out, img_size)


def _decode_clip_strict(path: str, start_frame: int, num_frames: int,
                        frame_stride: int, img_size: int) -> np.ndarray:
    """Fallback: walk every frame from the start. Slower but always correct."""
    target_idxs = set(start_frame + i * frame_stride for i in range(num_frames))
    last_needed = max(target_idxs)
    target_to_slot = {start_frame + i * frame_stride: i for i in range(num_frames)}
    out = [None] * num_frames
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if idx in target_to_slot:
                out[target_to_slot[idx]] = frame.to_ndarray(format="rgb24")
            if idx >= last_needed:
                break
    return _frames_to_array(out, img_size)


def _frames_to_array(frames: list[np.ndarray], img_size: int) -> np.ndarray:
    """Resize + centre-crop each frame to (img_size, img_size, 3) and stack."""
    import cv2  # opencv is available in our env; falls back to nearest if not
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        scale = img_size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        r = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
        # centre-crop
        y0 = max((nh - img_size) // 2, 0)
        x0 = max((nw - img_size) // 2, 0)
        r = r[y0:y0 + img_size, x0:x0 + img_size, :]
        if r.shape[0] != img_size or r.shape[1] != img_size:
            # pad if seek failed and frame is malformed
            r = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        resized.append(r)
    return np.stack(resized, axis=0)


def sample_clip_batch(index: list[dict], bs: int, num_frames: int,
                      frame_stride: int, img_size: int,
                      augment: bool = True) -> torch.Tensor:
    """Return [bs, T, H, W, 3] float tensor in [-1, 1]."""
    span = (num_frames - 1) * frame_stride + 1
    out = np.zeros((bs, num_frames, img_size, img_size, 3), dtype=np.uint8)
    i = 0
    tries = 0
    while i < bs and tries < bs * 8:
        tries += 1
        spec = random.choice(index)
        max_start = spec["n_frames"] - span - 1
        if max_start <= 0:
            continue
        start = random.randint(0, max_start)
        try:
            arr = _decode_clip(spec["path"], start, num_frames, frame_stride, img_size)
        except Exception:
            continue
        out[i] = arr
        i += 1
    if i < bs:
        raise RuntimeError(f"could only sample {i}/{bs} clips")
    arr = torch.from_numpy(out).float() / 127.5 - 1.0
    return arr  # [B, T, H, W, 3]
