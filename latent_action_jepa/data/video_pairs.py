"""Build a cached frame-pair tensor from a single video.

Sample N frames evenly from the video; resize to size; normalize with ImageNet
mean/std. Build pairs (i, i+k) for several k.
"""
import os
import cv2
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def decode_indices(path, indices, size=224):
    """Decode specific frame indices and return normalized tensor [N,3,H,W]."""
    import numpy as np
    cap = cv2.VideoCapture(path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if not ret:
            raise RuntimeError(f"failed at idx={idx}")
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    arr = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def decode_evenly(path, n_frames=400, size=224, start_skip=200, end_skip=200):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert total > start_skip + end_skip + n_frames, total
    idx = torch.linspace(start_skip, total - end_skip - 1, n_frames).round().long().tolist()
    return decode_indices(path, idx, size=size)


def decode_contiguous_pairs(path, n_pairs=200, k=2, size=224, stride=50, start_skip=200):
    """Sample n_pairs anchor frames at given stride; pair each with anchor+k (in video frames).
    Guarantees distinct anchor and future frames as long as stride > k.
    Returns x_t [n_pairs,3,H,W], x_tp [n_pairs,3,H,W].
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert stride >= 1
    anchors = [start_skip + i * stride for i in range(n_pairs)]
    assert anchors[-1] + k < total - 1, (anchors[-1], k, total, n_pairs, stride, start_skip)
    futures = [a + k for a in anchors]
    # Note: anchor/future overlap is acceptable as a codebook source — anchor_i==future_j
    # just means two entries share a frame value. The pair labels (i, i+k) remain valid.
    overlap = len(set(anchors) & set(futures))
    if overlap:
        print(f"  (note: {overlap}/{n_pairs} anchors overlap a future frame; ok for codebook)")
    frames = decode_indices(path, anchors + futures, size=size)
    x_t = frames[:n_pairs]
    x_tp = frames[n_pairs:]
    return x_t, x_tp, torch.tensor(anchors), torch.tensor([k] * n_pairs)


def build_pairs(frames, ks=(1, 2, 4, 8), per_anchor=1, seed=0):
    """Return (frame_t, frame_tp, t_idx, k_used) for valid (i, i+k)."""
    g = torch.Generator().manual_seed(seed)
    N = frames.shape[0]
    anchors, futures, ts, ks_out = [], [], [], []
    for i in range(N):
        valid_k = [k for k in ks if i + k < N]
        if not valid_k:
            continue
        for _ in range(per_anchor):
            k = valid_k[torch.randint(0, len(valid_k), (1,), generator=g).item()]
            anchors.append(i); futures.append(i + k)
            ts.append(i); ks_out.append(k)
    a = torch.tensor(anchors); f = torch.tensor(futures)
    return frames[a], frames[f], torch.tensor(ts), torch.tensor(ks_out)


def make_or_load_cache(video_path, cache_path, n_frames=400, size=224, ks=(1, 2, 4, 8)):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu")
        return d["x_t"], d["x_tp"], d["ts"], d["ks"]
    frames = decode_evenly(video_path, n_frames=n_frames, size=size)
    x_t, x_tp, ts, ks_used = build_pairs(frames, ks=ks)
    torch.save({"x_t": x_t, "x_tp": x_tp, "ts": ts, "ks": ks_used}, cache_path)
    return x_t, x_tp, ts, ks_used


def make_or_load_contig(video_path, cache_path, n_pairs=200, k=2, stride=50, size=224):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu")
        return d["x_t"], d["x_tp"], d["ts"], d["ks"]
    x_t, x_tp, ts, ks_t = decode_contiguous_pairs(video_path, n_pairs=n_pairs, k=k, stride=stride, size=size)
    torch.save({"x_t": x_t, "x_tp": x_tp, "ts": ts, "ks": ks_t}, cache_path)
    return x_t, x_tp, ts, ks_t
