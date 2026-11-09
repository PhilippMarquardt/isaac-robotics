"""Clip cache for V-JEPA-2 small-scale training.

Per film, decodes a long contiguous frame range as uint8 [N_frames,3,H,W].
At train time, each "clip" is `num_frames` frames sampled at `frame_stride`
starting from a random anchor.
"""
import os
import cv2
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def decode_film_uint8(path, lo, hi, size=224):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames = []
    for _ in range(hi - lo):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if len(frames) == 0:
        return None
    arr = np.stack(frames).transpose(0, 3, 1, 2)  # [N,3,H,W]
    return torch.from_numpy(arr)  # uint8


def build_cache(films, names, out_path, frame_lo=200, frame_hi=2400, size=224):
    """Decode frame_lo..frame_hi-1 from each film. Saves cache for clip sampling."""
    cache = {"film_names": [], "frame_lo": frame_lo, "frame_hi": frame_hi, "size": size,
             "frames": {}}
    out_fi = 0
    min_frames = max(64, (frame_hi - frame_lo) // 2)
    for path, name in zip(films, names):
        try:
            print(f"  [{name}] decoding frames [{frame_lo}, {frame_hi}) ...", flush=True)
            frames = decode_film_uint8(path, frame_lo, frame_hi, size=size)
            if frames is None or frames.shape[0] < min_frames:
                print(f"    SKIP {name}: only {0 if frames is None else frames.shape[0]} frames")
                continue
            print(f"    got {frames.shape[0]} frames", flush=True)
            cache["frames"][out_fi] = frames
            cache["film_names"].append(name)
            out_fi += 1
        except Exception as e:
            print(f"    SKIP {name}: decode error {e}")
            continue
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    total = sum(f.shape[0] for f in cache["frames"].values())
    print(f"saved -> {out_path}  total_frames={total}  bytes~={total*size*size*3}")


def sample_clip_batch(cache, batch_specs, num_frames, frame_stride, device):
    """batch_specs: list of (film_id, anchor_idx) relative to cache[fi] window.
    Returns clips [B, 3, num_frames, H, W] normalized, plus film_ids."""
    clips = []
    fids = []
    for fi, a in batch_specs:
        frames = cache["frames"][fi]
        idxs = [a + i * frame_stride for i in range(num_frames)]
        clip = frames[idxs]                                      # [T,3,H,W]
        clips.append(clip)
        fids.append(fi)
    clip = torch.stack(clips, dim=0).to(device).float() / 255.0   # [B,T,3,H,W]
    clip = clip.permute(0, 2, 1, 3, 4)                              # [B,3,T,H,W]
    clip = (clip - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return clip, torch.tensor(fids, dtype=torch.long, device=device)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame_lo", type=int, default=200)
    ap.add_argument("--frame_hi", type=int, default=2400)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    build_cache(args.films, args.names, args.out, args.frame_lo, args.frame_hi, args.size)
