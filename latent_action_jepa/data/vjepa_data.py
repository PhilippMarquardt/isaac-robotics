"""Cache raw frames (uint8 [H,W,3] at 224×224) per film in RAM for V-JEPA training.

Anchors at stride 1; context = [t-16, t-8, t]; target = t+16.
Each anchor requires frames at indices [t-16, t-8, t, t+16].
"""
import os
import cv2
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CTX_OFFSETS = [-16, -8, 0]
TGT_OFFSET = 16


def decode_film_uint8(path, lo, hi, size=224):
    """Decode frames lo..hi-1 from a video into uint8 tensor [N,3,H,W]."""
    cap = cv2.VideoCapture(path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for _ in range(hi - lo):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    arr = np.stack(frames)  # [N,H,W,3]
    arr = arr.transpose(0, 3, 1, 2)  # [N,3,H,W]
    return torch.from_numpy(arr)  # uint8


def build_vjepa_cache(films, names, out_path, n_train_per_film=2000, n_val_per_film=200,
                     stride=1, train_start=16, block_gap=80, size=224):
    """Build cache. Per film:
       train anchors: [train_start, train_start+1, ..., train_start+n_train-1]
       val anchors:   train_end + block_gap + i for i in range(n_val)
    Needs frames at anchor-16, anchor-8, anchor, anchor+16 for each anchor.
    Min cache index = train_start - 16. Max = val_end_anchor + 16.
    """
    train_start = max(train_start, abs(min(CTX_OFFSETS)))
    cache = {"film_names": names, "ctx_offsets": CTX_OFFSETS, "tgt_offset": TGT_OFFSET,
             "size": size, "train_anchors": {}, "val_anchors": {}, "frames": {}}
    for fi, (path, name) in enumerate(zip(films, names)):
        train_anchors = [train_start + i * stride for i in range(n_train_per_film)]
        train_end = train_anchors[-1] + TGT_OFFSET
        val_start = train_end + block_gap
        val_anchors = [val_start + i * stride for i in range(n_val_per_film)]
        val_end = val_anchors[-1] + TGT_OFFSET
        lo = train_anchors[0] + min(CTX_OFFSETS)  # min frame index needed
        hi = val_end + 1
        print(f"  [{name}] decoding frames [{lo}, {hi})  train [{train_anchors[0]},{train_end}]  val [{val_start},{val_end}]")
        frames_u8 = decode_film_uint8(path, lo, hi, size=size)
        print(f"    got {frames_u8.shape[0]} frames")
        cache["frames"][fi] = {"data": frames_u8, "lo": lo}
        cache["train_anchors"][fi] = train_anchors
        cache["val_anchors"][fi] = val_anchors
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    total_frames = sum(c["data"].shape[0] for c in cache["frames"].values())
    print(f"saved -> {out_path}  total_frames={total_frames}  bytes={total_frames*size*size*3}")


def load_clip_batch(cache, batch_specs, device, ctx_offsets=None):
    """batch_specs: list of (film_id, anchor_idx). ctx_offsets overrides default.
    Returns (context [B, T_ctx, 3, H, W], target [B,3,H,W], film_id [B])."""
    if ctx_offsets is None:
        ctx_offsets = CTX_OFFSETS
    ctx_list, tgt_list, fids = [], [], []
    for fi, a in batch_specs:
        frames = cache["frames"][fi]
        data = frames["data"]; lo = frames["lo"]
        ctx_frames = [data[a + off - lo] for off in ctx_offsets]
        tgt = data[a + TGT_OFFSET - lo]
        ctx_list.append(torch.stack(ctx_frames, dim=0))
        tgt_list.append(tgt)
        fids.append(fi)
    ctx = torch.stack(ctx_list, dim=0).to(device).float() / 255.0
    tgt = torch.stack(tgt_list, dim=0).to(device).float() / 255.0
    ctx = (ctx - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    tgt = (tgt - IMAGENET_MEAN.to(device).squeeze(0)) / IMAGENET_STD.to(device).squeeze(0)
    fids = torch.tensor(fids, dtype=torch.long, device=device)
    return ctx, tgt, fids


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train_per_film", type=int, default=2000)
    ap.add_argument("--n_val_per_film", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--train_start", type=int, default=200)
    ap.add_argument("--block_gap", type=int, default=80)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    build_vjepa_cache(args.films, args.names, args.out,
                      n_train_per_film=args.n_train_per_film,
                      n_val_per_film=args.n_val_per_film,
                      stride=args.stride, train_start=args.train_start,
                      block_gap=args.block_gap, size=args.size)
