"""Build a 13-film bundle with context frames.

Per film, per anchor t (drawn from [train_start, ...] with stride 1):
  context frames at t-32, t-16, t-8, t  → encode as H_ctx [4, 64, 384]
  anchor frame t                           → encode as H_t  [64, 384]   (== last context slot)
  target frame t+k                         → encode as H_tp [16, 384]
  apply frozen Stage 1 inverse encoder     → Z_star [4, 128]

Constraint: t - 32 >= 0. We start train_start at >= 32.
Val block kept disjoint with block_gap.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from data.video_pairs import decode_indices


CONTEXT_OFFSETS = [-32, -16, -8, 0]  # t-32, t-16, t-8, t


def build_one_film(path, name, film_id, args, enc, red_state, red_future, inv, device):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    train_start = max(args.train_start, 32)  # need t-32 >= 0
    n_tr = args.n_train_per_film
    stride_tr = args.stride_train
    anchors_tr = [train_start + i * stride_tr for i in range(n_tr)]
    train_end = anchors_tr[-1] + args.k
    val_start = train_end + args.block_gap
    n_va = args.n_val_per_film
    stride_va = args.stride_val
    anchors_va = [val_start + i * stride_va for i in range(n_va)]
    val_end = anchors_va[-1] + args.k
    assert val_end < total - 1, (val_end, total, name)
    print(f"  {name}: train [{train_start},{train_end}] n={n_tr}; val [{val_start},{val_end}] n={n_va}; total {total}")

    def encode_chunk(frame_indices):
        # decode + encode -> [N, 64, D] state tokens (and [N,16,D] future via the reducers as needed)
        # we re-decode each frame separately as before
        frames = decode_indices(path, frame_indices, size=args.size).to(device)
        Hs, Hf = [], []
        chunk = args.encode_chunk
        for s in range(0, frames.shape[0], chunk):
            e = min(s + chunk, frames.shape[0])
            feats = enc(frames[s:e])
            Hs.append(red_state(feats).cpu())
            Hf.append(red_future(feats).cpu())
        return torch.cat(Hs, 0), torch.cat(Hf, 0)

    def build_split(anchors):
        # frames needed: t-32, t-16, t-8, t (state encoded), t+k (future encoded)
        # for context we use STATE features; for future we use FUTURE reducer.
        ctx_inds = [a + off for a in anchors for off in CONTEXT_OFFSETS]
        tp_inds = [a + args.k for a in anchors]
        all_unique = sorted(set(ctx_inds + tp_inds))
        # decode each unique frame once; we need BOTH state and future features.
        H_state, H_future = encode_chunk(all_unique)
        idx_map = {fi: i for i, fi in enumerate(all_unique)}
        # H_ctx [N, 4, 64, D]
        N = len(anchors)
        H_ctx = torch.stack([
            torch.stack([H_state[idx_map[a + off]] for off in CONTEXT_OFFSETS], dim=0)
            for a in anchors
        ], dim=0)
        # H_t  = last context slot
        H_t = torch.stack([H_state[idx_map[a]] for a in anchors], dim=0)
        # H_tp [N, 16, D]
        H_tp = torch.stack([H_future[idx_map[a + args.k]] for a in anchors], dim=0)
        # Z_star from frozen inverse
        with torch.no_grad():
            Z = inv(H_t.to(device), H_tp.to(device)).cpu()
        return H_ctx, H_t, H_tp, Z

    H_ctx_tr, H_t_tr, H_tp_tr, Z_tr = build_split(anchors_tr)
    H_ctx_va, H_t_va, H_tp_va, Z_va = build_split(anchors_va)
    return (H_ctx_tr, H_t_tr, H_tp_tr, Z_tr, anchors_tr,
            H_ctx_va, H_t_va, H_tp_va, Z_va, anchors_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True, help="Stage 1 ckpt for inverse encoder (frozen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train_per_film", type=int, default=2000)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--train_start", type=int, default=232)  # >=32 for t-32 buffer
    ap.add_argument("--n_val_per_film", type=int, default=200)
    ap.add_argument("--stride_val", type=int, default=1)
    ap.add_argument("--block_gap", type=int, default=80)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    ap.add_argument("--encode_chunk", type=int, default=64)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    for p in inv.parameters(): p.requires_grad_(False)
    inv.eval()
    print(f"loaded inverse: action {cfg['action_tokens']}x{cfg['action_dim']}, depth {cfg['inverse_depth']}")

    H_ctx_tr_all, H_t_tr_all, H_tp_tr_all, Z_tr_all, vid_tr_all = [], [], [], [], []
    H_ctx_va_all, H_t_va_all, H_tp_va_all, Z_va_all, vid_va_all = [], [], [], [], []
    meta = []
    for fi, (path, name) in enumerate(zip(args.films, args.names)):
        print(f"\n--- film {fi}: {name} ---")
        out_tup = build_one_film(path, name, fi, args, enc, red_state, red_future, inv, device)
        (H_ctx_tr, H_t_tr, H_tp_tr, Z_tr, anchors_tr,
         H_ctx_va, H_t_va, H_tp_va, Z_va, anchors_va) = out_tup
        H_ctx_tr_all.append(H_ctx_tr); H_t_tr_all.append(H_t_tr); H_tp_tr_all.append(H_tp_tr); Z_tr_all.append(Z_tr)
        vid_tr_all.append(torch.full((H_ctx_tr.shape[0],), fi, dtype=torch.long))
        H_ctx_va_all.append(H_ctx_va); H_t_va_all.append(H_t_va); H_tp_va_all.append(H_tp_va); Z_va_all.append(Z_va)
        vid_va_all.append(torch.full((H_ctx_va.shape[0],), fi, dtype=torch.long))
        meta.append({"name": name, "path": path, "n_train": H_ctx_tr.shape[0], "n_val": H_ctx_va.shape[0]})

    bundle = {
        "H_ctx_train": torch.cat(H_ctx_tr_all, 0),
        "H_t_train": torch.cat(H_t_tr_all, 0),
        "H_tp_train": torch.cat(H_tp_tr_all, 0),
        "Z_star_train": torch.cat(Z_tr_all, 0),
        "video_id_train": torch.cat(vid_tr_all, 0),
        "H_ctx_val": torch.cat(H_ctx_va_all, 0),
        "H_t_val": torch.cat(H_t_va_all, 0),
        "H_tp_val": torch.cat(H_tp_va_all, 0),
        "Z_star_val": torch.cat(Z_va_all, 0),
        "video_id_val": torch.cat(vid_va_all, 0),
        "film_names": args.names,
        "k": args.k,
        "context_offsets": CONTEXT_OFFSETS,
        "meta": meta,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bundle, args.out)
    print(f"\nsaved -> {args.out}")
    print(f"  train pairs: {bundle['H_ctx_train'].shape[0]}  H_ctx shape {tuple(bundle['H_ctx_train'].shape)}")
    print(f"  val pairs:   {bundle['H_ctx_val'].shape[0]}")


if __name__ == "__main__":
    main()
