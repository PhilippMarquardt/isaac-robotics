"""Build a mixed multi-video bundle for Stage 1 real training.

Per-film: decodes train + val pairs (contiguous, block split with gap), encodes
through frozen ViT + reducers, stacks across films.

Output bundle keys:
  H_t_train, H_tp_train, video_id_train   (concatenated, per-pair film id)
  H_t_val,   H_tp_val,   video_id_val
  film_names                              (list of film names in order)

video_id is an integer 0..N_films-1.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from data.video_pairs import decode_contiguous_pairs


def encode_pair(enc, red_state, red_future, x_t, x_tp, chunk=64):
    Hts, Htps = [], []
    for s in range(0, x_t.shape[0], chunk):
        e = min(s + chunk, x_t.shape[0])
        Hts.append(red_state(enc(x_t[s:e])))
        Htps.append(red_future(enc(x_tp[s:e])))
    return torch.cat(Hts, 0), torch.cat(Htps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs="+", required=True, help="paths to mp4 files")
    ap.add_argument("--names", nargs="+", required=True, help="short ids per film")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train_per_film", type=int, default=3000)
    ap.add_argument("--stride_train", type=int, default=1)
    ap.add_argument("--train_start", type=int, default=200)
    ap.add_argument("--n_val_per_film", type=int, default=250)
    ap.add_argument("--stride_val", type=int, default=1)
    ap.add_argument("--block_gap", type=int, default=80)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    args = ap.parse_args()

    assert len(args.films) == len(args.names)
    device = "cuda"
    torch.manual_seed(0)

    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)
    print(f"loaded encoder ({args.encoder_name})")

    Ht_tr_all, Htp_tr_all, vid_tr_all = [], [], []
    Ht_va_all, Htp_va_all, vid_va_all = [], [], []
    meta = []

    for fi, (path, name) in enumerate(zip(args.films, args.names)):
        print(f"\n=== {name}  ({path}) ===")
        # train block
        x_t, x_tp, ts, _ = decode_contiguous_pairs(
            path, n_pairs=args.n_train_per_film, k=args.k,
            stride=args.stride_train, start_skip=args.train_start)
        train_end = int(ts[-1].item() + args.k)
        val_start = train_end + args.block_gap
        x_t = x_t.to(device); x_tp = x_tp.to(device)
        Ht, Htp = encode_pair(enc, red_state, red_future, x_t, x_tp)
        print(f"  train pairs: {Ht.shape[0]}, block [{args.train_start}, {train_end}]")
        Ht_tr_all.append(Ht.cpu()); Htp_tr_all.append(Htp.cpu())
        vid_tr_all.append(torch.full((Ht.shape[0],), fi, dtype=torch.long))
        del x_t, x_tp

        x_v_t, x_v_tp, vts, _ = decode_contiguous_pairs(
            path, n_pairs=args.n_val_per_film, k=args.k,
            stride=args.stride_val, start_skip=val_start)
        val_end = int(vts[-1].item() + args.k)
        x_v_t = x_v_t.to(device); x_v_tp = x_v_tp.to(device)
        Htv, Htpv = encode_pair(enc, red_state, red_future, x_v_t, x_v_tp)
        print(f"  val pairs: {Htv.shape[0]}, block [{val_start}, {val_end}]  gap={args.block_gap}")
        Ht_va_all.append(Htv.cpu()); Htp_va_all.append(Htpv.cpu())
        vid_va_all.append(torch.full((Htv.shape[0],), fi, dtype=torch.long))
        meta.append({"name": name, "path": path, "train": [args.train_start, train_end],
                     "val": [val_start, val_end], "n_train": Ht.shape[0], "n_val": Htv.shape[0]})

    out = {
        "H_t_train": torch.cat(Ht_tr_all, 0),
        "H_tp_train": torch.cat(Htp_tr_all, 0),
        "video_id_train": torch.cat(vid_tr_all, 0),
        "H_t_val": torch.cat(Ht_va_all, 0),
        "H_tp_val": torch.cat(Htp_va_all, 0),
        "video_id_val": torch.cat(vid_va_all, 0),
        "film_names": args.names,
        "k": args.k,
        "meta": meta,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nsaved -> {args.out}")
    print(f"  total train pairs: {out['H_t_train'].shape[0]}")
    print(f"  total val pairs:   {out['H_t_val'].shape[0]}")


if __name__ == "__main__":
    main()
