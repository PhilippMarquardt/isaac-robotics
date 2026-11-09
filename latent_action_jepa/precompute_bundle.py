"""Pre-compute H_t / H_tp / Z_star bundle for a contiguous video block.

Uses frozen ViT-S layer 12, FixedSpatialReducer (params-free), and a frozen
Stage 1C inverse encoder. Result is a pickle bundle compatible with train_stage2a.py.

Lets us scale Stage 2 data without retraining Stage 1C.
"""
import argparse
import os
import torch
import torch.nn.functional as F

from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
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
    ap.add_argument("--video", default="/workspace/latent_action_jepa/bbb.mp4")
    ap.add_argument("--ckpt", default="/workspace/latent_action_jepa/runs/stage1c/ckpt.pt")
    ap.add_argument("--out", required=True)
    # Train block
    ap.add_argument("--train_start", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--stride_train", type=int, default=3)
    # Val block
    ap.add_argument("--val_start", type=int, default=-1, help="-1 = auto (train_end + block_gap)")
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--stride_val", type=int, default=18)
    ap.add_argument("--block_gap", type=int, default=80)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--encoder_name", default="vit_small_patch16_224")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    enc = ViTEncoder(args.encoder_name, pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    red_state = FixedSpatialReducer(64).to(device)
    red_future = FixedSpatialReducer(16).to(device)
    print("loaded encoder")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    inv = InverseActionEncoder(cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"],
                               depth=cfg["inverse_depth"]).to(device)
    inv.load_state_dict(ckpt["inverse_encoder"])
    for p in inv.parameters(): p.requires_grad_(False)
    inv.eval()
    print(f"loaded inverse encoder (depth={cfg['inverse_depth']}, action {cfg['action_tokens']}x{cfg['action_dim']})")

    # Decode train block
    print(f"\ndecoding train block: start={args.train_start} n={args.n_train} stride={args.stride_train} k={args.k}")
    x_t, x_tp, ts, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_train, k=args.k,
                                              stride=args.stride_train, start_skip=args.train_start)
    train_end = int(ts[-1].item() + args.k)
    print(f"  train block end frame: {train_end}")
    x_t = x_t.to(device); x_tp = x_tp.to(device)
    H_t_train, H_tp_train = encode_pair(enc, red_state, red_future, x_t, x_tp)
    with torch.no_grad():
        Z_star_train = inv(H_t_train, H_tp_train)
    print(f"  H_t {tuple(H_t_train.shape)}  H_tp {tuple(H_tp_train.shape)}  Z* {tuple(Z_star_train.shape)}")
    del x_t, x_tp

    val_start = args.val_start if args.val_start >= 0 else train_end + args.block_gap
    print(f"\ndecoding val block: start={val_start} n={args.n_val} stride={args.stride_val} gap={val_start - train_end}")
    x_v_t, x_v_tp, ts_v, _ = decode_contiguous_pairs(args.video, n_pairs=args.n_val, k=args.k,
                                                    stride=args.stride_val, start_skip=val_start)
    print(f"  val block end frame: {int(ts_v[-1].item() + args.k)}")
    x_v_t = x_v_t.to(device); x_v_tp = x_v_tp.to(device)
    H_t_val, H_tp_val = encode_pair(enc, red_state, red_future, x_v_t, x_v_tp)
    with torch.no_grad():
        Z_star_val = inv(H_t_val, H_tp_val)
    print(f"  H_t {tuple(H_t_val.shape)}  H_tp {tuple(H_tp_val.shape)}  Z* {tuple(Z_star_val.shape)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "H_t_train": H_t_train.cpu(),
        "H_tp_train": H_tp_train.cpu(),
        "Z_star_train": Z_star_train.cpu(),
        "H_t_val": H_t_val.cpu(),
        "H_tp_val": H_tp_val.cpu(),
        "Z_star_val": Z_star_val.cpu(),
        "meta": {
            "video": args.video, "ckpt": args.ckpt, "k": args.k,
            "train_start": args.train_start, "n_train": args.n_train, "stride_train": args.stride_train,
            "val_start": val_start, "n_val": args.n_val, "stride_val": args.stride_val,
            "block_gap": val_start - train_end, "train_end": train_end,
        },
    }, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
