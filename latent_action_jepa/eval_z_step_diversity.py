"""C1 diagnostic: cross-step Z cosine on a LA-JEPA ckpt.

Loads (inverse, transition) from a la_jepa ckpt and the frozen V-JEPA-2 encoder
to produce Z* on val anchors. Reports:
  * mean cosine cos(Z_i, Z_j) for i != j   (across 8 step axis, per-sample then averaged)
  * mean diagonal cosine cos(Z_i, Z_i)     (should be 1)
  * per-step std of Z across batch
  * effective rank of the [N*S, action_dim] Z matrix
"""
import argparse
import json
import os
import random
import torch
import torch.nn.functional as F

from data.vjepa2_clips import sample_clip_batch
from models.vit3d import VisionTransformer3D
from models.la_modules import InverseActionMultiStep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vjepa_ckpt", required=True)
    ap.add_argument("--la_ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    vjepa = torch.load(args.vjepa_ckpt, map_location=device, weights_only=False)
    va = vjepa["args"]
    enc = VisionTransformer3D(
        num_frames=va["num_frames"], tubelet_size=va["tubelet_size"],
        patch_size=va["patch_size"], img_size=va["img_size"],
        embed_dim=va["embed_dim"], depth=va["depth"], num_heads=va["num_heads"],
        mlp_ratio=4.0,
    ).to(device).eval()
    enc.load_state_dict(vjepa["target"])
    for p in enc.parameters(): p.requires_grad_(False)
    grid_d = va["num_frames"] // va["tubelet_size"]
    grid_s = va["img_size"] // va["patch_size"]
    n_patch = grid_s * grid_s
    n_tokens = grid_d * n_patch
    state_dim = va["embed_dim"]

    la = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    la_args = la["args"]
    inv = InverseActionMultiStep(state_dim=state_dim, action_dim=la_args["action_dim"],
                                  n_steps=la_args["action_tokens"], n_patch=n_patch,
                                  depth=la_args["inverse_depth"]).to(device).eval()
    inv.load_state_dict(la["inv"])
    for p in inv.parameters(): p.requires_grad_(False)
    S = la_args["action_tokens"]; A = la_args["action_dim"]

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    ctx_span = (la_args["num_frames"] - 1) * la_args["frame_stride"]
    fut_span = ctx_span
    fo = la_args["future_offset"]
    anchor_stride = la_args["anchor_stride"]
    val_specs = []
    for fi in range(n_films):
        n_frames_film = cache["frames"][fi].shape[0]
        a_min = ctx_span
        a_max = n_frames_film - fo - fut_span - 1
        train_end = int(a_max * 0.9)
        val_start = train_end + ctx_span + fut_span + fo
        for a in range(val_start, a_max, anchor_stride):
            val_specs.append((fi, a))
    N_va = len(val_specs)
    print(f"val anchors N={N_va}  S={S}  A={A}")

    all_Z = []
    B_eval = 4
    with torch.no_grad():
        for s in range(0, N_va, B_eval):
            batch = val_specs[s:s+B_eval]
            ctx_clip, _ = sample_clip_batch(cache, [(fi, a - ctx_span) for (fi, a) in batch],
                                            la_args["num_frames"], la_args["frame_stride"], device)
            fut_clip, _ = sample_clip_batch(cache, [(fi, a + fo) for (fi, a) in batch],
                                            la_args["num_frames"], la_args["frame_stride"], device)
            H_ctx = enc(ctx_clip)
            H_fut = enc(fut_clip)
            H_t = H_ctx[:, -n_patch:, :]
            H_targets = H_fut[:, : S * n_patch, :].reshape(-1, S, n_patch, state_dim)
            Z = inv(H_t, H_targets)
            all_Z.append(Z)
    Z = torch.cat(all_Z, 0)  # [N, S, A]
    print(f"Z shape {tuple(Z.shape)}")

    # cross-step cosine per sample
    Zn = F.normalize(Z, dim=-1)
    cos_matrix = torch.einsum("nsa,nta->nst", Zn, Zn)  # [N, S, S]
    eye = torch.eye(S, device=cos_matrix.device).bool()
    off_mask = ~eye
    # per-sample mean off-diagonal
    off_cos = cos_matrix[:, off_mask].reshape(Z.shape[0], -1).mean(-1)  # [N]
    # per-pair (i,j) averaged across samples
    pair_cos = cos_matrix.mean(0)  # [S, S]
    print("per-pair cos(Z_i, Z_j) averaged across samples:")
    print(pair_cos.cpu().numpy().round(3))

    # std per step over the batch
    std_per_step = Z.std(dim=0).mean(dim=-1)  # [S]
    print(f"per-step std (mean over A): {std_per_step.cpu().tolist()}")

    # effective rank treating each (sample,step) as a token
    flat = Z.reshape(-1, A)
    flat = flat - flat.mean(0, keepdim=True)
    svals = torch.linalg.svdvals(flat.float())
    p = (svals ** 2) / (svals ** 2).sum().clamp_min(1e-12)
    H = -(p * (p.clamp_min(1e-12).log())).sum()
    eff_rank = torch.exp(H).item()
    print(f"effective rank of [N*S, A]: {eff_rank:.2f}  (max possible {A})")

    # also overall pairwise across (N*S) tokens
    off_diag_cos_overall = ((flat / (flat.norm(dim=-1, keepdim=True) + 1e-9))
                            @ (flat / (flat.norm(dim=-1, keepdim=True) + 1e-9)).T)
    # remove diagonal
    M = off_diag_cos_overall.shape[0]
    overall_mean = (off_diag_cos_overall.sum() - torch.diag(off_diag_cos_overall).sum()) / (M * (M - 1))
    print(f"overall pairwise cos across all (sample, step) tokens: {overall_mean.item():+.4f}")

    out = {
        "N": N_va, "S": S, "A": A,
        "mean_off_step_cos_per_sample": off_cos.mean().item(),
        "per_step_std_mean": std_per_step.mean().item(),
        "per_step_std": std_per_step.cpu().tolist(),
        "pair_cos_matrix": pair_cos.cpu().tolist(),
        "effective_rank_flat": eff_rank,
        "overall_pairwise_cos": overall_mean.item(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
