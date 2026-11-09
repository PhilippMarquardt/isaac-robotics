"""E2 — z-swap portability for the V1-architecture LA-JEPA on per-frame ViT-S.

Tests (matched to F7a's pair structure):
  (a) Portability. T(H_t_i, Z_j). Retrieval against H_tp_i (own) vs H_tp_j (donor's).
  (b) Controllability. Same-Z across many H_t vs diff-Z on same H_t.
"""
import argparse
import json
import os
import random
import torch
import torch.nn.functional as F

from data.vjepa2_clips import IMAGENET_MEAN, IMAGENET_STD
from models.encoder import ViTEncoder
from models.fixed_reducer import FixedSpatialReducer
from models.inverse_encoder import InverseActionEncoder
from models.transition_residual import ResidualTransitionModel


def sample_pair_batch(cache, anchors, delta, device):
    xt, xtp = [], []
    for fi, a in anchors:
        frames = cache["frames"][fi]
        data = frames["data"] if isinstance(frames, dict) else frames
        xt.append(data[a]); xtp.append(data[a + delta])
    xt = torch.stack(xt, 0).to(device).float() / 255.0
    xtp = torch.stack(xtp, 0).to(device).float() / 255.0
    m = IMAGENET_MEAN.squeeze(0).to(device).view(1, 3, 1, 1)
    s = IMAGENET_STD.squeeze(0).to(device).view(1, 3, 1, 1)
    return (xt - m) / s, (xtp - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--la_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_donors", type=int, default=24)
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.la_ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    enc = ViTEncoder(a["encoder_name"], pretrained=True, layer_idx=None).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    state_dim = enc.embed_dim
    red_state = FixedSpatialReducer(a["state_tokens"]).to(device)
    red_future = FixedSpatialReducer(a["future_tokens"]).to(device)
    inv = InverseActionEncoder(state_dim, a["action_dim"], a["action_tokens"],
                                depth=a["inverse_depth"]).to(device).eval()
    tr = ResidualTransitionModel(state_dim, a["action_dim"], a["future_tokens"],
                                  depth=a["transition_depth"]).to(device).eval()
    inv.load_state_dict(ck["inv"]); tr.load_state_dict(ck["tr"])
    for p in list(inv.parameters()) + list(tr.parameters()): p.requires_grad_(False)
    print(f"loaded F7 teacher. alpha={tr.alpha.item():.3f}")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n_films = len(cache["film_names"])
    val_specs = []
    fids = []
    for fi in range(n_films):
        data = cache["frames"][fi]["data"] if isinstance(cache["frames"][fi], dict) else cache["frames"][fi]
        n_frames = data.shape[0]
        last_anchor = n_frames - a["delta"] - 1
        train_end = int(last_anchor * 0.9)
        val_start = train_end + a["delta"] + 16
        for av in range(val_start, last_anchor, a["anchor_stride"]):
            val_specs.append((fi, av)); fids.append(fi)
    N = len(val_specs)
    print(f"val N={N}")

    all_Ht, all_Htp, all_Z = [], [], []
    B_eval = 16
    with torch.no_grad():
        for s in range(0, N, B_eval):
            specs = val_specs[s:s + B_eval]
            xt, xtp = sample_pair_batch(cache, specs, a["delta"], device)
            Ht = red_state(enc(xt)); Htp = red_future(enc(xtp))
            Z = inv(Ht, Htp)
            all_Ht.append(Ht); all_Htp.append(Htp); all_Z.append(Z)
    Ht = torch.cat(all_Ht, 0); Htp = torch.cat(all_Htp, 0); Z = torch.cat(all_Z, 0)

    def retr(q, k, name):
        qn = F.normalize(q, dim=-1); kn = F.normalize(k, dim=-1)
        sim = qn @ kn.T
        r = sim.argsort(dim=-1, descending=True)
        gt = torch.arange(q.shape[0], device=device).unsqueeze(-1)
        t1 = (r[:, :1] == gt).any(-1).float().mean().item()
        t5 = (r[:, :5] == gt).any(-1).float().mean().item()
        mr = (r == gt).float().argmax(-1).float().mean().item() + 1
        print(f"  {name:36s} top1={t1:.4f}  top5={t5:.4f}  mr={mr:.1f}")
        return {"top1": t1, "top5": t5, "mr": mr}

    out = {"N": N}

    # ---- (a) portability ----
    print("\n=== (a) portability ===")
    g = torch.Generator(device=device).manual_seed(7)
    perm = torch.randperm(N, device=device, generator=g)
    perm = torch.where(perm == torch.arange(N, device=device), (perm + 1) % N, perm)
    with torch.no_grad():
        H_pred_correct = tr(Ht, Z)
        H_pred_swap = tr(Ht, Z[perm])
        out["correct_vs_own"] = retr(H_pred_correct.mean(1), Htp.mean(1), "T(H_t, Z) vs own H_tp")
        out["swap_vs_own"] = retr(H_pred_swap.mean(1), Htp.mean(1),       "T(H_t_i, Z_j) vs OWN H_tp_i")
        out["swap_vs_donor"] = retr(H_pred_swap.mean(1), Htp[perm].mean(1),
                                      "T(H_t_i, Z_j) vs DONOR H_tp_j")

    # ---- (b) controllability ----
    print("\n=== (b) controllability ===")
    K = args.n_donors
    n_states = min(64, N)
    state_idx = torch.randperm(N, device=device,
                                generator=torch.Generator(device=device).manual_seed(11))[:n_states]
    donor_idx = torch.randperm(N, device=device,
                                generator=torch.Generator(device=device).manual_seed(13))[:K]
    deltas = torch.zeros(K, n_states, a["future_tokens"], state_dim, device=device)
    with torch.no_grad():
        for di, d in enumerate(donor_idx):
            z = Z[d:d+1].expand(n_states, -1, -1)
            pred = tr(Ht[state_idx], z)
            base = red_future(Ht[state_idx])
            deltas[di] = pred - base
    df = F.normalize(deltas.reshape(K, n_states, -1), dim=-1)
    same_z = 0.0
    for di in range(K):
        sim = df[di] @ df[di].T
        same_z += (sim.sum() - torch.diag(sim).sum()).item() / (n_states * (n_states - 1))
    same_z /= K
    diff_z = 0.0
    for si in range(n_states):
        sim = df[:, si] @ df[:, si].T
        diff_z += (sim.sum() - torch.diag(sim).sum()).item() / (K * (K - 1))
    diff_z /= n_states
    sel = same_z / max(abs(diff_z), 1e-6)
    print(f"  same-Z, diff-H delta-cos = {same_z:+.4f}")
    print(f"  diff-Z, same-H delta-cos = {diff_z:+.4f}")
    print(f"  selectivity (same/diff)  = {sel:+.2f}x")
    out["controllability"] = {"same_z": same_z, "diff_z": diff_z, "selectivity": sel}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
