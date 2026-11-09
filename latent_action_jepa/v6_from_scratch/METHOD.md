# V6 — Latent-Action JEPA, from-scratch method (math + verification)

Goal: a training METHOD that jointly learns (a) a video encoder from scratch,
(b) a continuous latent action code Z that explains transitions, (c) a
flow-matching prior over Z. No pretrained weights anywhere. Validated at small
scale; designed so nothing breaks when scaled.

---

## 0. Notation

Clip `x`: `T` frames, tokenized into tubelets (2 frames × 16×16 px).
Token grid: `d × h × w` (e.g. 8 × 14 × 14 for T=16, 224²). `P = h·w = 196`.

- `E_θ`   online encoder (ViT-3D, trained)
- `E_θ̄`  target encoder (EMA of θ, never gets gradient)
- `P_ρ`   JEPA predictor (narrow ViT, mask-token queries)
- `I_φ`   inverse (action tokenizer): pairs of states → Z
- `T_ψ`   transition (forward model): (state, Z) → next state
- `G_γ`   flow-grounding head: (state, Z) → coarse optical-flow field
- `v_ω`   flow-matching DiT over Z, conditioned on context

Temporal split: context tubelets `1..c`, future tubelets `c+1..d`.
States (all from the **target** encoder, stop-grad):

```
s_t      = sg( E_θ̄(x)[tubelet c] )         ∈ R^{P×D}     (last context tubelet)
s_{t+k}  = sg( E_θ̄(x)[tubelet c+k] )       ∈ R^{P×D}     k = 1..S future steps
```

---

## 1. Objective A — V-JEPA masked prediction (drives the encoder)

**This is the only loss that trains the encoder.** Everything else operates on
stop-gradded target features.

Sample a 3D multi-block mask `M` (spatiotemporal blocks, ~40-60% of tokens
masked — `masks/multiblock3d.py` already implements this). Visible set
`V = tokens \ M`.

```
H_V   = E_θ(x_V)                        # online sees ONLY visible tokens
Ĥ_M   = P_ρ(H_V, pos(M))                # predictor fills masked positions
H̄_M   = sg( E_θ̄(x) )[M]                 # target sees the FULL clip
L_A    = (1/|M|) Σ_{i∈M} || Ĥ_i − H̄_i ||₁
```

**Verification / what was wrong before:** v4/v5 used a *temporal-only* split
(context tubelets visible, future masked). That is a much weaker task than
V-JEPA's 3D multi-block masking — spatial infilling forces object/scene
structure; temporal-only extrapolation of slow features is nearly free, which
is one reason the v5 encoder solved its task at step 100 and learned little.
v6 restores true multi-block masking. Collapse is prevented the standard way:
EMA target + stop-grad + predictor asymmetry. Proven from scratch (V-JEPA).

## 2. Objective B — latent action bottleneck (defines Z)

**No learned base.** v4's `delta = H_target − StepBase(H_t)` was
unidentifiable: a learned base and Z trade information freely, so "action"
had no fixed meaning. v6 removes the base entirely:

```
z_k     = I_φ( s_t, s_{t+k} )            ∈ R^{d_z}, d_z small (16-32)
ŝ_{t+k} = T_ψ( s_t, z_k )
L_B     = (1/S) Σ_k || ŝ_{t+k} − s_{t+k} ||²      (targets already sg'd)
```

Identifiability argument (why Z becomes "the action" and not "the future"):

- `T_ψ(s_t, ·)` has full access to `s_t`, so everything *predictable* from
  the present is learned by T's weights.
- `z` is the only path carrying information about the *actual* future, and
  `dim(z) = 16 ≪ dim(s_{t+k}) = 196·384`. It physically cannot smuggle the
  whole future; the optimal use of 16 floats is the *residual mode choice* —
  i.e. which of the plausible futures happened. On agentic data that choice
  IS the action.
- Symmetric failure (T ignores z, "posterior collapse") happens exactly when
  futures are unimodal given the state — the v5 failure. This is a **data +
  horizon property**, not an architecture property, hence the launch gate in §6.

## 3. Objective C — composition consistency (makes Z an operator)

**Verification note — the naive cycle loss is vacuous.** I previously proposed
`z' = I(s', T(s', z)); L = ||z' − z||²`. That only trains `I` to invert `T`
— satisfiable for ANY Z semantics, including identity-leaking ones, because it
never touches real data at s'. Dropped.

The non-vacuous version uses real data at three timepoints (composition
through the transition):

```
z₁ = I_φ(s_t,     s_{t+k})
z₂ = I_φ(s_{t+k}, s_{t+2k})
L_C = || T_ψ( T_ψ(s_t, z₁), z₂ ) − s_{t+2k} ||²
```

Every term is grounded in observed states. To satisfy it, T's output for
(s, z) must be *usable as a state* — so Z must act as a composable operator,
not a one-shot residual code. This also stops T from ignoring its state input.

## 4. Objective D — classical-flow grounding (kills appearance leakage)

Optical flow computed with **Farneback / TV-L1** (classical, non-learned — no
pretrained weights involved). For each future step k, compute per-adjacent-
frame flow inside tubelet k, average, downsample to the token grid:

```
F_k ∈ R^{h×w×2}      (a 14×14 grid of mean displacement vectors)
F̂_k = G_γ( s_t, z_k )
L_D = (1/S) Σ_k Huber( F̂_k − F_k )
```

Rationale: flow fields contain zero appearance/identity information — grain,
color grading, scene layout style do not move pixels. Gradient through G
pushes z's content toward *motion*, directly attacking the 42×-chance
video-identity leakage measured in v4 (Prelinger) and v5 (Charades).

## 5. Objective E — flow-matching prior over Z (unchanged from v4/v5)

Rectified flow / conditional flow matching. This part was verified working in
v4 (goal-use curves) and was never the problem:

```
ε   ~ N(0, I),   τ ~ U(1e-3, 1−1e-3)
z_τ  = (1−τ)·ε + τ·sg(z)
v*   = sg(z) − ε                        # constant velocity along the path
L_E  = || v_ω(z_τ, τ, ctx(s_t)) − v* ||²
```

Inference: integrate `dz/dτ = v_ω(z_τ, τ, ctx)` from ε at τ=0 to τ=1 to
sample plausible actions given only the present. This is the planner/policy-
prior interface for robotics later.

## 6. Regularizers + gates

**VICReg on Z** (v5 dropped the covariance term — a bug; per-dim variance hit
1.0 while eff_rank collapsed to ~2, the signature of correlated dims):

```
L_var = (1/d_z) Σ_d max(0, 1 − √(Var_batch(z_d) + ε))
L_cov = (1/d_z) Σ_{i≠j} Cov(z)_{ij}²
```

**Launch gate (run BEFORE any long training):** train 300-500 steps, measure

```
U = 1 − L_B(z = z*) / L_B(z = shuffled-across-batch)
```

If `U < 0.2`, the futures are too predictable at this horizon on this data —
Z has nothing to encode. Abort, change data/horizon. (This one number would
have flagged v5_b at step 200 instead of after three 6k-step runs.)

**In-loop kill criteria (every 200 steps, on held-out clips):**
- `eff_rank(Z) < d_z/4` and falling → collapse, kill
- video-ID ridge-probe accuracy > 10× chance and rising → leakage, kill
- `U < 0.1` → Z unused, kill

## 7. Total loss & gradient topology

```
L = L_A + λ_B·L_B + λ_C·L_C + λ_D·L_D + λ_E·L_E + λ_v·L_var + λ_c·L_cov
```

Gradient flow (the critical design decision):

```
L_A               → E_θ, P_ρ          (ONLY loss touching the encoder)
L_B, L_C          → I_φ, T_ψ
L_D               → G_γ, I_φ          (through z)
L_E               → v_ω               (z is sg'd in the flow target)
regs              → I_φ
```

The action stack learns on top of sg(EMA features). Rationale: in v4/v5 the
action losses backpropagated into the encoder, letting encoder and shortcut
co-adapt (and letting the easy temporal task starve the encoder). Decoupling
means the encoder is driven purely by the proven V-JEPA objective, while the
action stack rides a consistent, improving feature stream. This also makes
the method modular — the identical action stack can later sit on any encoder.
(Ablation flag exists to re-enable λ_small backprop; default off.)

---

## 8. Dataset decision — BridgeData V2

Requirements: free direct download (no auth), agentic (static camera, one
agent whose choices dominate the future), scale enough for method validation,
and ideally ground-truth actions *for evaluation only*.

**BridgeData V2** (Berkeley, CC-BY): 60 496 robot-arm manipulation
trajectories, 24 environments, RGB 640×480 at **5 Hz**, direct HTTP download
from `rail.eecs.berkeley.edu/datasets/bridge_release/` — no registration.

Why it beats the alternatives:
- **Static third-person camera** → optical flow = object/arm motion only; the
  unpredictable residual IS the action. (Charades failed this: handheld shake.)
- **Ground-truth 7-DoF actions** ship with every trajectory. We never train on
  them, but `linear probe: Z → true EE-delta` is the definitive "is Z an
  action code?" metric — impossible on any web-video dataset.
- SSv2 needs a Qualcomm account (rejected earlier); Ego4D needs a license;
  Kinetics is link-rotted YouTube.

## 9. Horizon decision (how many frames apart)

BridgeV2 is recorded at 5 Hz (0.2 s/frame). With tubelet = 2 frames, one
future step k spans 0.4 s.

- 0.4 s at typical WidowX EE speeds ≈ several cm of arm travel → clearly
  visible in flow, clearly multimodal (which object, which direction).
- Clip layout: **T = 16 frames @ native 5 Hz (3.2 s), stride 1; context = 8
  frames (4 tubelets), future = 8 frames = S = 4 action steps of 0.4 s.**
- The composition loss uses (t, t+k, t+2k) = 0.8 s spans.

This is a hypothesis with an empirical check: after download, compute mean
EE displacement per Δt from the shipped actions and verify 0.4 s steps give
displacement ≫ tracking noise; the launch gate (§6) then confirms futures
diverge. If U < 0.2 at 8-frame context, halve the context to 4 frames first.

## 10. What v6 fixes vs v4/v5 (summary table)

| failure observed | cause | v6 fix |
|---|---|---|
| encoder solved task at step 100 | temporal-only masking, short horizon, slow features | true 3D multi-block masking (L_A) + agentic data + gate |
| Z collapsed (eff_rank 2/64) | no cov penalty + no signal left in task | restore VICReg cov + launch gate U |
| Z ↔ base unidentifiable | learned StepBase | base removed; T owns dynamics, Z owns choice |
| video-identity leakage 42× | appearance dominates residual; no grounding | flow-grounding head (L_D) + static-camera data |
| cycle loss would've been vacuous | I∘T=id satisfiable for any Z | composition loss on real (t, t+k, t+2k) triples |
| no honest metric during training | — | in-loop U, eff_rank, leakage probe with kill criteria; Z→true-action probe offline |
