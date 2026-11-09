"""LLM-driven component classifier.

Replaces the hardcoded heuristics in `classify_components.py` with a call to
an OpenAI model. The LLM gets:
  - object kind ("vintage cash register", "industrial pipe lamp", ...)
  - body bbox
  - per-component (id, size, world_center) records

and returns a JSON list of classifications, each with the real-world part
name, the physics joint that should drive it in Isaac Lab, and the
interaction effect.

The deterministic classifier sees only numbers and misses parts whose
identity depends on real-world context (a 43 × 4 × 1 mm flat bar near the
top-right of a vintage register IS a side hand crank, even though the
geometry alone doesn't say so). An LLM brings that context.

Use:
    python classify_with_llm.py \\
        --tree outputs/<asset>_tree.json \\
        --object_kind "vintage hand-crank cash register" \\
        --out_json outputs/<asset>_physics_plan.json \\
        --out_md   outputs/<asset>_physics_plan.md
"""
import argparse
import json
import os
import sys
from dotenv import load_dotenv

# Load .env from scene_pipeline root (one level up from scripts/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "..", ".env"))


def describe_cluster(c, body_min, body_max, others) -> str:
    """Return a rich human-readable description of one cluster: where it sits,
    what shape it has, how it relates to neighbours. The LLM reads THIS rather
    than raw JSON numbers."""
    cx, cy, cz = c["center"]
    sx, sy, sz = c["size"]
    bmn = body_min; bmx = body_max

    # Position relative to body, in fractions 0..1 along each axis.
    def frac(v, lo, hi):
        return (v - lo) / max(hi - lo, 1e-9)
    fx = frac(cx, bmn[0], bmx[0])
    fy = frac(cy, bmn[1], bmx[1])
    fz = frac(cz, bmn[2], bmx[2])

    horiz = ("left" if fx < 0.33 else ("right" if fx > 0.66 else "centre-X"))
    depth = ("front" if fy < 0.33 else ("back" if fy > 0.66 else "centre-Y"))
    vert  = ("bottom" if fz < 0.33 else ("top" if fz > 0.66 else "middle-Z"))

    # Shape descriptor.
    max_dim = max(sx, sy, sz)
    min_dim = min(sx, sy, sz) or 1e-9
    mid_dim = sorted([sx, sy, sz])[1]
    ratio = max_dim / min_dim
    if ratio > 4:
        shape = f"thin rod or strip {max_dim*1000:.0f} mm long"
    elif max_dim / mid_dim > 3:
        shape = f"thin tab {max_dim*1000:.0f} × {mid_dim*1000:.0f} mm"
    elif min_dim / max_dim > 0.7:
        shape = f"cube-like {max_dim*1000:.0f} mm"
    else:
        shape = f"slab {sx*1000:.0f} × {sy*1000:.0f} × {sz*1000:.0f} mm"

    face = c.get("nearest_body_face", "?")
    paxis = c.get("principal_axis_label", "?")

    # Nearest neighbours (by centre distance).
    me = (cx, cy, cz)
    dists = []
    for o in others:
        if o["cluster_id"] == c["cluster_id"]:
            continue
        ocx, ocy, ocz = o["center"]
        d = ((cx-ocx)**2 + (cy-ocy)**2 + (cz-ocz)**2) ** 0.5
        dists.append((d, o))
    dists.sort()
    near = []
    for d, o in dists[:3]:
        near.append(f"{o['cluster_id']} at {d*1000:.0f} mm ({max(o['size'])*1000:.0f} mm)")
    near_txt = "; ".join(near) if near else "(none nearby)"

    lines = [
        f"{c['cluster_id']} — {c['n_members']} piece(s); {shape}; principal axis {paxis};",
        f"  position: {vert} / {depth} / {horiz}  (rel-fracs x={fx:.2f}, y={fy:.2f}, z={fz:.2f})",
        f"  nearest body face: {face}",
        f"  nearest clusters: {near_txt}",
    ]
    if c["n_members"] > 1:
        lines.append(f"  member paths: {', '.join(c['members'][:4])}"
                     + ("" if len(c['members']) <= 4 else f", … +{len(c['members'])-4} more"))
    return "\n".join(lines)


def detect_grid(clusters):
    """Rough heuristic: detect groups of clusters with similar size arranged in
    a regular row/grid. Returns a list of group descriptions."""
    if len(clusters) < 6:
        return []
    # Bucket by similar max-dimension (10% tolerance).
    by_size = {}
    for c in clusters:
        md = max(c["size"])
        if md < 0.005:
            continue
        bucket = round(md * 1000)  # mm
        # join nearby buckets
        for b in list(by_size):
            if abs(b - bucket) <= 1:
                by_size.setdefault(b, []).append(c)
                break
        else:
            by_size[bucket] = [c]
    groups = []
    for size_mm, members in by_size.items():
        if len(members) >= 6:
            xs = [m["center"][0] for m in members]
            ys = [m["center"][1] for m in members]
            zs = [m["center"][2] for m in members]
            groups.append(
                f"GRID: {len(members)} clusters of ~{size_mm} mm "
                f"spanning x=[{min(xs)*100:.1f}..{max(xs)*100:.1f}cm] "
                f"y=[{min(ys)*100:.1f}..{max(ys)*100:.1f}cm] "
                f"z=[{min(zs)*100:.1f}..{max(zs)*100:.1f}cm]  "
                f"(likely a button/key bank — classify all uniformly)"
            )
    return groups


def call_llm(object_kind: str, body_bbox: dict, parts: list, model: str = None,
             timeout_s: float = 120, screenshot_b64: str = None,
             text_payload: str = None) -> list:
    """Ask the LLM to classify every component. Returns list of dicts."""
    from openai import OpenAI
    client = OpenAI(timeout=timeout_s)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system_prompt = """You are a robotics simulation expert classifying components of a 3D asset for Isaac Lab physics.

You will receive the prim tree of a single 3D asset, the real-world identity of the asset (e.g. "vintage hand-crank cash register"), and PER-PART ORIENTATION CUES:
  - principal_axis_label: signed cardinal axis of the part's PCA top eigenvector
        (the part's "long axis" in world space, e.g. "+X" = the part is mostly oriented along world X).
  - elongation: ratio of largest to second PCA eigenvalue. >>1 means the part is a stick/needle/rod;
        ~1 means it is roughly cube-like; intermediate values are slabs.
  - flatness: ratio of second to smallest PCA eigenvalue. >>1 means the part is a flat plate/wafer.
  - nearest_body_face: which face of the body bbox this part sits on (+X,-X,+Y,-Y,+Z,-Z).
  - distance_to_face_m: how close the part centre is to that face.
  - on_face: true when the part touches that face within ~1 cm.

Use these to choose physics axes correctly. Cheat sheet:

  | Part type                                    | joint axis follows...                                       |
  |----------------------------------------------|-------------------------------------------------------------|
  | Press button / key on top face (+Z)          | -Z (downward)                                               |
  | Press button / key on front face (-Y)        | +Y (inward) — i.e. push goes the opposite direction of face |
  | Drawer that pulls out the front face         | the face normal                                             |
  | Door swinging open about a vertical hinge    | +Z (vertical)                                               |
  | Hand crank / knob extending out a side face  | the face normal of the side it extends from                 |
  | Crank arm hinged about a horizontal pin      | the pin direction (use principal_axis_label of the pin)     |
  | Lever / handle being pulled                  | the perpendicular to the part's principal_axis              |
  | Rocker / toggle switch                       | the perpendicular to the part's principal_axis              |

For EVERY component, output a JSON object with:
- "id": the component path (copy verbatim)
- "real_world_part": short human label (e.g. "keypad key #7 (TOTAL)", "side hand crank", "bezel trim", "screw")
- "category": one of: keypad_key, action_button, total_key, hand_crank_lever, swing_door, drawer, switch_toggle, knob_dial, latch, display_panel, indicator_light, hinge_pin, bezel_or_moulding, screw_or_fastener, decorative_detail, body_shell, uncategorised
- "physics": one of:
    - null  (purely static decoration)
    - {"joint_type": "prismatic", "axis": one of [+X,-X,+Y,-Y,+Z,-Z], "lower_m": float, "upper_m": float, "mass_kg": float, "spring_k_N_per_m": float, "damping": float}
    - {"joint_type": "revolute",  "axis": one of [+X,-X,+Y,-Y,+Z,-Z], "lower_deg": float, "upper_deg": float, "mass_kg": float, "spring_k_Nm_per_rad": float, "damping": float}
    - {"joint_type": "fixed", "mass_kg": float}
- "interaction": null OR {"primary_action": str, "trigger_threshold_frac": float, "triggers": [paths]}
- "notes": one short sentence — INCLUDE which orientation cue drove your axis choice.

Hard rules:
- A part with elongation > 4 and on_face=true is almost always a knob/crank/lever sticking out of the body. Its rotation axis equals its principal_axis_label (it spins about its own shaft).
- A part on the top face with elongation ≈ 1 is a key/button. Prismatic axis = -Z, 4-6 mm travel, 5 g, spring 800 N/m.
- A part on the front face (typically -Y in this dataset) and flat (flatness > 3, max_dim > 5 cm) is a drawer. Prismatic axis = face normal (e.g. -Y), travel 100-250 mm.
- Tiny cubes max_dim < 8 mm and not at a face edge → screw_or_fastener, physics null.
- Thin flat strips with flatness > 5 and max_dim < 6 cm and NOT on a face the part should hinge from → bezel_or_moulding, physics null.

Return STRICT JSON: a single object with key "components" mapping to a list. Nothing else."""

    user_payload = {
        "object_kind": object_kind,
        "body_bbox": body_bbox,
        "components": parts,
    }

    print(f"[llm] model={model}, {len(parts)} components, calling API...", flush=True)
    if text_payload:
        user_text = (
            f"Object kind: {object_kind}\n"
            f"Body bbox size (m): {body_bbox.get('size')}\n\n"
            f"{text_payload}\n\n"
            f"Component IDs to classify (one entry per id): "
            f"{', '.join(p['id'] for p in parts)}\n"
            f"Output one JSON object per id."
        )
    else:
        user_text = json.dumps(user_payload)
    if screenshot_b64:
        user_msg_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ]
    else:
        user_msg_content = user_text
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt
                + "\n\nIMPORTANT: respond with ONLY a JSON object, no surrounding prose, "
                  "no markdown fences. The single top-level key must be \"components\"."},
            {"role": "user", "content": user_msg_content},
        ],
    )
    content = resp.choices[0].message.content
    print(f"[llm] response: {len(content)} chars", flush=True)
    # Tolerate ```json fences and surrounding prose.
    c = content.strip()
    if c.startswith("```"):
        c = c.split("```", 2)[1]
        if c.lower().startswith("json"):
            c = c[4:]
        c = c.strip()
    # First {...} block.
    first = c.find("{")
    last = c.rfind("}")
    if first >= 0 and last > first:
        c = c[first:last + 1]
    parsed = json.loads(c)
    comps = parsed.get("components", parsed)
    if not isinstance(comps, list):
        raise SystemExit(f"unexpected response shape: {type(comps)}")
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--augmented", default=None,
                    help="optional augment JSON from augment_components.py")
    ap.add_argument("--clusters", default=None,
                    help="optional cluster JSON from cluster_components.py — "
                         "when provided, the LLM classifies CLUSTERS (multi-piece "
                         "parts) instead of raw components")
    ap.add_argument("--screenshot", default=None,
                    help="optional PNG screenshot of the asset (vision input)")
    ap.add_argument("--object_kind", required=True,
                    help="real-world identity of the asset (free text)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--max_components", type=int, default=300,
                    help="if the tree has more parts than this, sample evenly")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (put it in scene_pipeline/.env)")

    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)

    # Find body Xform (largest top-level Xform that isn't a part_ extraction).
    body = None
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") == 2 and "/part_" not in path:
            if p.get("size") and (body is None or max(p["size"]) > max(body.get("size", [0, 0, 0]))):
                body = p
    if body is None:
        raise SystemExit("could not locate body Xform")
    body_bbox = {
        "min": body["bbox_min"], "max": body["bbox_max"],
        "center": body["center"], "size": body["size"],
    }

    parts = [p for p in tree["prims"]
             if p["type"] == "Xform" and "/part_" in p["path"]]
    # Include drawer / display / other top-level extracted Xforms too.
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") == 2 and p is not body and not any(pp["path"] == path for pp in parts):
            parts.append(p)

    aug_by_path = {}
    if args.augmented and os.path.exists(args.augmented):
        with open(args.augmented, "r", encoding="utf-8") as f:
            aug = json.load(f)
        for a in aug.get("parts", []):
            aug_by_path[a["path"]] = a
        print(f"[plan] loaded augmented data for {len(aug_by_path)} parts")

    clusters = None
    if args.clusters and os.path.exists(args.clusters):
        with open(args.clusters, "r", encoding="utf-8") as f:
            clusters = json.load(f).get("clusters", [])
        print(f"[plan] loaded {len(clusters)} clusters; sending those to LLM")

    screenshot_b64 = None
    if args.screenshot and os.path.exists(args.screenshot):
        import base64
        with open(args.screenshot, "rb") as f:
            screenshot_b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"[plan] attaching screenshot {args.screenshot} ({len(screenshot_b64)} chars b64)")

    if clusters:
        compact = []
        for c in clusters:
            rec = {
                "id": c["cluster_id"],
                "n_pieces": c["n_members"],
                "members": c["members"],
                "size": [round(x, 5) for x in c["size"]],
                "center": [round(x, 5) for x in c["center"]],
                "principal_axis_label": c.get("principal_axis_label"),
                "nearest_body_face": c.get("nearest_body_face"),
            }
            compact.append(rec)
        print(f"[plan] body={body['path']}  clusters={len(clusters)} (sending all)")
    else:
        compact = []
        for p in parts[: args.max_components]:
            rec = {
                "id": p["path"],
                "size": [round(x, 5) for x in p["size"]],
                "center": [round(x, 5) for x in p["center"]],
            }
            a = aug_by_path.get(p["path"])
            if a:
                rec["principal_axis_label"] = a.get("principal_axis_label")
                rec["elongation"] = a.get("elongation")
                rec["flatness"] = a.get("flatness")
                rec["nearest_body_face"] = a.get("nearest_body_face")
                rec["distance_to_face_m"] = a.get("distance_to_face_m")
                rec["on_face"] = a.get("on_face")
            compact.append(rec)
        print(f"[plan] body={body['path']}  parts={len(parts)} (sending {len(compact)})")

    # Build the rich text payload (cluster-mode only).
    overview_text = None
    if clusters:
        all_clusters = clusters  # full set for neighbour computation
        per_cluster_text = [describe_cluster(c, body["bbox_min"], body["bbox_max"], all_clusters)
                            for c in clusters]
        grid_text = detect_grid(clusters)
        overview_text = (
            "ASSET OVERVIEW\n"
            f"Body size (m): {body['size']}\n"
            f"Clusters: {len(clusters)}  (multi-piece: {sum(1 for c in clusters if c['n_members']>1)})\n\n"
            "GRID HINTS\n"
            + ("\n".join(grid_text) if grid_text else "(no obvious grid detected)\n")
            + "\n\nCLUSTER DETAILS\n"
            + "\n".join(per_cluster_text)
        )

    # Batch: gpt-5.4 drops the connection above ~80 records per call.
    BATCH = 50
    classifications = []
    for i in range(0, len(compact), BATCH):
        chunk = compact[i:i + BATCH]
        print(f"[plan] batch {i//BATCH + 1}: {len(chunk)} records")
        cls = call_llm(args.object_kind,
                       {"size": body["size"], "center": body["center"]},
                       chunk, screenshot_b64=screenshot_b64,
                       text_payload=overview_text if i == 0 else None)
        classifications.extend(cls)

    by_id = {c["id"]: c for c in classifications if "id" in c}
    plan = {
        "object_kind": args.object_kind,
        "body": {"path": body["path"], "bbox_min": body["bbox_min"],
                 "bbox_max": body["bbox_max"]},
        "parts": [],
    }
    counts = {}

    if clusters:
        # The LLM classified clusters; emit one plan entry per cluster, listing
        # all member component paths in `member_paths`. Downstream
        # spec generator treats this as one animated unit.
        cluster_by_id = {c["cluster_id"]: c for c in clusters}
        for cid, c in cluster_by_id.items():
            cls = by_id.get(cid, {})
            rec = {
                "cluster_id": cid,
                "member_paths": c["members"],
                "size": c["size"],
                "center": c["center"],
                "bbox_min": c["bbox_min"],
                "bbox_max": c["bbox_max"],
                "principal_axis_label": c.get("principal_axis_label"),
                "nearest_body_face": c.get("nearest_body_face"),
            }
            rec.update({k: v for k, v in cls.items() if k not in ("id",)})
            cat = rec.get("category", "unclassified")
            counts[cat] = counts.get(cat, 0) + 1
            plan["parts"].append(rec)
    else:
        for p in parts:
            rec = {"path": p["path"], "size": p["size"], "center": p["center"]}
            c = by_id.get(p["path"])
            if c:
                rec.update({k: v for k, v in c.items() if k != "id"})
                counts[c.get("category", "unknown")] = counts.get(c.get("category", "unknown"), 0) + 1
            else:
                rec["category"] = "unclassified"
                counts["unclassified"] = counts.get("unclassified", 0) + 1
            plan["parts"].append(rec)
    plan["category_counts"] = counts

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"wrote {args.out_json}")

    # Markdown.
    md = [
        f"# LLM physics plan — `{args.object_kind}`\n",
        f"Body: `{body['path']}` size {body['size']}\n",
        "## Category counts\n",
        "| category | count |",
        "|---|---|",
    ]
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| `{cat}` | {n} |")
    md.append("")
    md.append("## Per-part assignments\n")
    for rec in plan["parts"]:
        title = rec.get("path") or rec.get("cluster_id") or "(unknown)"
        md.append(f"### `{title}`")
        if rec.get("member_paths"):
            md.append(f"- members ({len(rec['member_paths'])}): "
                      f"{', '.join('`' + p + '`' for p in rec['member_paths'][:4])}"
                      + ("" if len(rec['member_paths']) <= 4 else f", … +{len(rec['member_paths'])-4} more"))
        md.append(f"- real_world_part: {rec.get('real_world_part', '?')}")
        md.append(f"- category: `{rec.get('category', '?')}`")
        md.append(f"- size: {rec.get('size')}  center: {rec.get('center')}")
        phys = rec.get("physics")
        if phys:
            md.append(f"- physics: `{json.dumps(phys)}`")
        else:
            md.append("- physics: static")
        inter = rec.get("interaction")
        if inter:
            md.append(f"- interaction: `{json.dumps(inter)}`")
        if rec.get("notes"):
            md.append(f"- notes: {rec['notes']}")
        md.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
