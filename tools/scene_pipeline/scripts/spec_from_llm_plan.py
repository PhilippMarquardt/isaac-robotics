"""Turn an LLM physics-plan JSON into a make_interactive.py spec.

Reads the per-part records produced by `classify_with_llm.py`, drops the
static parts (bezels, screws, decorative), turns the rest into `animated_part`
and `light` entries, and wires up trigger/light links by matching the LLM's
interaction.triggers paths to part names.

Use:
    python spec_from_llm_plan.py \\
        --plan outputs/<asset>_llm_plan.json \\
        --asset_rel ../assets/source/<asset>/<asset>_split.usd \\
        --out_rel  ../assets/usd/<asset>_v3.usd \\
        --spec spec/<asset>_full.json
"""
import argparse
import json
import os
from collections import defaultdict


STATIC_CATS = {
    "bezel_or_moulding", "screw_or_fastener", "decorative_detail",
    "hinge_pin", "body_shell", "uncategorised", "unclassified",
}
LIGHT_CATS = {"display_panel", "indicator_light"}


def axis_decode(axis_str):
    """Return (sign, lowercase_axis)."""
    s = (axis_str or "+Z").strip().upper()
    sign = -1 if s.startswith("-") else 1
    ax = s.lstrip("+-").lower()
    if ax not in ("x", "y", "z"):
        ax = "z"
    return sign, ax


def motion_amplitude(phys):
    """Return (motion_str, amplitude_value) for the make_interactive spec."""
    jt = phys.get("joint_type", "prismatic")
    sign, ax = axis_decode(phys.get("axis"))
    if jt == "prismatic":
        lo = float(phys.get("lower_m", 0.0))
        hi = float(phys.get("upper_m", 0.005))
        travel_m = abs(hi - lo)
        # Translate amplitude is in mm in the spec.
        amp_mm = sign * travel_m * 1000.0
        return f"translate_{ax}", amp_mm
    if jt == "revolute":
        lo = float(phys.get("lower_deg", -22.0))
        hi = float(phys.get("upper_deg", 22.0))
        amp_deg = sign * (abs(hi - lo) / 2.0)
        return f"rotate_{ax}", amp_deg
    return "translate_z", sign * 5.0


def slugify(text):
    return "".join(c.lower() if c.isalnum() else "_" for c in (text or "part")).strip("_")[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--asset_rel", required=True,
                    help="relative path FROM the spec file TO the source USD")
    ap.add_argument("--out_rel", required=True,
                    help="relative path FROM the spec file TO the wrapper USD output")
    ap.add_argument("--spec", required=True, help="path to write the spec JSON")
    args = ap.parse_args()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # 1) Assign a stable name to every animateable + light entry.
    # In cluster-mode the plan record carries `cluster_id` + `member_paths`;
    # in single-component mode each record carries `path`.
    name_for = {}        # cluster_id or part_path -> chosen unique name
    name_counts = defaultdict(int)
    def key_of(p):
        return p.get("cluster_id") or p.get("path")
    for p in plan["parts"]:
        cat = p.get("category", "")
        if cat in STATIC_CATS:
            continue
        base = slugify(p.get("real_world_part") or cat)
        name_counts[base] += 1
        name = f"{base}_{name_counts[base]:02d}" if name_counts[base] > 1 else base
        if not name:
            name = f"part_{len(name_for):03d}"
        name_for[key_of(p)] = name

    # 2) Build interactives.
    interactives = []
    inventory = []
    for p in plan["parts"]:
        cat = p.get("category", "")
        if cat in STATIC_CATS:
            continue
        k = key_of(p)
        name = name_for[k]
        member_paths = p.get("member_paths") or [p.get("path")]
        primary_prim = member_paths[0]
        if cat in LIGHT_CATS:
            entry = {
                "kind": "light",
                "name": name,
                "head_prim": primary_prim,
                "on_intensity": 1500.0 if cat == "display_panel" else 800.0,
                "on_color": [1.0, 0.55, 0.25] if cat == "display_panel" else [1.0, 0.3, 0.2],
            }
            interactives.append(entry)
            inventory.append(f"LIGHT  {name:32}  {primary_prim}  ({cat})")
            continue
        phys = p.get("physics")
        if not phys or phys.get("joint_type") == "fixed":
            inventory.append(f"SKIP   {name:32}  {primary_prim}  (no movable physics)")
            continue
        motion, amplitude = motion_amplitude(phys)
        entry = {
            "kind": "animated_part",
            "name": name,
            "prim": primary_prim,
            "motion": motion,
            "amplitude": amplitude,
        }
        if len(member_paths) > 1:
            entry["prim_list"] = member_paths
        # Per-part pivot center if available (cluster mode supplies it).
        if p.get("center"):
            entry["world_center"] = p["center"]
        # Resolve triggers — LLM may name cluster ids OR part paths.
        inter = p.get("interaction") or {}
        trig_paths = inter.get("triggers") or []
        trig_names = [name_for[t] for t in trig_paths if t in name_for and name_for[t] != name]
        light_names = {ie["name"] for ie in interactives if ie["kind"] == "light"}
        for t in trig_names:
            if t in light_names and "linked_light" not in entry:
                entry["linked_light"] = t
                continue
            if "triggers" not in entry:
                entry["triggers"] = t
        interactives.append(entry)
        members_str = f" (+{len(member_paths)-1} more)" if len(member_paths) > 1 else ""
        inventory.append(f"MOVE   {name:32}  {primary_prim}{members_str}  motion={motion}  amp={amplitude:.2f}  trig={trig_names}")

    out = {
        "_object_kind": plan.get("object_kind", ""),
        "_generated_from": os.path.basename(args.plan),
        "asset": args.asset_rel,
        "out":   args.out_rel,
        "interactives": interactives,
    }
    os.makedirs(os.path.dirname(args.spec) or ".", exist_ok=True)
    with open(args.spec, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.spec}  ({len(interactives)} interactives)")

    # Inventory log.
    inv_path = args.spec + ".inventory.txt"
    with open(inv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(inventory))
    print(f"wrote {inv_path}")


if __name__ == "__main__":
    main()
