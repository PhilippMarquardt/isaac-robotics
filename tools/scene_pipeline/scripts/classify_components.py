"""Classify every extracted component into a physics behavior.

Input: the tree JSON produced by `inspect_usd.py` on a fully-split asset.
Output: per-part assignment of physics joint type + interaction effect, both
as JSON and a human-readable markdown report.

Heuristics are tuned for "household electronic / vintage appliance" assets
(cash register, boombox, microwave, etc.). For very different objects (a
hand tool, a vehicle), the rules need to be retuned.

Use:
    python classify_components.py \\
        --tree outputs/<asset>_components.json \\
        --object_kind cash_register \\
        --out_json outputs/<asset>_physics_plan.json \\
        --out_md   outputs/<asset>_physics_plan.md
"""
import argparse
import json
import os
import sys


def cube_like(size, tol=0.4):
    """True when the three dims are within `tol` of each other (cube-ish)."""
    mn, mx = min(size), max(size)
    if mx < 1e-6:
        return False
    return (mx - mn) / mx <= tol


def long_thin(size, ratio=3.0):
    """True when one axis dominates the other two by `ratio` or more."""
    s = sorted(size, reverse=True)
    return s[1] < 1e-6 or s[0] / max(s[1], 1e-6) >= ratio


def flat_thin(size, ratio=4.0):
    """True when one axis is much smaller than the other two (a wafer)."""
    s = sorted(size)
    if s[2] < 1e-6:
        return False
    return s[2] / max(s[0], 1e-6) >= ratio


def classify_cash_register(part, body_bbox):
    """Return (category, physics dict, interaction dict) for a single part."""
    sz = part["size"]
    ctr = part["center"]
    max_dim = max(sz)
    min_dim = min(sz)
    body_x = body_bbox["x"]   # (min, max)
    body_y = body_bbox["y"]
    body_z = body_bbox["z"]

    # noise filter: ignore very small or very thin parts
    if max_dim < 0.006:
        return "decorative_detail", None, None
    # decorative thin moulding / bezel
    if flat_thin(sz) and max_dim < 0.06:
        return "bezel_or_moulding", None, None

    # keypad key: cube-ish, ~19-22 mm, in front-top area
    if cube_like(sz) and 0.013 <= max_dim <= 0.030:
        in_top = ctr[2] > body_z[0] + 0.55 * (body_z[1] - body_z[0])
        in_front = ctr[1] < body_y[0] + 0.55 * (body_y[1] - body_y[0])
        if in_top or in_front:
            return "keypad_key", {
                "joint_type": "prismatic",
                "axis": "-Z",
                "lower_limit_m": -0.005,
                "upper_limit_m": 0.0,
                "drive": {"type": "force", "target": 0.0,
                          "stiffness": 800, "damping": 4, "max_force": 30},
                "mass_kg": 0.005,
                "collider": "convex_hull",
            }, {
                "on_press_threshold_frac": 0.6,
                "triggers": ["display_glow"],
                "primary_action": "register_keypress",
            }

    # side lever: long thin cylinder, near right/left X edge, mid Z
    if long_thin(sz) and 0.04 <= max_dim <= 0.15:
        near_x_edge = abs(ctr[0]) > body_x[1] - 0.15 if body_x[1] > 0 else False
        if near_x_edge or max_dim > 0.08:
            return "side_lever", {
                "joint_type": "revolute",
                "axis": "X",
                "lower_limit_deg": -45.0,
                "upper_limit_deg": 45.0,
                "drive": {"type": "force", "target": 0.0,
                          "stiffness": 200, "damping": 10, "max_force": 100},
                "mass_kg": 0.05,
                "collider": "convex_hull",
            }, {
                "on_swing_threshold_deg": 30,
                "triggers": ["cash_drawer", "display_glow"],
                "primary_action": "ring_up_sale",
            }

    # generic small interactive cube (uncategorised button)
    if cube_like(sz) and 0.020 <= max_dim <= 0.05:
        return "auxiliary_button", {
            "joint_type": "prismatic",
            "axis": "-Z",
            "lower_limit_m": -0.005,
            "upper_limit_m": 0.0,
            "drive": {"type": "force", "target": 0.0,
                      "stiffness": 800, "damping": 4, "max_force": 30},
            "mass_kg": 0.008,
            "collider": "convex_hull",
        }, {
            "on_press_threshold_frac": 0.6,
            "triggers": ["display_glow"],
            "primary_action": "aux_button_press",
        }

    # everything else: probably geometry detail
    return "uncategorised_detail", None, None


def classify(part, kind, body_bbox):
    if kind == "cash_register":
        return classify_cash_register(part, body_bbox)
    return "unknown_object_kind", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="inspect_usd.py JSON output")
    ap.add_argument("--object_kind", required=True,
                    choices=("cash_register",),
                    help="real-world type, drives the rule set")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)

    # Find body bbox to anchor relative reasoning. We assume the largest Xform
    # under the default prim that isn't a part_/display_/drawer_ extraction.
    body = None
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") == 2 and "part_" not in path and "drawer" not in path.lower():
            if p.get("size") and (body is None or max(p["size"]) > max(body["size"])):
                body = p
    if body is None:
        raise SystemExit("could not locate body Xform in tree")
    bmn = body["bbox_min"]
    bmx = body["bbox_max"]
    body_bbox = {
        "x": (bmn[0], bmx[0]),
        "y": (bmn[1], bmx[1]),
        "z": (bmn[2], bmx[2]),
    }

    parts = [p for p in tree["prims"]
             if p["type"] == "Xform" and "/part_" in p["path"]]
    parts.sort(key=lambda p: p["path"])

    plan = {"body": {"path": body["path"], "bbox_min": bmn, "bbox_max": bmx},
            "categories": {}, "parts": []}
    counts = {}
    for p in parts:
        cat, phys, inter = classify(p, args.object_kind, body_bbox)
        counts[cat] = counts.get(cat, 0) + 1
        rec = {
            "path": p["path"],
            "center": p.get("center"),
            "size": p.get("size"),
            "category": cat,
        }
        if phys:
            rec["physics"] = phys
        if inter:
            rec["interaction"] = inter
        plan["parts"].append(rec)
    plan["categories"] = counts

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"wrote {args.out_json}")

    # Markdown.
    md = []
    md.append(f"# Physics plan for `{args.object_kind}`\n")
    md.append(f"Body Xform: `{body['path']}`  size {body['size']}\n")
    md.append("## Component counts\n")
    md.append("| category | count |")
    md.append("|---|---|")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| `{cat}` | {n} |")
    md.append("")

    by_cat = {}
    for r in plan["parts"]:
        by_cat.setdefault(r["category"], []).append(r)

    md.append("## Per-category physics behavior\n")
    template = {
        "keypad_key":         ("Prismatic joint along world Z (downward 5 mm). "
                               "Spring drive returns to rest. **On press**: light up "
                               "the totalizer display; if the key represents the "
                               "TOTAL key, also trigger the side lever / drawer."),
        "auxiliary_button":   ("Prismatic joint along Z (downward 5 mm). Spring "
                               "return. **On press**: light the display."),
        "side_lever":         ("Revolute joint about world X, ±45°. Soft spring "
                               "(stiff ~200) so the robot can move it but it "
                               "doesn't snap. **On swing past ~30°**: drawer "
                               "pops out, display lights up — the canonical "
                               "vintage register `ring up` flow."),
        "bezel_or_moulding":  ("Static. No collider needed beyond the body's "
                               "(decorative skin)."),
        "decorative_detail":  ("Static. Likely a screw head, weld seam, edge "
                               "bevel. Skip — animating it would be wrong."),
        "uncategorised_detail": ("Currently treated as static. Re-inspect by hand "
                                 "if a real-world inventory item is missing."),
    }
    for cat, recs in by_cat.items():
        md.append(f"### `{cat}` — {len(recs)} parts")
        md.append(template.get(cat, "(no template)"))
        md.append("")
        md.append("Sample paths:")
        for r in recs[:3]:
            md.append(f"- `{r['path']}` size={r['size']} ctr={r['center']}")
        if len(recs) > 3:
            md.append(f"- ... +{len(recs) - 3} more\n")
        md.append("")

    md.append("## Interaction graph\n")
    md.append("```\n"
              "keypad_key  --press-->  display_glow  (every key)\n"
              "TOTAL key   --press-->  side_lever    (cashier rings up)\n"
              "side_lever  --swing-->  cash_drawer + display_glow\n"
              "auxiliary_button --press--> display_glow\n"
              "```\n")
    md.append("## Suggested top-level scene wiring\n")
    md.append("Wrap the fully-split USD; on top of every animatable part attach:\n")
    md.append("- `UsdPhysics.RigidBodyAPI` (dynamic for keys/lever, kinematic for body)")
    md.append("- `UsdPhysics.CollisionAPI` + `MeshCollisionAPI(approximation=convexHull)`")
    md.append("- the joint described per-category above, with body0 = lamp body.")
    md.append("- `DriveAPI` per the spring values in the JSON `physics.drive` block.\n")
    md.append("The runtime can either:\n")
    md.append("a. read the joint state and trigger downstream effects in python "
              "(no PhysX needed for the effect, only for the motion), or\n")
    md.append("b. let PhysX handle everything if the downstream effect is also a "
              "physical body (e.g. the lever physically pulling the drawer "
              "through a linkage).\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
