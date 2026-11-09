"""Cluster extracted components by spatial proximity so a multi-piece part
(e.g. a hand crank = handle + shaft + boss + retainer) is treated as ONE
real-world part.

Two components join the same cluster when their axis-aligned bounding boxes
come within `--eps_mm` of each other. Union-Find groups transitive matches.

Output: cluster JSON with one record per cluster — member paths, union bbox,
union centre, and the average of member principal axes (if augmented JSON is
provided).

Use:
    python cluster_components.py \\
        --tree outputs/<asset>_tree.json \\
        --augmented outputs/<asset>_aug.json \\
        --eps_mm 6 \\
        --out outputs/<asset>_clusters.json
"""
import argparse
import json
import math
import os


def bbox_dist(a, b):
    """Min axis-aligned distance between two AABBs (0 if they overlap)."""
    dx = max(b["min"][0] - a["max"][0], a["min"][0] - b["max"][0], 0.0)
    dy = max(b["min"][1] - a["max"][1], a["min"][1] - b["max"][1], 0.0)
    dz = max(b["min"][2] - a["max"][2], a["min"][2] - b["max"][2], 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i
    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[ri] = rj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--augmented", default=None)
    ap.add_argument("--eps_mm", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.tree) as f:
        tree = json.load(f)
    aug_by_path = {}
    if args.augmented and os.path.exists(args.augmented):
        with open(args.augmented) as f:
            for a in json.load(f).get("parts", []):
                aug_by_path[a["path"]] = a

    # Pull every Xform under default prim that has a bbox.
    parts = []
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") < 2:
            continue
        if "/part_" not in path and "drawer" not in path.lower():
            # Skip body shell.
            continue
        if not p.get("bbox_min"):
            continue
        parts.append({
            "path": path,
            "min": p["bbox_min"], "max": p["bbox_max"],
            "center": p["center"], "size": p["size"],
            "principal_axis": (aug_by_path.get(path, {}).get("principal_axis_label")
                               if aug_by_path else None),
            "nearest_face": (aug_by_path.get(path, {}).get("nearest_body_face")
                             if aug_by_path else None),
        })

    eps = args.eps_mm * 1e-3
    n = len(parts)
    uf = UnionFind(n)
    # O(n^2) bbox-distance test. n is ~150 max -> 22k checks, instant.
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_dist(parts[i], parts[j]) <= eps:
                uf.union(i, j)
                edges += 1
    print(f"[cluster] {n} parts, eps={eps:.4f} m -> {edges} edges")

    by_root = {}
    for i in range(n):
        r = uf.find(i)
        by_root.setdefault(r, []).append(i)

    clusters = []
    for root, idxs in by_root.items():
        members = [parts[i] for i in idxs]
        union_min = [min(m["min"][k] for m in members) for k in range(3)]
        union_max = [max(m["max"][k] for m in members) for k in range(3)]
        union_ctr = [(union_min[k] + union_max[k]) / 2 for k in range(3)]
        union_size = [union_max[k] - union_min[k] for k in range(3)]
        # Adopt the principal axis with the longest projection (= the largest dim member).
        principal = None
        face = None
        if members:
            # take the principal axis of the longest member
            members_sorted = sorted(members, key=lambda m: -max(m["size"]))
            principal = members_sorted[0].get("principal_axis")
            face = members_sorted[0].get("nearest_face")
        clusters.append({
            "cluster_id": f"cluster_{root:04d}",
            "members": [m["path"] for m in members],
            "n_members": len(members),
            "bbox_min": [round(x, 5) for x in union_min],
            "bbox_max": [round(x, 5) for x in union_max],
            "center": [round(x, 5) for x in union_ctr],
            "size": [round(x, 5) for x in union_size],
            "principal_axis_label": principal,
            "nearest_body_face": face,
        })
    # Sort largest cluster first so it's easier to eyeball.
    clusters.sort(key=lambda c: -c["n_members"])
    with open(args.out, "w") as f:
        json.dump({"eps_m": eps, "clusters": clusters}, f, indent=2)
    print(f"[cluster] wrote {args.out}  ({len(clusters)} clusters; "
          f"{sum(1 for c in clusters if c['n_members'] > 1)} multi-piece)")


if __name__ == "__main__":
    main()
