"""Analyze a single baked mesh and list candidate interactive sub-parts.

Splits the mesh's faces into connected components by shared edges. Reports
each component's face count and world-space bbox so the LLM (or human) can
decide which look like keys, dials, levers, buttons, lids, etc.

Use:
    isaaclab.bat -p scripts/propose_interactives.py \\
        --usd .../asset.usdc \\
        --mesh /asset/body/body_mesh \\
        --out_json outputs/<asset>_components.json
"""
import argparse
import json
import os
from collections import defaultdict, deque
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--mesh", required=True, help="prim path of the mesh to analyze")
parser.add_argument("--out_json", default=None)
parser.add_argument("--min_faces", type=int, default=4,
                    help="ignore components smaller than this many faces (noise)")
parser.add_argument("--max_components", type=int, default=40,
                    help="report at most N components (largest dropped first by default)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Gf


def main():
    stage = Usd.Stage.Open(args.usd)
    mesh_prim = stage.GetPrimAtPath(args.mesh)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise SystemExit(f"mesh not found at {args.mesh}")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        raise SystemExit("mesh has no geometry data")

    n_faces = len(counts)
    print(f"[propose] mesh has {n_faces} faces, {len(points)} verts")

    # Build face index list per vertex.
    cursors = [0]
    for c in counts:
        cursors.append(cursors[-1] + c)
    vert_to_faces = defaultdict(list)
    for f in range(n_faces):
        for j in range(counts[f]):
            vert_to_faces[indices[cursors[f] + j]].append(f)

    # Build face adjacency: two faces are adjacent if they share at least one
    # edge (= two consecutive vertex indices in the same direction or reverse).
    face_edges = []  # list of sets of frozenset({v_a, v_b})
    for f in range(n_faces):
        c = counts[f]
        es = set()
        for j in range(c):
            a = indices[cursors[f] + j]
            b = indices[cursors[f] + (j + 1) % c]
            es.add(frozenset((a, b)))
        face_edges.append(es)

    edge_to_faces = defaultdict(list)
    for f, es in enumerate(face_edges):
        for e in es:
            edge_to_faces[e].append(f)

    adj = defaultdict(set)
    for e, fs in edge_to_faces.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                adj[fs[i]].add(fs[j])
                adj[fs[j]].add(fs[i])

    # BFS components.
    seen = [False] * n_faces
    comps = []
    for start in range(n_faces):
        if seen[start]:
            continue
        comp = []
        q = deque([start])
        seen[start] = True
        while q:
            f = q.popleft()
            comp.append(f)
            for nb in adj[f]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        comps.append(comp)

    print(f"[propose] found {len(comps)} components total")

    # World transform of the mesh (so we report world-space bboxes).
    xform = UsdGeom.XformCache(0).GetLocalToWorldTransform(mesh_prim)

    def world_bbox(face_list):
        mn = [1e30, 1e30, 1e30]
        mx = [-1e30, -1e30, -1e30]
        for f in face_list:
            for j in range(counts[f]):
                vi = indices[cursors[f] + j]
                p = points[vi]
                wp = xform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                for k in range(3):
                    if wp[k] < mn[k]:
                        mn[k] = wp[k]
                    if wp[k] > mx[k]:
                        mx[k] = wp[k]
        return mn, mx

    summaries = []
    for i, comp in enumerate(comps):
        if len(comp) < args.min_faces:
            continue
        mn, mx = world_bbox(comp)
        ctr = [(mn[k] + mx[k]) / 2 for k in range(3)]
        sz = [mx[k] - mn[k] for k in range(3)]
        summaries.append({
            "component_id": i,
            "face_count": len(comp),
            "bbox_min": [round(x, 4) for x in mn],
            "bbox_max": [round(x, 4) for x in mx],
            "center": [round(x, 4) for x in ctr],
            "size": [round(x, 4) for x in sz],
            "max_dim": round(max(sz), 4),
        })

    # Sort: smallest components first (likely individual parts), then by
    # ascending face count.
    summaries.sort(key=lambda s: (s["face_count"], s["max_dim"]))

    # Annotate a heuristic guess.
    if summaries:
        biggest = max(s["face_count"] for s in summaries)
        for s in summaries:
            ratio = s["face_count"] / biggest
            md = s["max_dim"]
            if ratio > 0.5:
                hint = "BODY (or main shell)"
            elif md < 0.03:
                hint = "small part (button / LED / screw)"
            elif md < 0.08:
                hint = "medium part (key / knob / latch)"
            elif md < 0.20:
                hint = "larger part (lever / handle / lid)"
            else:
                hint = "subassembly"
            s["hint"] = hint

    # Truncate report to top-N small components + the biggest body.
    truncated = summaries[: args.max_components]

    print("\n--- candidate components ---")
    for s in truncated:
        print(f"id={s['component_id']:4d}  faces={s['face_count']:6d}  "
              f"max_dim={s['max_dim']:.3f} m  ctr={s['center']}  -> {s['hint']}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "mesh_prim": args.mesh,
                "n_faces": n_faces,
                "n_components": len(comps),
                "components": truncated,
            }, f, indent=2)
        print(f"\n[propose] wrote {args.out_json}")

    simulation_app.close()


if __name__ == "__main__":
    main()
