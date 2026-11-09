"""Precompute orientation cues for every part Xform.

For each `/asset/part_*` (or any Xform with a child Mesh), output:
  - principal_axis        : top PCA eigenvector of the mesh's vertices (world)
  - elongation            : ratio of largest to second eigenvalue (>>1 = thin/long)
  - flatness              : ratio of second to smallest eigenvalue (>>1 = flat plate)
  - nearest_face          : one of +X,-X,+Y,-Y,+Z,-Z (closest body bbox face)
  - distance_to_face_m    : perpendicular distance to that face
  - on_face_flag          : whether the part centre lies on that face within tol

Writes an augmented JSON that classify_with_llm.py can consume.

Use:
    isaaclab.bat -p scripts/augment_components.py \\
        --usd .../asset_split.usd \\
        --tree outputs/<asset>_tree.json \\
        --out outputs/<asset>_components_aug.json
"""
import argparse
import json
import math
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--tree", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
from pxr import Usd, UsdGeom, Gf


def pca(points_world):
    arr = np.asarray(points_world, dtype=np.float64)
    if arr.shape[0] < 3:
        return None
    mean = arr.mean(axis=0)
    centred = arr - mean
    cov = (centred.T @ centred) / max(arr.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; reverse to descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def axis_label(v):
    """Return signed-axis label '+X','-X','+Y','-Y','+Z','-Z' for the
    closest cardinal direction to vector v."""
    ax = ["X", "Y", "Z"]
    i = int(np.argmax(np.abs(v)))
    sign = "+" if v[i] >= 0 else "-"
    return f"{sign}{ax[i]}"


def nearest_face(ctr, body_min, body_max):
    """Closest face of the body bbox to point ctr."""
    dists = {
        "-X": ctr[0] - body_min[0], "+X": body_max[0] - ctr[0],
        "-Y": ctr[1] - body_min[1], "+Y": body_max[1] - ctr[1],
        "-Z": ctr[2] - body_min[2], "+Z": body_max[2] - ctr[2],
    }
    face = min(dists, key=dists.get)
    return face, dists[face]


def main():
    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise SystemExit(f"can't open: {args.usd}")
    xcache = UsdGeom.XformCache(0)

    # Body bbox.
    body = None
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") == 2 and "/part_" not in path:
            if p.get("size") and (body is None or max(p["size"]) > max(body.get("size", [0, 0, 0]))):
                body = p
    if body is None:
        raise SystemExit("body Xform not found")
    bmn, bmx = body["bbox_min"], body["bbox_max"]
    body_size = body["size"]
    print(f"[aug] body {body['path']} size {body_size}")

    # Walk every Xform that has a Mesh child (extracted parts + originals).
    aug = []
    for p in tree["prims"]:
        if p["type"] != "Xform":
            continue
        path = p["path"]
        if path.count("/") < 2:
            continue
        if path == body["path"]:
            continue

        # Try to find a child Mesh prim.
        usd_prim = stage.GetPrimAtPath(path)
        if not usd_prim.IsValid():
            continue
        mesh_prim = None
        for ch in usd_prim.GetChildren():
            if ch.IsA(UsdGeom.Mesh):
                mesh_prim = ch
                break
        # If the Xform itself IS the mesh path's parent, also accept the meshes
        # whose path is `path/Cash_register` etc.
        if mesh_prim is None:
            # walk descendants up to depth 2
            for desc in usd_prim.GetDescendants():
                if desc.IsA(UsdGeom.Mesh):
                    mesh_prim = desc
                    break
        if mesh_prim is None:
            continue

        mesh = UsdGeom.Mesh(mesh_prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is None or len(pts) < 3:
            continue
        # to world
        xform = xcache.GetLocalToWorldTransform(mesh_prim)
        wpts = [(xform.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))) for pt in pts]
        wpts = [(wp[0], wp[1], wp[2]) for wp in wpts]
        out = pca(wpts)
        if out is None:
            continue
        eigvals, eigvecs = out
        principal = eigvecs[:, 0]
        # Force a consistent sign: max component positive.
        if principal[int(np.argmax(np.abs(principal)))] < 0:
            principal = -principal
        elongation = float(eigvals[0] / max(eigvals[1], 1e-12))
        flatness = float(eigvals[1] / max(eigvals[2], 1e-12))

        ctr = [(min(wp[k] for wp in wpts) + max(wp[k] for wp in wpts)) / 2
               for k in range(3)]
        face, dist = nearest_face(ctr, bmn, bmx)

        aug.append({
            "path": path,
            "principal_axis_world": [round(float(principal[k]), 4) for k in range(3)],
            "principal_axis_label": axis_label(principal),
            "elongation": round(elongation, 3),
            "flatness": round(flatness, 3),
            "nearest_body_face": face,
            "distance_to_face_m": round(float(dist), 5),
            "on_face": bool(dist < 0.01),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"body_path": body["path"],
                   "body_bbox_min": bmn, "body_bbox_max": bmx,
                   "body_size": body_size,
                   "parts": aug}, f, indent=2)
    print(f"[aug] wrote {args.out}  ({len(aug)} parts)")
    simulation_app.close()


if __name__ == "__main__":
    main()
