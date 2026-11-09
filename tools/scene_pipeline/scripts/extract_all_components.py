"""Extract every connected component of a baked mesh into its own Xform/Mesh.

Wraps `extract_part.py`'s mesh-splitting logic but runs over the whole mesh
in one pass:
  1. Connected-component analysis (faces sharing edges -> same component).
  2. Filter by face count and bbox size to keep candidate "parts" only —
     skip the body (largest component) and noise (tiny components).
  3. Emit each kept component as `<asset_default_prim>/<part_name>` with its
     own Mesh prim, naming parts `comp_<id>`, `comp_<id>_b`, etc.
  4. Optionally remove those faces from the source mesh so the body becomes
     just the leftover geometry.

Use:
    isaaclab.bat -p scripts/extract_all_components.py \\
        --src .../asset.usdc --dst .../asset_split.usd \\
        --mesh /asset/body/mesh \\
        --min_faces 4 --max_faces 50 --max_dim 0.08
"""
import argparse
import os
from collections import defaultdict, deque
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--mesh", required=True)
parser.add_argument("--min_faces", type=int, default=4)
parser.add_argument("--max_faces", type=int, default=200)
parser.add_argument("--max_dim", type=float, default=0.10,
                    help="ignore components whose max bbox dim exceeds this (meters). "
                         "Filters out the body shell.")
parser.add_argument("--name_prefix", default="part_")
parser.add_argument("--cut", action="store_true", default=True,
                    help="remove extracted faces from source mesh (default)")
parser.add_argument("--no_cut", dest="cut", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf


def connected_components(counts, indices):
    n_faces = len(counts)
    cursors = [0]
    for c in counts:
        cursors.append(cursors[-1] + c)
    # edge -> faces
    edge_to_faces = defaultdict(list)
    for f in range(n_faces):
        c = counts[f]
        for j in range(c):
            a = indices[cursors[f] + j]
            b = indices[cursors[f] + (j + 1) % c]
            edge_to_faces[frozenset((a, b))].append(f)
    adj = defaultdict(set)
    for fs in edge_to_faces.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                adj[fs[i]].add(fs[j])
                adj[fs[j]].add(fs[i])
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
    return comps, cursors


def world_bbox(face_list, counts, cursors, indices, points, world_xform):
    mn = [1e30] * 3
    mx = [-1e30] * 3
    for f in face_list:
        for j in range(counts[f]):
            p = points[indices[cursors[f] + j]]
            wp = world_xform.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            for k in range(3):
                if wp[k] < mn[k]:
                    mn[k] = wp[k]
                if wp[k] > mx[k]:
                    mx[k] = wp[k]
    return mn, mx


def build_subprim(stage, src_mesh, face_list, new_xform_path, counts, cursors,
                  indices, points, normals, uv_pv, uv_data, uv_interp):
    from pxr import Vt
    new_xform_path = new_xform_path
    new_mesh_path = f"{new_xform_path}/Mesh"
    UsdGeom.Xform.Define(stage, new_xform_path)
    new_mesh = UsdGeom.Mesh.Define(stage, new_mesh_path)
    new_point_map = {}
    new_points = []
    new_counts = []
    new_indices = []
    new_normals = []
    new_uvs = []
    for f in face_list:
        c = counts[f]
        new_counts.append(c)
        for j in range(c):
            old_vi = indices[cursors[f] + j]
            if old_vi not in new_point_map:
                new_point_map[old_vi] = len(new_points)
                new_points.append(points[old_vi])
            new_indices.append(new_point_map[old_vi])
            if normals is not None and uv_interp != "uniform":
                n_idx = cursors[f] + j
                if n_idx < len(normals):
                    new_normals.append(normals[n_idx])
            if uv_data is not None:
                if uv_interp == "faceVarying":
                    uv_idx = cursors[f] + j
                    if uv_idx < len(uv_data):
                        new_uvs.append(uv_data[uv_idx])
                elif uv_interp == "vertex":
                    new_uvs.append(uv_data[old_vi])
    new_mesh.CreatePointsAttr(new_points)
    new_mesh.CreateFaceVertexCountsAttr(new_counts)
    new_mesh.CreateFaceVertexIndicesAttr(new_indices)
    if new_normals:
        new_mesh.CreateNormalsAttr(new_normals)
        new_mesh.SetNormalsInterpolation("faceVarying")
    if new_uvs and uv_pv:
        new_uv_pv = UsdGeom.PrimvarsAPI(new_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, uv_interp or "faceVarying"
        )
        new_uv_pv.Set(new_uvs)
    src_bind = UsdShade.MaterialBindingAPI(src_mesh.GetPrim()).GetDirectBinding().GetMaterial()
    if src_bind:
        UsdShade.MaterialBindingAPI(new_mesh.GetPrim()).Bind(src_bind)
    return new_mesh_path


def remap_geomsubsets(mesh_prim, kept_face_mask):
    """For every UsdGeomSubset child of `mesh_prim`, filter+remap its `indices`
    so they reference faces that still exist after `kept_face_mask` was applied.

    Without this, leftover GeomSubsets point at gone face IDs and Isaac/Hydra
    refuses to render the mesh:
        '[omni.hydra] Face maps onto an non-existent geomSubset, this means the USD is malformed.'
    """
    # Build old_face -> new_face_index mapping.
    new_idx_of = {}
    new = 0
    for old, keep in enumerate(kept_face_mask):
        if keep:
            new_idx_of[old] = new
            new += 1
    for child in list(mesh_prim.GetChildren()):
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        old_indices = subset.GetIndicesAttr().Get()
        if old_indices is None:
            continue
        new_indices = [new_idx_of[i] for i in old_indices if i in new_idx_of]
        if not new_indices:
            # Subset is now empty: delete it to keep the file clean.
            child.GetStage().RemovePrim(child.GetPath())
            continue
        subset.GetIndicesAttr().Set(new_indices)


def remove_faces(mesh, kept_face_mask, counts, cursors, indices, normals,
                 uv_pv, uv_data, uv_interp):
    new_counts = []
    new_indices = []
    new_normals = []
    new_uvs = []
    for f, keep in enumerate(kept_face_mask):
        if not keep:
            continue
        c = counts[f]
        new_counts.append(c)
        for j in range(c):
            new_indices.append(indices[cursors[f] + j])
            if normals is not None and uv_interp != "uniform":
                n_idx = cursors[f] + j
                if n_idx < len(normals):
                    new_normals.append(normals[n_idx])
            if uv_data is not None and uv_interp == "faceVarying":
                uv_idx = cursors[f] + j
                if uv_idx < len(uv_data):
                    new_uvs.append(uv_data[uv_idx])
            elif uv_data is not None and uv_interp == "vertex":
                new_uvs.append(uv_data[indices[cursors[f] + j]])
    mesh.CreateFaceVertexCountsAttr(new_counts)
    mesh.CreateFaceVertexIndicesAttr(new_indices)
    if new_normals:
        mesh.CreateNormalsAttr(new_normals)
    if uv_pv and new_uvs:
        uv_pv.Set(new_uvs)


def main():
    src_stage = Usd.Stage.Open(args.src)
    if src_stage is None:
        raise SystemExit(f"can't open: {args.src}")
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    src_stage.Export(args.dst)
    dst = Usd.Stage.Open(args.dst)

    mesh_prim = dst.GetPrimAtPath(args.mesh)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise SystemExit(f"mesh not found at {args.mesh}")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    normals = mesh.GetNormalsAttr().Get()
    uv_pv = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
    uv_data = uv_pv.Get() if uv_pv else None
    uv_interp = uv_pv.GetInterpolation() if uv_pv else None

    comps, cursors = connected_components(counts, indices)
    print(f"[extract_all] mesh has {len(counts)} faces, {len(comps)} components")

    world_xform = UsdGeom.XformCache(0).GetLocalToWorldTransform(mesh_prim)

    default_prim = dst.GetDefaultPrim()
    if default_prim is None:
        raise SystemExit("no defaultPrim on source")

    kept_face_mask = [True] * len(counts)
    extracted = []  # (xform_path, comp_id, bbox_center, bbox_size, face_count)

    for comp_id, comp in enumerate(comps):
        if len(comp) < args.min_faces or len(comp) > args.max_faces:
            continue
        mn, mx = world_bbox(comp, counts, cursors, indices, points, world_xform)
        sz = [mx[k] - mn[k] for k in range(3)]
        if max(sz) > args.max_dim:
            continue
        new_xform_path = f"{default_prim.GetPath()}/{args.name_prefix}{comp_id:04d}"
        new_mesh_path = build_subprim(dst, mesh, comp, new_xform_path, counts,
                                      cursors, indices, points, normals, uv_pv,
                                      uv_data, uv_interp)
        ctr = [(mn[k] + mx[k]) / 2 for k in range(3)]
        extracted.append((new_xform_path, comp_id, ctr, sz, len(comp)))
        if args.cut:
            for f in comp:
                kept_face_mask[f] = False

    print(f"[extract_all] extracted {len(extracted)} components into separate Xforms")
    for p, cid, ctr, sz, fc in extracted[:5]:
        print(f"   {p}  faces={fc}  ctr={[round(x,3) for x in ctr]}  sz={[round(x,3) for x in sz]}")
    if len(extracted) > 5:
        print(f"   ... and {len(extracted) - 5} more")

    if args.cut:
        remove_faces(mesh, kept_face_mask, counts, cursors, indices, normals,
                     uv_pv, uv_data, uv_interp)
        remap_geomsubsets(mesh_prim, kept_face_mask)
        kept = sum(kept_face_mask)
        print(f"[extract_all] body mesh now has {kept} / {len(counts)} faces")

    dst.GetRootLayer().Save()
    print(f"[extract_all] wrote {args.dst}")
    simulation_app.close()


if __name__ == "__main__":
    main()
