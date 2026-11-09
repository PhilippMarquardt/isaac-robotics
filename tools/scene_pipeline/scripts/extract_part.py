"""Extract faces from a baked-in body mesh into a new Mesh + Xform.

When the artist baked all geometry into a single mesh (keys, levers, display
all just painted into the body), this script splits the requested faces into
their own prim so they can be animated independently.

Three selection modes:

  --by-subset    : promote an existing UsdGeomSubset to a top-level mesh.
                   Use when the asset already tagged the faces with a subset.

  --by-material  : extract every face whose material:binding points at the
                   named material. Use when keys / glass / etc. have a
                   different material slot than the body.

  --by-bbox      : extract every face whose centroid sits inside the given
                   world-space bbox. Use when you have neither signal but
                   know the spatial layout of the real object.

The extracted faces are *removed* from the source mesh by default, so the
visual scene shows the extraction as separate geometry (no Z-fighting).

Outputs a new USD that references the source AND overrides the source body
mesh to drop the extracted faces, while adding a new top-level Mesh prim
with just the extracted faces.

Use:
    isaaclab.bat -p scripts/extract_part.py \\
        --src .../asset.usdc \\
        --dst .../asset_split.usd \\
        --mesh /asset_root/body_xform/body_mesh \\
        --part_name key_total \\
        --by_bbox -0.10,-0.20,0.44 -0.06,-0.16,0.50 \\
        --target_xform /asset_root/key_total
"""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="source USD")
parser.add_argument("--dst", required=True, help="destination USD (single-layer copy of source, with edits)")
parser.add_argument("--mesh", required=True, help="prim path of the baked body mesh")
parser.add_argument("--part_name", required=True, help="name for the extracted Xform/Mesh under the asset root")
parser.add_argument("--by_subset", default=None, help="GeomSubset prim path under --mesh")
parser.add_argument("--by_material", default=None, help="material prim path; extract faces bound to it")
parser.add_argument("--bbox_min", default=None,
                    help="world-space bbox lower corner 'x,y,z' (use with --bbox_max)")
parser.add_argument("--bbox_max", default=None,
                    help="world-space bbox upper corner 'x,y,z' (use with --bbox_min)")
parser.add_argument("--cut", action="store_true", default=True,
                    help="remove extracted faces from the source mesh (default true)")
parser.add_argument("--no_cut", dest="cut", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt


def parse_vec3(s):
    # Accept comma- or space-separated values (Windows shells sometimes strip
    # commas inside quoted args).
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers, got: {s!r}")
    return [float(x) for x in parts]


def faces_in_bbox(stage, mesh_prim, bbox_min, bbox_max):
    """Return list of face indices whose centroid falls inside bbox."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    # Convert points to world frame for bbox check.
    xform = UsdGeom.XformCache(0).GetLocalToWorldTransform(mesh_prim)
    selected = []
    cursor = 0
    for f, c in enumerate(counts):
        cx = cy = cz = 0.0
        for j in range(c):
            v = points[indices[cursor + j]]
            wp = xform.Transform(Gf.Vec3d(v[0], v[1], v[2]))
            cx += wp[0]; cy += wp[1]; cz += wp[2]
        cursor += c
        cx /= c; cy /= c; cz /= c
        if (bbox_min[0] <= cx <= bbox_max[0] and
            bbox_min[1] <= cy <= bbox_max[1] and
            bbox_min[2] <= cz <= bbox_max[2]):
            selected.append(f)
    return selected


def faces_from_subset(mesh_prim, subset_path):
    sub = UsdGeom.Subset(mesh_prim.GetStage().GetPrimAtPath(subset_path))
    if not sub:
        raise SystemExit(f"subset not found: {subset_path}")
    return list(sub.GetIndicesAttr().Get())


def faces_from_material(stage, mesh_prim, material_path):
    """Find subsets on this mesh whose material binding points to material_path."""
    out = []
    for child in mesh_prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        bind_api = UsdShade.MaterialBindingAPI(child)
        bound = bind_api.GetDirectBinding().GetMaterial()
        if bound and str(bound.GetPath()) == material_path:
            out.extend(list(UsdGeom.Subset(child).GetIndicesAttr().Get()))
    return sorted(set(out))


def extract(stage, mesh_prim, face_idxs, new_mesh_path):
    mesh = UsdGeom.Mesh(mesh_prim)
    src_points = mesh.GetPointsAttr().Get()
    src_counts = mesh.GetFaceVertexCountsAttr().Get()
    src_indices = mesh.GetFaceVertexIndicesAttr().Get()
    src_normals = mesh.GetNormalsAttr().Get()
    primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    src_uv = primvars_api.GetPrimvar("st")  # standard UV name
    uv_data = src_uv.Get() if src_uv else None
    uv_interp = src_uv.GetInterpolation() if src_uv else None

    # Build face index → cursor mapping.
    cursors = [0]
    for c in src_counts:
        cursors.append(cursors[-1] + c)

    keep_set = set(face_idxs)
    # Collect new face data.
    new_counts = []
    new_indices = []
    new_normals = []
    new_uvs = []
    new_point_map = {}   # old vertex idx -> new vertex idx
    new_points = []

    for f, c in enumerate(src_counts):
        if f not in keep_set:
            continue
        new_counts.append(c)
        for j in range(c):
            old_vi = src_indices[cursors[f] + j]
            if old_vi not in new_point_map:
                new_point_map[old_vi] = len(new_points)
                new_points.append(src_points[old_vi])
            new_indices.append(new_point_map[old_vi])
            if src_normals is not None and uv_interp != "uniform":
                # normals are typically faceVarying
                n_idx = cursors[f] + j
                if n_idx < len(src_normals):
                    new_normals.append(src_normals[n_idx])
            if uv_data is not None:
                if uv_interp == "faceVarying":
                    uv_idx = cursors[f] + j
                    if uv_idx < len(uv_data):
                        new_uvs.append(uv_data[uv_idx])
                elif uv_interp == "vertex":
                    new_uvs.append(uv_data[old_vi])

    # Create new mesh prim.
    new_mesh = UsdGeom.Mesh.Define(stage, new_mesh_path)
    new_mesh.CreatePointsAttr(new_points)
    new_mesh.CreateFaceVertexCountsAttr(new_counts)
    new_mesh.CreateFaceVertexIndicesAttr(new_indices)
    if new_normals:
        new_mesh.CreateNormalsAttr(new_normals)
        new_mesh.SetNormalsInterpolation("faceVarying")
    if new_uvs:
        new_uv_pv = UsdGeom.PrimvarsAPI(new_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, uv_interp or "faceVarying"
        )
        new_uv_pv.Set(new_uvs)
    # Copy material binding from source.
    src_bind = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBinding().GetMaterial()
    if src_bind:
        UsdShade.MaterialBindingAPI(new_mesh.GetPrim()).Bind(src_bind)

    return new_mesh


def remove_faces_from_source(mesh_prim, face_idxs):
    mesh = UsdGeom.Mesh(mesh_prim)
    src_counts = list(mesh.GetFaceVertexCountsAttr().Get())
    src_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    src_normals = mesh.GetNormalsAttr().Get()
    uv_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    uv_pv = uv_api.GetPrimvar("st")
    uv_data = uv_pv.Get() if uv_pv else None
    uv_interp = uv_pv.GetInterpolation() if uv_pv else None

    drop = set(face_idxs)
    cursors = [0]
    for c in src_counts:
        cursors.append(cursors[-1] + c)

    new_counts = []
    new_indices = []
    new_normals = []
    new_uvs = []
    for f, c in enumerate(src_counts):
        if f in drop:
            continue
        new_counts.append(c)
        for j in range(c):
            new_indices.append(src_indices[cursors[f] + j])
            if src_normals is not None and uv_interp != "uniform":
                n_idx = cursors[f] + j
                if n_idx < len(src_normals):
                    new_normals.append(src_normals[n_idx])
            if uv_data is not None and uv_interp == "faceVarying":
                uv_idx = cursors[f] + j
                if uv_idx < len(uv_data):
                    new_uvs.append(uv_data[uv_idx])
            elif uv_data is not None and uv_interp == "vertex":
                new_uvs.append(uv_data[src_indices[cursors[f] + j]])

    mesh.CreateFaceVertexCountsAttr(new_counts)
    mesh.CreateFaceVertexIndicesAttr(new_indices)
    if new_normals:
        mesh.CreateNormalsAttr(new_normals)
    if new_uvs and uv_pv:
        uv_pv.Set(new_uvs)


def main():
    # Open source, flatten to single layer for destination (so we own all the
    # geometry data and can mutate it freely).
    src_stage = Usd.Stage.Open(args.src)
    if src_stage is None:
        raise SystemExit(f"can't open: {args.src}")
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    src_stage.Export(args.dst)
    dst = Usd.Stage.Open(args.dst)

    mesh_prim = dst.GetPrimAtPath(args.mesh)
    if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
        raise SystemExit(f"mesh not found at {args.mesh}")

    if args.by_subset:
        face_idxs = faces_from_subset(mesh_prim, args.by_subset)
        mode = f"subset={args.by_subset}"
    elif args.by_material:
        face_idxs = faces_from_material(dst, mesh_prim, args.by_material)
        mode = f"material={args.by_material}"
    elif args.bbox_min and args.bbox_max:
        bmn = parse_vec3(args.bbox_min)
        bmx = parse_vec3(args.bbox_max)
        face_idxs = faces_in_bbox(dst, mesh_prim, bmn, bmx)
        mode = f"bbox={bmn} -> {bmx}"
    else:
        raise SystemExit("must give one of --by_subset / --by_material / (--bbox_min + --bbox_max)")

    print(f"[extract] {mode}: selected {len(face_idxs)} faces")
    if not face_idxs:
        raise SystemExit("no faces matched; check your selection criteria")

    # New mesh path: as a child of the source's default prim so paths inside
    # the asset stay consistent.
    default_prim = dst.GetDefaultPrim()
    if default_prim is None:
        raise SystemExit("source has no defaultPrim")
    new_xform_path = f"{default_prim.GetPath()}/{args.part_name}"
    new_mesh_path = f"{new_xform_path}/Mesh"
    UsdGeom.Xform.Define(dst, new_xform_path)
    extract(dst, mesh_prim, face_idxs, new_mesh_path)
    print(f"[extract] wrote {new_mesh_path}")

    if args.cut:
        remove_faces_from_source(mesh_prim, face_idxs)
        print(f"[extract] removed {len(face_idxs)} faces from {args.mesh}")

    dst.GetRootLayer().Save()
    print(f"[extract] wrote {args.dst}")
    simulation_app.close()


if __name__ == "__main__":
    main()
