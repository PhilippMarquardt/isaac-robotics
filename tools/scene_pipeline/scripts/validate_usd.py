"""USD asset validator for Isaac Lab pipeline.

Opens a USD file with pxr.Usd and reports:
  - defaultPrim, metersPerUnit, upAxis
  - counts of meshes, materials, textures, lights, cameras
  - missing texture asset paths
  - suspicious unit scale

Use:
    python validate_usd.py --usd assets/usd/imported_scene.usd
"""
import argparse
import os
from collections import Counter

from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf


def list_texture_paths(stage):
    paths = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            tn = attr.GetTypeName()
            if tn == Sdf.ValueTypeNames.Asset:
                v = attr.Get()
                if v is None:
                    continue
                p = v.resolvedPath or v.path
                if p:
                    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.usd):
        raise SystemExit(f"FAIL: {args.usd} not found")

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise SystemExit(f"FAIL: stage failed to open {args.usd}")

    default = stage.GetDefaultPrim()
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)

    counts = Counter()
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if prim.IsA(UsdGeom.Mesh):
            counts["mesh"] += 1
        if prim.IsA(UsdShade.Material):
            counts["material"] += 1
        if prim.IsA(UsdLux.LightAPI) or "Light" in t:
            counts["light"] += 1
        if "Camera" in t:
            counts["camera"] += 1
        if "Xform" in t:
            counts["xform"] += 1

    textures = list_texture_paths(stage)
    missing = [p for p in textures if p and not os.path.exists(p) and not p.startswith(("http", "omniverse://", "@"))]

    print(f"file:             {args.usd}")
    print(f"defaultPrim:      {default.GetName() if default else '(none)'}")
    print(f"metersPerUnit:    {mpu}")
    print(f"upAxis:           {up}")
    print(f"mesh:             {counts['mesh']}")
    print(f"material:         {counts['material']}")
    print(f"light:            {counts['light']}")
    print(f"camera:           {counts['camera']}")
    print(f"xform:            {counts['xform']}")
    print(f"texture refs:     {len(textures)}")
    print(f"missing textures: {len(missing)}")

    warns = []
    if default is None:
        warns.append("no defaultPrim — Isaac Sim may misroute references")
    if counts["mesh"] == 0:
        warns.append("no meshes")
    if counts["material"] == 0:
        warns.append("no materials")
    if counts["light"] == 0:
        warns.append("no lights (will add fallback dome)")
    if mpu and abs(mpu - 1.0) > 1e-3:
        warns.append(f"metersPerUnit={mpu} not 1.0 (scale issues likely)")
    if missing:
        warns.append(f"{len(missing)} missing texture paths")
        for p in missing[:5]:
            warns.append(f"    {p}")

    if warns:
        print("\nWARNINGS:")
        for w in warns:
            print(f"  - {w}")
    else:
        print("\nOK")


if __name__ == "__main__":
    main()
