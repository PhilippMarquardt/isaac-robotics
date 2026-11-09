"""Dump the prim tree of a USD asset so we (and an LLM) can reason about it.

Use through isaaclab.bat so pxr is available:
    isaaclab.bat -p scripts/inspect_usd.py --usd path/to/asset.usd[c]
"""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--out_json", default=None, help="optional: dump tree to this JSON file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf


def prim_summary(prim):
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    out = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "name": prim.GetName(),
        "is_mesh": prim.IsA(UsdGeom.Mesh),
        "is_material": prim.IsA(UsdShade.Material),
        "is_light": prim.IsA(UsdLux.LightAPI) or "Light" in prim.GetTypeName(),
        "is_xform": prim.IsA(UsdGeom.Xform),
    }
    if out["is_mesh"] or out["is_xform"]:
        try:
            bbox = bbox_cache.ComputeWorldBound(prim).GetRange()
            mn, mx = bbox.GetMin(), bbox.GetMax()
            if not bbox.IsEmpty():
                out["bbox_min"] = [mn[0], mn[1], mn[2]]
                out["bbox_max"] = [mx[0], mx[1], mx[2]]
                out["center"] = [(mn[i] + mx[i]) / 2 for i in range(3)]
                out["size"] = [mx[i] - mn[i] for i in range(3)]
        except Exception:
            pass
    return out


def main():
    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise SystemExit(f"FAIL: could not open {args.usd}")
    tree = []
    for prim in stage.Traverse():
        tree.append(prim_summary(prim))

    default = stage.GetDefaultPrim()
    info = {
        "file": os.path.abspath(args.usd),
        "default_prim": default.GetName() if default else None,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "prims": tree,
    }
    print(json.dumps(info, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\n[inspect] wrote {args.out_json}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
