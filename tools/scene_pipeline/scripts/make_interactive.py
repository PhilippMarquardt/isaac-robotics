"""Generalized asset->interactive wrapper.

Input: a JSON spec describing the source asset + one or more "interactives"
(lights, animated parts). Output: a wrapper USD that references the source and
carries metadata describing how each interactive should be driven at runtime.

Spec schema:

    {
      "asset":   "../source/foo/foo.usdc",        # path to source (rel or abs)
      "out":     "../usd/foo_interactive.usd",    # path to wrapper to write
      "interactives": [
        {
          "kind": "light",
          "name": "main_light",
          "head_prim": "/foo/bulb",               # path inside source
          "emissive_material": "/foo/_materials/bulb_glass",  # optional
          "on_intensity": 5000,
          "on_color": [1.0, 0.95, 0.85]
        },
        {
          "kind": "animated_part",
          "name": "power_switch",
          "prim":  "/foo/switch_xform",           # path inside source
          "motion": "rotate_y",                   # rotate_y|rotate_x|translate_z
          "amplitude": 22,                        # degrees (rot) or millimeters (trans)
          "linked_light": "main_light"            # optional: toggle this light on press
        }
      ]
    }

Use:
    isaaclab.bat -p scripts/make_interactive.py --spec my_asset.json
"""
import argparse
import json
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True, help="path to JSON spec")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf


def _bbox_center(stage, prim_path):
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise SystemExit(f"prim not found in source: {prim_path}")
    r = cache.ComputeWorldBound(prim).GetRange()
    if r.IsEmpty():
        raise SystemExit(f"empty bbox: {prim_path}")
    mn, mx = r.GetMin(), r.GetMax()
    return tuple((mn[i] + mx[i]) / 2 for i in range(3))


def _strip_default_prefix(stage, path):
    dp = stage.GetDefaultPrim()
    if not dp:
        return path
    p = path if path.startswith("/") else "/" + path
    prefix = f"/{dp.GetName()}"
    if p == prefix:
        return "/"
    if p.startswith(prefix + "/"):
        return p[len(prefix):]
    return p


def main():
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    spec_dir = os.path.dirname(os.path.abspath(args.spec))
    src_path = spec["asset"] if os.path.isabs(spec["asset"]) else os.path.join(spec_dir, spec["asset"])
    out_path = spec["out"] if os.path.isabs(spec["out"]) else os.path.join(spec_dir, spec["out"])
    src_path = os.path.abspath(src_path)
    out_path = os.path.abspath(out_path)

    src_stage = Usd.Stage.Open(src_path)
    if src_stage is None:
        raise SystemExit(f"could not open: {src_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dst_stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst_stage, 1.0)
    root = UsdGeom.Xform.Define(dst_stage, "/Prop")
    dst_stage.SetDefaultPrim(root.GetPrim())

    rel_src = os.path.relpath(src_path, os.path.dirname(out_path)).replace("\\", "/")
    visual = UsdGeom.Xform.Define(dst_stage, "/Prop/Visual")
    visual.GetPrim().GetReferences().AddReference(rel_src)

    interactives_meta = []
    synthetic_meta = []
    # First, materialize synthetic parts as new prims under /Prop/Synth/<name>.
    # They live OUTSIDE /Prop/Visual so they aren't ambiguous with referenced
    # geometry, but they share the scene transform with the asset.
    synth_root = UsdGeom.Xform.Define(dst_stage, "/Prop/Synth")
    for sp in spec.get("synthetic_parts", []):
        name = sp["name"]
        prim_path = f"/Prop/Synth/{name}"
        wp = sp["world_pos"]
        wr = sp.get("world_rot_deg", [0.0, 0.0, 0.0])
        color = sp.get("color", [0.7, 0.7, 0.75])

        if sp.get("shape") == "cube":
            sz = sp.get("size_mm", [15, 15, 15])
            cube = UsdGeom.Cube.Define(dst_stage, prim_path)
            cube.CreateSizeAttr(1.0)
            xapi = UsdGeom.XformCommonAPI(cube.GetPrim())
            xapi.SetTranslate(Gf.Vec3d(*wp))
            xapi.SetRotate(Gf.Vec3f(*wr))
            xapi.SetScale(Gf.Vec3f(sz[0] * 1e-3, sz[1] * 1e-3, sz[2] * 1e-3))
            # Visual material via displayColor (simpler than full Material binding).
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        elif sp.get("shape") == "sphere":
            r = sp.get("radius_mm", 8) * 1e-3
            sph = UsdGeom.Sphere.Define(dst_stage, prim_path)
            sph.CreateRadiusAttr(r)
            xapi = UsdGeom.XformCommonAPI(sph.GetPrim())
            xapi.SetTranslate(Gf.Vec3d(*wp))
            xapi.SetRotate(Gf.Vec3f(*wr))
            sph.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        elif sp.get("shape") == "cylinder":
            r = sp.get("radius_mm", 8) * 1e-3
            h = sp.get("height_mm", 30) * 1e-3
            cyl = UsdGeom.Cylinder.Define(dst_stage, prim_path)
            cyl.CreateRadiusAttr(r)
            cyl.CreateHeightAttr(h)
            cyl.CreateAxisAttr("Z")
            xapi = UsdGeom.XformCommonAPI(cyl.GetPrim())
            xapi.SetTranslate(Gf.Vec3d(*wp))
            xapi.SetRotate(Gf.Vec3f(*wr))
            cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        elif sp.get("shape") == "sphere_light":
            r = sp.get("radius_mm", 8) * 1e-3
            slight = UsdLux.SphereLight.Define(dst_stage, prim_path)
            slight.CreateRadiusAttr(r)
            slight.CreateIntensityAttr(0.0)
            slight.CreateColorAttr(Gf.Vec3f(*sp.get("on_color", [1.0, 0.95, 0.85])))
            UsdGeom.XformCommonAPI(slight.GetPrim()).SetTranslate(Gf.Vec3d(*wp))
        else:
            print(f"[make] unknown synthetic shape, skipping: {sp.get('shape')}")
            continue

        rec = {
            "kind": sp["kind"],
            "name": name,
            "prim_in_synth": f"/{name}",
            "world_center": list(wp),
        }
        if sp["kind"] == "light":
            rec.update({
                "light_prim": prim_path,
                "on_intensity": float(sp.get("on_intensity", 200.0)),
                "on_color": list(sp.get("on_color", [1.0, 0.95, 0.85])),
            })
        else:
            rec.update({
                "motion": sp.get("motion", "rotate_y"),
                "amplitude": float(sp.get("amplitude", 22.0)),
                "linked_light": sp.get("linked_light"),
                "triggers": sp.get("triggers"),
            })
        synthetic_meta.append(rec)
        print(f"[make] synth +{sp['kind']} '{name}' shape={sp.get('shape')}")

    for it in spec.get("interactives", []):
        kind = it["kind"]
        name = it["name"]
        rec = {"kind": kind, "name": name}

        if kind == "light":
            head_center = _bbox_center(src_stage, it["head_prim"])
            light_path = f"/Prop/Visual/_{name}_light"
            light = UsdLux.SphereLight.Define(dst_stage, light_path)
            light.CreateRadiusAttr(float(it.get("radius", 0.04)))
            light.CreateIntensityAttr(0.0)
            color = it.get("on_color", [1.0, 0.95, 0.85])
            light.CreateColorAttr(Gf.Vec3f(*color))
            UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*head_center))

            rec.update({
                "light_prim": light_path,
                "on_intensity": float(it.get("on_intensity", 5000.0)),
                "on_color": list(color),
            })
            if "emissive_material" in it:
                rec["emissive_material_in_visual"] = _strip_default_prefix(src_stage, it["emissive_material"])

        elif kind == "animated_part":
            sub = _strip_default_prefix(src_stage, it["prim"])
            ctr = it.get("world_center")
            if ctr is None:
                ctr = _bbox_center(src_stage, it["prim"])
            extra_subs = []
            for extra in (it.get("prim_list") or []):
                if extra == it["prim"]:
                    continue
                extra_subs.append(_strip_default_prefix(src_stage, extra))
            rec.update({
                "prim_in_visual": sub,
                "extra_prims_in_visual": extra_subs,
                "world_center": list(ctr),
                "motion": it.get("motion", "rotate_y"),
                "amplitude": float(it.get("amplitude", 22.0)),
                "linked_light": it.get("linked_light"),
                "triggers": it.get("triggers"),
            })

        else:
            print(f"[make] unknown kind, skipping: {kind}")
            continue

        interactives_meta.append(rec)
        print(f"[make] +{kind} '{name}'")

    custom = {
        "asset_source": os.path.basename(src_path),
        "interactives_json": json.dumps(interactives_meta),
        "synthetic_json": json.dumps(synthetic_meta),
    }
    root.GetPrim().SetCustomDataByKey("interactive_v2", custom)

    dst_stage.GetRootLayer().Save()
    print(f"[make] wrote {out_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
