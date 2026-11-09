"""LLM-as-converter step: take a "dumb" lamp USD asset and produce a robot-usable
interactive variant.

What this does:
  - References the source lamp USD (read-only) under `/Prop`.
  - Adds a `UsdLuxSphereLight` at the lamp-head bbox center, starting at intensity 0.
  - Records the light path and (optionally) bulb-material path in custom userProperties
    so `ToggleableLight` can find them by inspection rather than hard-coding.

Use through isaaclab.bat so pxr is available:
    isaaclab.bat -p scripts/make_interactive_lamp.py \\
        --src .../desk_lamp_arm_01_1k.usdc \\
        --dst .../desk_lamp_interactive.usd \\
        --head_path /desk_lamp_arm_01/geo_lamp_head \\
        --bulb_material_path /desk_lamp_arm_01/_materials/desk_lamp_arm_01_light
"""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="source lamp USD")
parser.add_argument("--dst", required=True, help="destination interactive USD")
parser.add_argument("--head_path", required=True,
                    help="prim path of the lamp head Xform in the source USD")
parser.add_argument("--bulb_material_path", default=None,
                    help="optional: prim path of the bulb material in source USD "
                         "(used to drive emissive on/off)")
parser.add_argument("--switch_path", default=None,
                    help="optional: prim path of the switch/button geometry in source USD. "
                         "Records its world center so a robot can target it.")
parser.add_argument("--body_path", default=None,
                    help="optional: prim path of the static lamp body mesh in source USD. "
                         "Gets a CollisionAPI so the robot can't phase through it.")
parser.add_argument("--on_intensity", type=float, default=20000.0)
parser.add_argument("--on_color", type=float, nargs=3, default=[1.0, 0.95, 0.85])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Sdf, Gf


def _bbox_center(stage, prim_path):
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise SystemExit(f"head_path not found in source: {prim_path}")
    r = cache.ComputeWorldBound(prim).GetRange()
    if r.IsEmpty():
        raise SystemExit(f"head_path has empty bbox: {prim_path}")
    mn, mx = r.GetMin(), r.GetMax()
    return tuple((mn[i] + mx[i]) / 2 for i in range(3))


def _strip_default_prefix(src_stage, path: str) -> str:
    """Translate a source-USD prim path to the path it lives at after being
    referenced (because USD drops the defaultPrim name on reference)."""
    dp = src_stage.GetDefaultPrim()
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
    # Open the source read-only to compute the head center.
    src_stage = Usd.Stage.Open(args.src)
    if src_stage is None:
        raise SystemExit(f"could not open source: {args.src}")
    head_center = _bbox_center(src_stage, args.head_path)
    print(f"[make] head center: {head_center}")

    # Build the destination stage. The default prim wraps everything so the
    # asset can be `def "Prop"` in a parent stage and referenced cleanly.
    dst_stage = Usd.Stage.CreateNew(args.dst)
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dst_stage, 1.0)
    root = UsdGeom.Xform.Define(dst_stage, "/Prop")
    dst_stage.SetDefaultPrim(root.GetPrim())

    # Layered reference to the source lamp USD. Path is stored relative so the
    # consumer can move both files together.
    src_rel = os.path.relpath(os.path.abspath(args.src),
                              os.path.dirname(os.path.abspath(args.dst)))
    visual = UsdGeom.Xform.Define(dst_stage, "/Prop/Visual")
    visual.GetPrim().GetReferences().AddReference(src_rel.replace("\\", "/"))

    # Sphere light at the head center, starts OFF (intensity=0).
    light_path = "/Prop/Visual/InteractiveLight"
    light = UsdLux.SphereLight.Define(dst_stage, light_path)
    light.CreateRadiusAttr(0.04)
    light.CreateIntensityAttr(0.0)  # off by default
    light.CreateColorAttr(Gf.Vec3f(*args.on_color))
    # XformOp translate so the light sits at the head bbox center.
    UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(
        Gf.Vec3d(*head_center)
    )

    # Record metadata so `ToggleableLight` (or any consumer) can find the knobs
    # by reading the asset rather than hard-coding paths.
    prop_prim = root.GetPrim()
    custom = prop_prim.GetCustomDataByKey("interactive") or {}
    custom["kind"] = "lamp"
    custom["light_prim"] = light_path
    custom["on_intensity"] = args.on_intensity
    custom["on_color"] = list(args.on_color)
    if args.bulb_material_path:
        custom["bulb_material_in_visual"] = _strip_default_prefix(src_stage, args.bulb_material_path)

    if args.switch_path:
        sw_center = _bbox_center(src_stage, args.switch_path)
        sw_sub = _strip_default_prefix(src_stage, args.switch_path)
        custom["switch_prim_in_visual"] = sw_sub
        custom["switch_world_center"] = list(sw_center)
        custom["switch_joint_travel"] = 0.003
        print(f"[make] switch center: {sw_center}")
        # Physics (rigid body, joint, drive) is attached at RUNTIME by
        # runtime.physics_attach — authoring it here as USD overrides did not
        # compose correctly through references for PhysX.

    if args.body_path:
        body_sub = _strip_default_prefix(src_stage, args.body_path)
        custom["lamp_body_in_visual"] = body_sub  # path inside /Prop/Visual
        print(f"[make] body sub-path recorded: {body_sub}")

    prop_prim.SetCustomDataByKey("interactive", custom)

    dst_stage.GetRootLayer().Save()
    print(f"[make] wrote {args.dst}")
    simulation_app.close()


if __name__ == "__main__":
    main()
