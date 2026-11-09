"""Convert FBX / glTF / GLB / OBJ → USD via Isaac Sim's asset_converter extension.

Use when a Fab/Sketchfab/Poly-Haven asset ships in a format other than USD and
you want to skip the Unity round-trip.

    isaaclab.bat -p scripts/convert_to_usd.py --src path/in.fbx --dst path/out.usd
"""
import argparse
import asyncio
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="input mesh: .fbx / .gltf / .glb / .obj")
parser.add_argument("--dst", required=True, help="output .usd path")
parser.add_argument("--no_materials", action="store_true", help="skip material/texture import")
parser.add_argument("--no_textures", action="store_true", help="skip texture import (keep materials)")
parser.add_argument("--up", default="Z", choices=["Y", "Z"], help="up axis (Isaac expects Z)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# headless+RTX crashes on this machine when --enable_cameras is set; conversion
# doesn't need a render context so leave both off.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.app
_ext_mgr = omni.kit.app.get_app().get_extension_manager()
for ext in ("omni.kit.asset_converter", "omni.kit.tool.asset_importer"):
    try:
        _ext_mgr.set_extension_enabled_immediate(ext, True)
    except Exception as e:
        print(f"[convert] could not enable {ext}: {e}")

import omni.kit.asset_converter as asset_converter


async def _convert(src: str, dst: str) -> bool:
    task_manager = asset_converter.get_instance()
    cfg = asset_converter.AssetConverterContext()
    cfg.ignore_materials = args.no_materials
    cfg.ignore_textures = args.no_textures
    cfg.embed_textures = True
    cfg.use_meter_as_world_unit = True
    cfg.create_world_as_default_root_prim = True
    cfg.merge_all_meshes = False
    cfg.smooth_normals = True
    cfg.export_preview_surface = True
    cfg.support_point_instancer = True
    task = task_manager.create_converter_task(src, dst, None, cfg)
    ok = await task.wait_until_finished()
    if not ok:
        print(f"[convert] FAIL: {task.get_status()} -- {task.get_error_message()}")
        return False
    print(f"[convert] OK -> {dst}")
    return True


def main():
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.exists(src):
        raise SystemExit(f"[convert] src not found: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[convert] {src} -> {dst}")
    ok = asyncio.get_event_loop().run_until_complete(_convert(src, dst))
    simulation_app.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
