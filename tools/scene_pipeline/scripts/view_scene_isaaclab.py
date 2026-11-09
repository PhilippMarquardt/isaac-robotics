"""Load a USD scene in Isaac Lab and render an RGB screenshot.

Adds a fallback dome light + a camera. Saves outputs/render.png.

Defaults to a Nucleus-hosted textured table USD so the pipeline can be
verified end-to-end before any local asset is provided.

Use:
    isaaclab.bat -p scripts/view_scene_isaaclab.py
    isaaclab.bat -p scripts/view_scene_isaaclab.py --usd path/to/scene.usd
    isaaclab.bat -p scripts/view_scene_isaaclab.py --interactive
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default=None,
                    help="path to USD file. defaults to Nucleus textured table sample.")
parser.add_argument("--interactive", action="store_true",
                    help="keep app open until window closed; otherwise capture+exit.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--out", type=str, default="outputs/render.png")
parser.add_argument("--cam_pos", type=float, nargs=3, default=[2.5, 2.5, 1.8])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.0, 0.5])
parser.add_argument("--y_up", action="store_true",
                    help="source asset is Y-up (e.g. glTF/FBX). Rotates +90° on X "
                         "so Z is up in Isaac.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# headless + RTX Hydra engine deadlocks/crashes on this machine, so default to
# a windowed render. `--headless_render` re-enables the offscreen path if/when fixed.
# enable_cameras must be on for the camera sensor to produce RGB.
if not hasattr(args, "enable_cameras") or not args.enable_cameras:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg, Camera


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(args.cam_pos, args.cam_target)

    if args.usd:
        print(f"[scene] loading: {args.usd}")
        scene_spawn = sim_utils.UsdFileCfg(usd_path=args.usd)
        # +90° rotation about X maps Y-up source -> Z-up Isaac (quat w,x,y,z).
        orientation = (0.7071068, 0.7071068, 0.0, 0.0) if args.y_up else (1.0, 0.0, 0.0, 0.0)
        scene_spawn.func("/World/ImportedScene", scene_spawn,
                         translation=(0.0, 0.0, 0.0), orientation=orientation)
    else:
        # no USD given: spawn a minimal in-process scene (a colored cube on the
        # ground) so the pipeline can be smoke-tested without any Nucleus or
        # local asset.
        print("[scene] no --usd given; spawning built-in smoke-test scene (cube)")
        sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2), metallic=0.1
            ),
        ).func("/World/SmokeTestCube", sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)),
               translation=(0.0, 0.0, 0.25))

    # fallback ground in case scene has none
    sim_utils.GroundPlaneCfg().func("/World/GroundFallback", sim_utils.GroundPlaneCfg())

    # fallback dome light so textures are visible even if scene lights failed export
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.9)).func(
        "/World/FallbackDomeLight",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.9)),
    )

    # RGB camera for capture
    cam_cfg = CameraCfg(
        prim_path="/World/CaptureCam",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(args.cam_pos), rot=(0.0, 0.0, 0.0, 1.0), convention="world",
        ),
    )
    cam = Camera(cfg=cam_cfg)

    sim.reset()

    # Point the capture camera at the target (offset's identity rot would
    # otherwise look down its local axis).
    import torch
    cam.set_world_poses_from_view(
        eyes=torch.tensor([args.cam_pos], device=sim.device),
        targets=torch.tensor([args.cam_target], device=sim.device),
    )

    # warm up a few frames so renderer fills RGB
    for _ in range(10):
        sim.step()
        cam.update(dt=sim_cfg.dt)

    import numpy as np
    from PIL import Image
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rgb = cam.data.output["rgb"]
    if rgb is None:
        print("[error] camera produced no RGB. is --enable_cameras set?")
    else:
        img = rgb[0].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(img[..., :3]).save(args.out)
        print(f"[scene] saved {args.out}  shape={img.shape}")

    if args.interactive:
        while simulation_app.is_running():
            sim.step()
            cam.update(dt=sim_cfg.dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
