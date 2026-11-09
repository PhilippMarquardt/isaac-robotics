"""Smallest interactive scene demo: ground + lamp base + sphere light.

Runs `set_on(False)` -> render -> save `lamp_off.png`.
Runs `set_on(True)`  -> render -> save `lamp_on.png`.

Proves the runtime-state pattern: a python call changes light intensity at
sim time and the renderer reflects it. Same shape will be used for drawers,
doors, switches, water taps, etc.

Use:
    isaaclab.bat -p scripts/demo_lamp_toggle.py --out_dir outputs/lamp_demo
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="outputs/lamp_demo")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# RTX+--headless+--enable_cameras crashes on this machine; force windowed cam mode.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg, Camera

# bolt scripts/ onto sys.path so we can import the sibling `runtime` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.interactions import ToggleableLight  # noqa: E402


CAM_POS = (1.8, 1.8, 1.4)
CAM_TARGET = (0.0, 0.0, 0.3)


def _save_rgb(cam: Camera, path: str):
    import numpy as np
    from PIL import Image
    rgb = cam.data.output["rgb"]
    if rgb is None:
        print(f"[demo] camera produced no rgb for {path}")
        return False
    img = rgb[0].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(img[..., :3]).save(path)
    print(f"[demo] saved {path}  shape={img.shape}")
    return True


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
    sim.set_camera_view(list(CAM_POS), list(CAM_TARGET))

    # Ground + dim ambient dome so the "off" state is dark but the lamp shape
    # is still visible. The dome is intentionally dim so the toggleable light
    # produces a visible delta.
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=80.0, color=(0.4, 0.45, 0.6)).func(
        "/World/AmbientDome",
        sim_utils.DomeLightCfg(intensity=80.0, color=(0.4, 0.45, 0.6)),
    )

    # Lamp base — a small cube on the ground.
    base_cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.3, 0.35), metallic=0.6, roughness=0.4,
        ),
    )
    base_cfg.func("/World/Lamp/Base", base_cfg, translation=(0.0, 0.0, 0.2))

    # Lamp bulb — a small sphere with emissive material, sitting on top of the base.
    bulb_cfg = sim_utils.SphereCfg(
        radius=0.08,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.95, 0.85),
            emissive_color=(0.0, 0.0, 0.0),  # starts off
        ),
    )
    bulb_cfg.func("/World/Lamp/Bulb", bulb_cfg, translation=(0.0, 0.0, 0.48))

    # The actual light source — a sphere light at the bulb center, starts off.
    light_cfg = sim_utils.SphereLightCfg(intensity=0.0, color=(1.0, 0.95, 0.85), radius=0.08)
    light_cfg.func("/World/Lamp/Light", light_cfg, translation=(0.0, 0.0, 0.48))

    # Camera for capture. Pose set by camera_view above; offset is identity.
    cam_cfg = CameraCfg(
        prim_path="/World/CaptureCam",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAM_POS, rot=(1.0, 0.0, 0.0, 0.0), convention="world",
        ),
    )
    cam = Camera(cfg=cam_cfg)
    sim.reset()
    import torch
    cam.set_world_poses_from_view(
        eyes=torch.tensor([CAM_POS], device=sim.device),
        targets=torch.tensor([CAM_TARGET], device=sim.device),
    )

    # Interaction handle. Talks straight to USD attributes via omni.usd.
    lamp = ToggleableLight(light_prim_path="/World/Lamp/Light", on_intensity=12000.0)

    # Warm up the renderer.
    for _ in range(6):
        sim.step(); cam.update(dt=1.0 / 60.0)

    # Frame 1: light off (initial state).
    lamp.set_on(False)
    for _ in range(6):
        sim.step(); cam.update(dt=1.0 / 60.0)
    _save_rgb(cam, os.path.join(args.out_dir, "lamp_off.png"))

    # Frame 2: light on.
    lamp.set_on(True)
    for _ in range(8):
        sim.step(); cam.update(dt=1.0 / 60.0)
    _save_rgb(cam, os.path.join(args.out_dir, "lamp_on.png"))

    simulation_app.close()


if __name__ == "__main__":
    main()
