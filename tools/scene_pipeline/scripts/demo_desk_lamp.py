"""Demo: render the Poly Haven desk lamp with the runtime-attached light off vs on.

Loads the interactive-wrapper USD produced by `make_interactive_lamp.py`, reads
its custom metadata to find the light prim path, then toggles via
`ToggleableLight`.
"""
import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_usd", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "usd", "pipe_lamp_interactive.usd"
)))
parser.add_argument("--out_dir", default="outputs/pipe_lamp_demo")
parser.add_argument("--cam_pos", type=float, nargs=3, default=[0.45, -0.65, 0.40])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, -0.03, 0.18])
parser.add_argument("--dome_intensity", type=float, default=400.0)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg, Camera

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.interactions import ToggleableLight  # noqa: E402




def _save_rgb(cam, path):
    import numpy as np
    from PIL import Image
    rgb = cam.data.output["rgb"]
    if rgb is None:
        print(f"[demo] no rgb for {path}")
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
    sim.set_camera_view(list(args.cam_pos), list(args.cam_target))

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)).func(
        "/World/AmbientDome",
        sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)),
    )

    # Spawn the interactive lamp USD under /World/Lamp.
    lamp_spawn = sim_utils.UsdFileCfg(usd_path=args.lamp_usd)
    lamp_spawn.func("/World/Lamp", lamp_spawn, translation=(0.0, 0.0, 0.0))

    # Read interactive metadata from the spawned prim to discover the light path.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    spawned = stage.GetPrimAtPath("/World/Lamp")
    meta = spawned.GetCustomDataByKey("interactive") or {}
    print(f"[demo] interactive meta: {dict(meta)}")
    light_sub_path = meta.get("light_prim", "/Visual/InteractiveLight")
    bulb_mat_sub = meta.get("bulb_material_in_visual")
    on_intensity = float(meta.get("on_intensity", 20000.0))
    on_color = tuple(meta.get("on_color", (1.0, 0.95, 0.85)))

    light_full_path = f"/World/Lamp{light_sub_path[len('/Prop'):]}" if light_sub_path.startswith("/Prop") else f"/World/Lamp{light_sub_path}"
    bulb_mat_full = (f"/World/Lamp/Visual{bulb_mat_sub}" if bulb_mat_sub else None)
    print(f"[demo] light path:   {light_full_path}")
    print(f"[demo] bulb mat:     {bulb_mat_full}")

    # Camera.
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
            pos=tuple(args.cam_pos), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
        ),
    )
    cam = Camera(cfg=cam_cfg)
    sim.reset()
    import torch
    cam.set_world_poses_from_view(
        eyes=torch.tensor([tuple(args.cam_pos)], device=sim.device),
        targets=torch.tensor([tuple(args.cam_target)], device=sim.device),
    )

    lamp = ToggleableLight(
        light_prim_path=light_full_path,
        on_intensity=on_intensity,
        on_color=on_color,
        emissive_material_path=bulb_mat_full,
    )

    for _ in range(6):
        sim.step(); cam.update(dt=1.0 / 60.0)

    lamp.set_on(False)
    for _ in range(6):
        sim.step(); cam.update(dt=1.0 / 60.0)
    _save_rgb(cam, os.path.join(args.out_dir, "desk_lamp_off.png"))

    lamp.set_on(True)
    for _ in range(10):
        sim.step(); cam.update(dt=1.0 / 60.0)
    _save_rgb(cam, os.path.join(args.out_dir, "desk_lamp_on.png"))

    simulation_app.close()


if __name__ == "__main__":
    main()
