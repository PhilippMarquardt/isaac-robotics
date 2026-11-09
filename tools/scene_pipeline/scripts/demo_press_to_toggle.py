"""Robot-finger pushes the lamp's switch -> lamp turns on.

For now the "robot" is a kinematic sphere that we drive in a straight line toward
the switch world position recorded in the lamp's interactive metadata. Same code
will work when the finger is replaced by a real arm end-effector — the press
detection is decoupled from how the probe got there.

Renders 4 frames:
    01_approach_far.png   - finger above & away
    02_approach_near.png  - finger ~halfway
    03_pressed.png        - moment of press (light just toggled on)
    04_after.png          - finger withdrawn, light still on
"""
import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_usd", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "usd", "pipe_lamp_interactive.usd"
)))
parser.add_argument("--out_dir", default="outputs/press_demo")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--cam_pos", type=float, nargs=3, default=[0.45, -0.55, 0.40])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, -0.05, 0.15])
parser.add_argument("--dome_intensity", type=float, default=400.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import CameraCfg, Camera

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.interactions import ToggleableLight, PressButton  # noqa: E402


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

    # Lamp.
    lamp_spawn = sim_utils.UsdFileCfg(usd_path=args.lamp_usd)
    lamp_spawn.func("/World/Lamp", lamp_spawn, translation=(0.0, 0.0, 0.0))

    # Read interactive metadata.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    spawned = stage.GetPrimAtPath("/World/Lamp")
    meta = spawned.GetCustomDataByKey("interactive") or {}
    print(f"[demo] meta keys: {list(meta.keys())}")
    light_sub = meta.get("light_prim", "/Visual/InteractiveLight")
    bulb_mat_sub = meta.get("bulb_material_in_visual")
    on_intensity = float(meta.get("on_intensity", 20000.0))
    on_color = tuple(meta.get("on_color", (1.0, 0.95, 0.85)))
    switch_world = tuple(meta.get("switch_world_center", (0.0, -0.024, 0.09)))

    def to_world(sub_path):
        if sub_path.startswith("/Prop"):
            return f"/World/Lamp{sub_path[len('/Prop'):]}"
        return f"/World/Lamp{sub_path}"

    light_full_path = to_world(light_sub)
    bulb_mat_full = (to_world(bulb_mat_sub) if bulb_mat_sub else None)
    print(f"[demo] switch world center: {switch_world}")
    print(f"[demo] light: {light_full_path}")

    # Finger probe — a small bright sphere as a kinematic rigid body. Kinematic
    # = position set by user, but participates in physics collision, so it
    # actually pushes the switch joint when it touches.
    finger_radius = 0.012
    start_pos = (switch_world[0], switch_world[1], switch_world[2] + 0.10)
    finger_cfg = sim_utils.SphereCfg(
        radius=finger_radius,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.2, 0.2), metallic=0.1, roughness=0.4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    finger_cfg.func("/World/Finger", finger_cfg, translation=start_pos)

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

    joint_sub = meta.get("switch_joint_path")  # e.g. "/Prop/SwitchJoint"
    joint_full = to_world(joint_sub) if joint_sub else None
    joint_travel = float(meta.get("switch_joint_travel", 0.003))
    if joint_full:
        print(f"[demo] using joint-based press detection at {joint_full}")
        button = PressButton(joint_prim_path=joint_full, joint_travel=joint_travel,
                             press_fraction=0.5)
    else:
        print("[demo] no joint metadata, falling back to trigger volume")
        button = PressButton(world_pos=switch_world, radius=finger_radius + 0.012)
    lamp.set_on(False)

    # Finger trajectory: descend straight down from above the switch, dwell
    # pushing the joint, then retract.
    sx, sy, sz = switch_world
    apex_z = sz + 0.10                       # 10 cm above the switch center
    contact_z = sz - 0.005                   # finger center 5 mm below switch center
                                             # (fully depressed against joint stop)
    above = (sx, sy, apex_z)
    pressed = (sx, sy, contact_z)
    descend_steps = 30
    dwell_steps = 10
    retract_steps = 20

    from pxr import Gf, UsdGeom
    finger_xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Finger"))

    def set_finger(pos):
        finger_xform.SetTranslate(Gf.Vec3d(*pos))

    total_steps = descend_steps + dwell_steps + retract_steps
    save_at = {
        2:                                       "01_approach_far.png",
        descend_steps - 4:                       "02_approach_near.png",
        descend_steps + dwell_steps // 2:        "03_pressed.png",
        total_steps - 2:                         "04_after.png",
    }

    pressed_once = False
    light_state = False
    for s in range(total_steps + 1):
        if s <= descend_steps:
            t = s / descend_steps
            pos = tuple(above[i] + (pressed[i] - above[i]) * t for i in range(3))
        elif s <= descend_steps + dwell_steps:
            pos = pressed
        else:
            t = (s - descend_steps - dwell_steps) / max(1, retract_steps)
            t = min(1.0, t)
            pos = tuple(pressed[i] + (above[i] - pressed[i]) * t for i in range(3))
        set_finger(pos)

        # Press detection — joint-position based now.
        if button.step(pos):
            pressed_once = True
            light_state = not light_state
            lamp.set_on(light_state)
            print(f"[demo] step {s}: PRESS -> light_state={light_state}")

        sim.step()
        cam.update(dt=1.0 / 60.0)

        if s in save_at:
            _save_rgb(cam, os.path.join(args.out_dir, save_at[s]))

    if not pressed_once:
        print("[demo] WARNING: finger never triggered the press detector. "
              "Check switch_world_center vs waypoint geometry.")

    simulation_app.close()


if __name__ == "__main__":
    main()
