"""Auto-toggle the lamp switch every 5 seconds.

Same physics setup as `demo_press_to_toggle.py` but loops: the finger descends,
presses, retracts, waits, repeats. Every cycle saves a PNG snapshot after the
press so you can flip through the sequence and see the light alternating.

Run with `--cycles N` to stop after N presses; otherwise it runs until the Kit
window is closed.
"""
import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_usd", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "usd", "pipe_lamp_interactive.usd"
)))
parser.add_argument("--out_dir", default="outputs/auto_toggle")
parser.add_argument("--cycles", type=int, default=6,
                    help="number of press cycles; <=0 runs indefinitely")
parser.add_argument("--period_s", type=float, default=5.0,
                    help="seconds between presses")
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
from runtime.physics_attach import attach_lamp_switch_physics  # noqa: E402


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
    print(f"[demo] saved {path}")
    return True


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
    sim.set_camera_view(list(args.cam_pos), list(args.cam_target))

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)).func(
        "/World/AmbientDome",
        sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)),
    )

    lamp_spawn = sim_utils.UsdFileCfg(usd_path=args.lamp_usd)
    lamp_spawn.func("/World/Lamp", lamp_spawn, translation=(0.0, 0.0, 0.0))

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    spawned = stage.GetPrimAtPath("/World/Lamp")
    meta = spawned.GetCustomDataByKey("interactive") or {}

    # Apply rigid bodies + joint live on the composed scene.
    switch_sub_in_visual = meta.get("switch_prim_in_visual")
    body_sub_in_visual = meta.get("lamp_body_in_visual")
    sw_center = tuple(meta.get("switch_world_center", (0.0, -0.024, 0.09)))
    if switch_sub_in_visual and body_sub_in_visual:
        phys = attach_lamp_switch_physics(
            lamp_root_path="/World/Lamp",
            body_in_visual=body_sub_in_visual,
            switch_in_visual=switch_sub_in_visual,
            switch_world_center=sw_center,
        )
        meta = dict(meta)
        meta["switch_joint_path"] = "/Prop/SwitchJoint"  # to_world() -> /World/Lamp/SwitchJoint
        print(f"[demo] attached physics: {phys}")
    light_sub = meta.get("light_prim", "/Visual/InteractiveLight")
    bulb_mat_sub = meta.get("bulb_material_in_visual")
    on_intensity = float(meta.get("on_intensity", 20000.0))
    on_color = tuple(meta.get("on_color", (1.0, 0.95, 0.85)))
    switch_world = tuple(meta.get("switch_world_center", (0.0, -0.024, 0.09)))
    joint_sub = meta.get("switch_joint_path")
    joint_travel = float(meta.get("switch_joint_travel", 0.003))

    def to_world(sub_path):
        if sub_path and sub_path.startswith("/Prop"):
            return f"/World/Lamp{sub_path[len('/Prop'):]}"
        return f"/World/Lamp{sub_path}" if sub_path else None

    finger_radius = 0.012
    sx, sy, sz = switch_world
    above = (sx, sy, sz + 0.10)
    pressed = (sx, sy, sz - 0.005)
    sim_utils.SphereCfg(
        radius=finger_radius,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.2, 0.2), metallic=0.1, roughness=0.4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ).func("/World/Finger", sim_utils.SphereCfg(radius=finger_radius),
           translation=above)

    cam_cfg = CameraCfg(
        prim_path="/World/CaptureCam", update_period=0.0,
        height=args.height, width=args.width, data_types=["rgb"],
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
        light_prim_path=to_world(light_sub),
        on_intensity=on_intensity, on_color=on_color,
        emissive_material_path=to_world(bulb_mat_sub),
    )
    joint_full = to_world(joint_sub)
    button = PressButton(joint_prim_path=joint_full, joint_travel=joint_travel,
                         press_fraction=0.5) if joint_full else \
             PressButton(world_pos=switch_world, radius=finger_radius + 0.012)
    print(f"[demo] joint={joint_full}  travel={joint_travel}")

    from pxr import Gf, UsdGeom
    finger_xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Finger"))

    def set_finger(pos):
        finger_xform.SetTranslate(Gf.Vec3d(*pos))

    DT = 1.0 / 60.0
    steps_per_period = int(round(args.period_s / DT))
    # Within a period: 0..D = descend, D..D+H = hold, D+H..D+H+R = retract, rest idle.
    descend = int(steps_per_period * 0.20)
    hold = int(steps_per_period * 0.10)
    retract = int(steps_per_period * 0.20)
    light_state = False
    lamp.set_on(False)
    cycles_done = 0

    s = 0
    while simulation_app.is_running():
        in_cycle = s % steps_per_period
        if in_cycle == 0 and cycles_done > 0:
            # nothing to do at the boundary
            pass
        if in_cycle < descend:
            t = in_cycle / max(1, descend)
            pos = tuple(above[i] + (pressed[i] - above[i]) * t for i in range(3))
        elif in_cycle < descend + hold:
            pos = pressed
        elif in_cycle < descend + hold + retract:
            t = (in_cycle - descend - hold) / max(1, retract)
            pos = tuple(pressed[i] + (above[i] - pressed[i]) * t for i in range(3))
        else:
            pos = above
        set_finger(pos)

        if button.step(pos):
            light_state = not light_state
            lamp.set_on(light_state)
            print(f"[demo] cycle {cycles_done}: PRESS -> light_state={light_state}")
            _save_rgb(cam, os.path.join(args.out_dir, f"cycle_{cycles_done:02d}_on{int(light_state)}.png"))

        sim.step()
        cam.update(dt=DT)

        s += 1
        if in_cycle == steps_per_period - 1:
            cycles_done += 1
            if args.cycles > 0 and cycles_done >= args.cycles:
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
