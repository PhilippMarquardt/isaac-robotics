"""Scripted switch animation. No finger, no physics. Every period the switch
mesh visibly depresses by ~3 mm, latches the light state, then springs back.

This is the simplest possible "see it moving" demo. It bypasses PhysX entirely;
the switch Xform is translated kinematically via XformCommonAPI.

For the physics-correct version (robot finger pushes a prismatic joint) see
`demo_auto_toggle.py`.
"""
import argparse
import math
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_usd", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "usd", "pipe_lamp_interactive.usd"
)))
parser.add_argument("--out_dir", default="outputs/scripted_switch")
parser.add_argument("--cycles", type=int, default=8)
parser.add_argument("--period_s", type=float, default=5.0)
parser.add_argument("--motion", choices=("rotate_y", "rotate_x", "translate_z"),
                    default="rotate_y", help="how the switch animates each press")
parser.add_argument("--angle_deg", type=float, default=22.0,
                    help="rotation amplitude (rotate_y/rotate_x modes)")
parser.add_argument("--depress_mm", type=float, default=3.0,
                    help="translation amplitude (translate_z mode)")
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
from runtime.interactions import ToggleableLight  # noqa: E402


def _save_rgb(cam, path):
    import numpy as np
    from PIL import Image
    rgb = cam.data.output["rgb"]
    if rgb is None:
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
    print(f"[demo] meta: {list(meta.keys())}")

    light_sub = meta.get("light_prim", "/Visual/InteractiveLight")
    bulb_mat_sub = meta.get("bulb_material_in_visual")
    on_intensity = float(meta.get("on_intensity", 20000.0))
    on_color = tuple(meta.get("on_color", (1.0, 0.95, 0.85)))
    switch_sub = meta.get("switch_prim_in_visual")
    if switch_sub is None:
        raise SystemExit("[demo] need switch_prim_in_visual in metadata")

    def to_world(sub_path):
        if sub_path and sub_path.startswith("/Prop"):
            return f"/World/Lamp{sub_path[len('/Prop'):]}"
        return f"/World/Lamp{sub_path}" if sub_path else None

    light_full = to_world(light_sub)
    bulb_mat_full = to_world(bulb_mat_sub) if bulb_mat_sub else None
    switch_full = f"/World/Lamp/Visual{switch_sub}"
    print(f"[demo] switch prim: {switch_full}")

    switch_prim = stage.GetPrimAtPath(switch_full)
    if not switch_prim.IsValid():
        raise SystemExit(f"[demo] switch prim not found at {switch_full}")

    from pxr import UsdGeom, Gf
    switch_xform = UsdGeom.XformCommonAPI(switch_prim)
    rest_t, rest_r, rest_s, rest_pivot, _ = switch_xform.GetXformVectors(0.0)
    print(f"[demo] switch rest translate: {rest_t}  rotateXYZ: {rest_r}")

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
        light_prim_path=light_full,
        on_intensity=on_intensity, on_color=on_color,
        emissive_material_path=bulb_mat_full,
    )
    lamp.set_on(False)

    DT = 1.0 / 60.0
    period_steps = int(round(args.period_s / DT))
    depress = args.depress_mm * 1e-3
    angle_amp = args.angle_deg
    D = max(1, int(period_steps * 0.10))
    held_until = 2 * D
    return_until = 3 * D
    light_state = False
    cycles_done = 0
    s = 0

    # Side toggles flip between two states; track sign so the switch latches in
    # whichever direction the previous press left it.
    flip_sign = -1  # next press goes negative first

    def set_animation(progress: float, sign: int):
        """progress is 0..1 across the press half; sign is +1 or -1 for direction.
        Final resting attitude is 'sign * amplitude' so the switch latches."""
        if args.motion == "translate_z":
            offset = -depress * progress * sign  # depress sign chooses direction
            switch_xform.SetTranslate(Gf.Vec3d(rest_t[0], rest_t[1], rest_t[2] + offset))
        elif args.motion == "rotate_x":
            switch_xform.SetRotate(Gf.Vec3f(rest_r[0] + angle_amp * progress * sign,
                                            rest_r[1], rest_r[2]))
        else:  # rotate_y
            switch_xform.SetRotate(Gf.Vec3f(rest_r[0],
                                            rest_r[1] + angle_amp * progress * sign,
                                            rest_r[2]))

    # Latched toggle: after each press the switch stays at sign*amplitude until
    # the next cycle flips it the other way.
    current_sign = 1  # rest at center; first press goes to sign = +1 (light ON)
    pending_sign = current_sign
    while simulation_app.is_running():
        in_cycle = s % period_steps
        if in_cycle < D:
            # flip from current to new direction over D steps
            t = in_cycle / D
            progress = (1 - 2 * t)  # 1 -> -1 over D
            sign_now = current_sign if t < 0.5 else pending_sign
            # smoother: linearly cross from current to pending
            blend = current_sign + (pending_sign - current_sign) * t
            set_animation(abs(blend), 1 if blend >= 0 else -1)
            if in_cycle == D - 1:
                current_sign = pending_sign
                light_state = (current_sign > 0)
                lamp.set_on(light_state)
                print(f"[demo] cycle {cycles_done}: latched sign={current_sign} -> light={light_state}")
                _save_rgb(cam, os.path.join(args.out_dir, f"cycle_{cycles_done:02d}_on{int(light_state)}.png"))
        else:
            # hold at the latched position
            set_animation(1.0, current_sign)

        sim.step()
        cam.update(dt=DT)

        s += 1
        if in_cycle == period_steps - 1:
            pending_sign = -current_sign  # next cycle will flip
            cycles_done += 1
            if args.cycles > 0 and cycles_done >= args.cycles:
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
