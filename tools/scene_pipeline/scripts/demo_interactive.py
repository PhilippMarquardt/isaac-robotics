"""Generic interactive-asset demo runner.

Spawns one or more wrapper-USD assets (built by `make_interactive.py`), reads
the `interactive_v2` metadata, and animates every described part on a schedule.
`animated_part` entries flip between two states each cycle; `linked_light`
entries toggle a `light` entry's intensity at each flip.
"""
import argparse
import os
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", action="append", required=True,
                    help="path to a wrapper USD (built by make_interactive.py). "
                         "Pass multiple --usd flags to spawn several assets.")
parser.add_argument("--pos", action="append", default=None,
                    help="per-USD world translation 'x,y,z'. Repeat to match --usd.")
parser.add_argument("--out_dir", default="outputs/interactive_demo")
parser.add_argument("--cycles", type=int, default=6)
parser.add_argument("--period_s", type=float, default=5.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cam_pos", type=float, nargs=3, default=[1.5, -2.0, 1.4])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.0, 0.7])
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


def parse_pos(s):
    return tuple(float(x) for x in s.split(","))


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
    sim.set_camera_view(list(args.cam_pos), list(args.cam_target))

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)).func(
        "/World/AmbientDome",
        sim_utils.DomeLightCfg(intensity=args.dome_intensity, color=(0.7, 0.75, 0.85)),
    )

    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()

    # Spawn each USD and gather actors.
    positions = [parse_pos(p) for p in (args.pos or [])]
    actors = []  # list of dicts describing one asset's runtime handles
    for i, usd in enumerate(args.usd):
        prop_path = f"/World/Asset_{i}"
        pos = positions[i] if i < len(positions) else (i * 2.0, 0.0, 0.0)
        spawn = sim_utils.UsdFileCfg(usd_path=usd)
        spawn.func(prop_path, spawn, translation=pos)
        raw = stage.GetPrimAtPath(prop_path).GetCustomDataByKey("interactive_v2") or {}
        import json as _json
        items = _json.loads(raw.get("interactives_json", "[]")) if raw else []
        synth = _json.loads(raw.get("synthetic_json", "[]")) if raw else []
        meta = {"items": items, "synth": synth,
                "asset_source": raw.get("asset_source") if raw else None}
        if not items and not synth:
            print(f"[demo] WARN: {usd} has no interactives")
        actor = {"prop_path": prop_path, "meta": meta,
                 "lights": {}, "parts": [], "world_pos": pos}

        def resolve_light_path(sub):
            return f"{prop_path}{sub[len('/Prop'):]}" if sub.startswith("/Prop") else f"{prop_path}{sub}"

        # Lights — from both asset-sourced and synthetic.
        for it in items + synth:
            if it.get("kind") != "light":
                continue
            light_sub = it["light_prim"]
            full_light = resolve_light_path(light_sub)
            mat_sub = it.get("emissive_material_in_visual")
            full_mat = f"{prop_path}/Visual{mat_sub}" if mat_sub else None
            actor["lights"][it["name"]] = ToggleableLight(
                light_prim_path=full_light,
                on_intensity=float(it.get("on_intensity", 5000.0)),
                on_color=tuple(it.get("on_color", (1.0, 0.95, 0.85))),
                emissive_material_path=full_mat,
            )

        # Animated parts — asset-sourced live under /Visual; synthetic under /Synth.
        def collect_part(it, source):
            if source == "asset":
                sub = it["prim_in_visual"]
                full = f"{prop_path}/Visual{sub}"
            else:
                sub = it["prim_in_synth"]
                full = f"{prop_path}/Synth{sub}"
            prim = stage.GetPrimAtPath(full)
            if not prim.IsValid():
                print(f"[demo] WARN: part prim not found at {full}")
                return
            # All prims of this cluster — each gets the same xform delta.
            extra_subs = it.get("extra_prims_in_visual") or []
            full_paths = [full] + [f"{prop_path}/Visual{e}" for e in extra_subs]
            xforms = []
            rest_states = []
            motion = it.get("motion", "rotate_y")
            world_center = it.get("world_center")
            for fp in full_paths:
                pr = stage.GetPrimAtPath(fp)
                if not pr.IsValid():
                    continue
                xa = UsdGeom.XformCommonAPI(pr)
                rt, rr, _, _, _ = xa.GetXformVectors(0.0)
                if motion.startswith("rotate") and world_center is not None:
                    try:
                        xa.SetPivot(Gf.Vec3f(*world_center))
                    except Exception:
                        pass
                xforms.append(xa)
                rest_states.append((tuple(rt), tuple(rr)))
            if not xforms:
                print(f"[demo] WARN: no valid prims for {it['name']}")
                return
            actor["parts"].append({
                "name": it["name"],
                "xforms": xforms,
                "rests": rest_states,
                "world_center": world_center,
                "motion": motion,
                "amplitude": float(it.get("amplitude", 22.0)),
                "linked_light": it.get("linked_light"),
                "triggers": it.get("triggers"),
                "current_sign": 1,
                "pending_sign": 1,
            })

        for it in items:
            if it.get("kind") == "animated_part":
                collect_part(it, "asset")
        for it in synth:
            if it.get("kind") == "animated_part":
                collect_part(it, "synth")
        actors.append(actor)
        print(f"[demo] spawned {prop_path} from {os.path.basename(usd)}  "
              f"lights={list(actor['lights'])}  parts={[p['name'] for p in actor['parts']]}")

    # Initialize all lights OFF.
    for a in actors:
        for lamp in a["lights"].values():
            lamp.set_on(False)

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

    DT = 1.0 / 60.0
    period_steps = int(round(args.period_s / DT))
    D = max(1, int(period_steps * 0.20))   # animation window
    light_states = {(a_i, lname): False for a_i, a in enumerate(actors) for lname in a["lights"]}

    def apply_part(part, sign_now: float):
        """sign_now in [-1, +1] continuous. Sets the same transform offset on
        every member of this cluster (so multi-piece parts move as one)."""
        amp = part["amplitude"]
        motion = part["motion"]
        for xform, (rt, rr) in zip(part["xforms"], part["rests"]):
            if motion == "rotate_y":
                xform.SetRotate(Gf.Vec3f(rr[0], rr[1] + amp * sign_now, rr[2]))
            elif motion == "rotate_x":
                xform.SetRotate(Gf.Vec3f(rr[0] + amp * sign_now, rr[1], rr[2]))
            elif motion == "rotate_z":
                xform.SetRotate(Gf.Vec3f(rr[0], rr[1], rr[2] + amp * sign_now))
            elif motion in ("translate_x", "translate_y", "translate_z"):
                axis = "xyz".index(motion[-1])
                delta = [0.0, 0.0, 0.0]
                delta[axis] = (amp * 1e-3) * sign_now
                xform.SetTranslate(Gf.Vec3d(rt[0] + delta[0], rt[1] + delta[1], rt[2] + delta[2]))

    cycles_done = 0
    s = 0
    while simulation_app.is_running():
        in_cycle = s % period_steps
        # For each part, animate this step.
        for a_i, a in enumerate(actors):
            for part in a["parts"]:
                cur, pen = part["current_sign"], part["pending_sign"]
                if in_cycle < D:
                    t = in_cycle / D
                    blend = cur + (pen - cur) * t
                else:
                    blend = pen
                apply_part(part, blend)
                # Toggle linked light at midpoint of transition (so light state
                # changes when the switch is roughly centered).
                if in_cycle == D // 2 and part["linked_light"] is not None and cur != pen:
                    lname = part["linked_light"]
                    if lname in a["lights"]:
                        key = (a_i, lname)
                        light_states[key] = not light_states[key]
                        a["lights"][lname].set_on(light_states[key])
                        print(f"[demo] cycle {cycles_done}: {a['prop_path']}.{lname} -> {light_states[key]}")

        sim.step()
        cam.update(dt=DT)

        if in_cycle == D and cycles_done == 0:
            _save_rgb(cam, os.path.join(args.out_dir, f"cycle_00_after_flip.png"))
        elif in_cycle == D:
            _save_rgb(cam, os.path.join(args.out_dir, f"cycle_{cycles_done:02d}.png"))

        s += 1
        if in_cycle == period_steps - 1:
            # Latch new state, set next pending = flip.
            for a in actors:
                # First decide each part's own next pending.
                for part in a["parts"]:
                    part["current_sign"] = part["pending_sign"]
                    part["pending_sign"] = -part["current_sign"]
                # Then sync triggered parts: a "trigger" relationship means
                # when source part flips, target part flips with it.
                by_name = {p["name"]: p for p in a["parts"]}
                for part in a["parts"]:
                    target_name = part.get("triggers")
                    if target_name and target_name in by_name:
                        # match the target's pending to mirror source's pending.
                        by_name[target_name]["pending_sign"] = part["pending_sign"]
            cycles_done += 1
            if args.cycles > 0 and cycles_done >= args.cycles:
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
