"""Runtime control of interactive scene elements in Isaac Lab.

Isaac Lab natively wraps rigid bodies, articulations, and sensors. Things like
lights and material properties are not articulations — but every attribute on a
USD prim is editable at sim time. This module is the thin shim that exposes
those as friendly python objects: `lamp.set_on(True)`, `door.set_open(0.7)`, etc.

Each interaction holds a reference to one or more USD prims and writes
attribute values when called. Call from any step of the sim loop.

Design rule: an `Interaction` does not own physics. The scene config still
owns articulations / rigid bodies. An `Interaction` only touches USD attributes
that the scene config doesn't already manage (light intensity, emissive color,
visibility, custom userProps signals).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# pxr is only available after AppLauncher has booted Isaac kit.
# Import lazily so this module can be referenced before app launch.


@dataclass
class ToggleableLight:
    """Wraps a UsdLuxLight prim + optional emissive material binding.

    set_on(True)  -> intensity = on_intensity, emissive_color = on_color
    set_on(False) -> intensity = 0,           emissive_color = (0,0,0)
    """

    light_prim_path: str
    on_intensity: float = 3000.0
    on_color: tuple = (1.0, 0.95, 0.85)
    emissive_material_path: Optional[str] = None  # binds to the bulb mesh's material

    def _light_attr(self, name: str):
        from pxr import Usd, UsdLux
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.light_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"light prim not found: {self.light_prim_path}")
        # UsdLux light attributes live under `inputs:` namespace in USD 21.05+.
        attr = prim.GetAttribute(f"inputs:{name}")
        if not attr or not attr.IsValid():
            attr = prim.GetAttribute(name)  # older variant
        return attr

    def _material_attr(self, name: str):
        if self.emissive_material_path is None:
            return None
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        # PreviewSurface input lives on the shader prim under the material.
        shader_path = f"{self.emissive_material_path}/Shader"
        prim = stage.GetPrimAtPath(shader_path)
        if not prim or not prim.IsValid():
            return None
        return prim.GetAttribute(f"inputs:{name}")

    def set_on(self, on: bool):
        from pxr import Gf
        i = self._light_attr("intensity")
        if i is not None and i.IsValid():
            i.Set(self.on_intensity if on else 0.0)
        c = self._light_attr("color")
        if c is not None and c.IsValid():
            c.Set(Gf.Vec3f(*self.on_color) if on else Gf.Vec3f(0.0, 0.0, 0.0))
        ec = self._material_attr("emissive_color")
        if ec is not None and ec.IsValid():
            ec.Set(Gf.Vec3f(*self.on_color) if on else Gf.Vec3f(0.0, 0.0, 0.0))
        ei = self._material_attr("emissive_intensity")
        if ei is not None and ei.IsValid():
            ei.Set(self.on_intensity if on else 0.0)


@dataclass
class PressButton:
    """Pressable button. Two modes:

    Trigger-volume (legacy): set `world_pos` + `radius`, call `step(probe_xyz)`
    every sim step. Fires when probe enters the sphere.

    Joint-position (preferred): set `joint_prim_path`, `joint_travel`, optionally
    `press_fraction`. Reads the PhysX joint state and fires when the joint is
    depressed past `press_fraction * joint_travel`. Call `step()` with no args.

    Rising-edge only; debounced via `cooldown_steps`.
    """

    world_pos: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.02
    joint_prim_path: Optional[str] = None
    joint_travel: float = 0.003  # meters; assumed lower=-travel, upper=0
    press_fraction: float = 0.6
    cooldown_steps: int = 12
    _was_pressed: bool = False
    _cooldown_left: int = 0

    def _joint_position(self) -> float:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.joint_prim_path)
        if not prim or not prim.IsValid():
            return 0.0
        # PhysX writes current joint state into `state:linear:physics:position`
        for attr_name in (
            "state:linear:physics:position",
            "physics:state:linear:position",
            "physics:jointPosition",
        ):
            a = prim.GetAttribute(attr_name)
            if a and a.IsValid() and a.HasAuthoredValue():
                v = a.Get()
                if v is not None:
                    return float(v)
        return 0.0

    def step(self, probe_world_pos=None) -> bool:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if self.joint_prim_path is not None:
            jp = self._joint_position()
            # Joint travel runs from 0 (rest, upper) down to -joint_travel (pressed).
            depress_frac = -jp / self.joint_travel if self.joint_travel > 0 else 0.0
            currently_pressed = depress_frac >= self.press_fraction
        elif probe_world_pos is not None:
            x, y, z = probe_world_pos
            bx, by, bz = self.world_pos
            d2 = (x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2
            currently_pressed = d2 < self.radius ** 2
        else:
            currently_pressed = False

        rising = currently_pressed and not self._was_pressed and self._cooldown_left == 0
        self._was_pressed = currently_pressed
        if rising:
            self._cooldown_left = self.cooldown_steps
        return rising


# Future siblings (interface sketch — implement when each is needed):
#
# @dataclass
# class Drawer:
#     articulation_view_key: str   # e.g. "drawer" in the InteractiveScene dict
#     joint_name: str              # e.g. "slider"
#     def set_open(self, t: float):  # t in [0,1]; maps to joint pos target
#         ...
#
# @dataclass
# class Door:
#     articulation_view_key: str
#     joint_name: str              # revolute hinge
#     def set_angle(self, rad: float): ...
