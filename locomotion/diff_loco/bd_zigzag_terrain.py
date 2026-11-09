"""Boston-Dynamics-style zigzag angled-plate heightfield terrain.

Robot spawns on central flat platform. Alternating angled plates extend
outward in +x and -x directions. Each successive plate sits offset to the
opposite side in y, so the robot must hop diagonally left-right-left.

Each plate is tilted from inner-edge low to outer-edge high (cup-like
landing surface). Difficulty scales: gap width, plate vertical step, and
tilt angle. At difficulty=0 the plates form an easy zigzag walk; at
difficulty=1 they require committed diagonal jumps.

Field layout (top-down, +x right, +y up):

      [P3+y][   ][P5+y]
   [P1-y][PLATFORM][P2-y][P4-y]
      [P3+y][   ][P5+y]    (mirror on -x side)

Function signature follows IsaacLab `@height_field_to_mesh` convention.
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def zigzag_plates_terrain(difficulty: float, cfg: "HfZigzagPlatesTerrainCfg") -> np.ndarray:
    """Generate a Boston-Dynamics-style zigzag of angled plates.

    Args:
        difficulty: 0..1.
        cfg: HfZigzagPlatesTerrainCfg.

    Returns:
        2D int16 heightfield of shape (width_pixels, length_pixels).
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    plate_x_pix = max(1, int(cfg.plate_size[0] / cfg.horizontal_scale))
    plate_y_pix = max(1, int(cfg.plate_size[1] / cfg.horizontal_scale))
    platform_pix = max(1, int(cfg.platform_width / cfg.horizontal_scale))
    offset_y_pix = int(cfg.plate_offset_y / cfg.horizontal_scale)

    # difficulty interpolation
    gap = cfg.gap_x_range[0] + difficulty * (cfg.gap_x_range[1] - cfg.gap_x_range[0])
    angle = cfg.angle_range[0] + difficulty * (cfg.angle_range[1] - cfg.angle_range[0])
    base_h = cfg.plate_height_range[0] + difficulty * (cfg.plate_height_range[1] - cfg.plate_height_range[0])

    gap_pix = max(1, int(gap / cfg.horizontal_scale))
    base_h_pix = int(base_h / cfg.vertical_scale)
    holes_pix = int(cfg.holes_depth / cfg.vertical_scale)

    # max rise from inner edge to outer edge across plate width (in vert pixels)
    rise_pix = int(plate_y_pix * cfg.horizontal_scale * np.tan(angle) / cfg.vertical_scale)

    cx = width_pixels // 2
    cy = length_pixels // 2

    hf_raw = np.full((width_pixels, length_pixels), holes_pix, dtype=np.float32)

    # central flat platform at height 0
    px0 = cx - platform_pix // 2
    px1 = cx + platform_pix // 2
    py0 = cy - platform_pix // 2
    py1 = cy + platform_pix // 2
    hf_raw[px0:px1, py0:py1] = 0

    def place_plate(x_center: int, y_center: int) -> None:
        x0 = max(0, x_center - plate_x_pix // 2)
        x1 = min(width_pixels, x_center + plate_x_pix // 2)
        y0 = max(0, y_center - plate_y_pix // 2)
        y1 = min(length_pixels, y_center + plate_y_pix // 2)
        if x0 >= x1 or y0 >= y1:
            return
        # outer side = side further from terrain centerline (cy)
        # if y_center > cy: outer is +y → height grows toward y1
        # if y_center < cy: outer is -y → height grows toward y0
        side = 1 if y_center > cy else (-1 if y_center < cy else 1)
        ys = np.arange(y0, y1)
        if side == 1:
            frac = (ys - y0) / max(1, (y1 - 1 - y0))
        else:
            frac = (y1 - 1 - ys) / max(1, (y1 - 1 - y0))
        col_h = base_h_pix + frac * rise_pix  # shape (yspan,)
        hf_raw[x0:x1, y0:y1] = col_h[None, :]

    # plate stride along x (from one plate center to the next)
    stride = plate_x_pix + gap_pix
    n_per_side = cfg.num_plates_per_side

    # +x side
    for i in range(1, n_per_side + 1):
        x_c = px1 + gap_pix + plate_x_pix // 2 + (i - 1) * stride
        if x_c >= width_pixels:
            break
        sgn = 1 if (i % 2 == 1) else -1
        y_c = cy + sgn * offset_y_pix
        place_plate(x_c, y_c)

    # -x side (mirror)
    for i in range(1, n_per_side + 1):
        x_c = px0 - gap_pix - plate_x_pix // 2 - (i - 1) * stride
        if x_c < 0:
            break
        sgn = 1 if (i % 2 == 1) else -1
        y_c = cy + sgn * offset_y_pix
        place_plate(x_c, y_c)

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfZigzagPlatesTerrainCfg(HfTerrainBaseCfg):
    """Config for zigzag angled-plate heightfield terrain."""

    function = zigzag_plates_terrain

    plate_size: tuple[float, float] = (0.7, 1.0)
    """Plate footprint (x, y) in meters. y must be < 2*plate_offset_y so the
    centerline strip stays empty (forces zigzag)."""

    plate_offset_y: float = 0.7
    """How far each plate is offset from the centerline in y (meters).
    Plates alternate ±this offset. With plate_size_y=1.0 the centerline gap is
    2*0.7 - 1.0 = 0.4m wide."""

    gap_x_range: tuple[float, float] = (0.18, 0.40)
    """Min/max gap between adjacent plates along x (meters). Scales with difficulty."""

    angle_range: tuple[float, float] = (0.0, 0.25)
    """Min/max plate tilt angle in radians. Outer edge is raised; inner edge is base."""

    plate_height_range: tuple[float, float] = (0.05, 0.20)
    """Min/max plate base height above the platform (meters). Scales with difficulty."""

    num_plates_per_side: int = 3
    """Number of plates extending out from the platform on each side (±x)."""

    platform_width: float = 2.0
    """Width of the central flat platform (meters)."""

    holes_depth: float = -0.5
    """Depth of the pit between plates (meters, negative). Forces jumping."""
