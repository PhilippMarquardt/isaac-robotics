"""Headless Open3D renderer — outputs paper-style PNG with G1 + lidar + GT.

No viewer needed; produces a high-resolution PNG directly.

Robot is built from primitive meshes (head, torso, hips, legs, arms) sized to real
G1 dimensions. Lidar + GT rendered as point clouds with elevation coloring.

Usage:
    python viz_open3d.py --chunk /root/lidar_dataset_v3/chunk_00080.pt \\
        --frame 500 --window 200 --out viz.png
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import open3d as o3d


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def world_to_local(pts_w: torch.Tensor, origin_w: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ]).reshape(3, 3)
    delta = pts_w - origin_w.unsqueeze(0)
    return delta @ R


def make_box(center, size, color):
    box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
    box.translate(np.array(center) - np.array(size) / 2)
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def make_sphere(center, radius, color, resolution=20):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def build_g1(base_pos: np.ndarray, base_quat: np.ndarray):
    """Assemble a G1 humanoid from primitive meshes at the given world pose."""
    parts: list[o3d.geometry.TriangleMesh] = []
    orange = [1.0, 0.50, 0.18]
    dark_orange = [0.70, 0.32, 0.08]
    darker = [0.45, 0.20, 0.05]
    head = [0.95, 0.55, 0.20]

    # all coords in body frame (Z up)
    parts.append(make_box([0.0, 0.0, 0.30], [0.20, 0.16, 0.40], orange))   # torso
    parts.append(make_box([0.0, 0.0, 0.0], [0.22, 0.16, 0.10], dark_orange))  # hips
    parts.append(make_sphere([0.0, 0.0, 0.62], 0.08, head))   # head
    parts.append(make_box([0.0, 0.0, 0.55], [0.10, 0.12, 0.10], head))  # neck box

    # legs
    for sy in (-1, 1):
        parts.append(make_box([0.0, 0.08 * sy, -0.20], [0.10, 0.10, 0.30], dark_orange))   # thigh
        parts.append(make_box([0.0, 0.08 * sy, -0.50], [0.09, 0.09, 0.28], darker))        # calf
        parts.append(make_box([0.06, 0.08 * sy, -0.66], [0.20, 0.10, 0.05], [0.25, 0.12, 0.05]))  # foot

    # arms (slightly out from torso)
    for sy in (-1, 1):
        parts.append(make_box([0.0, 0.18 * sy, 0.45], [0.08, 0.10, 0.25], orange))   # upper arm
        parts.append(make_box([0.0, 0.20 * sy, 0.20], [0.07, 0.08, 0.22], dark_orange))   # forearm

    # combine, rotate, translate
    g1 = o3d.geometry.TriangleMesh()
    for p in parts:
        g1 += p
    R = quat_xyzw_to_rotmat(base_quat)
    g1.rotate(R, center=(0, 0, 0))
    g1.translate(base_pos)
    g1.compute_vertex_normals()
    return g1


def colorize_z(pts: np.ndarray, low_rgb=(0.20, 0.30, 0.45), high_rgb=(0.85, 0.90, 0.95)) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((0, 3))
    z = pts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    rng = max(zmax - zmin, 1e-6)
    t = ((z - zmin) / rng).reshape(-1, 1)
    lo = np.array(low_rgb)
    hi = np.array(high_rgb)
    return lo + t * (hi - lo)


def make_pcd(pts: np.ndarray, colors: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors.shape[0] == pts.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_offscreen(geometries, view_extent: float, base_pos: np.ndarray,
                      out_path: str, width: int = 1600, height: int = 1200,
                      elev_deg: float = 28.0, azim_deg: float = -55.0,
                      bg_color=(1.0, 1.0, 1.0)):
    """Render the scene to a PNG via OffscreenRenderer."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([*bg_color, 1.0]))

    # standard pbr-ish material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = 0.7

    pt_mat = o3d.visualization.rendering.MaterialRecord()
    pt_mat.shader = "defaultUnlit"
    pt_mat.point_size = 6.0

    for i, g in enumerate(geometries):
        name = f"obj_{i}"
        if isinstance(g, o3d.geometry.PointCloud):
            renderer.scene.add_geometry(name, g, pt_mat)
        else:
            renderer.scene.add_geometry(name, g, mat)

    # camera placement: orbit around the robot at view_extent radius, at given elev/azim
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    r = view_extent
    eye = base_pos + np.array([
        r * math.cos(elev) * math.cos(azim),
        r * math.cos(elev) * math.sin(azim),
        r * math.sin(elev),
    ])
    center = base_pos + np.array([0.0, 0.0, 0.5])  # look slightly above base
    up = np.array([0.0, 0.0, 1.0])
    renderer.scene.camera.look_at(center, eye, up)
    renderer.scene.camera.set_projection(
        45.0, width / height, 0.1, 200.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )

    # lighting
    renderer.scene.scene.set_sun_light(
        direction=[0.4, -0.6, -0.7], color=[1.0, 0.95, 0.9], intensity=80000.0,
    )
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.show_axes(False)

    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img, 9)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frame", type=int, default=500)
    ap.add_argument("--window", type=int, default=0,
                    help="0 = single frame; otherwise accumulate ±window/2 frames")
    ap.add_argument("--out", default="viz_open3d.png")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=-55.0)
    ap.add_argument("--view_extent", type=float, default=14.0,
                    help="camera distance from robot (m)")
    ap.add_argument("--show_gt", action="store_true", default=True)
    args = ap.parse_args()

    print(f"[INFO] loading {args.chunk}")
    d = torch.load(args.chunk, map_location="cpu", weights_only=True)
    F = d["episode_id"].shape[0]
    c = max(0, min(args.frame, F - 1))

    base_pos = d["base_pos_world"][c].numpy()
    base_quat = d["base_quat_world"][c].numpy()

    # gather lidar (current frame or windowed) in WORLD frame
    if args.window <= 0:
        lidar_pts = d["lidar_points_world"][c][d["lidar_hit_mask"][c]].numpy()
    else:
        same_ep = d["episode_id"][c]
        lo = max(0, c - args.window // 2)
        hi = min(F, c + args.window // 2)
        accum = []
        for t in range(lo, hi):
            if d["episode_id"][t] != same_ep:
                continue
            ptsw = d["lidar_points_world"][t][d["lidar_hit_mask"][t]].numpy()
            if ptsw.shape[0] > 0:
                accum.append(ptsw)
        lidar_pts = np.concatenate(accum, axis=0) if accum else np.zeros((0, 3))
    print(f"[INFO] lidar pts: {lidar_pts.shape[0]} (window {args.window})")

    # GT terrain (single frame is fine — dense already)
    gt_pts = d["gt_points_world"][c][d["gt_hit_mask"][c]].numpy() if args.show_gt else np.zeros((0, 3))

    # build geometries
    g1 = build_g1(base_pos, base_quat)

    geoms = [g1]
    if lidar_pts.shape[0] > 0:
        lidar_pcd = make_pcd(lidar_pts, colorize_z(lidar_pts, (0.18, 0.28, 0.40), (0.85, 0.90, 0.95)))
        geoms.append(lidar_pcd)
    if gt_pts.shape[0] > 0:
        gt_pcd = make_pcd(gt_pts, colorize_z(gt_pts, (0.55, 0.40, 0.20), (0.30, 0.65, 0.45)))
        geoms.append(gt_pcd)

    # ground plane disc (subtle, makes the scene readable)
    ground = o3d.geometry.TriangleMesh.create_cylinder(radius=args.view_extent * 0.85, height=0.005)
    ground.translate(base_pos - np.array([0, 0, 0.71]))
    ground.paint_uniform_color([0.92, 0.92, 0.92])
    ground.compute_vertex_normals()
    geoms.insert(0, ground)

    render_offscreen(
        geoms,
        view_extent=args.view_extent,
        base_pos=base_pos,
        out_path=args.out,
        width=args.width,
        height=args.height,
        elev_deg=args.elev,
        azim_deg=args.azim,
    )


if __name__ == "__main__":
    main()
