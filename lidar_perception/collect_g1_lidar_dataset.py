"""Collect G1 lidar dataset for offline scene-encoder training (Hoeller 2022 setup).

Spawns N parallel G1 envs on rough terrain, drives each with the stock RSL-RL G1 rough
locomotion policy + randomized velocity commands, and dumps per-step (lidar pcl,
GT pcl, lidar mount pose, base pose, episode/step ids) to chunked .pt files.

Usage (head-less):

    cd /workspace/diff_loco/lidar_perception
    /workspace/IsaacLab/_isaac_sim/python.sh collect_g1_lidar_dataset.py \\
        --headless --num_envs 16 --num_frames 20000 \\
        --checkpoint /workspace/logs/rsl_rl/g1_rough/2026-04-26_16-12-00/model_last.pt \\
        --output_dir /workspace/lidar_dataset_v1

For a quick smoke test on flat ground / 1 env / few frames, pass `--smoke`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# -------- argparse --------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Collect G1 lidar dataset.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_frames", type=int, default=20000)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/workspace/lidar_dataset_v1",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/workspace/logs/rsl_rl/g1_rough/2026-04-26_16-12-00/model_last.pt",
)
parser.add_argument("--episode_length_s", type=float, default=50.0)
parser.add_argument("--frames_per_chunk", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Smoke test: 1 env, flat ground, 100 frames, no chunked save.",
)
parser.add_argument(
    "--terrain",
    choices=["plane", "rough", "diverse"],
    default="rough",
    help="plane: flat (smoke). rough: ROUGH_TERRAINS_CFG (Stage 2). diverse: LIDAR_DIVERSE_TERRAINS_CFG (Stage 4).",
)
parser.add_argument(
    "--motion",
    choices=["default", "forward", "backward", "diverse"],
    default="default",
    help="default: random. forward/backward: sustained ±1 m/s. diverse: variable speed + small yaw + lateral.",
)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    help="Enable RayCaster debug viz markers (requires non-headless).",
)
parser.add_argument(
    "--check_only",
    action="store_true",
    help="Run mesh-GT pre-checks (mesh extraction, single-frame .ply, voxel density), "
         "then exit without launching the full collection loop.",
)
parser.add_argument(
    "--check_ply_out",
    type=str,
    default="/tmp/gt_check.ply",
    help="Where to dump the single-frame .ply for sanity inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch sim
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# -------- imports (after app launch) --------------------------------------------------

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  (registers stock G1 task)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# our cfg lives next to this script — make it importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import lidar_env_cfg  # noqa: F401  (registers Isaac-Velocity-Rough-G1-LidarCollect-v0)
from lidar_env_cfg import (G1RoughLidarCollectCfg, G1FwdLidarCollectCfg,
                            G1BwdLidarCollectCfg, G1DiverseLidarCollectCfg)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)


# -------- helpers ---------------------------------------------------------------------


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """IsaacLab stores quaternions as (w, x, y, z); brief asks for (x, y, z, w)."""
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


def make_env_cfg(args) -> G1RoughLidarCollectCfg:
    if args.motion == "forward":
        cfg = G1FwdLidarCollectCfg()
    elif args.motion == "backward":
        cfg = G1BwdLidarCollectCfg()
    elif args.motion == "diverse":
        cfg = G1DiverseLidarCollectCfg()
    else:
        cfg = G1RoughLidarCollectCfg()
    cfg.scene.num_envs = 1 if args.smoke else args.num_envs
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = args.episode_length_s
    cfg.seed = args.seed
    if args.debug_vis:
        cfg.scene.lidar_scanner.debug_vis = True
        cfg.scene.gt_scanner.debug_vis = True
    terrain_choice = "plane" if args.smoke else args.terrain
    if terrain_choice == "plane":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        # parent's terrain_levels curriculum reads terrain_generator.size; disable on plane
        if hasattr(cfg, "curriculum") and getattr(cfg.curriculum, "terrain_levels", None) is not None:
            cfg.curriculum.terrain_levels = None
    elif terrain_choice == "diverse":
        from lidar_terrain_cfg import LIDAR_DIVERSE_TERRAINS_CFG
        cfg.scene.terrain.terrain_generator = LIDAR_DIVERSE_TERRAINS_CFG
    # "rough" keeps the parent ROUGH_TERRAINS_CFG (default)
    return cfg


# -------- main ------------------------------------------------------------------------


MESH_GT_PRIM_PATH = "/World/ground"
# half-extent of axis-aligned cube around base_pos for GT sampling.
# 3.0 m → 6 m cube — only slightly larger than training cube (HALF=1.6 m). Smaller
# cube means smaller KDTree query radius → ~5× faster collection. Visualization
# scripts that want the full lidar-range cloud should re-run with bigger cube.
MESH_GT_HALF = 3.0
MESH_GT_MAX_PTS = 30000  # ~13× smaller volume than 7.07 → scale point cap


def _build_mesh_gt_index(env):
    """Pre-sample dense surface points from the same terrain mesh the lidar raycasts.

    Returns:
        kdtree: scipy cKDTree of world-frame surface samples
        points: (N, 3) float32 array of those samples
    """
    from isaaclab.sensors.ray_caster.ray_caster import RayCaster as _RC
    import trimesh
    from scipy.spatial import cKDTree

    wp_mesh = _RC.meshes.get(MESH_GT_PRIM_PATH)
    if wp_mesh is None:
        raise RuntimeError(
            f"warp mesh {MESH_GT_PRIM_PATH} not loaded; available: "
            f"{list(_RC.meshes.keys())}"
        )
    verts = wp_mesh.points.numpy()  # (V, 3)
    faces = wp_mesh.indices.numpy().reshape(-1, 3)  # (F, 3)
    print(
        f"[MESH-GT] prim_path={MESH_GT_PRIM_PATH}  verts={verts.shape[0]}  "
        f"faces={faces.shape[0]}",
        flush=True,
    )

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    n_samples = min(30_000_000, max(5_000_000, int(tm.area * 600)))
    print(
        f"[MESH-GT] area={tm.area:.1f} m²  → sampling {n_samples} surface points",
        flush=True,
    )
    sampled, _ = trimesh.sample.sample_surface(tm, n_samples)
    sampled = sampled.astype(np.float32)
    print(
        f"[MESH-GT] samples bounds: {sampled.min(axis=0).tolist()} → "
        f"{sampled.max(axis=0).tolist()}",
        flush=True,
    )
    tree = cKDTree(sampled)
    print("[MESH-GT] KDTree built", flush=True)
    return tree, sampled


def _query_gt(tree, points: np.ndarray, base_pos: np.ndarray,
              half: float = MESH_GT_HALF, max_pts: int = MESH_GT_MAX_PTS):
    """Return up to `max_pts` mesh-surface points within the AABB cube around base_pos."""
    radius = (3 ** 0.5) * half  # bounding-sphere radius for a cube
    idx = tree.query_ball_point(base_pos, r=radius)
    if not idx:
        return np.zeros((0, 3), dtype=np.float32)
    cand = points[idx]
    diff = np.abs(cand - base_pos)
    inside = (diff[:, 0] <= half) & (diff[:, 1] <= half) & (diff[:, 2] <= half)
    pts = cand[inside]
    if pts.shape[0] > max_pts:
        sel = np.random.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[sel]
    return pts


def _voxel_density_check(gt_pts: np.ndarray, voxel_size: float = 0.05):
    """Return (occupied_voxels, mean_pts_per_voxel, min_pts_per_voxel)."""
    if gt_pts.shape[0] == 0:
        return 0, 0.0, 0
    keys = np.floor(gt_pts / voxel_size).astype(np.int64)
    keys_packed = keys[:, 0] * 100003 + keys[:, 1] * 10007 + keys[:, 2]
    uniq, counts = np.unique(keys_packed, return_counts=True)
    return int(uniq.shape[0]), float(counts.mean()), int(counts.min())


def main():
    print("[STAGE] make_env_cfg", flush=True)
    env_cfg = make_env_cfg(args)
    print("[STAGE] gym.make", flush=True)
    if args.motion == "forward":
        task_id = "Isaac-Velocity-Rough-G1-LidarFwd-v0"
    elif args.motion == "backward":
        task_id = "Isaac-Velocity-Rough-G1-LidarBwd-v0"
    elif args.motion == "diverse":
        task_id = "Isaac-Velocity-Rough-G1-LidarDiverse-v0"
    else:
        task_id = "Isaac-Velocity-Rough-G1-LidarCollect-v0"
    env = gym.make(task_id, cfg=env_cfg)
    print("[STAGE] RslRlVecEnvWrapper", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    print("[STAGE] build agent_cfg + handle_deprecated", flush=True)
    agent_cfg = G1RoughPPORunnerCfg()
    agent_cfg.device = "cuda:0"
    agent_cfg.experiment_name = "g1_rough"
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        import importlib.metadata as md
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
    except Exception as e:
        print(f"[WARN] handle_deprecated_rsl_rl_cfg skipped: {e}", flush=True)
    print("[STAGE] OnPolicyRunner", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[STAGE] runner.load ckpt: {args.checkpoint}", flush=True)
    runner.load(args.checkpoint)
    print("[STAGE] get_inference_policy", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[STAGE] policy ready", flush=True)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    lidar = scene.sensors["lidar_scanner"]
    gt = scene.sensors["gt_scanner"]
    device = env.unwrapped.device

    # Build perfect-GT KDTree from the actual terrain mesh (same one lidar raycasts).
    print("[STAGE] building mesh-GT KDTree", flush=True)
    gt_tree, gt_pts_world = _build_mesh_gt_index(env)

    if args.check_only:
        # === Pre-flight checks before committing to a full collection run ===
        print("[CHECK] running pre-flight checks", flush=True)

        body_names = robot.body_names
        print(f"[CHECK 0] robot.body_names ({len(body_names)}):", flush=True)
        for bn in body_names:
            print(f"  - {bn}", flush=True)
        print(
            f"[CHECK 0] lidar prim_path: {lidar.cfg.prim_path}  "
            f"offset.pos: {lidar.cfg.offset.pos}",
            flush=True,
        )
        # all body world Z heights for env 0 (sorted by Z so we can spot the head)
        body_pos = robot.data.body_pos_w[0].cpu().numpy()  # (B, 3)
        order = np.argsort(body_pos[:, 2])[::-1]  # tallest first
        print("[CHECK 0] all bodies sorted by Z:")
        for j in order:
            print(f"  z={body_pos[j, 2]:.3f}  {body_names[j]}  pos_w={body_pos[j].tolist()}", flush=True)
        print(f"[CHECK 0] root_pos_w: {robot.data.root_pos_w[0].cpu().numpy().tolist()}", flush=True)

        with torch.inference_mode():
            obs0 = env.get_observations()
            for _ in range(20):
                actions0 = policy(obs0)
                obs0, _, _, _ = env.step(actions0)

        i = 0
        frame_idx = 20  # we stepped 20 times before measuring
        base_pos0 = robot.data.root_pos_w[i].cpu().numpy()
        # lidar mount position (apply offset to torso link pose, same as _record_step)
        import isaaclab.utils.math as math_utils
        offset_local = torch.tensor(lidar.cfg.offset.pos, device=lidar.data.pos_w.device).expand(
            lidar.data.pos_w.shape[0], 3)
        lidar_pos_w = (lidar.data.pos_w + math_utils.quat_apply(lidar.data.quat_w, offset_local)).cpu().numpy()
        lidar_origin0 = lidar_pos_w[i]

        gt_pts0 = _query_gt(gt_tree, gt_pts_world, base_pos0,
                             half=MESH_GT_HALF, max_pts=MESH_GT_MAX_PTS)
        n_vox, mean_pts, min_pts = _voxel_density_check(gt_pts0 - base_pos0, voxel_size=0.05)

        try:
            lidar_pts0 = lidar.data.ray_hits_w[i][
                torch.isfinite(lidar.data.ray_hits_w[i]).all(dim=-1)
            ].cpu().numpy()
        except Exception as e:
            lidar_pts0 = np.zeros((0, 3), dtype=np.float32)
            print(f"[WARN] lidar pts read: {e}", flush=True)

        # ---- diagnostic prints (9 lines exactly as requested) -------------
        gt_min = gt_pts0.min(axis=0).tolist() if gt_pts0.shape[0] else [float('nan')]*3
        gt_max = gt_pts0.max(axis=0).tolist() if gt_pts0.shape[0] else [float('nan')]*3
        cube_min = (base_pos0 - MESH_GT_HALF).tolist()
        cube_max = (base_pos0 + MESH_GT_HALF).tolist()
        if gt_pts0.shape[0]:
            d = np.abs(gt_pts0 - base_pos0)
            inside = ((d[:, 0] <= MESH_GT_HALF) & (d[:, 1] <= MESH_GT_HALF) & (d[:, 2] <= MESH_GT_HALF))
            n_in = int(inside.sum()); n_out = int((~inside).sum())
        else:
            n_in = 0; n_out = 0
        print("=" * 64, flush=True)
        print(f"frame_idx: {frame_idx}", flush=True)
        print(f"base_pos_world: {base_pos0.tolist()}", flush=True)
        print(f"lidar_origin_world: {lidar_origin0.tolist()}", flush=True)
        print(f"gt_n_points: {gt_pts0.shape[0]}", flush=True)
        print(f"gt_points_world min: {gt_min}", flush=True)
        print(f"gt_points_world max: {gt_max}", flush=True)
        print(f"expected cube min (base - 1.6): {cube_min}", flush=True)
        print(f"expected cube max (base + 1.6): {cube_max}", flush=True)
        print(f"count GT points inside cube: {n_in}, outside: {n_out}", flush=True)
        print("=" * 64, flush=True)

        # ---- markers -------------------------------------------------------
        # Red pelvis: 500 pts in 10 cm sphere
        rng = np.random.default_rng(0)
        pelvis_dirs = rng.normal(size=(500, 3))
        pelvis_dirs /= np.linalg.norm(pelvis_dirs, axis=1, keepdims=True)
        pelvis_r = 0.10 * (rng.random(500) ** (1.0 / 3.0)).reshape(-1, 1)
        pelvis_pts = base_pos0 + pelvis_dirs * pelvis_r
        # Yellow lidar: 200 pts in 5 cm sphere
        lidar_dirs = rng.normal(size=(200, 3))
        lidar_dirs /= np.linalg.norm(lidar_dirs, axis=1, keepdims=True)
        lidar_r = 0.05 * (rng.random(200) ** (1.0 / 3.0)).reshape(-1, 1)
        lidar_marker_pts = lidar_origin0 + lidar_dirs * lidar_r
        # Green cube wireframe: 12 edges × 100 pts
        h = MESH_GT_HALF
        corners = np.array([
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h],
        ])
        edges_idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
            (4, 5), (5, 6), (6, 7), (7, 4),  # top square
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        cube_pts = []
        for a, b in edges_idx:
            t = np.linspace(0, 1, 100).reshape(-1, 1)
            cube_pts.append(corners[a] + t * (corners[b] - corners[a]))
        cube_pts = np.concatenate(cube_pts, axis=0) + base_pos0  # (1200, 3)

        # ---- assemble PLY --------------------------------------------------
        all_pts = np.concatenate([
            gt_pts0,                    # brown/tan
            lidar_pts0,                 # blue
            pelvis_pts,                 # red
            lidar_marker_pts,           # yellow
            cube_pts,                   # green
        ], axis=0).astype(np.float32)
        all_cols = np.concatenate([
            np.tile(np.array([180, 130, 60], dtype=np.uint8), (gt_pts0.shape[0], 1)),
            np.tile(np.array([60, 110, 200], dtype=np.uint8), (lidar_pts0.shape[0], 1)),
            np.tile(np.array([255,  0,  0], dtype=np.uint8), (pelvis_pts.shape[0], 1)),
            np.tile(np.array([255, 255,  0], dtype=np.uint8), (lidar_marker_pts.shape[0], 1)),
            np.tile(np.array([  0, 255,  0], dtype=np.uint8), (cube_pts.shape[0], 1)),
        ], axis=0)
        n_pts = all_pts.shape[0]
        with open(args.check_ply_out, "wb") as f:
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n_pts}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode("ascii"))
            body = np.empty(n_pts, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("r", "u1"), ("g", "u1"), ("b", "u1")])
            body["x"] = all_pts[:, 0]; body["y"] = all_pts[:, 1]; body["z"] = all_pts[:, 2]
            body["r"] = all_cols[:, 0]; body["g"] = all_cols[:, 1]; body["b"] = all_cols[:, 2]
            f.write(body.tobytes())
        print(
            f"[CHECK 2] saved {args.check_ply_out}  "
            f"(GT={gt_pts0.shape[0]}  lidar={lidar_pts0.shape[0]}  "
            f"pelvis={pelvis_pts.shape[0]}  lidar_marker={lidar_marker_pts.shape[0]}  "
            f"cube_wire={cube_pts.shape[0]})",
            flush=True,
        )
        print(f"[CHECK 3] @5cm voxels: occupied={n_vox}  mean={mean_pts:.2f}  min={min_pts}", flush=True)
        print("[CHECK] pre-flight done; exiting", flush=True)
        env.close()
        return

    n_envs = env_cfg.scene.num_envs
    n_lidar_rays = lidar.num_rays  # 40 × 500 = 20000
    n_gt_rays = gt.num_rays  # 64 × 64 = 4096
    print(
        f"[INFO] envs={n_envs}  lidar_rays={n_lidar_rays}  gt_rays={n_gt_rays}  "
        f"target_frames={args.num_frames}  smoke={args.smoke}"
    )

    if args.smoke:
        target_frames = 100
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        target_frames = args.num_frames
        os.makedirs(args.output_dir, exist_ok=True)

    # bookkeeping
    episode_id = torch.zeros(n_envs, dtype=torch.long, device=device)
    step_in_ep = torch.zeros(n_envs, dtype=torch.long, device=device)

    # per-chunk staging buffers (lists of cpu tensors, one entry per recorded sample)
    bufs = _new_chunk_bufs()
    chunk_idx = 0
    frames_total = 0
    t0 = time.time()

    obs = env.get_observations()
    sim = env.unwrapped.sim
    cmd_vx_default = {"forward": 1.0, "backward": -1.0}.get(args.motion, 0.0)
    # record the initial t=0 frame for ep 0 across all envs
    take = min(n_envs, target_frames - frames_total)
    _record_step(bufs, take, episode_id, step_in_ep, lidar, gt, robot, sim,
                 env=env, gt_tree=gt_tree, gt_pts_world=gt_pts_world,
                 mesh_gt_half=MESH_GT_HALF, mesh_gt_max=MESH_GT_MAX_PTS,
                 cmd_vx_default=cmd_vx_default)
    frames_total += take

    while frames_total < target_frames:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if dones.any():
            done_idx = dones.nonzero(as_tuple=True)[0]
            episode_id[done_idx] += 1
            step_in_ep[done_idx] = 0
            mask = torch.ones(n_envs, dtype=torch.bool, device=device)
            mask[done_idx] = False
            step_in_ep[mask] += 1
        else:
            step_in_ep += 1

        take = min(n_envs, target_frames - frames_total)
        _record_step(bufs, take, episode_id, step_in_ep, lidar, gt, robot, sim,
                     env=env, gt_tree=gt_tree, gt_pts_world=gt_pts_world,
                     mesh_gt_half=MESH_GT_HALF, mesh_gt_max=MESH_GT_MAX_PTS,
                     cmd_vx_default=cmd_vx_default)
        frames_total += take

        # flush full chunks (count is over records, not list entries)
        while not args.smoke and sum(t.shape[0] for t in bufs["episode_id"]) >= args.frames_per_chunk:
            chunk_path = os.path.join(args.output_dir, f"chunk_{chunk_idx:05d}.pt")
            _flush_chunk(bufs, args.frames_per_chunk, chunk_path)
            elapsed = time.time() - t0
            print(
                f"[FLUSH] {chunk_path}  total={frames_total}/{target_frames}  "
                f"elapsed={elapsed:.1f}s  fps={frames_total / max(elapsed, 1e-6):.1f}",
                flush=True,
            )
            chunk_idx += 1

    # final flush (all remaining)
    remaining_records = sum(t.shape[0] for t in bufs["episode_id"]) if bufs["episode_id"] else 0
    if remaining_records > 0:
        chunk_path = os.path.join(
            args.output_dir,
            "smoke_chunk.pt" if args.smoke else f"chunk_{chunk_idx:05d}.pt",
        )
        _flush_chunk(bufs, remaining_records, chunk_path)
        print(f"[FLUSH-FINAL] {chunk_path}  records={remaining_records}  total={frames_total}", flush=True)

    elapsed = time.time() - t0
    print(
        f"[DONE] {frames_total} frames in {elapsed:.1f}s "
        f"({frames_total / max(elapsed, 1e-6):.1f} fps) -> {args.output_dir}"
    )
    env.close()


# -------- chunk-buffer helpers --------------------------------------------------------


def _new_chunk_bufs() -> dict:
    return {
        "env_id": [],
        "episode_id": [],
        "step_in_episode": [],
        "timestamp_sim": [],
        "lidar_points_world": [],
        "lidar_hit_mask": [],
        "lidar_origin_world": [],
        "lidar_quat_world": [],
        "base_pos_world": [],
        "base_quat_world": [],
        "gt_points_world": [],   # (F, MESH_GT_MAX_PTS, 3) padded with zeros
        "gt_n_points": [],       # (F,) int32 valid prefix length
        "commanded_velocity": [],     # (F, 3) [vx_cmd, vy_cmd, wz_cmd] in base frame
        "actual_base_velocity": [],   # (F, 3) [vx, vy, vz] in world frame (from root_lin_vel_w)
    }


_CMD_WARNED = False


def _record_step(
    bufs: dict,
    n_to_record: int,
    episode_id: torch.Tensor,
    step_in_ep: torch.Tensor,
    lidar,
    gt,
    robot,
    sim,
    env=None,
    gt_tree=None,
    gt_pts_world=None,
    mesh_gt_half: float = 1.6,
    mesh_gt_max: int = 4096,
    cmd_vx_default: float = 0.0,
) -> None:
    """Pull current sensor data and append per-env rows to staging buffers.

    If `gt_tree` provided, replaces the downward raycaster GT with mesh-sampled
    points from the actual terrain trimesh inside the ±mesh_gt_half cube around
    each robot's base pose.
    """
    if n_to_record <= 0:
        return

    import isaaclab.utils.math as math_utils

    n_envs = lidar.data.pos_w.shape[0]
    sim_t = float(sim.current_time)

    # IsaacLab's RayCaster.data.pos_w reports the *tracked link's* world pose, NOT the
    # offset-applied sensor mount. Apply cfg.offset.pos manually so the saved pose is
    # the actual lidar mount in world frame.
    offset_local = torch.tensor(lidar.cfg.offset.pos, device=lidar.data.pos_w.device).expand(n_envs, 3)
    lidar_pos = lidar.data.pos_w + math_utils.quat_apply(lidar.data.quat_w, offset_local)  # (N, 3)
    lidar_quat = lidar.data.quat_w  # (N, 4) wxyz; lidar offset rot is identity by default
    lidar_hits = lidar.data.ray_hits_w  # (N, R, 3); inf where missed
    gt_hits = gt.data.ray_hits_w  # (N, 4096, 3)
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w

    lidar_hit_mask = torch.isfinite(lidar_hits).all(dim=-1)  # (N, R)
    lidar_safe = torch.where(lidar_hit_mask.unsqueeze(-1), lidar_hits, torch.zeros_like(lidar_hits))

    # Mesh-sampled GT — vectorized batch query: ONE cKDTree.query_ball_point call
    # for all envs (returns list of arrays), then per-env box filter + sub-sample.
    base_pos_np = base_pos.cpu().numpy()
    mesh_pts_per_env = np.zeros((n_envs, mesh_gt_max, 3), dtype=np.float32)
    n_pts_per_env = np.zeros(n_envs, dtype=np.int32)
    if gt_tree is not None and gt_pts_world is not None:
        radius = (3 ** 0.5) * mesh_gt_half
        # cKDTree.query_ball_point on (N, 3) array returns object array of index lists
        cand_lists = gt_tree.query_ball_point(base_pos_np, r=radius)
        for i in range(n_envs):
            idx = cand_lists[i]
            if not idx:
                continue
            cand = gt_pts_world[idx]
            diff = np.abs(cand - base_pos_np[i])
            inside = (diff[:, 0] <= mesh_gt_half) & (diff[:, 1] <= mesh_gt_half) & (diff[:, 2] <= mesh_gt_half)
            cand = cand[inside]
            if cand.shape[0] == 0:
                continue
            if cand.shape[0] > mesh_gt_max:
                sel = np.random.choice(cand.shape[0], mesh_gt_max, replace=False)
                cand = cand[sel]
            n = cand.shape[0]
            mesh_pts_per_env[i, :n] = cand
            n_pts_per_env[i] = n
    gt_safe = torch.from_numpy(mesh_pts_per_env)
    gt_n_pts = torch.from_numpy(n_pts_per_env)

    take = min(n_to_record, n_envs)
    sl = slice(0, take)

    # commanded velocity:
    #  - directed motion (forward/backward) → trust the cfg constant; IsaacLab's
    #    command_manager.get_command() in heading-mode stores a different tensor
    #    that reads as 0 even though the policy walks forward.
    #  - default motion → try real command_manager (may still be flaky).
    global _CMD_WARNED
    cmd_vel = torch.zeros((n_envs, 3), dtype=torch.float32)
    cmd_vel[:, 0] = cmd_vx_default
    if env is not None and cmd_vx_default == 0.0:
        try:
            inner = env
            while hasattr(inner, "unwrapped") and inner.unwrapped is not inner:
                inner = inner.unwrapped
            cmd = inner.command_manager.get_command("base_velocity")
            cmd_vel = cmd[:, :3].cpu().clone()
        except Exception as e:
            if not _CMD_WARNED:
                print(f"[WARN] command_manager access failed: {e}", flush=True)
                _CMD_WARNED = True
    actual_vel = robot.data.root_lin_vel_w  # (N, 3) world-frame linear vel

    bufs["env_id"].append(torch.arange(take, dtype=torch.long))
    bufs["episode_id"].append(episode_id[sl].cpu().clone())
    bufs["step_in_episode"].append(step_in_ep[sl].cpu().clone())
    bufs["timestamp_sim"].append(torch.full((take,), sim_t, dtype=torch.float32))
    bufs["lidar_points_world"].append(lidar_safe[sl].cpu().clone())
    bufs["lidar_hit_mask"].append(lidar_hit_mask[sl].cpu().clone())
    bufs["lidar_origin_world"].append(lidar_pos[sl].cpu().clone())
    bufs["lidar_quat_world"].append(wxyz_to_xyzw(lidar_quat[sl]).cpu().clone())
    bufs["base_pos_world"].append(base_pos[sl].cpu().clone())
    bufs["base_quat_world"].append(wxyz_to_xyzw(base_quat[sl]).cpu().clone())
    bufs["gt_points_world"].append(gt_safe[:take].clone())  # already CPU, padded
    bufs["gt_n_points"].append(gt_n_pts[:take].clone())
    bufs["commanded_velocity"].append(cmd_vel[sl].clone())
    bufs["actual_base_velocity"].append(actual_vel[sl].cpu().clone())


def _flush_chunk(bufs: dict, n_frames: int, path: str) -> None:
    """Concat staging lists, save first n_frames rows, retain the rest in `bufs`."""
    out = {}
    leftover = {}
    for k, v_list in bufs.items():
        full = torch.cat(v_list, dim=0)
        out[k] = full[:n_frames].contiguous()
        leftover[k] = full[n_frames:].contiguous()

    torch.save(out, path)

    for k in bufs:
        bufs[k][:] = [leftover[k]] if leftover[k].shape[0] > 0 else []


if __name__ == "__main__":
    try:
        main()
    finally:
        sim_app.close()
