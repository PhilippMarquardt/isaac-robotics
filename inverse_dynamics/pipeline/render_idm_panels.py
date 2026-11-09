"""Render 3-panel video: cosmos (left) | MuJoCo sim (mid) | joint-dots (right).

Input:
  - cosmos video mp4 (93 frames typical, ~16fps)
  - traj_pillow.npz (traj_phys (N, 41): left_leg6, right_leg6, waist3,
    left_arm7, right_arm7, left_hand6, right_hand6)

Mapping (IDM dex5 -> MuJoCo g1_29dof_inspire):
  - body 29 dims 1:1 into BODY_JOINT_NAMES (Unitree SDK order)
  - hands 6 dex5 dims -> 12 inspire URDF joints via inspire6_to_inspire_urdf

Panels:
  - L: cosmos frames resized to H x H
  - M: mujoco 3rd-person render at H x H
  - R: black bg H x H, 41 red dots per frame (x=dim_idx, y=joint_angle)
"""
import argparse, sys, numpy as np, cv2, mujoco
from pathlib import Path

SIM_VIZ = Path(__file__).resolve().parents[2] / "gr00t_finetune" / "sim_viz"
sys.path.insert(0, str(SIM_VIZ))
from viz_mujoco import (
    BODY_JOINT_NAMES, HAND_JOINT_NAMES_INSPIRE_L, HAND_JOINT_NAMES_INSPIRE_R,
    build_joint_qpos_map, inspire6_to_inspire_urdf,
)

SCENE_XML = str(SIM_VIZ / "assets" / "scene_pillow_inspire_official.xml")


def third_person_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.4
    cam.lookat[:] = [0.0, 0.4, 0.8]
    cam.azimuth = 100.0
    cam.elevation = -12.0
    return cam


def render_sim_frames(traj_phys: np.ndarray, H: int) -> list[np.ndarray]:
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    body_map = build_joint_qpos_map(m)
    hand_l = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                       for n in HAND_JOINT_NAMES_INSPIRE_L], dtype=np.int32)
    hand_r = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                       for n in HAND_JOINT_NAMES_INSPIRE_R], dtype=np.int32)

    # freejoint root for robot: pose robot upright, feet near ground
    # find "pelvis" or root joint — G1 URDF root is pelvis via freejoint
    renderer = mujoco.Renderer(m, width=H, height=H)
    cam = third_person_camera()

    frames = []
    for i in range(len(traj_phys)):
        a = traj_phys[i]
        # root: fixed upright standing pose
        d.qpos[0:3] = [0.0, 0.0, 0.745]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # body 29 dims -> 29 joints
        for k in range(29):
            d.qpos[body_map[k]] = a[k]
        # hands: dex5 6-dim -> inspire 12-dim (reverse inspire6_to_inspire_urdf expects
        # [0,1] where 1=open. Our IDM outputs are joint angles directly; just clip and
        # scale to approximate close amount 0-1).
        lh = np.clip(a[29:35], 0.0, 1.0)
        rh = np.clip(a[35:41], 0.0, 1.0)
        lh_urdf = inspire6_to_inspire_urdf(lh, 'left')
        rh_urdf = inspire6_to_inspire_urdf(rh, 'right')
        for k in range(12):
            d.qpos[hand_l[k]] = lh_urdf[k]
            d.qpos[hand_r[k]] = rh_urdf[k]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frame = renderer.render()  # RGB H,W,3
        frames.append(frame.copy())
    return frames


def render_dots_frames(traj_phys: np.ndarray, H: int) -> list[np.ndarray]:
    """Red dots on black bg. x = joint idx (0..40), y = angle normalized per dim."""
    traj = traj_phys.copy()
    # normalize each dim to [0, 1] using global min/max across trajectory
    lo, hi = traj.min(0), traj.max(0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (traj - lo) / span  # (N, 41)

    frames = []
    pad = 20
    for i in range(len(traj)):
        canvas = np.zeros((H, H, 3), dtype=np.uint8)
        for k in range(41):
            x = int(pad + k * (H - 2 * pad) / 40)
            y = int(pad + (1 - norm[i, k]) * (H - 2 * pad))
            cv2.circle(canvas, (x, y), 4, (60, 60, 255), -1)  # BGR red
        # axes labels
        cv2.putText(canvas, "legs", (pad, H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(canvas, "arms", (int(H * 0.4), H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(canvas, "hands", (int(H * 0.75), H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(canvas, f"t={i}", (pad, pad + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # BGR -> RGB for consistency with mujoco
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return frames


def load_cosmos_frames(path: str, n_target: int, H: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    # resample/subsample to n_target
    idx = np.linspace(0, len(raw) - 1, n_target).astype(int)
    out = []
    for j in idx:
        f = raw[j]
        h, w = f.shape[:2]
        # center-crop to square, resize
        side = min(h, w)
        y0 = (h - side) // 2; x0 = (w - side) // 2
        f = f[y0:y0 + side, x0:x0 + side]
        f = cv2.resize(f, (H, H))
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    data = np.load(args.traj, allow_pickle=True)
    traj = data['traj_phys']  # (N, 41)
    print(f"trajectory: {traj.shape}")

    print("rendering mujoco sim...")
    sim_frames = render_sim_frames(traj, args.h)
    print("rendering joint dots...")
    dot_frames = render_dots_frames(traj, args.h)
    print("loading cosmos video...")
    cos_frames = load_cosmos_frames(args.video, len(traj), args.h)

    H, W = args.h, args.h * 3
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for cos, sim, dot in zip(cos_frames, sim_frames, dot_frames):
        row = np.concatenate([cos, sim, dot], axis=1)  # RGB
        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved: {args.out} ({len(sim_frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
