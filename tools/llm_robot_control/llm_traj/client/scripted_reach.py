"""Hard-coded pick-and-lift of the cube. Sanity check: if this fails, the IK
layer is broken; debug that before bringing an LLM in.
"""
from __future__ import annotations

import argparse
import json

from llm_traj.client.tools import SimClient

# Franka "pointing-down" EE orientation in robot base frame.
# quat (w, x, y, z) for a 180deg rotation about X — flips tool-z to point toward -Z_world.
DOWN_QUAT = [0.0, 1.0, 0.0, 0.0]


def run(seed: int = 0, verbose: bool = True) -> dict:
    c = SimClient()
    try:
        st = c.reset(seed=seed, task="scripted pick")
        cube = st["cube_pos"]
        if verbose:
            print(f"[reset] cube_pos={cube}  ee_pos={st['ee_pos']}  ee_quat={st['ee_quat_wxyz']}")

        # 1. pre-grasp: hover 12 cm above cube
        pre = [cube[0], cube[1], cube[2] + 0.12]
        r1 = c.move_ee_to(pre, DOWN_QUAT, gripper="open", tol_pos=0.015, max_steps=300)
        if verbose:
            print(f"[pre-grasp] arrived={r1['arrived']} pos_err={r1['pos_err']:.3f}")

        # 2. descend to cube center so fingertips straddle the cube
        grasp = [cube[0], cube[1], cube[2] - 0.005]
        r2 = c.move_ee_to(grasp, DOWN_QUAT, tol_pos=0.012, max_steps=240)
        if verbose:
            print(f"[descend] arrived={r2['arrived']} pos_err={r2['pos_err']:.3f}")

        # 3. close gripper
        r3 = c.set_gripper("close", hold_steps=30)
        if verbose:
            print(f"[grip close] opening={r3['gripper_opening']:.3f}")

        # 4. lift 20 cm straight up
        lift = [cube[0], cube[1], cube[2] + 0.22]
        r4 = c.move_ee_to(lift, DOWN_QUAT, gripper="close", tol_pos=0.015, max_steps=240)
        if verbose:
            print(f"[lift] arrived={r4['arrived']} cube_height={r4['cube_height_world']:.3f}")

        # 5. success check
        lifted = c.check_lifted(min_lift=0.10)
        grasped = c.is_grasped()
        if verbose:
            print(f"[check] lifted={lifted['lifted']} grasped={grasped['grasped']} "
                  f"cube_z={lifted['cube_height']:.3f}")

        return {
            "seed": seed,
            "lifted": lifted["lifted"],
            "grasped": grasped["grasped"],
            "cube_height": lifted["cube_height"],
        }
    finally:
        c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run(seed=args.seed)
    print(json.dumps(result, indent=2))
