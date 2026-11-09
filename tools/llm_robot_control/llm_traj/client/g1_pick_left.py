"""Closed-loop scripted left-hand pick: small steps, observe between each."""
from llm_traj.client.tools import SimClient

QUAT = [0, 0, 1, 0]  # palm-down via Y180 (verified via test_quats.py)
HAND = "left"


def step_towards(c, target, *, gripper, step=0.03, max_iters=20,
                 tol=0.015, max_steps=50, tol_rot=0.4):
    """Move EE towards target in ~step-size chunks, reading pose each time."""
    for i in range(max_iters):
        s = c.get_state()
        cur = s["left_ee_pos"]
        dx = target[0] - cur[0]; dy = target[1] - cur[1]; dz = target[2] - cur[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        print(f"  iter {i:2d} cur={cur} dist={dist:.4f}")
        if dist < tol:
            print(f"  reached (dist {dist:.4f} < tol {tol})")
            return cur
        scale = min(step / dist, 1.0)
        wp = [cur[0] + dx * scale, cur[1] + dy * scale, cur[2] + dz * scale]
        r = c.move_ee_to(wp, QUAT, hand=HAND, gripper=gripper,
                         tol_pos=0.01, tol_rot=tol_rot, max_steps=max_steps)
        # detect IK stall: moved <20% of requested step
        achieved = r["left_ee_pos"]
        moved = ((achieved[0] - cur[0]) ** 2 + (achieved[1] - cur[1]) ** 2
                 + (achieved[2] - cur[2]) ** 2) ** 0.5
        if moved < 0.2 * min(step, dist):
            print(f"  IK stall (moved {moved:.4f} of requested {scale * dist:.4f}); stopping")
            return achieved
    print(f"  hit max_iters {max_iters}")
    return c.get_state()["left_ee_pos"]


c = SimClient()
try:
    s = c.reset(seed=0, task="pick cube left hand, lift 15cm (closed-loop)")
    cx, cy, cz = s["cube_pos"]
    print(f"reset: cube=({cx:.3f},{cy:.3f},{cz:.3f}) rest_z={s['cube_rest_z_world']:.4f}")
    print(f"start left_ee={s['left_ee_pos']}")

    print("\n[1] approach hover 16cm above cube")
    step_towards(c, [cx, cy, cz + 0.16], gripper="open", step=0.04, max_iters=25)

    print("\n[2] descend onto cube (small steps)")
    grasp = step_towards(c, [cx, cy, cz + 0.00], gripper="open", step=0.015,
                         max_iters=25, tol=0.012)
    print(f"  grasp pose = {grasp}")

    print("\n[3] close fingers")
    r = c.set_gripper("close", hold_steps=120, hand=HAND)
    print(f"  left_opening={r['left_gripper_opening']:.3f}")

    lift0 = c.check_lifted(min_lift=0.02)
    print(f"  cube_z post-close = {lift0['cube_height']:.4f} "
          f"delta={lift0['delta_from_rest']:.4f}")

    print("\n[4] lift straight up (small steps)")
    final = step_towards(c, [grasp[0], grasp[1], grasp[2] + 0.20], gripper="close",
                         step=0.02, max_iters=25, tol=0.02)
    print(f"  final ee = {final}")

    lift = c.check_lifted(min_lift=0.05)
    print(f"\nlifted={lift['lifted']} cube_z={lift['cube_height']:.4f} "
          f"delta={lift['delta_from_rest']:.4f}")
finally:
    c.close()
