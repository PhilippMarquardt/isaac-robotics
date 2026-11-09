"""Probe which wrist orientation puts left-hand fingertips on the cube."""
from llm_traj.client.tools import SimClient

# Candidate wrist quats (wxyz). Goal: palm faces cube from above or front.
CANDIDATES = [
    ("palm_down_x180",  [0, 1, 0, 0]),
    ("palm_down_y180",  [0, 0, 1, 0]),
    ("palm_fwd_z90",    [0.7071, 0, 0, 0.7071]),
    ("palm_fwd_nz90",   [0.7071, 0, 0, -0.7071]),
    ("rx90_left",       [0.7071, 0.7071, 0, 0]),
    ("rx-90_left",      [0.7071, -0.7071, 0, 0]),
    ("mixed_1",         [0.5, 0.5, 0.5, 0.5]),
    ("mixed_2",         [0.5, -0.5, 0.5, -0.5]),
    ("ry90",            [0.7071, 0, 0.7071, 0]),
    ("ry-90",           [0.7071, 0, -0.7071, 0]),
]

c = SimClient()
try:
    for name, quat in CANDIDATES:
        s = c.reset(seed=0, task=f"probe {name}")
        cx, cy, cz = s["cube_pos"]
        # Hover directly above cube
        c.move_ee_to([cx, cy, cz + 0.18], quat, hand="left",
                     gripper="open", tol_pos=0.05, tol_rot=0.5, max_steps=250)
        # Descend
        r = c.move_ee_to([cx, cy, cz - 0.03], quat, hand="left",
                         gripper="open", tol_pos=0.02, tol_rot=0.5, max_steps=300)
        c.set_gripper("close", hold_steps=60, hand="left")
        # Check: does cube move with wrist after close+tiny lift?
        pre = c.check_lifted(min_lift=0.01)
        lifted = c.move_ee_to([cx, cy, cz + 0.15], quat, hand="left",
                              gripper="close", tol_pos=0.02, tol_rot=0.5, max_steps=250)
        post = c.check_lifted(min_lift=0.01)
        print(f"{name:20s}  wrist={r['left_ee_pos']}  "
              f"close_opening={r.get('left_gripper_opening', 0):.2f}  "
              f"cube_rise_post_close={pre['delta_from_rest']:.4f}  "
              f"cube_rise_after_lift={post['delta_from_rest']:.4f}  "
              f"lift_err={lifted['pos_err']:.3f}")
finally:
    c.close()
