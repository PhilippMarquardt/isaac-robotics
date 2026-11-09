"""Probe hand joints + test raw_hand to bypass pink IK action."""
import json
from llm_traj.client.tools import SimClient

c = SimClient()
try:
    c.reset(seed=0, task="debug fingers")
    pre = c._rpc("debug_hands")
    print("PRE CLOSE (joint info):")
    for n, info in pre.get("joints", {}).items():
        print(f"  {n}: id={info['id']} pos={info['pos']:.4f} target={info['target']:.4f} "
              f"limits=[{info['limit_low']:.3f}, {info['limit_high']:.3f}]")
    print("\nJoint order in articulation:")
    for i, n in enumerate(pre.get("all_joint_names", [])):
        if "hand" in n:
            print(f"  [{i}] {n}")
    c.set_gripper("close", hold_steps=60, hand="left")
    post = c._rpc("debug_hands")
    print("\nPOST left close (via pink IK action):")
    for n, info in post.get("joints", {}).items():
        if "left" in n:
            print(f"  {n}: pos={info['pos']:.4f} target={info['target']:.4f}")

    # Now try direct raw_hand bypass
    raw = c._rpc("raw_hand", targets={
        "left_hand_index_0_joint": 1.0,
        "left_hand_middle_0_joint": 1.0,
        "left_hand_index_1_joint": 1.0,
        "left_hand_middle_1_joint": 1.0,
    }, hold_steps=60)
    print("\nPOST raw_hand direct (bypass pink):")
    print(json.dumps(raw.get("after", {}), indent=2))
finally:
    c.close()
