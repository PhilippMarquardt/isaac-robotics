# LLM robot control through tool calls

An Isaac Lab process holds one environment and serves a small JSON-over-TCP API. A language model gets a
system prompt describing the tools and must solve the task within 20 calls; each tool call is one CLI
invocation, and every call is appended to an episode log.

Tools: `reset`, `get_state`, `move_ee_to(pos, quat, gripper)` (absolute end-effector target, the
environment's IK does the rest), `set_gripper`, `check_lifted`, `is_grasped`, and
`walk_to` on the locomanipulation server. The model ends an episode with `declare_done` (prompt protocol).

## Layout

| Path | What it does |
|---|---|
| `llm_traj/server/sim_server.py` | Franka cube lift (`Isaac-Lift-Cube-Franka-IK-Abs-v0`) |
| `llm_traj/server/sim_server_g1.py` | G1 fixed-base upper body with IK and dexterous hands (`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`) |
| `llm_traj/server/sim_server_g1_loco.py` | G1 locomanipulation (`Isaac-PickPlace-Locomanipulation-G1-Abs-v0`, pretrained lower-body policy) |
| `llm_traj/server/protocol.py` | Length-prefixed JSON framing, default `127.0.0.1:9876` |
| `llm_traj/client/tools.py`, `cli.py` | Python client and the one-shot CLI the model calls |
| `llm_traj/client/scripted_reach.py`, `g1_pick_left.py`, `test_quats.py`, `debug_hands.py` | Scripted sanity checks of the IK / hand layer before an LLM is involved |
| `llm_traj/client/plot_traj.py` | Plots planned targets vs actual end-effector path from the episode log |
| `llm_traj/prompts/system_prompt.md` | System prompt (Franka cube task) |
| `example_episode/g1_cube_episode_failed.jsonl` | Logged G1 upper-body episode (16 calls, cube not lifted) |

## Run

```bash
cd tools/llm_robot_control
# terminal 1: sim server (IsaacLab python)
python -m llm_traj.server.sim_server --headless            # or sim_server_g1 / sim_server_g1_loco
# terminal 2: tool calls (what the model issues)
python -m llm_traj.client.cli reset --seed 0
python -m llm_traj.client.cli get_state
python -m llm_traj.client.cli move_ee_to "[0.5, 0.0, 0.2]" "[0, 1, 0, 0]" --gripper open
python -m llm_traj.client.cli check_lifted --min_lift 0.10
python -m llm_traj.client.plot_traj                          # plot logs/episode_live.jsonl
```

## Results

On the Franka arm the model picks up the cube in 5 calls; on the G1 upper body the same cube task did not
succeed. These are single episodes, not a measured success rate. Figure:
[`../../docs/media/llm_tool_pick.png`](../../docs/media/llm_tool_pick.png).
