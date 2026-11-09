"""Thin synchronous client over the sim_server TCP socket."""
from __future__ import annotations

import socket
from typing import Any

from llm_traj.server.protocol import HOST, PORT, recv_message, send_message


class SimClient:
    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def _rpc(self, cmd: str, **payload: Any) -> dict:
        send_message(self.sock, {"cmd": cmd, **payload})
        resp = recv_message(self.sock)
        if resp.get("status") == "error":
            raise RuntimeError(f"sim_server error on {cmd}: {resp.get('reason')}")
        return resp

    # --- public API ------------------------------------------------------
    def reset(self, seed: int | None = None, task: str | None = None) -> dict:
        return self._rpc("reset", seed=seed, task=task)

    def get_state(self) -> dict:
        return self._rpc("get_state")

    def move_ee_to(
        self,
        pos: list[float],
        quat_wxyz: list[float],
        gripper: str | None = None,
        tol_pos: float = 0.01,
        tol_rot: float = 0.05,
        max_steps: int = 240,
        hand: str | None = None,
    ) -> dict:
        return self._rpc(
            "move_ee_to",
            pos=list(pos),
            quat_wxyz=list(quat_wxyz),
            gripper=gripper,
            tol_pos=tol_pos,
            tol_rot=tol_rot,
            max_steps=max_steps,
            hand=hand,
        )

    def set_gripper(self, state: str, hold_steps: int = 20, hand: str | None = None) -> dict:
        assert state in ("open", "close")
        return self._rpc("set_gripper", state=state, hold_steps=hold_steps, hand=hand)

    def walk_to(self, x: float, y: float, yaw: float = 0.0, tol_pos: float = 0.15,
                tol_yaw: float = 0.2, max_steps: int = 900) -> dict:
        return self._rpc("walk_to", x=x, y=y, yaw=yaw, tol_pos=tol_pos,
                         tol_yaw=tol_yaw, max_steps=max_steps)

    def is_grasped(self) -> dict:
        return self._rpc("is_grasped")

    def check_lifted(self, min_lift: float = 0.10) -> dict:
        return self._rpc("check_lifted", min_lift=min_lift)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass
