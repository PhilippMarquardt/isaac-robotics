"""Length-prefixed JSON framing over TCP, shared by server and client."""
from __future__ import annotations

import json
import socket
import struct

HOST = "127.0.0.1"
PORT = 9876
_HEADER = struct.Struct(">I")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def send_message(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv_message(sock: socket.socket) -> dict:
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))
