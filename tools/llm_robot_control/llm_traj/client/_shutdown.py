from llm_traj.server.protocol import HOST, PORT, send_message, recv_message
import socket
s = socket.create_connection((HOST, PORT), timeout=5)
send_message(s, {"cmd": "shutdown"})
try:
    print(recv_message(s))
except Exception as e:
    print("no response (server likely closed):", e)
s.close()
