"""Stand-in api and dist for scripts/quettaserve-serve.sh; argv[1] says which.

api listens on --port, takes --num-nodes - 1 secondaries, then answers /health and /v1/models (the
--served-model-name names) on --http-addr. dist connects to --primary and sends its --gpu. Each writes
<name>.json with its pid, argv and CUDA_VISIBLE_DEVICES to FAKE_QS_STATE; api writes it once listening and
again with the secondaries' GPUs in connection order.
dist exits FAKE_QS_DIST_EXIT's status when its --gpu equals FAKE_QS_DIST_GPU.
"""

import http.server
import json
import os
import socket
import sys
import time

role, argv = sys.argv[1], sys.argv[2:]
state = os.environ["FAKE_QS_STATE"]


def value(flag):
    return argv[argv.index(flag) + 1]


def record(name, **extra):
    doc = {"pid": os.getpid(), "argv": argv, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    tmp = os.path.join(state, name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, os.path.join(state, name + ".json"))


if role == "dist":
    gpu = value("--gpu")
    record(f"dist{gpu}", model=os.environ.get("MODEL"))
    if os.environ.get("FAKE_QS_DIST_GPU") == gpu:
        sys.exit(int(os.environ["FAKE_QS_DIST_EXIT"]))
    host, port = value("--primary").rsplit(":", 1)
    conn = socket.create_connection((host, int(port)))
    conn.sendall(gpu.encode() + b"\n")
    time.sleep(600)
    sys.exit(0)

names = argv[argv.index("--served-model-name") + 1 :] if "--served-model-name" in argv else []
peer = socket.socket()
peer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
peer.bind(("127.0.0.1", int(value("--port"))))
peer.listen()
record("api", ranks=[])
conns = [peer.accept()[0] for _ in range(int(value("--num-nodes")) - 1)]
ranks = [c.makefile().readline().strip() for c in conns]


class Api(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path in ("/health", "/v1/models") else 404)
        self.end_headers()
        if self.path == "/v1/models":
            self.wfile.write(json.dumps({"data": [{"id": n} for n in names]}).encode())

    def log_message(self, *args):
        pass


host, port = value("--http-addr").rsplit(":", 1)
server = http.server.HTTPServer((host, int(port)), Api)
record("api", ranks=ranks)
server.serve_forever()
