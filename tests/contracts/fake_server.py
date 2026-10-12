"""Stand-in engine server: answers 200 on /health and keeps one child alive,
as a tensor-parallel server keeps its workers.

usage: fake_server.py <port> <state.json>; with --child it only sleeps.
It binds first (port 0 takes a free one; a taken port exits 98), then
state.json gets {pid, child_pid, mark, port}: mark is FAKE_MARK, port the
one it bound, so a test can check it reached this server and no other.
"""

import errno
import http.server
import json
import os
import subprocess
import sys
import time

if sys.argv[1] == "--child":
    time.sleep(600)
    sys.exit(0)

port, state = int(sys.argv[1]), sys.argv[2]


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


try:
    server = http.server.HTTPServer(("127.0.0.1", port), Health)
except OSError as e:
    if e.errno == errno.EADDRINUSE:
        sys.exit(98)
    raise
child = subprocess.Popen([sys.executable, __file__, "--child"])
tmp = state + ".tmp"
with open(tmp, "w") as fh:
    json.dump(
        {"pid": os.getpid(), "child_pid": child.pid, "mark": os.environ.get("FAKE_MARK"), "port": server.server_port},
        fh,
    )
os.replace(tmp, state)
server.serve_forever()
