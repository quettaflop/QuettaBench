"""Stand-in engine server: answers 200 on /health and keeps one child alive,
as a tensor-parallel server keeps its workers.

usage: fake_server.py <port> <state.json>; with --child it only sleeps.
state.json gets {pid, child_pid, mark} once the child is up (mark is FAKE_MARK).
"""

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
child = subprocess.Popen([sys.executable, __file__, "--child"])
with open(state, "w") as fh:
    json.dump({"pid": os.getpid(), "child_pid": child.pid,
               "mark": os.environ.get("FAKE_MARK")}, fh)


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
