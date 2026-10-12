"""Stand-in server: 200 on /health, one child alive, writes {pid, child_pid, mark, port} to state.json; a taken port exits 98."""

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
