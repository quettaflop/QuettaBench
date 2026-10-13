"""Stand-in server: /health 200, /v1/models lists the --served-model-name names, one child alive, state.json; a taken port exits 98."""

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
names = sys.argv[sys.argv.index("--served-model-name") + 1 :] if "--served-model-name" in sys.argv else []


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path in ("/health", "/v1/models") else 404)
        self.end_headers()
        if self.path == "/v1/models":
            self.wfile.write(json.dumps({"data": [{"id": n} for n in names]}).encode())

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
