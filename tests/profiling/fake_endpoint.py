"""A local stand-in for an attached vLLM server: GET /metrics returns the
given Prometheus text, POST /v1/completions answers `#### <prompt length>`,
or with "stream" set streams max_tokens one-token chunks and a usage chunk,
and POST /reset_prefix_cache answers 200. Every POST body is recorded in
`posts` (None for a bodiless POST)."""

import http.server
import json
import threading
from contextlib import contextmanager

METRICS = """\
# HELP vllm:gpu_prefix_cache_queries_total GPU prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:gpu_prefix_cache_queries_total counter
vllm:gpu_prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:gpu_prefix_cache_queries_total{engine="1",model_name="m"} 600.0
vllm:gpu_prefix_cache_queries_created{engine="0",model_name="m"} 1.7e9
# TYPE vllm:gpu_prefix_cache_hits_total counter
vllm:gpu_prefix_cache_hits_total{engine="0",model_name="m"} 800.0
vllm:gpu_prefix_cache_hits_total{engine="1",model_name="m"} 400.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
process_open_fds 42.0
odd line without a value
vllm:label_escape{path="a\\"b"} 1.0 1712345678000
"""


@contextmanager
def serve(metrics=METRICS):
    """Yields (endpoint, posts) for a server on a free localhost port."""
    posts = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            self._reply(metrics.encode(), "text/plain")

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(size)) if size else None
            posts.append((self.path, body))
            if self.path == "/reset_prefix_cache":
                return self._reply(b"", "text/plain")
            if body.get("stream"):
                return self._stream(body)
            text = f"#### {len(body['prompt'])}"
            self._reply(json.dumps({"choices": [{"text": text}]}).encode(), "application/json")

        def _stream(self, body):
            n = body["max_tokens"]
            chunks = [{"choices": [{"text": "t"}]} for _ in range(n)]
            chunks.append({"choices": [], "usage": {"prompt_tokens": len(body["prompt"]),
                                                    "completion_tokens": n}})
            lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
            self._reply("".join(lines).encode(), "text/event-stream")

        def _reply(self, data, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", posts
    finally:
        server.shutdown()
        server.server_close()
