"""Shared test plumbing: a fake OpenAI endpoint and stub run dirs for compare."""

import http.server
import json
import shutil
import threading
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.engines.drivers import BenchDriver
from quettabench.metrics.results import RunWriter
from quettabench.metrics.step_fit import fit_run
from tests.golden._harness import grid_argv, run_grid_cli

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


def cache_config_text(engines=1, prefix_caching=True, kv_tokens=82777364):
    """vLLM 0.31.0's cache_config_info gauge, one series per engine."""
    return "".join(
        'vllm:cache_config_info{_block_size_resolved="True",block_size="128",cache_dtype="auto",'
        'effective_attention_block_size="128",enable_mamba_shared_prefix_checkpoint="False",'
        f'enable_prefix_caching="{prefix_caching}",engine="{i}",gpu_memory_utilization="0.85",'
        f'is_attention_free="False",kv_cache_size_tokens="{kv_tokens + i}",model_name="a,b"}} 1.0\n'
        for i in range(engines)
    )


@contextmanager
def serve(metrics=METRICS):
    posts = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            self._reply((metrics() if callable(metrics) else metrics).encode(), "text/plain")

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
            if self.path.endswith("/chat/completions"):
                chunks = [{"choices": [{"delta": {"content": "t"}}]} for _ in range(n)]
                prompt_len = sum(len(m.get("content") or "") for m in body["messages"])
            else:
                chunks = [{"choices": [{"text": "t"}]} for _ in range(n)]
                prompt_len = len(body["prompt"])
            chunks.append({"choices": [], "usage": {"prompt_tokens": prompt_len, "completion_tokens": n}})
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


RECORDED = "2026-09-21T14:13:20+00:00"
FAKE_BENCH_SHA = "0123abc"
RAW_STEP = {
    "ctx": 4096,
    "bs": 64,
    "tp": 1,
    "ms_per_step": 19.874,
    "ms_at_eval": None,
    "eval_pos": 0,
    "r2": 0.9995,
    "prefill_s": 1.1,
    "raw": [2.29, 5.67, 9.05],
}


def _edit_run_json(path, edit):
    run = json.loads(path.read_text())
    edit(run)
    path.write_text(json.dumps(run, indent=2, default=str) + "\n")


def grid_run_dir(out):
    out = Path(out)
    fake_git = types.SimpleNamespace(stdout=FAKE_BENCH_SHA + "\n")
    with mock.patch("subprocess.run", return_value=fake_git):
        run = run_grid_cli(grid_argv("llama", "--set", f"run_name={out.name}", "--results-dir", str(out.parent)), {})
    if run["exit"] is not None:
        raise RuntimeError(f"grid run exited: {run['exit']}")

    def edit(run_json):
        run_json["steps"].append(dict(RAW_STEP))
        run_json["provenance"]["finished"] = RECORDED

    _edit_run_json(out / "run.json", edit)
    return out


def bench_run_dir(out, bench_log):
    out = Path(out)
    out.mkdir(parents=True)
    shutil.copy(bench_log, out / "bench.log")
    rows = list(BenchDriver(out / "bench.log").rows())
    writer = RunWriter(out, bench_config(out.name))
    for r in rows:
        writer.append(r)
    writer.finish(None, steps=fit_run(rows, 0, 1, 1.0))
    return out


def bench_config(run_name):
    return Config(
        engine=EngineCfg(name="quettaserve"),
        workload=WorkloadCfg(),
        mode=ModeCfg(),
        hardware=HardwareCfg(),
        run_name=run_name,
    )
