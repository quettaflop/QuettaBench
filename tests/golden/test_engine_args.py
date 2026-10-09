"""Golden: the vLLM engine kwargs a grid run and trace_serve.py build per
workload, captured through the tests/_stubs/vllm recorder.

The grid side is run as vllm.sh runs it (serve --grid with the workload
name and weights dir; the fixture's "vmin_fit" keys name the script it was
captured from, which serve --grid replaced); trace_serve
as xval.sh runs it (trace, --model, --workload, --tp from the workload).
Both now build through engines/vllm.py translate(); the per-key diff is
pinned and should be empty. qwen's trace_serve side is the loader's
refusal (hybrid without kv_bytes), recorded as its exit message.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.golden._harness import (
    ENV_OVERRIDE, REPO, check_json, crossval_path, fresh_import, run_grid_cli,
)

WORKLOADS = json.loads((REPO / "configs" / "workloads.json").read_text())["workloads"]
ABSENT = "<absent>"


def vmin_fit_kwargs(name, env, extra_argv=()):
    run = run_grid_cli(["--grid", "--workload", name, "--model", "<MODEL>", "--allow-unverified",
                        *extra_argv], env)
    (kwargs,) = run["llm_kwargs"]
    return kwargs


def trace_serve_kwargs(name):
    """AsyncEngineArgs kwargs from trace_serve.main() with xval.sh's argv."""
    tp = str(WORKLOADS[name].get("tp", 1))
    with tempfile.TemporaryDirectory() as td:
        trace = Path(td) / "trace.jsonl"
        trace.write_text(json.dumps({"prompt_token_ids": [1, 2, 3], "output_length": 4}) + "\n")
        argv = ["trace_serve.py", str(trace), "--model", "<MODEL>", "--workload", name,
                "--tp", tp]
        with crossval_path(stubs=True), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            vllm = fresh_import("vllm")
            trace_serve = fresh_import("legacy.trace_serve")
            try:
                with halted_engine(vllm):
                    trace_serve.main()
            except SystemExit as e:
                return {"exit": str(e.code).replace(str(REPO), "<REPO>")}
            ((ctor, kwargs),) = vllm.CALLS
    assert ctor == "AsyncEngineArgs", ctor
    return kwargs


@contextlib.contextmanager
def halted_engine(vllm):
    """The stub engine raises StubHalt once its args are built; swallow it."""
    try:
        yield
    except vllm.StubHalt:
        pass


def kwargs_diff(a, b):
    """{key: [a value, b value]} for every key whose value differs."""
    return {k: [a.get(k, ABSENT), b.get(k, ABSENT)]
            for k in sorted(set(a) | set(b)) if a.get(k, ABSENT) != b.get(k, ABSENT)}


class EngineArgsGolden(unittest.TestCase):
    def test_engine_kwargs_per_workload(self):
        record = {}
        for name in sorted(WORKLOADS):
            vmin = vmin_fit_kwargs(name, {})
            serve = trace_serve_kwargs(name)
            record[name] = {
                "vmin_fit": vmin,
                "vmin_fit_env_override": vmin_fit_kwargs(name, ENV_OVERRIDE),
                "vmin_fit_nograph": vmin_fit_kwargs(name, {}, ["--nograph"]),
                "trace_serve": serve,
                "diff_vmin_fit_vs_trace_serve": None if "exit" in serve else kwargs_diff(vmin, serve),
            }
        check_json(self, "engine_args", record)


if __name__ == "__main__":
    unittest.main()
