"""Golden test plumbing: fixture compare and a CPU grid run through `qbench run` on the stubs; QBENCH_GOLDEN_REGEN=1 rewrites fixtures."""

import asyncio
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
import urllib.error
from pathlib import Path
from unittest import mock

import aiohttp

REPO = Path(__file__).resolve().parents[2]
STUBS = REPO / "tests" / "_stubs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REGEN = os.environ.get("QBENCH_GOLDEN_REGEN") == "1"

CROSSVAL_MODULES = ("quettabench",)
STUB_MODULE_ROOTS = ("vllm", "torch", "transformers")

# Llama-3.1-8B shaped config.json, 131072 KV bytes per token
DENSE_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "head_dim": 128,
}

ENV_OVERRIDE = {
    "XVAL_LINK_PROFILE": "pcie",
    "XVAL_PP": "2",
    "XVAL_BACKEND": "cuda",
    "GPU_UTIL": "0.85",
    "PROMPT_MODE": "clone",
    "VMIN_ATTEMPTS": "2",
    "XVAL_EVAL_POS": "25",
}

RUNS = REPO / "configs" / "runs"

# the loader refuses hybrid attention without kv_bytes, so the qwen golden run says dense
AS_VMIN_FIT = {"qwen": ["engine.attention=dense"]}

# the one (ctx, bs) cell the fake clock makes noisy, to produce a RESULT line with r2 < 0.999
NOISY_CELL = (1024, 16)


def check_json(case, name, actual):
    """Compare actual against fixtures/<name>.json (or rewrite it under REGEN)."""
    path = FIXTURES / f"{name}.json"
    actual = json.loads(json.dumps(actual))  # tuples become lists, as on disk
    if REGEN:
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        return
    case.assertEqual(actual, json.loads(path.read_text()), f"golden drift in {path.name}")


def check_text(case, name, actual):
    """Compare actual text against fixtures/<name> byte for byte (or rewrite it)."""
    path = FIXTURES / name
    if REGEN:
        path.write_text(actual)
        return
    case.assertEqual(actual, path.read_text(), f"golden drift in {path.name}")


def _drop_modules(roots):
    for name in list(sys.modules):
        if name.split(".")[0] in roots:
            del sys.modules[name]


@contextlib.contextmanager
def crossval_path(stubs=False):
    """Fresh imports of quettabench, with the vllm/torch stubs if asked; dropped again on exit."""
    added = [str(REPO)] + ([str(STUBS)] if stubs else [])
    roots = CROSSVAL_MODULES + (STUB_MODULE_ROOTS if stubs else ())
    _drop_modules(roots)
    for p in added:
        sys.path.insert(0, p)
    try:
        yield
    finally:
        for p in added:
            sys.path.remove(p)
        _drop_modules(roots)


def fresh_import(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def fake_generate_seconds(prompt_len, batch, max_tokens, call_no):
    """Linear decode cost in max_tokens (r2 is 1); NOISY_CELL bends one point so every attempt fits r2 < 0.999."""
    step_ms = 4.0 + 0.0005 * prompt_len * batch / 8
    prefill_s = 0.02 + 2e-6 * prompt_len * batch
    bend = 0.3 if (prompt_len, batch) == NOISY_CELL and max_tokens == 230 else 0.0
    return prefill_s + max_tokens * step_ms / 1000.0 + bend


class StandInServer:
    """`vllm serve` on the stubs, reached through the aiohttp session HttpDriver opens; a grid batch runs as one vllm.run_batch."""

    URL = "http://stand-in:8000"

    def __init__(self, vllm, launcher):
        self.vllm, self.launcher, self.pending = vllm, launcher, []

    def start(self, spec, results_dir, **kwargs):
        """launcher.start: the spec must name a server; the handle points here."""
        if not spec.server_argv:
            raise ValueError(f"{spec.adapter}: the spec has no server to launch")
        return self.launcher.NoServer(Path(results_dir), endpoint=self.URL)

    def urlopen(self, url, timeout=None):
        if url != self.URL + "/version":
            raise urllib.error.URLError(url)
        return io.BytesIO(json.dumps({"version": self.vllm.__version__}).encode())

    def session(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @contextlib.asynccontextmanager
    async def post(self, url, headers=None, json=None, timeout=None):
        yield await self.answer(url, json)

    async def answer(self, url, body):
        """Every post of a gather is pending before the first one resumes, so that one runs them all as one batch."""
        if url.endswith("/reset_prefix_cache"):
            return Reply([])
        entry = {"body": body}
        self.pending.append(entry)
        await asyncio.sleep(0)
        if self.pending:
            batch, self.pending = self.pending, []
            params = self.vllm.SamplingParams(
                temperature=0.0, ignore_eos=True, max_tokens=batch[0]["body"]["max_tokens"]
            )
            outs = self.vllm.run_batch([{"prompt_token_ids": e["body"]["prompt"]} for e in batch], params)
            for e, out in zip(batch, outs):
                e["tokens"] = len(out.outputs[0].token_ids)
        usage = {"prompt_tokens": len(body["prompt"]), "completion_tokens": entry["tokens"]}
        return Reply([{"choices": [{"text": "t"}]}, {"choices": [], "usage": usage}])


class Reply:
    """An SSE response: one `data:` line per chunk, then [DONE]."""

    status = 200

    def __init__(self, chunks):
        self.lines = [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"data: [DONE]"]

    @property
    def content(self):
        async def lines():
            for line in self.lines:
                yield line

        return lines()

    async def text(self):
        return ""


def grid_argv(name, *extra):
    """`qbench run` argv for a workload's grid run on the stub weights dir."""
    sets = ["engine.model=<MODEL>", *AS_VMIN_FIT.get(name, [])]
    return ["run", str(RUNS / f"{name}.yaml"), *[a for s in sets for a in ("--set", s)], *extra]


def loader_message(exit_msg):
    """A loader refusal without its `<file> [<run>]: ` prefix."""
    return exit_msg.split(": ", 1)[1]


def run_grid_cli(argv, env, model_config=None):
    """Run cli.main(argv) on the stubs under `env`; returns stdout lines, LLM kwargs, the generate() count and the SystemExit message if any."""
    with tempfile.TemporaryDirectory() as model_dir:
        Path(model_dir, "config.json").write_text(json.dumps(model_config or DENSE_CONFIG))
        argv = [a.replace("<MODEL>", model_dir) for a in argv]
        if "--results-dir" not in argv:
            argv += ["--results-dir", str(Path(model_dir, "results"))]
        out = io.StringIO()
        with (
            crossval_path(stubs=True),
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("shutil.which", return_value=None),
        ):
            vllm = importlib.import_module("vllm")
            vllm.reset()
            vllm.generate_seconds = fake_generate_seconds
            cli = fresh_import("quettabench.cli")
            drivers = importlib.import_module("quettabench.engines.drivers")
            launcher = importlib.import_module("quettabench.engines.launcher")
            torch = importlib.import_module("torch")
            clock = types.SimpleNamespace(perf_counter=vllm.CLOCK.perf_counter)
            server = StandInServer(vllm, launcher)
            gpu = (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
            exit_msg = None
            with (
                mock.patch.object(drivers, "time", clock),
                mock.patch.object(launcher, "idle_gpus", return_value="0"),
                mock.patch.object(launcher, "start", server.start),
                mock.patch.object(launcher, "urlopen", server.urlopen),
                mock.patch.object(launcher, "accelerator", return_value=gpu),
                mock.patch.object(aiohttp, "ClientSession", server.session),
                mock.patch.object(aiohttp, "TCPConnector", lambda **kw: None),
                contextlib.redirect_stdout(out),
            ):
                try:
                    cli.main(argv)
                except SystemExit as e:
                    exit_msg = str(e.code)
            llm_kwargs = [kw for name, kw in vllm.CALLS if name == "LLM"]
            calls = vllm.CLOCK.generate_calls
    text = out.getvalue().replace(model_dir, "<MODEL>")
    return {
        "stdout": text.splitlines(),
        "llm_kwargs": [
            {k: (v.replace(model_dir, "<MODEL>") if isinstance(v, str) else v) for k, v in kw.items()}
            for kw in llm_kwargs
        ],
        "generate_calls": calls,
        "exit": exit_msg,
    }
