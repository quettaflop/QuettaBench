"""Shared plumbing for the golden tests: fixture compare/regen, crossval
imports, and a CPU run of vmin_fit.main() against the vllm/torch stubs.

QBENCH_GOLDEN_REGEN=1 rewrites every fixture from the current code instead of
comparing; run it only when a behaviour change is intended.
"""

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
CROSSVAL = REPO / "scripts" / "crossval"
STUBS = REPO / "tests" / "_stubs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REGEN = os.environ.get("QBENCH_GOLDEN_REGEN") == "1"

CROSSVAL_MODULES = ("xval_config", "synth_bench", "trace_serve", "vmin_fit", "cache", "table")
STUB_MODULE_ROOTS = ("vllm", "torch")

# Llama-3.1-8B shaped config.json: the dense kv formula gives 131072 B/token.
DENSE_CONFIG = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "head_dim": 128,
}

# One representative override of every env knob the crossval path reads.
ENV_OVERRIDE = {
    "XVAL_LINK_PROFILE": "pcie",
    "XVAL_PP": "2",
    "XVAL_BACKEND": "cuda",
    "GPU_UTIL": "0.85",
    "PROMPT_MODE": "clone",
    "VMIN_ATTEMPTS": "2",
    "XVAL_EVAL_POS": "25",
}

# The one (ctx, bs) cell the fake clock makes noisy, so a RESULT line with
# r2 < 0.999 (table.py's LOW R2) comes out of the real vmin_fit code path.
NOISY_CELL = (1024, 16)


def check_json(case, name, actual):
    """Compare actual against fixtures/<name>.json (or rewrite it under REGEN)."""
    path = FIXTURES / f"{name}.json"
    actual = json.loads(json.dumps(actual))  # tuples -> lists, as on disk
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
    """Fresh imports of the crossval scripts (and the vllm/torch stubs when
    asked); everything imported inside is dropped again on exit."""
    added = [str(CROSSVAL)] + ([str(STUBS)] if stubs else [])
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
    """Fake decode cost: exact line in max_tokens, so r2 is 1 and the fit stops
    after one attempt, except NOISY_CELL, whose middle step point is bent off
    the line so every attempt fits r2 < 0.999 and VMIN_ATTEMPTS is used up."""
    step_ms = 4.0 + 0.0005 * prompt_len * batch / 8
    prefill_s = 0.02 + 2e-6 * prompt_len * batch
    bend = 0.3 if (prompt_len, batch) == NOISY_CELL and max_tokens == 230 else 0.0
    return prefill_s + max_tokens * step_ms / 1000.0 + bend


def run_vmin_fit(argv, env, model_config=None, extra_workloads=None):
    """Run vmin_fit.main(argv) on the stubs under exactly `env` (os.environ is
    replaced, not merged). nvidia-smi is hidden so backend and link-profile
    detection never depend on the host. Returns stdout lines, the LLM kwargs,
    the number of generate() calls and the SystemExit message if one fired."""
    with tempfile.TemporaryDirectory() as model_dir:
        Path(model_dir, "config.json").write_text(json.dumps(model_config or DENSE_CONFIG))
        argv = [a.replace("<MODEL>", model_dir) for a in argv]
        out = io.StringIO()
        with crossval_path(stubs=True), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("shutil.which", return_value=None):
            vllm = importlib.import_module("vllm")
            vllm.reset()
            vllm.generate_seconds = fake_generate_seconds
            vmin_fit = fresh_import("vmin_fit")
            if extra_workloads:
                vmin_fit.CFG["workloads"].update(extra_workloads)
            exit_msg = None
            with mock.patch.object(vmin_fit, "time", types.SimpleNamespace(
                    perf_counter=vllm.CLOCK.perf_counter)), \
                    mock.patch.object(sys, "argv", ["vmin_fit.py"] + argv), \
                    contextlib.redirect_stdout(out):
                try:
                    vmin_fit.main()
                except SystemExit as e:
                    exit_msg = str(e.code)
            llm_kwargs = [kw for name, kw in vllm.CALLS if name == "LLM"]
            calls = vllm.CLOCK.generate_calls
    text = out.getvalue().replace(model_dir, "<MODEL>")
    return {
        "stdout": text.splitlines(),
        "llm_kwargs": [{k: (v.replace(model_dir, "<MODEL>") if isinstance(v, str) else v)
                        for k, v in kw.items()} for kw in llm_kwargs],
        "generate_calls": calls,
        "exit": exit_msg,
    }
