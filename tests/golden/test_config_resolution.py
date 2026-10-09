"""Golden: crossval config resolution for every workload, and the failures
that must stay at parse time.

Each workload is resolved twice: with os.environ empty and with one
representative override of every env knob the crossval path reads. The
record joins what xval_config.py hands the shell (its cli) with what a
grid run resolves from it (its META/CAPACITY/SKIP/RESULT lines on the
stubs, plus how many generate() calls the fit made). The "vmin_fit" keys
name the script the fixture was captured from; serve --grid replaced it.
"""

import contextlib
import io
import json
import os
import unittest
from unittest import mock

from tests.golden._harness import (
    ENV_OVERRIDE, REPO, check_json, crossval_path, fresh_import, run_grid_cli,
)

SCENARIOS = {"cleared": {}, "override": ENV_OVERRIDE}
WORKLOAD_NAMES = sorted(json.loads((REPO / "configs" / "workloads.json").read_text())["workloads"])

# An MLA stack (DeepSeek-V3 shaped config) declared without kv_bytes.
MLA_CONFIG = {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "num_hidden_layers": 61,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
}
MLA_WORKLOAD = {
    "mla_no_kv_bytes": {
        "name": "mla-probe", "dtype": "bfloat16", "weights_gib": 10, "maxlen": 9216,
        "tp": 1, "grid": "smoke", "kv_dtype": "fp8_ds_mla", "verified": True,
    }
}


def _cli(xval_config, *args):
    """xval_config.py's shell interface: stdout, or the exit message."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            xval_config._cli(list(args))
        except SystemExit as e:
            return {"exit": str(e.code)}
    return {"stdout": out.getvalue()}


def _error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:  # the type is part of what is pinned
        return {"type": type(e).__name__, "message": str(e)}
    return None


def resolve_xval(name, env):
    """What xval_config.py resolves for one workload under exactly `env`."""
    with crossval_path(), mock.patch.dict(os.environ, env, clear=True), \
            mock.patch("shutil.which", return_value=None):
        xc = fresh_import("quettabench.config.xval_config")
        wl = xc.workloads()[name]
        return {
            "workload": wl,
            "grid_cells": xc.grids()[wl["grid"]],
            "serving_style": xc.parse_serving_style(wl.get("serving_style")),
            "cli": {cmd: _cli(xc, cmd, name) for cmd in ("tp", "devices", "cells", "bench")},
        }


def resolve_global(env):
    """Workload-independent resolution under exactly `env`."""
    with crossval_path(), mock.patch.dict(os.environ, env, clear=True), \
            mock.patch("shutil.which", return_value=None):
        xc = fresh_import("quettabench.config.xval_config")
        return {
            "grids": xc.grids(),
            "step_points": xc.step_points(),
            "run_params": xc.run_params(),
            "provision": xc.provision(),
            "backend": xc.backend(),
            "link_profile": xc.link_profile(),
            "collective": xc.collective(),
            "collective_by_profile": {p: xc.collective(profile=p) for p in ("pcie", "nvlink")},
            "cli": {cmd: _cli(xc, cmd) for cmd in
                    ("collective", "run-params", "provision", "backend", "link-profile",
                     "serving-style", "engine")},
        }


def resolve_vmin_fit(name, env):
    run = run_grid_cli(["--grid", "--workload", name, "--model", "<MODEL>",
                        "--allow-unverified"], env)
    return {"stdout": run["stdout"], "generate_calls": run["generate_calls"], "exit": run["exit"]}


class ConfigResolutionGolden(unittest.TestCase):
    def test_global_resolution(self):
        check_json(self, "config_global", {s: resolve_global(env) for s, env in SCENARIOS.items()})

    def test_every_workload_resolves_as_captured(self):
        record = {
            name: {s: {"xval_config": resolve_xval(name, env),
                       "vmin_fit": resolve_vmin_fit(name, env)}
                   for s, env in SCENARIOS.items()}
            for name in WORKLOAD_NAMES
        }
        check_json(self, "config_workloads", record)


class ParseTimeFailuresGolden(unittest.TestCase):
    def test_failures_keep_type_and_message(self):
        with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
            xc = fresh_import("quettabench.config.xval_config")
            failures = {
                "serving_style": {spec: _error(xc.parse_serving_style, spec) for spec in
                                  ("disagg:0p1d", "disagg:2p", "ep4", "aggregate",
                                   "disagg:1p1d:x", "EP8DP2")},
                "disagg_oversubscribed": _error(
                    xc.disagg_gpu_sets, xc.parse_serving_style("disagg:4p5d"), list(range(8))),
                "ep_degree": {f"ep{ep}_experts{n}": _error(xc.check_ep_legal, ep, n)
                              for ep, n in ((3, 256), (7, 288), (8, 256), (16, 288))},
                "link_profile": _error(xc.collective, profile="infiniband"),
                "kv_role": _error(xc.disagg_kv_config, "producer", 0, "NixlConnector"),
                "engine": {e: _error(xc.resolve_engine, e) for e in ("rocm", "tgi")},
            }
        unverified = [n for n in WORKLOAD_NAMES
                      if not resolve_xval(n, {})["workload"].get("verified")]
        failures["vmin_fit_unverified"] = {
            n: run_grid_cli(["--grid", "--workload", n, "--model", "<MODEL>"], {})["exit"]
            for n in unverified}
        failures["vmin_fit_unknown_crate"] = run_grid_cli(
            ["--grid", "--workload", "nope", "--model", "<MODEL>"], {})["exit"]
        failures["vmin_fit_unknown_grid"] = run_grid_cli(
            ["--grid", "nope", "--workload", "llama", "--model", "<MODEL>"], {})["exit"]
        failures["vmin_fit_unknown_prompt_mode"] = run_grid_cli(
            ["--grid", "--workload", "llama", "--model", "<MODEL>"], {"PROMPT_MODE": "random"})["exit"]
        # Not a failure today: an MLA stack without kv_bytes falls back to the
        # dense per-head formula silently (see REPORT.md).
        mla = run_grid_cli(["--grid", "--workload", "mla_no_kv_bytes", "--model", "<MODEL>"], {},
                           model_config=MLA_CONFIG, extra_workloads=MLA_WORKLOAD)
        failures["mla_without_kv_bytes"] = {
            "exit": mla["exit"],
            "capacity": [line for line in mla["stdout"] if line.startswith("CAPACITY")],
        }
        check_json(self, "config_failures", failures)

    def test_trace_serve_rejects_malformed_serving_style_before_loading(self):
        with crossval_path(), mock.patch.dict(os.environ, {}, clear=True):
            trace_serve = fresh_import("legacy.trace_serve")
            argv = ["trace_serve.py", "/nonexistent.jsonl", "--model", "m",
                    "--serving-style", "disagg:2p"]
            with mock.patch("sys.argv", argv):
                err = _error(trace_serve.main)
        check_json(self, "config_trace_serve_style_error", err)


if __name__ == "__main__":
    unittest.main()
