"""Golden: config resolution for every run yaml through the loader, and the
failures that must stay at parse time.

Each workload is resolved twice: with os.environ empty and with one
representative override of every env knob the crossval path reads. The
"xval_config" keys hold what the retired xval_config.py resolved (merged
workload record, grid cells, serving style); the loader must resolve the
same values from configs/, field by field. "vmin_fit" is what a grid run
prints on the stubs (META/CAPACITY/SKIP/RESULT lines, generate() calls),
named for the script it was captured from; qbench run replaced it.
"dry_run" pins the tp, pp, grid cells and QuettaServe bench block that
`qbench run --dry-run` prints, which xval_config's tp/devices/cells/bench
shell commands used to.
"""

import contextlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench.config import loader
from quettabench.config.validate import (
    check_ep_legal,
    disagg_gpu_sets,
    disagg_kv_config,
    parse_serving_style,
)
from quettabench.engines import base
from tests.golden._harness import (
    AS_VMIN_FIT,
    ENV_OVERRIDE,
    FIXTURES,
    REPO,
    RUNS,
    check_json,
    crossval_path,
    fresh_import,
    grid_argv,
    loader_message,
    run_grid_cli,
)

SCENARIOS = {"cleared": {}, "override": ENV_OVERRIDE}
# Run yamls and grids written after the capture have no old xval_config to
# match; their own tests pin them (test_crossval_contracts ConfigTree).
ADDED_RUNS = {"decode_batch_llama"}
ADDED_GRIDS = {"decode_batch"}
WORKLOAD_NAMES = sorted(p.stem for p in RUNS.glob("*.yaml") if p.stem not in ADDED_RUNS)
WORKLOADS = json.loads((FIXTURES / "config_workloads.json").read_text())
DRY_RUN_KEYS = ("engine.tp ", "engine.pp ", "engine.extra.quettaserve ", "grid.cells ")

# An MLA stack (DeepSeek-V3 shaped config) declared without kv_bytes.
MLA_CONFIG = {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "num_hidden_layers": 61,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
}
MLA_MODEL = {"name": "mla-probe", "dtype": "bfloat16", "weights_gib": 10, "max_model_len": 9216,
             "tp": 1, "kv_dtype": "fp8_ds_mla", "verified": True, "attention": "mla"}
MLA_RUN = {"model": "mla_probe", "workload": {"kind": "grid"}, "mode": {"grid": "smoke"}}


@contextlib.contextmanager
def mla_run_yaml():
    """configs/ copied to a scratch dir with the MLA model and its run added."""
    with tempfile.TemporaryDirectory() as td:
        configs = Path(td, "configs")
        shutil.copytree(REPO / "configs", configs)
        (configs / "models" / "mla_probe.yaml").write_text(yaml.safe_dump(MLA_MODEL))
        run = configs / "runs" / "mla_no_kv_bytes.yaml"
        run.write_text(yaml.safe_dump(MLA_RUN))
        yield run


def _error(fn, *args, **kwargs):
    """The exception's builtin class (a ConfigError is a ValueError) and message."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # the type is part of what is pinned
        kind = next(c.__name__ for c in type(e).__mro__ if c.__module__ == "builtins")
        return {"type": kind, "message": str(e)}
    return None


def resolve(name, env, *sets):
    """The loader's Config for one run yaml under exactly `env`, no tooling on
    PATH; qwen as vmin_fit sized it (dense)."""
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch("shutil.which", return_value=None):
        (cfg,) = loader.load(RUNS / f"{name}.yaml",
                             ["allow_unverified=true", *AS_VMIN_FIT.get(name, []), *sets])
    return cfg


def check_record(case, cfg, old, env):
    """cfg holds every value xval_config resolved for the workload."""
    wl, e, m = old["workload"], cfg.engine, cfg.mode
    case.assertEqual((e.model_name, e.dtype, e.weights_gib, e.max_model_len),
                     (wl["name"], wl["dtype"], wl["weights_gib"], wl["maxlen"]))
    case.assertEqual((e.tp, e.pp), (int(wl.get("tp", 1)), int(env.get("XVAL_PP", wl.get("pp", 1)))))
    case.assertEqual((e.ep, e.kv_dtype, e.kv_bytes, e.quant, e.verified),
                     (bool(wl.get("expert_parallel")), wl.get("kv_dtype") or "auto",
                      wl.get("kv_bytes"), wl.get("quant"), wl["verified"]))
    case.assertEqual(e.gpu_util, float(env.get("GPU_UTIL", wl.get("gpu_util", 0.90))))
    bench = wl.get("bench")
    case.assertEqual(e.extra.get("quettaserve"), bench and {
        "crate": bench["crate"], "test": bench["test"],
        "env_prefix": bench["env"], "family": bench["family"]})
    case.assertEqual(cfg.crossval.get("comm_bound_bs"), wl.get("comm_bound_bs"))
    for key in ("engine_commit", "checkpoint"):
        case.assertEqual(cfg.provenance.get(key), wl.get(key))
    case.assertEqual(cfg.workload.prompt_mode,
                     env.get("PROMPT_MODE", wl.get("prompt_mode", "distinct")))
    case.assertEqual((m.grid, cfg.grid.cells), (wl["grid"], tuple(map(tuple, old["grid_cells"]))))
    case.assertEqual(parse_serving_style(e.serving_style), old["serving_style"])


def resolve_vmin_fit(name, env):
    run = run_grid_cli(grid_argv(name, "--allow-unverified"), env)
    return {"stdout": run["stdout"], "generate_calls": run["generate_calls"], "exit": run["exit"]}


def resolve_dry_run(name, env):
    stdout = run_grid_cli(grid_argv(name, "--allow-unverified", "--dry-run"), env)["stdout"]
    return [line for line in stdout if line.startswith(DRY_RUN_KEYS)]


def resolve_global(env):
    """Workload-independent values, as the loader resolves them for llama."""
    cfg = resolve("llama", env)
    grids, step_points = loader._read_grids(REPO / "configs")
    h = cfg.hardware
    # xval_config.collective() took the profile from XVAL_LINK_PROFILE, else pcie.
    default = env.get("XVAL_LINK_PROFILE", "pcie")
    by_profile = {p: dict(resolve("llama", env, f"hardware.link_profile={p}").hardware.collective)
                  for p in ("pcie", "nvlink")}
    return {
        "grids": {k: v for k, v in grids.items() if k not in ADDED_GRIDS},
        "step_points": step_points,
        "run_params": {"world": h.world, "timing_steps": h.timing_steps,
                       "max_seq_headroom": h.max_seq_headroom},
        "provision": dict(h.provision),
        "backend": h.backend,
        "link_profile": h.link_profile,
        "collective": by_profile[default],
        "collective_by_profile": by_profile,
    }


class ConfigResolutionGolden(unittest.TestCase):
    def test_global_resolution(self):
        check_json(self, "config_global", {s: resolve_global(env) for s, env in SCENARIOS.items()})

    def test_every_workload_resolves_as_captured(self):
        self.assertEqual(WORKLOAD_NAMES, sorted(WORKLOADS))
        record = {}
        for name in WORKLOAD_NAMES:
            record[name] = {}
            for s, env in SCENARIOS.items():
                old = WORKLOADS[name][s]["xval_config"]
                with self.subTest(workload=name, env=s):
                    check_record(self, resolve(name, env), old, env)
                record[name][s] = {"xval_config": {k: v for k, v in old.items() if k != "cli"},
                                   "vmin_fit": resolve_vmin_fit(name, env),
                                   "dry_run": resolve_dry_run(name, env)}
        check_json(self, "config_workloads", record)

    def test_qwen_needs_the_dense_override(self):
        with self.assertRaisesRegex(loader.ConfigError, "attention=hybrid needs kv_bytes"):
            loader.load(RUNS / "qwen.yaml", ["allow_unverified=true"], env={})


class ParseTimeFailuresGolden(unittest.TestCase):
    def test_failures_keep_type_and_message(self):
        link = _error(resolve, "llama", {}, "hardware.link_profile=infiniband")
        failures = {
            "serving_style": {spec: _error(parse_serving_style, spec) for spec in
                              ("disagg:0p1d", "disagg:2p", "ep4", "aggregate",
                               "disagg:1p1d:x", "EP8DP2")},
            "disagg_oversubscribed": _error(
                disagg_gpu_sets, parse_serving_style("disagg:4p5d"), list(range(8))),
            "ep_degree": {f"ep{ep}_experts{n}": _error(check_ep_legal, ep, n)
                          for ep, n in ((3, 256), (7, 288), (8, 256), (16, 288))},
            "link_profile": {**link, "message": loader_message(link["message"])},
            "kv_role": _error(disagg_kv_config, "producer", 0, "NixlConnector"),
            "engine_unknown": _error(base.get, "tgi"),
        }
        unverified = [n for n in WORKLOAD_NAMES
                      if not WORKLOADS[n]["cleared"]["xval_config"]["workload"].get("verified")]
        # The loader's refusals, without their `<file> [<run>]: ` prefix.
        failures["vmin_fit_unverified"] = {
            n: loader_message(run_grid_cli(grid_argv(n), {})["exit"]) for n in unverified}
        failures["run_yaml_missing"] = run_grid_cli(grid_argv("nope"), {})["exit"].replace(
            str(REPO), "<REPO>")
        # The message lists every grid; the ones added since the capture are dropped.
        unknown_grid = loader_message(run_grid_cli(
            grid_argv("llama", "--set", "mode.grid=nope"), {})["exit"])
        for added in ADDED_GRIDS:
            unknown_grid = unknown_grid.replace(f", {added}", "")
        failures["vmin_fit_unknown_grid"] = unknown_grid
        failures["vmin_fit_unknown_prompt_mode"] = loader_message(run_grid_cli(
            grid_argv("llama"), {"PROMPT_MODE": "random"})["exit"])
        # vmin_fit sized an MLA stack without kv_bytes with the dense per-head
        # formula, silently (REPORT.md). The loader refuses it; the CAPACITY
        # vmin_fit printed is what the run gives when told dense explicitly.
        with mla_run_yaml() as run_yaml:
            refused = run_grid_cli(["run", str(run_yaml), "--set", "engine.model=<MODEL>"], {},
                                   model_config=MLA_CONFIG)
            self.assertIn("attention=mla needs kv_bytes", refused["exit"])
            mla = run_grid_cli(["run", str(run_yaml), "--set", "engine.model=<MODEL>",
                                "--set", "engine.attention=dense"], {}, model_config=MLA_CONFIG)
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
