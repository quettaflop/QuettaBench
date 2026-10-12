"""Golden: the engine args each run yaml translates to."""

import json
import unittest

from quettabench.config import loader
from quettabench.config.schema import ConfigError
from quettabench.engines import vllm as vllm_adapter
from tests.golden._harness import ENV_OVERRIDE, FIXTURES, REPO, RUNS, check_json, grid_argv, run_grid_cli

WORKLOADS = {
    name: rec["cleared"]["captured_config"]["workload"]
    for name, rec in json.loads((FIXTURES / "config_workloads.json").read_text()).items()
}
ABSENT = "<absent>"


def returns_grid_kwargs(name, env, extra_argv=()):
    """The one path that still builds vllm.LLM."""
    run = run_grid_cli(grid_argv(name, "--allow-unverified", "--set", "mode.return_outputs=text", *extra_argv), env)
    (kwargs,) = run["llm_kwargs"]
    return kwargs


def serve_kwargs(name):
    tp = WORKLOADS[name].get("tp", 1)
    try:
        (cfg,) = loader.load(
            RUNS / f"{name}.yaml", ["allow_unverified=true", "engine.model=<MODEL>", f"engine.tp={tp}"], env={}
        )
    except ConfigError as e:
        return {"exit": "trace_serve: " + str(e).replace(str(REPO), "<REPO>")}
    return dict(vllm_adapter.ADAPTER.translate(cfg).engine_kwargs)


def kwargs_diff(a, b):
    return {
        k: [a.get(k, ABSENT), b.get(k, ABSENT)] for k in sorted(set(a) | set(b)) if a.get(k, ABSENT) != b.get(k, ABSENT)
    }


class EngineArgsGolden(unittest.TestCase):
    def test_engine_kwargs_per_workload(self):
        record = {}
        for name in sorted(WORKLOADS):
            vmin = returns_grid_kwargs(name, {})
            serve = serve_kwargs(name)
            record[name] = {
                "vmin_fit": vmin,
                "vmin_fit_env_override": returns_grid_kwargs(name, ENV_OVERRIDE),
                "vmin_fit_nograph": returns_grid_kwargs(name, {}, ["--set", "engine.cudagraph=false"]),
                "trace_serve": serve,
                "diff_vmin_fit_vs_trace_serve": None if "exit" in serve else kwargs_diff(vmin, serve),
            }
        check_json(self, "engine_args", record)


if __name__ == "__main__":
    unittest.main()
