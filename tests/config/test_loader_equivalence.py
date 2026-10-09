"""The new yaml tree resolves to what xval_config and vmin_fit resolve today.

For every workload in configs/workloads.json, configs/runs/<name>.yaml is
loaded through config/loader.py with env cleared and with the golden override
env, and compared field by field against xval_config under the same env, and
against the META/SKIP lines vmin_fit printed in the golden fixture.
"""

import json
import os
import unittest
from unittest import mock

from quettabench.config import xval_config
from quettabench.config.loader import ConfigError, load
from tests.golden._harness import ENV_OVERRIDE, FIXTURES, REPO

SCENARIOS = {"cleared": {}, "override": ENV_OVERRIDE}
WORKLOADS = json.loads((REPO / "configs" / "workloads.json").read_text())["workloads"]
GOLDEN = json.loads((FIXTURES / "config_workloads.json").read_text())
# Hybrid attention without kv_bytes: the loader refuses on purpose (REPORT.md
# "What the current code contradicts" item 4); vmin_fit used the dense formula.
REFUSED = {"qwen"}


def resolve(name, env):
    """(Config, xval_config's view) for one workload under exactly `env`; no
    tooling on PATH, as in the golden capture."""
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch("shutil.which", return_value=None):
        profile = xval_config.link_profile()
        old = {
            "wl": xval_config.workloads()[name],
            "grids": xval_config.grids(),
            "step_points": xval_config.step_points(),
            "run": xval_config.run_params(),
            "provision": xval_config.provision(),
            "backend": xval_config.backend(),
            "link_profile": profile,
            # vmin_fit exports link_profile() before collective() reads it.
            "collective": xval_config.collective(profile=profile),
        }
        configs = load(REPO / "configs" / "runs" / f"{name}.yaml", ["allow_unverified=true"], env)
    assert len(configs) == 1
    return configs[0], old


def meta(lines):
    return dict(line[5:].split("=", 1) for line in lines if line.startswith("META "))


class LoaderEquivalence(unittest.TestCase):
    def test_every_run_yaml_matches_xval_config(self):
        for name in sorted(set(WORKLOADS) - REFUSED):
            for scenario, env in SCENARIOS.items():
                with self.subTest(workload=name, env=scenario):
                    self.check(name, env, GOLDEN[name][scenario]["vmin_fit"]["stdout"])

    def check(self, name, env, golden_stdout):
        cfg, old = resolve(name, env)
        wl, e, m = old["wl"], cfg.engine, cfg.mode
        self.assertEqual(e.model_name, wl["name"])
        self.assertEqual(e.dtype, wl["dtype"])
        self.assertEqual(e.weights_gib, wl["weights_gib"])
        self.assertEqual(e.max_model_len, wl["maxlen"])
        self.assertEqual(e.tp, int(wl.get("tp", 1)))
        self.assertEqual(e.pp, int(env.get("XVAL_PP", wl.get("pp", 1))))
        self.assertEqual(e.ep, bool(wl.get("expert_parallel")))
        self.assertEqual(e.kv_dtype, wl.get("kv_dtype") or "auto")
        self.assertEqual(e.kv_bytes, wl.get("kv_bytes"))
        self.assertEqual(e.quant, wl.get("quant"))
        self.assertEqual(e.verified, wl["verified"])
        self.assertEqual(e.gpu_util, float(env.get("GPU_UTIL", wl.get("gpu_util", 0.90))))
        bench = wl.get("bench")
        self.assertEqual(e.extra.get("quettaserve"), bench and {
            "crate": bench["crate"], "test": bench["test"],
            "env_prefix": bench["env"], "family": bench["family"]})
        self.assertEqual(cfg.crossval.get("comm_bound_bs"), wl.get("comm_bound_bs"))
        for key in ("engine_commit", "checkpoint"):
            self.assertEqual(cfg.provenance.get(key), wl.get(key))

        self.assertEqual(cfg.workload.prompt_mode,
                         env.get("PROMPT_MODE", wl.get("prompt_mode", "distinct")))
        self.assertEqual(m.vmin_attempts, int(env.get("VMIN_ATTEMPTS", "4")))
        eval_pos = m.eval_pos if m.eval_pos is not None else cfg.hardware.timing_steps // 2
        self.assertEqual(eval_pos, int(env.get("XVAL_EVAL_POS", old["run"]["timing_steps"] // 2)))
        self.assertEqual(m.grid, wl["grid"])
        self.assertEqual(cfg.grid.cells, tuple(map(tuple, old["grids"][wl["grid"]])))
        self.assertEqual(cfg.grid.step_points, tuple(old["step_points"]))

        h = cfg.hardware
        self.assertEqual((h.backend, h.link_profile), (old["backend"], old["link_profile"]))
        self.assertEqual(dict(h.collective), old["collective"])
        self.assertEqual({"world": h.world, "timing_steps": h.timing_steps,
                          "max_seq_headroom": h.max_seq_headroom}, old["run"])
        self.assertEqual(dict(h.provision), old["provision"])

        # Against what vmin_fit itself printed on the stubs.
        got = meta(golden_stdout)
        self.assertEqual(got["model"], e.model_name)
        self.assertEqual(got["dtype"], e.dtype)
        self.assertEqual(got["grid"], m.grid)
        self.assertEqual(got["eval_pos"], str(eval_pos))
        self.assertEqual(got["link_profile"], h.link_profile)
        self.assertEqual(got["prompt_mode"], cfg.workload.prompt_mode)
        self.assertEqual((got["tp"], got["pp"]), (str(e.tp), str(e.pp)))
        self.assertEqual(got["quant"], e.quant or e.dtype)
        self.assertEqual("expert_parallel" in got, e.ep)
        skipped = {tuple(int(t.split("=")[1]) for t in line.split()[1:3])
                   for line in golden_stdout if line.endswith("reason=maxlen")}
        self.assertEqual(set(cfg.grid.cells_over_maxlen), skipped)

    def test_refused_workloads_fail_for_the_deliberate_reason(self):
        for name in sorted(REFUSED):
            for scenario, env in SCENARIOS.items():
                with self.subTest(workload=name, env=scenario), \
                        self.assertRaisesRegex(ConfigError, "attention=hybrid needs kv_bytes"):
                    resolve(name, env)

    def test_every_workload_has_a_run_yaml(self):
        runs = {p.stem for p in (REPO / "configs" / "runs").glob("*.yaml")}
        self.assertEqual(runs, set(WORKLOADS))


if __name__ == "__main__":
    unittest.main()
