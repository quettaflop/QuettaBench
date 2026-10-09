"""legacy_cli.grid_config resolves a workload name as vmin_fit did: the
engine kwargs equal vmin_fit's golden LLM kwargs, the bad-name exits keep
vmin_fit's messages, and the env knobs land in the Config fields."""

import json
import os
import unittest
from unittest import mock

from quettabench.engines import vllm as vllm_adapter
from quettabench.modes import legacy_cli
from tests.golden._harness import ENV_OVERRIDE, FIXTURES, REPO

WORKLOADS = sorted(json.loads((REPO / "configs" / "workloads.json").read_text())["workloads"])
ENGINE_ARGS = json.loads((FIXTURES / "engine_args.json").read_text())
FAILURES = json.loads((FIXTURES / "config_failures.json").read_text())


def grid_config(argv, env):
    with mock.patch.dict(os.environ, env, clear=True), mock.patch("shutil.which", return_value=None):
        return legacy_cli.grid_config(legacy_cli.get_args(argv))


def exit_message(argv, env=None):
    """The SystemExit message grid_config stops with, or None."""
    try:
        grid_config(argv, env or {})
    except SystemExit as e:
        return str(e.code)
    return None


class EngineKwargs(unittest.TestCase):
    def kwargs(self, name, env, *extra):
        cfg = grid_config(["--grid", "--workload", name, "--model", "<MODEL>",
                           "--allow-unverified", *extra], env)
        return vllm_adapter.ADAPTER.translate(cfg).engine_kwargs

    def test_every_workload_matches_vmin_fit(self):
        for name in WORKLOADS:
            with self.subTest(name):
                golden = ENGINE_ARGS[name]
                self.assertEqual(self.kwargs(name, {}), golden["vmin_fit"])
                self.assertEqual(self.kwargs(name, ENV_OVERRIDE), golden["vmin_fit_env_override"])
                self.assertEqual(self.kwargs(name, {}, "--nograph"), golden["vmin_fit_nograph"])


class Exits(unittest.TestCase):
    def test_bad_names_keep_vmin_fit_messages(self):
        base = ["--grid", "--model", "m"]
        self.assertEqual(exit_message(["--grid", "nope", "--workload", "llama", "--model", "m"]),
                         FAILURES["vmin_fit_unknown_grid"])
        self.assertEqual(exit_message(base + ["--workload", "llama"], {"PROMPT_MODE": "random"}),
                         FAILURES["vmin_fit_unknown_prompt_mode"])
        for name, message in FAILURES["vmin_fit_unverified"].items():
            self.assertEqual(exit_message(base + ["--workload", name]), message)


class EnvKnobs(unittest.TestCase):
    def test_override_scenario_fields(self):
        cfg = grid_config(["--grid", "--workload", "llama", "--model", "m"], ENV_OVERRIDE)
        self.assertEqual((cfg.engine.pp, cfg.engine.gpu_util), (2, 0.85))
        self.assertEqual((cfg.mode.vmin_attempts, cfg.mode.eval_pos, cfg.mode.grid), (2, 25, "llama"))
        self.assertEqual((cfg.workload.prompt_mode, cfg.hardware.link_profile), ("clone", "pcie"))
        self.assertEqual(cfg.hardware.collective["NCCL_ALGO"], "allreduce:tree;allgather:ring")
        self.assertEqual(cfg.run_name, "llama")

    def test_cleared_scenario_defaults(self):
        cfg = grid_config(["--grid", "Smoke", "--workload", "llama", "--model", "m"], {})
        self.assertEqual((cfg.mode.vmin_attempts, cfg.mode.eval_pos), (4, None))
        self.assertEqual((cfg.hardware.link_profile, cfg.hardware.timing_steps), ("nvlink", 100))
        self.assertNotIn("NCCL_ALGO", cfg.hardware.collective)
        self.assertEqual((cfg.mode.grid, cfg.run_name), ("smoke", "llama-Smoke"))
        self.assertEqual(cfg.grid.step_points, (60, 230, 400))

    def test_caller_nccl_wins(self):
        cfg = grid_config(["--grid", "--workload", "llama", "--model", "m"],
                          {"XVAL_LINK_PROFILE": "pcie", "NCCL_ALGO": "Ring"})
        self.assertEqual(cfg.hardware.collective["NCCL_ALGO"], "Ring")


if __name__ == "__main__":
    unittest.main()
