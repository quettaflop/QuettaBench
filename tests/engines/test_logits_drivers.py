"""The FIDELITY gate's two halves on the stubs: InProcessDriver scores with
logit_agreement.py's vLLM call, ReferenceDriver with its transformers
forward, and both summaries line up position for position; then the same
through `qbench run` twice and `qbench crossval`."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.crossval import agreement
from quettabench.engines.base import LaunchSpec
from quettabench.engines.drivers import InProcessDriver, ReferenceDriver
from quettabench.workloads.base import Request
from tests.golden._harness import crossval_path, fresh_import, grid_argv, run_grid_cli

CELL = {"ctx": 5, "bs": 2, "tp": 1, "step_point": None, "phase": "logits", "attempt": None}
PROMPTS = [[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]]


def requests():
    return [Request(ids, len(ids)) for ids in PROMPTS]


class Halves(unittest.TestCase):
    def vllm_rows(self):
        vllm = fresh_import("vllm")
        vllm.reset()
        driver = InProcessDriver(LaunchSpec(adapter="vllm", engine_kwargs={"model": "m"}))
        with mock.patch.object(vllm.LLM, "generate", autospec=True,
                               side_effect=vllm.LLM.generate) as generate:
            rows = list(driver.run(requests(), cell=CELL, returns="logits"))
        (_, prompts, params), kwargs = generate.call_args
        self.assertEqual(params.kwargs, {"temperature": 0, "max_tokens": 1, "prompt_logprobs": 2})
        self.assertEqual(prompts, [{"prompt_token_ids": ids} for ids in PROMPTS])
        self.assertEqual(kwargs, {"use_tqdm": False})
        return rows

    def reference_rows(self):
        transformers = fresh_import("transformers")
        torch = fresh_import("torch")
        transformers.CALLS.clear()
        driver = ReferenceDriver(LaunchSpec(adapter="reference",
                                            engine_kwargs={"model": "/w", "dtype": "bfloat16"}))
        rows = list(driver.run(requests(), cell=CELL))
        self.assertEqual(transformers.CALLS, [
            ("AutoModelForCausalLM.from_pretrained", "/w", {"torch_dtype": torch.bfloat16}),
            ("to", "cuda:0"), ("eval",)])
        return rows

    def test_layers_cut_the_config_as_slice_truncate_did(self):
        with crossval_path(stubs=True):
            transformers = fresh_import("transformers")
            torch = fresh_import("torch")
            transformers.CALLS.clear()
            driver = ReferenceDriver(LaunchSpec(adapter="reference", engine_kwargs={
                "model": "/w", "dtype": "float32", "layers": 2}))
            list(driver.run(requests(), cell=CELL))
            (_, path, kwargs) = transformers.CALLS[1]
            self.assertEqual(transformers.CALLS[0], ("AutoConfig.from_pretrained", "/w", {}))
            self.assertEqual((path, kwargs["torch_dtype"]), ("/w", torch.float32))
            self.assertEqual((kwargs["config"].num_hidden_layers, kwargs["config"].layer_types),
                             (2, ["a", "b"]))
            self.assertEqual(len(driver.model().model.layers), 2)
            deep = ReferenceDriver(LaunchSpec(adapter="reference", engine_kwargs={
                "model": "/w", "dtype": "float32", "layers": 5}))
            with self.assertRaisesRegex(ValueError, "exceeds the checkpoint's 4 layers"):
                deep.model()

    def test_summaries_line_up(self):
        with crossval_path(stubs=True):
            ours, ref = self.vllm_rows(), self.reference_rows()
        for o, r, ids in zip(ours, ref, PROMPTS):
            self.assertEqual(o.cell, CELL)
            self.assertEqual(o.logits["argmax"], [t + 1 for t in ids[:-1]])
            self.assertEqual(r.logits["argmax"], o.logits["argmax"])
            self.assertEqual(len(r.logits["logprob"]), len(ids) - 1)
            for a, b in zip(o.logits["logprob"] + o.logits["top2_gap"],
                            r.logits["logprob"] + r.logits["top2_gap"]):
                self.assertAlmostEqual(a, b, places=12)
        summary = agreement.logit([{"cell": CELL, "logits": o.logits} for o in ours],
                                  [{"cell": CELL, "logits": r.logits} for r in ref])[-1]
        self.assertEqual((summary["positions"], summary["argmax_agree"], summary["verdict"]),
                         (8, 8, "PASS"))


class Recipe(unittest.TestCase):
    def test_run_run_crossval_is_fidelity_and_band_pass(self):
        sets = ["--set", "mode.grid=smoke"]
        with tempfile.TemporaryDirectory() as td:
            for name, extra in (("ours", ["--set", "mode.returns=logits"]),
                                ("ref", ["--set", "engine.name=reference"]),
                                ("slice", ["--set", "engine.name=reference",
                                           "--set", "engine.layers=2"])):
                run = run_grid_cli(grid_argv("llama", *sets, *extra, "--set", f"run_name={name}",
                                             "--results-dir", td), {})
                self.assertIsNone(run["exit"])
            from quettabench import crossval
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                crossval.main([str(Path(td, "ours")), str(Path(td, "ref")),
                               "--floor", str(Path(td, "slice")),
                               "--json", str(Path(td, "x.json"))])
        self.assertIn("SUMMARY positions=99 argmax_agree=99 (100.00%)", out.getvalue())
        self.assertIn("FIDELITY verdict=PASS hard_disagreements=0", out.getvalue())
        self.assertIn("BAND verdict=PASS flips=0 floor=0 floor_runs=0 positions=99",
                      out.getvalue())


if __name__ == "__main__":
    unittest.main()
