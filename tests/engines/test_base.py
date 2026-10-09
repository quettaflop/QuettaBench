"""LaunchSpec is plain frozen data; child_argv carries env without reading ours."""

import dataclasses
import unittest

from quettabench.engines import base


class LaunchSpecTests(unittest.TestCase):
    def test_frozen(self):
        spec = base.LaunchSpec(adapter="x")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.port = 1

    def test_child_argv_sets_and_drops_env(self):
        spec = base.LaunchSpec(adapter="x", env={"A": "1", "B": "two"}, env_drop=("C",))
        self.assertEqual(spec.child_argv(["run", "--x"]),
                         ["env", "-u", "C", "A=1", "B=two", "run", "--x"])

    def test_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "unknown engine 'nope'"):
            base.get("nope")

    def test_unsupported_field_names_engine_and_field(self):
        err = base.UnsupportedField("vllm", "kv_dtype", "no flag")
        self.assertEqual((err.engine, err.field), ("vllm", "kv_dtype"))
        self.assertIn("kv_dtype", str(err))


if __name__ == "__main__":
    unittest.main()
