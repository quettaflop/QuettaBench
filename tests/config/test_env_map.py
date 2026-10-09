"""env_map.read_env: each old env var lands on its Config path, typed the way
its old reader parsed it."""

import unittest

from quettabench.config.env_map import ENV_MAP, read_env
from quettabench.config.schema import ConfigError, default_values, dict_field


class ReadEnv(unittest.TestCase):
    def test_every_target_is_a_config_path(self):
        paths = default_values()
        for var, (path, _, _) in ENV_MAP.items():
            with self.subTest(var):
                self.assertTrue(path in paths or dict_field(path) or path.startswith("detect."))

    def test_values_are_typed_per_var(self):
        got = read_env({"XVAL_PP": "2", "GPU_UTIL": "0.85", "ENABLE_EP": "Yes",
                        "QS_SHA": "1234567", "CUDA_VISIBLE_DEVICES": "0",
                        "XVAL_ALLOW_METHOD_MISMATCH": "0", "TPU_NAME": "v5"})
        self.assertEqual(got, {"engine.pp": 2, "engine.gpu_util": 0.85, "engine.ep": True,
                               "provenance.engine_sha": "1234567",
                               "hardware.visible_devices": "0",
                               "crossval.allow_method_mismatch": True,
                               "detect.tpu_name": "v5"})

    def test_unset_and_empty_vars_are_skipped(self):
        self.assertEqual(read_env({"XVAL_PP": "", "UNRELATED": "1"}), {})

    def test_enable_ep_falsy_values_turn_ep_off(self):
        self.assertEqual(read_env({"ENABLE_EP": "0"}), {"engine.ep": False})

    def test_qs_sha_wins_over_ds_engine_sha(self):
        got = read_env({"QS_SHA": "aaa", "DS_ENGINE_SHA": "bbb"})
        self.assertEqual(got, {"provenance.engine_sha": "aaa"})
        self.assertEqual(read_env({"DS_ENGINE_SHA": "bbb"}), {"provenance.engine_sha": "bbb"})

    def test_bad_number_names_the_var(self):
        with self.assertRaisesRegex(ConfigError, "XVAL_PP='two'"):
            read_env({"XVAL_PP": "two"})

    def test_corpus_and_trace_both_set_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "XVAL_CORPUS and XVAL_TRACE"):
            read_env({"XVAL_CORPUS": "a.txt", "XVAL_TRACE": "b.jsonl"})
        self.assertEqual(read_env({"XVAL_TRACE": "b.jsonl"}), {"workload.source": "b.jsonl"})


if __name__ == "__main__":
    unittest.main()
