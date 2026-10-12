"""env_map.read_env maps each env var to its Config path, and no other module reads os.environ."""

import ast
import unittest

from quettabench.config.env_map import ENV_MAP, read_env
from quettabench.config.schema import ConfigError, default_values, dict_field
from tests.golden._harness import REPO


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

    def test_afd_pools_become_a_serving_style(self):
        self.assertEqual(read_env({"XVAL_AFD": "attn=2,ffn=6"}),
                         {"engine.serving_style": "afd:2a6f"})
        for bad in ("attn=2", "attn=2,ffn=6,moe=1", "attn=x,ffn=1", "2,6"):
            with self.subTest(bad), self.assertRaisesRegex(ConfigError, "XVAL_AFD"):
                read_env({"XVAL_AFD": bad})

    def test_afd_beside_another_serving_style_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "XVAL_SERVING_STYLE and XVAL_AFD"):
            read_env({"XVAL_SERVING_STYLE": "aggregated", "XVAL_AFD": "attn=1,ffn=1"})


ROOTS = ("quettabench",)
ENV_NAMES = ("environ", "getenv")
OWNER = "quettabench/config/env_map.py"

ALLOWLIST = {
    # kernel probes: torchrun's RANK/WORLD_SIZE/LOCAL_RANK are the launcher protocol, NCU_BIN names the profiler binary
    "quettabench/probes/flash_attn.py",
    "quettabench/probes/gdn.py",
    "quettabench/probes/gemm.py",
    "quettabench/probes/reparallel.py",
    "quettabench/probes/vllm_allreduce.py",
    "quettabench/profiling/kv_transfer.py",
}


def reads_env(source):
    """True when the module reads the process environment in any of the scanned forms."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in ENV_NAMES \
                and isinstance(node.value, ast.Name) and node.value.id == "os":
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "environ":
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os" \
                and any(a.name in ENV_NAMES for a in node.names):
            return True
    return False


def env_names(path):
    """The literal names a module reads or sets through os.environ / os.getenv."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            names.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and (node.func.attr == "getenv" or (isinstance(node.func.value, ast.Attribute)
                                                    and node.func.value.attr == "environ")):
            names.add(node.args[0].value)
    return names


def env_readers():
    found = set()
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if reads_env(path.read_text()):
                found.add(path.relative_to(REPO).as_posix())
    return found


class NoEnviron(unittest.TestCase):
    def test_only_env_map_and_allowlisted_files_read_env(self):
        readers = env_readers()
        self.assertIn(OWNER, readers)
        self.assertEqual(sorted(readers - ALLOWLIST - {OWNER}), [],
                         "read env through config/env_map.read_env instead")

    def test_allowlist_has_no_stale_entries(self):
        stale = sorted(ALLOWLIST - env_readers())
        self.assertEqual(stale, [], "no longer reads env: drop it from ALLOWLIST")

    def test_agents_law_6_names_every_variable_the_allowlist_reads(self):
        """AGENTS.md law 6 lists the env vars the allowlisted files read; a
        new one read without being named there fails."""
        law = (REPO / "AGENTS.md").read_text().split("6. No module outside", 1)[1].split("\n7. ", 1)[0]
        for path in sorted(ALLOWLIST):
            for name in env_names(REPO / path):
                self.assertIn(f"`{name}`", law, path)

    def test_scanner_catches_each_form(self):
        for src in ("import os\nos.environ.get('X')", "import os\nos.getenv('X')",
                    "from os import environ\nenviron['X']", "from os import getenv"):
            with self.subTest(src):
                self.assertTrue(reads_env(src))
        self.assertFalse(reads_env("import os\nos.path.exists('x')"))


if __name__ == "__main__":
    unittest.main()
