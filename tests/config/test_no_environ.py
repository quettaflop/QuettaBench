"""os.environ is read only by quettabench/config/env_map.py.

AST scan of quettabench/ and legacy/ for os.environ, os.getenv, environ[...]
and `from os import environ|getenv`. Files that still read env today are on
ALLOWLIST with the commit that removes the read (Stage 2 numbering). The test
fails on a new reader, and on an allowlisted file that no longer reads env,
so the list only shrinks.
"""

import ast
import unittest

from tests.golden._harness import REPO

ROOTS = ("quettabench", "legacy")
ENV_NAMES = ("environ", "getenv")
OWNER = "quettabench/config/env_map.py"

ALLOWLIST = {
    # removed in commit 7: runner.py replaces the shell cli; detection lives in loader.py
    "quettabench/config/xval_config.py",
    # removed in commit 6: step_grid mode takes a Config
    "quettabench/modes/vmin_fit.py",
    # removed in Phase 3: crossval.py reads results files; cache.py is deleted
    "quettabench/crossval/cache.py",
    # removed in Phase 3: table logic moves to crossval.py and reads Config
    "quettabench/crossval/table.py",
    # removed in Phase 3: agreement becomes a report row (copies env for a subprocess)
    "quettabench/crossval/greedy_agreement.py",
    # removed in Phase 3 exit: legacy/ is deleted
    "legacy/trace_serve.py",
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

    def test_scanner_catches_each_form(self):
        for src in ("import os\nos.environ.get('X')", "import os\nos.getenv('X')",
                    "from os import environ\nenviron['X']", "from os import getenv"):
            with self.subTest(src):
                self.assertTrue(reads_env(src))
        self.assertFalse(reads_env("import os\nos.path.exists('x')"))


if __name__ == "__main__":
    unittest.main()
