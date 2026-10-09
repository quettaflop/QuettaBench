"""qbench's doors: each hands its argv to the module's main(argv), and the
command runs from a checkout with python3 -m, nothing installed."""

import subprocess
import sys
import unittest
from unittest import mock

from quettabench import cli
from tests.golden._harness import REPO


class Doors(unittest.TestCase):
    def test_each_door_gets_the_rest_of_argv(self):
        for door, module in cli.DOORS.items():
            with self.subTest(door=door), mock.patch("importlib.import_module") as imp:
                imp.return_value.main.return_value = 0
                self.assertEqual(cli.main([door, "--x", "y"]), 0)
                imp.assert_called_once_with(module)
                imp.return_value.main.assert_called_once_with(["--x", "y"])

    def test_runs_without_installing(self):
        proc = subprocess.run([sys.executable, "-m", "quettabench.cli", "export", "--help"],
                              cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("usage: qbench export"))

    def test_no_door_prints_the_doors(self):
        proc = subprocess.run([sys.executable, "-m", "quettabench.cli"], cwd=REPO,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("{profile,crossval,export}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
