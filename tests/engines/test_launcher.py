"""start/health/engine.json/stop on a fake server, and no child survives stop."""

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quettabench.engines import launcher
from quettabench.engines.base import LaunchSpec, UnsupportedField

FAKE = Path(__file__).with_name("fake_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reparented zombie no longer runs; count it dead.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(") ")[1].startswith("Z"))


class LocalLauncher(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.state = self.dir / "state.json"
        self.port = free_port()

    def spec(self, **kw):
        return LaunchSpec(adapter="fake", port=self.port, health_path="/health",
                          server_argv=(sys.executable, str(FAKE), str(self.port), str(self.state)),
                          env={"FAKE_MARK": "from-spec", "HF_TOKEN": "secret",
                               "DB_SECRET": "secret"}, **kw)

    def test_start_health_engine_json_stop(self):
        handle = launcher.start(self.spec(), self.dir / "run", timeout_s=30, poll_s=0.1)
        state = json.loads(self.state.read_text())
        self.assertEqual(state["mark"], "from-spec")
        record = json.loads((self.dir / "run" / "engine.json").read_text())
        self.assertEqual(record["adapter"], "fake")
        self.assertEqual(record["endpoint"], f"http://127.0.0.1:{self.port}")
        self.assertEqual(record["pid"], handle.pid)
        self.assertEqual(record["instrumentation"], "none")
        self.assertEqual(record["launch_spec"]["env"],
                         {"FAKE_MARK": "from-spec", "HF_TOKEN": "REDACTED",
                          "DB_SECRET": "REDACTED"})  # the results.redact rule
        self.assertNotIn("secret", (self.dir / "run" / "engine.json").read_text())
        self.assertTrue(alive(state["child_pid"]))
        handle.stop(grace_s=5)
        self.assertFalse(alive(handle.pid))
        self.assertFalse(alive(state["child_pid"]))

    def test_server_that_dies_fails_fast(self):
        spec = LaunchSpec(adapter="fake", port=self.port,
                          server_argv=(sys.executable, "-c", "raise SystemExit(4)"))
        with self.assertRaisesRegex(RuntimeError, "exited with 4"):
            launcher.start(spec, self.dir / "dead", timeout_s=30, poll_s=0.1)

    def test_refusals(self):
        with self.assertRaises(NotImplementedError):
            launcher.start(self.spec(), self.dir, backend="docker")
        with self.assertRaises(UnsupportedField):
            launcher.start(LaunchSpec(adapter="vllm", server_unsupported=("pp=2",)), self.dir)
        with self.assertRaisesRegex(ValueError, "no server"):
            launcher.start(LaunchSpec(adapter="quettaserve"), self.dir)


class IdleGpus(unittest.TestCase):
    def test_non_cuda_takes_visible_devices_or_the_first_n(self):
        self.assertEqual(launcher.idle_gpus(3, "rocm"), "0,1,2")
        self.assertEqual(launcher.idle_gpus(2, "cpu", "4,5"), "4,5")

    def test_cuda_picks_gpus_under_1000_mib(self):
        smi = "0, 70000 MiB\n1, 3 MiB\n2, 999 MiB\n3, 1000 MiB\n"
        with mock.patch.object(launcher.subprocess, "run",
                               return_value=mock.Mock(stdout=smi)):
            self.assertEqual(launcher.idle_gpus(2, "cuda"), "1,2")
            with self.assertRaisesRegex(RuntimeError, "NEED 3 FREE GPUS, have: 1 2"):
                launcher.idle_gpus(3, "cuda")


if __name__ == "__main__":
    unittest.main()
