"""Every quettabench module imports on its own in a fresh interpreter, so no
import order hides a cycle. A module whose optional dependency (torch, vllm,
QuettaSim's engine, ...) is not installed here is skipped by name; a missing
quettabench name is a failure. tests/_stubs is on the path, as the golden
harness puts it, so the probes that need only a torch or vllm name load too.
__main__ modules run a CLI on import and are left out."""

import os
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import quettabench

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "tests" / "_stubs"
MODULES = sorted(m.name for m in pkgutil.walk_packages(quettabench.__path__, "quettabench.")
                 if not m.name.endswith(".__main__"))


def fresh_import(code):
    env = {**os.environ, "PYTHONPATH": str(STUBS)}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_native_then_profiler_imports():
    proc = fresh_import("import quettabench.profiling.native; import quettabench.profiler")
    assert proc.returncode == 0, proc.stderr


def test_every_module_imports_alone():
    with ThreadPoolExecutor(max_workers=8) as pool:
        procs = dict(zip(MODULES, pool.map(lambda m: fresh_import(f"import {m}"), MODULES)))
    failures, skipped = [], []
    for name, proc in procs.items():
        if proc.returncode == 0:
            continue
        last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        if last.startswith("ModuleNotFoundError") and "'quettabench" not in last:
            skipped.append(name)
        else:
            failures.append(f"{name}: {last}")
    assert not failures, failures
    if len(skipped) == len(MODULES):
        pytest.skip("no module importable here")
