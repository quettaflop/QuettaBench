"""Every quettabench module imports on its own, and the package never imports analysis/ or data/."""

import ast
import os
import pkgutil
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import quettabench

ROOT = Path(__file__).resolve().parents[2]
STUBS = ROOT / "tests" / "_stubs"
MODULES = sorted(
    m.name for m in pkgutil.walk_packages(quettabench.__path__, "quettabench.") if not m.name.endswith(".__main__")
)
# GPU-only or external packages the CPU suite lacks; any other missing module is a typo or a bug.
OPTIONAL_PACKAGES = {"torch", "vllm", "triton", "fla", "safetensors", "engine"}


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
        missing = re.match(r"ModuleNotFoundError: No module named '([^']+)'", last)
        if missing and missing[1].split(".")[0] in OPTIONAL_PACKAGES:
            skipped.append(name)
        else:
            failures.append(f"{name}: {last}")
    assert not failures, failures
    if len(skipped) == len(MODULES):
        pytest.skip("no module importable here")


PACKAGE = Path(__file__).resolve().parents[2] / "quettabench"


def imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_package_never_imports_analysis_or_data():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in imported_modules(tree):
            if module.split(".")[0] in ("analysis", "data"):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}: {module}")
    assert not offenders, offenders
