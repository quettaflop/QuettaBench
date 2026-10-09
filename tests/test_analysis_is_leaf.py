"""quettabench must not depend on the one-off studies under analysis/."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "quettabench"

# dataset.py still builds synthetic profiles through a lazy import. It goes
# when the synthetic sampler is dropped from workloads/; do not add to this.
KNOWN_EXCEPTIONS = {"quettabench/workloads/dataset.py"}


def imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_package_never_imports_analysis():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if str(path.relative_to(PACKAGE.parent)) in KNOWN_EXCEPTIONS:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in imported_modules(tree):
            if module == "analysis" or module.startswith("analysis."):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}: {module}")
    assert not offenders, offenders
