# profiling/tests/test_probe_contracts.py

"""Recipe-to-probe contract for kernel_composed: every probe a justfile recipe
invokes must exist, parse, and define every flag the recipe passes.

The probes need a GPU and torch to RUN, so CI cannot execute them; what it can
pin is the wiring that otherwise only fails at probe start on a GPU box: the
recipe paths, the argparse surface behind each recipe, run_probe.sh's --out-dir
injection, and the sibling _helper imports. Flags written --flag=value in a
recipe belong to the launcher (torch.distributed.run), not the probe, and are
skipped.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

KERNEL_COMPOSED = Path(__file__).resolve().parents[1] / "kernel_composed"
PROBES_DIR = KERNEL_COMPOSED / "probes"
RUN_PROBE = KERNEL_COMPOSED / "run_probe.sh"


def argparse_flags(probe: Path) -> set[str]:
    """Literal --flags the probe's add_argument calls define."""
    flags = set()
    for node in ast.walk(ast.parse(probe.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).startswith("--")
        ):
            flags.add(node.args[0].value)
    return flags


def defined_names(module: Path) -> set[str]:
    """Top-level function, class and assignment names in a module."""
    names = set()
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def recipe_blocks(justfile: Path) -> list[str]:
    """Justfile split at non-indented lines, backslash continuations folded."""
    text = justfile.read_text().replace("\\\n", " ")
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def probe_invocations() -> list[tuple[str, set[str], bool]]:
    """(probe filename, flags passed, invoked via run_probe.sh) per recipe."""
    out = []
    for block in recipe_blocks(KERNEL_COMPOSED / "justfile"):
        names = set(re.findall(r"(?:run_probe\.sh\s+|probes/)([a-z0-9_]+\.py)", block))
        if not names:
            continue
        flags = {t for t in block.split() if t.startswith("--") and "=" not in t}
        for name in names:
            out.append((name, flags, "run_probe.sh" in block))
    return out


def test_every_probe_parses():
    probes = sorted(PROBES_DIR.glob("*.py"))
    assert probes, f"no probes found under {PROBES_DIR}"
    for probe in probes:
        ast.parse(probe.read_text(), filename=str(probe))


def test_recipe_probes_exist_and_accept_their_flags():
    invocations = probe_invocations()
    assert invocations, "no probe invocations found in the justfile"
    for name, flags, via_run_probe in invocations:
        probe = PROBES_DIR / name
        assert probe.is_file(), f"justfile invokes {name} but probes/{name} is missing"
        defined = argparse_flags(probe)
        missing = sorted(flags - defined)
        assert not missing, f"{name} does not define recipe flags: {missing}"
        if via_run_probe:
            assert "--out-dir" in defined, (
                f"{name} runs through run_probe.sh, which injects --out-dir"
            )


def test_sibling_helper_imports_resolve():
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        for node in ast.walk(ast.parse(probe.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("_")
                and not node.module.startswith("__")
            ):
                helper = PROBES_DIR / f"{node.module}.py"
                assert helper.is_file(), f"{probe.name} imports {node.module}, no {helper.name}"
                exported = defined_names(helper)
                for alias in node.names:
                    assert alias.name in exported, (
                        f"{probe.name} imports {node.module}.{alias.name}, not defined there"
                    )


def test_run_probe_sh_contract():
    subprocess.run(["bash", "-n", str(RUN_PROBE)], check=True)
    text = RUN_PROBE.read_text()
    assert '--out-dir "$KDATA"' in text, "run_probe.sh must inject --out-dir"
    assert 'probes/_*.py' in text, "run_probe.sh must ship the _helper modules to remote hosts"
