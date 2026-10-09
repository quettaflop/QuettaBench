# tests/probes/test_probe_contracts.py

"""Recipe-to-probe contract: every probe a root justfile recipe invokes must exist,
parse, and define every flag the recipe passes.

The probes need a GPU and torch to RUN, so CI cannot execute them; what it can
pin is the wiring that otherwise only fails at probe start on a GPU box: the
recipe paths, the argparse surface behind each recipe, the remote recipe shipping
the probes package, and the sibling _helper imports. Flags written --flag=value in a
recipe belong to the launcher (torch.distributed.run), not the probe, and are
skipped.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = ROOT / "justfile"
PROBES_DIR = ROOT / "quettabench" / "probes"
UNMOVED_PROBES = ROOT / "profiling" / "kernel_composed" / "probes"


def argparse_flags(probe: Path) -> set[str]:
    """Literal --flags the probe's add_argument calls define, shared base.py flags included."""
    flags = set()
    if probe.name != "base.py" and "add_output_args" in probe.read_text():
        flags |= argparse_flags(PROBES_DIR / "base.py")
    for node in ast.walk(ast.parse(probe.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            flags.update(
                arg.value for arg in node.args
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--")
            )
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


def probe_invocations() -> list[tuple[Path, set[str]]]:
    """(probe file, flags passed) per recipe."""
    out = []
    for block in recipe_blocks(JUSTFILE):
        probes = [PROBES_DIR / f"{m}.py" for m in re.findall(r"-m quettabench\.probes\.([a-z0-9_]+)", block)]
        probes += [UNMOVED_PROBES / m for m in re.findall(r"kernel_composed/probes/([a-z0-9_]+\.py)", block)]
        flags = {t for t in block.split() if t.startswith("--") and "=" not in t}
        out.extend((probe, flags) for probe in probes)
    return out


def test_every_probe_parses():
    probes = sorted(PROBES_DIR.glob("*.py"))
    assert probes, f"no probes found under {PROBES_DIR}"
    for probe in probes:
        ast.parse(probe.read_text(), filename=str(probe))


def test_recipe_probes_exist_and_accept_their_flags():
    invocations = probe_invocations()
    assert invocations, "no probe invocations found in the justfile"
    for probe, flags in invocations:
        assert probe.is_file(), f"justfile invokes {probe.name} but {probe} is missing"
        defined = argparse_flags(probe)
        missing = sorted(flags - defined)
        assert not missing, f"{probe.name} does not define recipe flags: {missing}"


# Live-server probes name their output through --out/--out-raw, not a GPU label.
LIVE_SERVER_PROBES = {"serving_decode_grid.py", "serving_stage_split.py"}


def test_every_probe_labels_its_gpu():
    """Tables are keyed by GPU name, so every kernel probe must take --gpu-label,
    recipe-invoked or not."""
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        if probe.name in LIVE_SERVER_PROBES:
            continue
        assert "--gpu-label" in argparse_flags(probe), f"{probe.name} has no --gpu-label"


def test_sibling_helper_imports_resolve():
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        for node in ast.walk(ast.parse(probe.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("quettabench.probes.")
            ):
                module = node.module.rsplit(".", 1)[-1]
                helper = PROBES_DIR / f"{module}.py"
                assert helper.is_file(), f"{probe.name} imports {node.module}, no {helper.name}"
                exported = defined_names(helper)
                for alias in node.names:
                    assert alias.name in exported, (
                        f"{probe.name} imports {node.module}.{alias.name}, not defined there"
                    )


def test_every_moved_probe_takes_out():
    """The remote recipe and `just probe` pass --out; every packaged probe must accept it."""
    for probe in sorted(PROBES_DIR.glob("[!_]*.py")):
        if probe.name != "base.py" and probe.name not in LIVE_SERVER_PROBES:
            assert "--out" in argparse_flags(probe), f"{probe.name} has no --out"


def test_remote_recipe_ships_the_probes_package():
    text = JUSTFILE.read_text()
    assert "quettabench/probes/" in text and "rsync" in text, "remote-probe must ship quettabench/probes"
    assert "quettabench/__init__.py" in text, "remote-probe must ship the package root"
    assert "--out $rdir/out" in text, "remote-probe must point --out at the pulled directory"
