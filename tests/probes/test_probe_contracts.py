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
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = ROOT / "justfile"
PROBES_DIR = ROOT / "quettabench" / "probes"
PROFILING_DIR = ROOT / "quettabench" / "profiling"


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
        probes += [PROFILING_DIR / f"{m}.py" for m in re.findall(r"-m quettabench\.profiling\.([a-z0-9_]+)", block)]
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


def test_remote_recipe_is_one_line_into_base():
    block = next(b for b in recipe_blocks(JUSTFILE) if b.startswith("remote-probe "))
    body = [ln.strip() for ln in block.splitlines()[1:] if ln.strip() and not ln.strip().startswith("#")]
    assert len(body) == 1 and "-m quettabench.probes.base remote-probe" in body[0], body
    assert "rm -rf" not in block and "/tmp/qbench-probes" not in block


def test_remote_probe_ships_what_base_imports():
    """Exactly the files remote_probe ships, in a clean dir, import
    quettabench.probes.base under an isolated interpreter."""
    from quettabench.probes.base import SHIPPED
    assert "quettabench/probes/" in SHIPPED and "quettabench/__init__.py" in SHIPPED
    with tempfile.TemporaryDirectory() as td:
        for rel in SHIPPED:
            src, dst = ROOT / rel, Path(td) / rel
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import quettabench.probes.base as b; "
                "assert b.__file__.startswith(sys.argv[1]), b.__file__")
        proc = subprocess.run([sys.executable, "-I", "-c", code, td], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


class FakeHost:
    """subprocess.run for remote_probe: records each command, answers mktemp
    with rdir, and fails the remote probe run and the pull as told."""

    def __init__(self, rdir="/tmp/qbench-probes.Ab12Cd", probe_rc=0, pull_rc=0):
        self.rdir, self.probe_rc, self.pull_rc, self.calls = rdir, probe_rc, pull_rc, []

    def __call__(self, cmd, check=False, **kw):
        self.calls.append(cmd)
        rc = 0
        if cmd[0] == "rsync" and cmd[-2].endswith("/out/"):
            rc = self.pull_rc
        elif cmd[0] == "ssh" and "quettabench.probes." in cmd[-1]:
            rc = self.probe_rc
        if check and rc:
            raise subprocess.CalledProcessError(rc, cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=self.rdir + "\n" if "mktemp" in cmd[-1] else "")


NOW = datetime(2026, 10, 11, 12, 0, 0, 123456, tzinfo=timezone.utc)
REMOTE = ["--host", "a100", "--py", "/v/bin/python", "--gpu-id", "3"]


def remote(argv, host, tmp_path, monkeypatch):
    from quettabench.probes.base import remote_probe
    monkeypatch.chdir(tmp_path)
    return remote_probe([*REMOTE, *argv], run=host, now=NOW)


def test_remote_probe_runs_in_its_own_dir_and_always_pulls(tmp_path, monkeypatch):
    """A failed remote run is still pulled back, then only its own mktemp dir
    is removed, and the remote run's exit code is returned."""
    host = FakeHost(probe_rc=3)
    rc = remote(["gemm", "A100", "--probe-args=--ncu --wide"], host, tmp_path, monkeypatch)
    rdir = host.rdir
    assert rc == 3
    mktemp, ship, probe, pull, rm = host.calls
    assert mktemp == ["ssh", "-o", "BatchMode=yes", "a100", "mktemp -d /tmp/qbench-probes.XXXXXX"]
    assert ship[:4] == ["rsync", "-a", "--relative", "-e"] and ship[-1] == f"a100:{rdir}/"
    assert probe[-1] == (f"cd {rdir} && CUDA_VISIBLE_DEVICES=3 /v/bin/python -m quettabench.probes.gemm "
                         f"--gpu-label A100 --out {rdir}/out --run-id 20261011T120000.123456Z --ncu --wide")
    assert pull[-2:] == [f"a100:{rdir}/out/", "results/probes/A100/"] and "--ignore-existing" in pull
    assert rm == ["ssh", "-o", "BatchMode=yes", "a100", f"rm -rf {rdir}"]
    assert (tmp_path / "results" / "probes" / "A100").is_dir()


def test_remote_probe_under_torchrun_and_ncu(tmp_path, monkeypatch):
    host = FakeHost()
    remote(["--nproc", "2", "--gpu-ids", "0,1", "collective", "H100"], host, tmp_path, monkeypatch)
    assert "CUDA_VISIBLE_DEVICES=0,1 /v/bin/python -m torch.distributed.run --nproc_per_node=2 " \
        "-m quettabench.probes.collective --gpu-label H100" in host.calls[2][-1]
    host = FakeHost()
    remote(["--ncu-bin", "/opt/ncu", "gemm", "H100"], host, tmp_path, monkeypatch)
    assert "CUDA_VISIBLE_DEVICES=3 NCU_BIN=/opt/ncu /v/bin/python -m quettabench.probes.gemm" \
        in host.calls[2][-1]


def test_a_failed_pull_keeps_the_remote_dir(tmp_path, monkeypatch):
    host = FakeHost(pull_rc=23)
    with pytest.raises(SystemExit, match="kept"):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert not any("rm -rf" in c[-1] for c in host.calls)


def test_a_failed_ship_removes_the_remote_dir(tmp_path, monkeypatch):
    class ShipFails(FakeHost):
        def __call__(self, cmd, check=False, **kw):
            if cmd[:3] == ["rsync", "-a", "--relative"]:
                self.calls.append(cmd)
                raise subprocess.CalledProcessError(12, cmd)
            return super().__call__(cmd, check=check, **kw)
    host = ShipFails()
    with pytest.raises(subprocess.CalledProcessError):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert host.calls[-1][-1] == f"rm -rf {host.rdir}"


@pytest.mark.parametrize("rdir", ["", "/tmp", "/tmp/qbench-probes", "/tmp/qbench-probes.x y", "/"])
def test_an_odd_mktemp_answer_is_never_removed(rdir, tmp_path, monkeypatch):
    host = FakeHost(rdir=rdir)
    with pytest.raises(SystemExit, match="mktemp"):
        remote(["gemm", "A100"], host, tmp_path, monkeypatch)
    assert len(host.calls) == 1


@pytest.mark.parametrize("argv", [["../x", "A100"], ["gemm", "A 100"], ["gemm", "A100", "--host", ""]])
def test_remote_probe_refuses_odd_names(argv, tmp_path, monkeypatch):
    host = FakeHost()
    with pytest.raises(SystemExit):
        remote(argv, host, tmp_path, monkeypatch)
    assert host.calls == []


def test_the_r2_mirror_has_its_own_prefix_and_never_deletes():
    """mirror-up/down sync results/probes/ with profiling/qbench_probes/ only;
    a --delete onto a shared prefix would remove what other writers keep."""
    text = JUSTFILE.read_text()
    assert 'r2_prefix := "profiling/qbench_probes"' in text
    assert "--delete" not in text and "mirror-up-prune" not in text
    assert "kernel_data" not in "".join(ln for ln in text.splitlines() if "r2_" in ln)


def test_probes_import_their_siblings_at_the_top():
    """_graph and _ncu ship with every probe, so no probe imports them inside a
    function; an inline import names its reason in a docstring, not a noqa."""
    for probe in sorted(PROBES_DIR.glob("*.py")):
        text = probe.read_text()
        assert "noqa: PLC0415" not in text, probe.name
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = [n.module for n in ast.walk(node) if isinstance(n, ast.ImportFrom)]
                assert not any(m and m.startswith("quettabench.probes._") for m in inner), \
                    f"{probe.name}:{node.name}"


def inline_imports_without_a_reason(path: Path) -> list[str]:
    """Function-level imports in path whose reason is stated nowhere: neither a
    comment line just above the import (or its run of imports) nor the
    enclosing function's docstring mentions the import."""
    text = path.read_text()
    lines = text.splitlines()
    bare = []
    for fn in ast.walk(ast.parse(text)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = (ast.get_docstring(fn) or "").lower()
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            above = node.lineno - 2
            while above >= 0 and lines[above].strip().startswith(("from ", "import ")):
                above -= 1
            comment = [ln.lower() for ln in lines[node.lineno - 1:node.end_lineno] if "#" in ln]
            while above >= 0 and lines[above].strip().startswith("#"):
                comment.append(lines[above].lower())
                above -= 1
            if not any("import" in c for c in comment) and "import" not in doc:
                bare.append(f"{path.name}:{node.lineno}")
    return bare


def test_every_inline_import_states_why():
    """An import inside a function says why it is not at the top, in a
    comment above it or in the function's docstring; never a noqa."""
    bare = []
    for path in sorted([*PROBES_DIR.glob("*.py"), *PROFILING_DIR.rglob("*.py")]):
        assert "noqa: PLC0415" not in path.read_text(), path.name
        bare += inline_imports_without_a_reason(path)
    assert bare == []


def test_moe_docstrings_name_only_real_backends():
    text = (PROBES_DIR / "moe.py").read_text()
    assert set(re.findall(r"--backend (\w+)", text)) <= {"triton", "flashinfer"}


QUETTASIM_RECIPES = ("collectives", "cross-attn", "vllm-allreduce", "reparallel-probe",
                     "reparallel-probe-val")


def readme_section(title: str) -> str:
    text = (ROOT / "README.md").read_text()
    assert f"## {title}\n" in text, title
    return text.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_readme_maps_each_old_quettasim_probe_recipe_to_a_working_command():
    """QuettaSim's justfile still calls QuettaBench/profiling/kernel_composed/...;
    each such recipe has a replacement whose recipe and flags exist here."""
    section = readme_section("QuettaSim recipes that call the old probe paths")
    recipes = {b.split()[0].rstrip(":") for b in recipe_blocks(JUSTFILE)}
    for name in QUETTASIM_RECIPES:
        assert f"`{name}`" in section, name
    lines = [ln for ln in section.replace("\\\n", " ").splitlines() if "just " in ln or "-m quettabench." in ln]
    assert lines
    for line in lines:
        for recipe in re.findall(r"\bjust (?:gpu=\S+ )?([a-z][a-z0-9-]*)", line):
            assert recipe in recipes, (recipe, line)
        for probe in re.findall(r"-m quettabench\.probes\.([a-z0-9_]+)", line):
            flags = {t for t in line.split() if t.startswith("--") and "=" not in t}
            missing = flags - argparse_flags(PROBES_DIR / f"{probe}.py") - {"--only", "--results", "--kernel-dest"}
            assert not missing, (probe, missing)
