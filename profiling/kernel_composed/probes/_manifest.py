"""Sidecar manifest next to every table a probe writes: <table>.meta.json.

The QuettaSim loader (kernel_composed/provenance.py) reads it to report what a
prediction priced from; tools/kernel_manifest.py checks every table has one.
Schema (keys the loader relies on: mode, dir, tool):

    mode     graph_replay | ncu | eager | amortized_wallclock | host_wallclock
    dir      graph | eager   (must agree with mode; see provenance.MODES)
    tool     probe file + the flags that select the method
    reduce   min | median | ...   (what one cell is, over reps)
    reps / warmup, gpu, device (torch's name), host, measured (date),
    software {torch, cuda, vllm, driver}, argv, repo_commit, notes

Upserting probes (flash, cross_attn, collectives, reparallel, kv_transfer) keep a
``runs`` history: one entry per run, newest last, so one file can honestly hold
geometries measured on different days.
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import subprocess
import sys
from pathlib import Path

SCHEMA = 1
_KEEP_RUNS = 20


def _software() -> dict:
    sw: dict = {}
    try:
        import torch  # noqa: PLC0415
        sw["torch"] = torch.__version__
        sw["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            sw["device"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - probe hosts vary
        pass
    try:
        import vllm  # noqa: PLC0415
        sw["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if out:
            sw["driver"] = out[0].strip()
    except Exception:
        pass
    return sw


def _repo_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def write_manifest(table: Path, *, mode: str, tool: str, reduce: str, gpu_label: str,
                   reps: int | None = None, warmup: int | None = None,
                   notes: str = "", upsert: bool = False, extra: dict | None = None) -> Path:
    """Write (or, for ``upsert`` probes, extend) the manifest for ``table``."""
    table = Path(table)
    mp = table.with_name(table.name + ".meta.json")
    parts = table.resolve().parts
    d = next((p for p in reversed(parts[:-1]) if p in ("graph", "eager")), None)
    run = {
        "measured": dt.date.today().isoformat(),
        "host": socket.gethostname(),
        "argv": sys.argv[1:],
        "software": _software(),
        "repo_commit": _repo_commit(),
    }
    doc = {
        "schema": SCHEMA, "table": "/".join(parts[parts.index(d):]) if d else table.name,
        "dir": d, "mode": mode, "tool": tool, "reduce": reduce, "reps": reps, "warmup": warmup,
        "gpu": gpu_label, "notes": notes, "provenance": "probe",
        **(extra or {}), **run,
    }
    if upsert and mp.exists():
        try:
            prev = json.loads(mp.read_text())
        except ValueError:
            prev = {}
        runs = list(prev.get("runs") or [])
        if prev.get("measured"):
            runs.append({k: prev.get(k) for k in ("measured", "host", "argv", "software", "repo_commit", "tool")})
        doc["runs"] = runs[-_KEEP_RUNS:]
        doc["first_measured"] = prev.get("first_measured") or prev.get("measured") or doc["measured"]
    mp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return mp
