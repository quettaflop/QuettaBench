"""Resolve the QuettaSim checkout (device YAMLs + kernel_data)."""
from __future__ import annotations

import os
from pathlib import Path


def quettasim_root() -> Path:
    env = os.environ.get("QUETTASIM")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "device_spec").is_dir() and (p / "engine").is_dir():
            return p
    raise SystemExit(
        "set QUETTASIM to the QuettaSim checkout (device_spec not found)"
    )


def kernel_data_root() -> Path:
    env = os.environ.get("KDATA") or os.environ.get("KERNEL_DATA")
    if env:
        return Path(env)
    return quettasim_root() / "data" / "kernel_data"
