# Load xval.yaml merged over workloads.json; fall back to hardcoded defaults
# when xval.yaml is absent. Runs as a library and as a shell-callable cli.
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_YAML_PATH = _HERE / "xval.yaml"

# Defaults mirror what was hardcoded before xval.yaml existed.
_DEFAULTS = {
    "run": {
        "world": 8,
        "timing_steps": 100,
        "max_seq_headroom": 28,
    },
    "collective": {
        # per-collective syntax required: NCCL rejects plain Tree for all-gather.
        "nccl_algo": "allreduce:tree;allgather:ring",
        "nccl_proto": "Simple",
        # SYS enables cross-socket P2P; without it tree links fall back to TCP (~4x slower).
        "nccl_p2p_level": "SYS",
        "nccl_ib_disable": 1,
        # Empty string: caller must set NCCL_SOCKET_IFNAME for their node's NIC.
        "nccl_socket_ifname": "",
        "llmsrv_no_nvls": 1,
        "llmsrv_twoshot_min_bytes": 0,
    },
    "provision": {
        # bootstrap.sh reads these; empty means supply via env or optional.
        "cache_root": "/workspace/.xval-cache",
        "py": "",
        "vllm_spec": "",
        "qs_repo": "",
        "qs_commit": "",
        "nccl_lib": "",
    },
}


def _load_yaml():
    """Return parsed xval.yaml, or {} if the file is absent."""
    if not _YAML_PATH.exists():
        return {}
    import yaml
    with open(_YAML_PATH) as fh:
        return yaml.safe_load(fh) or {}


def load():
    """Return the merged config: xval.yaml over _DEFAULTS."""
    raw = _load_yaml()

    def _merge(base, override):
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                out[k] = _merge(base[k], v)
            else:
                out[k] = v
        return out

    return _merge(_DEFAULTS, raw)


def collective(cfg=None, profile=None):
    """Collective env block as uppercase env-var names, keyed by link profile.

    "pcie" (default) exports the tuned NCCL pins from xval.yaml. "nvlink" blanks
    them (empty = do not export): on NVLink, NCCL's own selection and the engine's
    peer-allreduce beat the pins, and vLLM's TP path ignores NCCL_ALGO anyway.
    Profile resolves from the argument, then $XVAL_LINK_PROFILE, then "pcie".
    """
    if profile is None:
        profile = os.environ.get("XVAL_LINK_PROFILE", "pcie")
    if profile not in ("pcie", "nvlink"):
        raise ValueError(f"unknown link profile {profile!r}; use pcie or nvlink")
    if cfg is None:
        cfg = load()
    c = cfg.get("collective", _DEFAULTS["collective"])
    out = {
        "NCCL_ALGO": str(c.get("nccl_algo", _DEFAULTS["collective"]["nccl_algo"])),
        "NCCL_PROTO": str(c.get("nccl_proto", _DEFAULTS["collective"]["nccl_proto"])),
        "NCCL_P2P_LEVEL": str(c.get("nccl_p2p_level", _DEFAULTS["collective"]["nccl_p2p_level"])),
        "NCCL_IB_DISABLE": str(c.get("nccl_ib_disable", _DEFAULTS["collective"]["nccl_ib_disable"])),
        "NCCL_SOCKET_IFNAME": str(c.get("nccl_socket_ifname", _DEFAULTS["collective"]["nccl_socket_ifname"])),
        "LLMSRV_NO_NVLS": str(c.get("llmsrv_no_nvls", _DEFAULTS["collective"]["llmsrv_no_nvls"])),
        "LLMSRV_TWOSHOT_MIN_BYTES": str(c.get("llmsrv_twoshot_min_bytes", _DEFAULTS["collective"]["llmsrv_twoshot_min_bytes"])),
    }
    if profile == "nvlink":
        ib = out["NCCL_IB_DISABLE"]
        out = {k: "" for k in out}
        out["NCCL_IB_DISABLE"] = ib
        for k, v in cfg.get("collective_nvlink", {}).items():
            out[k.upper()] = str(v)
    return out


def run_params(cfg=None):
    """Return run-section values as a dict."""
    if cfg is None:
        cfg = load()
    r = cfg.get("run", _DEFAULTS["run"])
    return {
        "world": int(r.get("world", _DEFAULTS["run"]["world"])),
        "timing_steps": int(r.get("timing_steps", _DEFAULTS["run"]["timing_steps"])),
        "max_seq_headroom": int(r.get("max_seq_headroom", _DEFAULTS["run"]["max_seq_headroom"])),
    }


_WL_JSON = _HERE / "workloads.json"


def _load_json():
    """Return parsed workloads.json."""
    return json.loads(_WL_JSON.read_text())


def workloads(cfg=None):
    """Return workloads dict: workloads.json deep-merged with xval.yaml overrides.

    workloads.json is the complete base. xval.yaml adds or overrides per-crate
    keys. JSON-only crates are never dropped.
    """
    if cfg is None:
        cfg = load()
    base = _load_json()["workloads"]
    yaml_wls = cfg.get("workloads", {})
    if not yaml_wls:
        return base
    merged = dict(base)
    for crate, overrides in yaml_wls.items():
        if crate in merged:
            merged[crate] = {**merged[crate], **overrides}
        else:
            merged[crate] = overrides
    return merged


def grids(cfg=None):
    """Return grids dict: workloads.json base merged with xval.yaml overrides.

    JSON-only grids are never dropped. xval.yaml may override a grid by name
    or add new named grids.
    """
    if cfg is None:
        cfg = load()
    base = _load_json()["grids"]
    yaml_grids = cfg.get("grids", {})
    if not yaml_grids:
        return base
    return {**base, **yaml_grids}


def step_points(cfg=None):
    """Return step_points list: xval.yaml value when present, else workloads.json."""
    if cfg is None:
        cfg = load()
    if "step_points" in cfg:
        return list(cfg["step_points"])
    return _load_json()["step_points"]


def provision(cfg=None):
    """Return the provision block (bootstrap.sh inputs) merged over defaults."""
    if cfg is None:
        cfg = load()
    p = dict(_DEFAULTS["provision"])
    p.update(cfg.get("provision", {}))
    return p


def backend():
    """Detect the accelerator backend: XVAL_BACKEND override, else from the tooling
    present (nvidia-smi -> cuda, rocm-smi -> rocm, neuron-ls -> neuron, libtpu ->
    tpu, else cpu). Only device picking and the NCCL pins are cuda-specific; the
    rest of the harness is backend-agnostic."""
    b = os.environ.get("XVAL_BACKEND")
    if b:
        return b
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocm-smi"):
        return "rocm"
    if shutil.which("neuron-ls"):
        return "neuron"
    if os.environ.get("TPU_NAME") or os.path.exists("/lib/libtpu.so"):
        return "tpu"
    return "cpu"


def link_profile():
    """Interconnect profile for the collective env: XVAL_LINK_PROFILE override,
    else nvlink when nvidia-smi topo shows an NV* link, else pcie. Non-cuda
    backends take nvlink, the no-pins profile."""
    p = os.environ.get("XVAL_LINK_PROFILE")
    if p:
        return p
    if backend() != "cuda":
        return "nvlink"
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"],
                              capture_output=True, text=True).stdout
    except OSError:
        topo = ""
    return "nvlink" if re.search(r"NV\d", topo) else "pcie"


def _cli(args):
    """Shell entry: collective | run-params | provision [key] | tp <wl> | cells <wl> | bench <wl>."""
    cmd = args[0] if args else ""
    if cmd == "collective":
        for k, v in collective().items():
            print(f"{k}={v}")
        return
    if cmd == "run-params":
        p = run_params()
        print(p["timing_steps"], p["max_seq_headroom"])
        return
    if cmd == "provision":
        p = provision()
        if len(args) == 2:
            print(p.get(args[1], ""))
        else:
            for k, v in p.items():
                print(f"{k}={v}")
        return
    if cmd == "backend":
        print(backend())
        return
    if cmd == "link-profile":
        print(link_profile())
        return
    if cmd not in ("tp", "devices", "cells", "bench") or len(args) != 2:
        sys.exit("usage: xval_config.py collective|run-params|provision|backend|link-profile|tp|devices|cells|bench [workload]")
    wl = workloads().get(args[1])
    if wl is None:
        sys.exit(f"unknown workload {args[1]}")
    if cmd == "tp":
        print(wl.get("tp", 1))
    elif cmd == "devices":
        pp = int(os.environ.get("XVAL_PP", wl.get("pp", 1)))
        print(int(wl.get("tp", 1)) * pp)
    elif cmd == "cells":
        for ctx, bs in grids()[wl["grid"]]:
            print(ctx, bs)
    else:
        b = wl.get("bench")
        if not b:
            sys.exit(f"workload {args[1]} has no bench config")
        for key in ("crate", "test", "env", "family", "ok"):
            print(b[key])
        print(wl.get("tp", 1))


if __name__ == "__main__":
    _cli(sys.argv[1:])
