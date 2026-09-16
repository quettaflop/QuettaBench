# xval_config.py -- load xval.yaml merged over workloads.json; fall back to
# hardcoded defaults when xval.yaml is absent.
# Used by: ds_bench.sh (via python3 -c inline), vmin_fit.py, table.py.
# PyYAML is available in the crossval environment (yaml 6.0.x).
import json
import os
from pathlib import Path

_HERE = Path(__file__).parent
_YAML_PATH = _HERE / "xval.yaml"

# Defaults mirror what was hardcoded before xval.yaml existed.
_DEFAULTS = {
    "run": {
        "world": 8,
        "timing_steps": 100,
        "max_seq_headroom": 28,
        "clone_slots": 1,
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
    "runtime": {
        # Blackwell sm_120 requires NCCL >= 2.28; 2.25.1 fails with "invalid argument".
        "nccl_min_version": "2.28",
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
    """Return the collective env block as a dict of uppercase env-var names.

    profile "pcie" (the default) applies the tuned pins from xval.yaml.
    profile "nvlink" returns them as empty strings (empty = do not export):
    on NVLink boxes NCCL's own NVLS/LL128 selection and the engine's
    peer-allreduce beat the PCIe pins, and vLLM's TP path bypasses NCCL_ALGO
    entirely, so exporting the pins there slows only the engine. A
    collective_nvlink block in xval.yaml overrides individual values.
    Resolution order: argument, then $XVAL_LINK_PROFILE, then "pcie".
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
        "clone_slots": int(r.get("clone_slots", _DEFAULTS["run"]["clone_slots"])),
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
