# yaml + env + --set -> Config. Precedence, highest first: --set (cli), env
# (via env_map), run yaml, model yaml, hardware yaml, schema defaults.
import itertools
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from quettabench.config import env_map
from quettabench.config.schema import (
    CHOICES,
    DICT_KEYS,
    SECTIONS,
    Config,
    ConfigError,
    GridCfg,
    default_values,
    dict_field,
)
from quettabench.config.validate import check_ep_legal, parse_serving_style

# model yaml key -> Config path
MODEL_KEYS = {
    "name": "engine.model_name", "dtype": "engine.dtype", "tp": "engine.tp",
    "weights_gib": "engine.weights_gib", "max_model_len": "engine.max_model_len",
    "kv_dtype": "engine.kv_dtype", "kv_bytes": "engine.kv_bytes", "quant": "engine.quant",
    "ep": "engine.ep", "attention": "engine.attention", "num_experts": "engine.num_experts",
    "verified": "engine.verified", "quettaserve": "engine.extra.quettaserve",
    "provenance": "provenance", "crossval": "crossval",
}
QUETTASERVE_KEYS = ("crate", "test", "env_prefix", "family")
RUN_KEYS = ("model", "engine", "workload", "mode", "hardware", "matrix")
HARDWARE_FILE_KEYS = ("world", "timing_steps", "max_seq_headroom", "collective", "provision")
GRIDS_KEYS = ("grids", "step_points")
RATE_ARRIVALS = ("poisson", "ramp", "burst")
# path -> (lowest, highest) allowed; None is no bound
RANGES = {"mode.concurrency": (1, None), "workload.num_requests": (1, None),
          "mode.warmup": (0, None), "mode.discard_first": (0, None),
          "mode.min_success_rate": (0.0, 1.0), "workload.tool_wait_ms": (0.0, None),
          "workload.human_wait_ms": (0.0, None)}


def _read_yaml(path):
    """Parsed mapping from a yaml file; a missing file or a non-mapping is an error."""
    if not path.exists():
        raise ConfigError(f"{path}: no such file")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


def _check_keys(raw, allowed, where):
    for key in raw:
        if key not in allowed:
            raise ConfigError(f"{where}: unknown key {key!r}")


def _check_path(path, where):
    """Raise unless path names a schema field or a key of a dict field."""
    if path in default_values():
        return
    df = dict_field(path)
    if df is not None:
        key = path[len(df) + 1:]
        allowed = DICT_KEYS[df]
        if "." not in key and (allowed is None or key in allowed):
            return
    raise ConfigError(f"{where}: unknown key {path!r}")


def _flatten(raw, prefix, where):
    """{dotted path: value} for one yaml section, every path checked."""
    out = {}
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if dict_field(path) == path and isinstance(value, dict):
            for k, v in value.items():
                _check_path(f"{path}.{k}", where)
                out[f"{path}.{k}"] = v
        else:
            _check_path(path, where)
            out[path] = value
    return out


def _model_layer(path):
    raw = _read_yaml(path)
    _check_keys(raw, MODEL_KEYS, path)
    _check_keys(raw.get("quettaserve") or {}, QUETTASERVE_KEYS, f"{path} quettaserve")
    out = {}
    for key, value in raw.items():
        target = MODEL_KEYS[key]
        if target in DICT_KEYS:
            out.update(_flatten({target: value}, "", path))
        else:
            out[target] = value
    return out


def _hardware_layer(configs, profile):
    path = configs / "hardware" / f"{profile}.yaml"
    raw = _read_yaml(path)
    _check_keys(raw, HARDWARE_FILE_KEYS, path)
    return _flatten(raw, "hardware.", path)


def _read_grids(configs):
    path = configs / "grids.yaml"
    raw = _read_yaml(path)
    _check_keys(raw, GRIDS_KEYS, path)
    step_points = raw.get("step_points") or []
    if not step_points:
        raise ConfigError(f"{path}: step_points must be a non-empty list")
    return {k.lower(): v for k, v in raw.get("grids", {}).items()}, step_points


def parse_set(item):
    """'a.b=c' -> ('a.b', c) with c parsed as a yaml scalar (int, float, bool, null)."""
    path, sep, text = item.partition("=")
    if not sep or not path:
        raise ConfigError(f"--set {item!r}: expected key=value")
    return path.strip(), yaml.safe_load(text)


def parse_arrival(spec):
    """(kind, params) from steady | trace | poisson:<r> | ramp:<a>:<b> | burst:<n>x<size>@<gap>."""
    kind, _, rest = str(spec).partition(":")
    try:
        if kind in ("steady", "trace") and not rest:
            return kind, ()
        if kind == "poisson":
            params = (float(rest),)
        elif kind == "ramp":
            a, b = rest.split(":")
            params = (float(a), float(b))
        elif kind == "burst":
            n, tail = rest.split("x")
            size, gap = tail.split("@")
            params = (int(n), int(size), float(gap))
        else:
            raise ValueError
    except ValueError:
        raise ConfigError(f"arrival {spec!r}: use steady, trace, poisson:<r>, "
                          "ramp:<a>:<b> or burst:<n>x<size>@<gap>") from None
    if kind in ("poisson", "ramp") and min(params) <= 0:
        raise ConfigError(f"arrival {spec!r}: rate must be > 0")
    return kind, params


def detect_backend(tpu_name):
    """The accelerator from the tooling on PATH: nvidia-smi cuda, rocm-smi rocm,
    neuron-ls neuron, libtpu (or TPU_NAME) tpu, else cpu."""
    for tool, backend in (("nvidia-smi", "cuda"), ("rocm-smi", "rocm"), ("neuron-ls", "neuron")):
        if shutil.which(tool):
            return backend
    if tpu_name or Path("/lib/libtpu.so").exists():
        return "tpu"
    return "cpu"


def detect_link_profile(backend):
    """nvlink when nvidia-smi topo shows an NV* link, else pcie; non-cuda
    backends take nvlink, the profile without NCCL pins."""
    if backend != "cuda":
        return "nvlink"
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
                              check=False).stdout
    except OSError:
        topo = ""
    return "nvlink" if re.search(r"NV\d", topo) else "pcie"


def _apply(values, origins, layer, origin):
    for path, value in layer.items():
        df = dict_field(path)
        if df is not None and path != df:
            values[df] = {**values[df], path[len(df) + 1:]: value}
        else:
            values[path] = value
        origins[path] = origin


def _force(values, origins, path, value, by, where):
    """Set a value a serve profile requires; an explicit contrary setting is refused."""
    if values[path] != value and origins.get(path, "default") != "default":
        raise ConfigError(f"{where}: {path}={values[path]!r} contradicts "
                          f"mode.serve_profile={values['mode.serve_profile']}")
    values[path] = value
    origins[path] = by


def _one_of(allowed):
    """'a, b or c'."""
    words = [str(a) for a in allowed]
    return ", ".join(words[:-1]) + " or " + words[-1]


def _validate(values, origins, allow_unverified, where):
    for path, allowed in CHOICES.items():
        if values[path] not in allowed:
            raise ConfigError(f"{where}: unknown {path.rsplit('.', 1)[-1]} {values[path]!r}; "
                              f"use {_one_of(allowed)}")
    for path, (low, high) in RANGES.items():
        if isinstance(values[path], bool) or not isinstance(values[path], (int, float)):
            raise ConfigError(f"{where}: {path} must be a number, got {values[path]!r}")
        if (low is not None and values[path] < low) or (high is not None and values[path] > high):
            raise ConfigError(f"{where}: {path} must be in [{low}, {high or 'inf'}], "
                              f"got {values[path]!r}")
    # validate.py raises ValueError; as ConfigError the CLI prints one line, message unchanged.
    try:
        parse_serving_style(values["engine.serving_style"])
        # num_experts None (no MoE fact recorded) skips the check.
        if values["engine.ep"] and values["engine.num_experts"] is not None:
            check_ep_legal(values["engine.tp"], values["engine.num_experts"])
    except ValueError as e:
        raise ConfigError(str(e)) from None
    if not values["engine.verified"] and not allow_unverified:
        raise ConfigError(f"{where}: {values['run_name']} is not validated yet; "
                          "pass --allow-unverified to bring it up")
    if values["mode.mode"] == "probe" and not values["mode.probe"]:
        raise ConfigError(f"{where}: mode=probe needs mode.probe, a quettabench.probes module")
    if values["engine.attention"] in ("mla", "hybrid") and values["engine.kv_bytes"] is None:
        raise ConfigError(f"{where}: attention={values['engine.attention']} needs kv_bytes; "
                          "the dense per-head formula overestimates its KV")
    profile = values["mode.serve_profile"]
    by = origins.get("mode.serve_profile", "default")
    if profile == "stress-test":
        _force(values, origins, "mode.ignore_eos", True, by, where)
        _force(values, origins, "engine.prefix_caching", False, by, where)
    elif profile in ("single-turn", "multi-turn"):
        _force(values, origins, "engine.prefix_caching", True, by, where)
    arrival, _ = parse_arrival(values["workload.arrival"])
    if values["mode.load"] == "open" and arrival not in RATE_ARRIVALS + ("trace",):
        raise ConfigError(f"{where}: open load needs a rate-bearing arrival "
                          f"(poisson, ramp, burst) or trace, got {values['workload.arrival']!r}")
    if values["mode.load"] == "closed" and arrival in RATE_ARRIVALS:
        raise ConfigError(f"{where}: arrival {values['workload.arrival']!r} is open-loop "
                          "offered load; set mode.load=open or arrival=steady")
    if values["mode.goodput_sweep"] and (values["mode.load"] != "open" or not (
            values["mode.sla_ttft_ms"] or values["mode.sla_tpot_ms"])):
        raise ConfigError(f"{where}: goodput sweep needs open load and mode.sla_ttft_ms "
                          "and/or sla_tpot_ms")
    if (values["mode.grid"] and values["workload.prompt_mode"] == "corpus"
            and values["engine.driver"] == "http"):
        raise ConfigError(f"{where}: prompt_mode=corpus tokenizes in-process; "
                          "engine.driver=http has no tokenizer")


def _grid(values, grids, step_points, where):
    name = values["mode.grid"]
    if name is None:
        return None
    if name.lower() not in grids:
        raise ConfigError(f"{where}: unknown grid {name!r}; grids: {', '.join(grids)}")
    cells = tuple(tuple(c) for c in grids[name.lower()])
    maxlen = values["engine.max_model_len"]
    over = tuple(c for c in cells
                 if maxlen is not None and c[0] + max(step_points) + 2 > maxlen)
    return GridCfg(cells=cells, step_points=tuple(step_points), cells_over_maxlen=over)


def _build(values, origins, grid):
    parts = {s: {} for s in SECTIONS}
    for path, value in values.items():
        section, _, name = path.partition(".")
        if section in SECTIONS:
            parts[section][name] = value
    for path, value in values.items():
        if not (isinstance(value, dict) and value):  # a filled dict has per-key origins
            origins.setdefault(path, "default")
    return Config(
        **{s: cls(**parts[s]) for s, cls in SECTIONS.items()},
        grid=grid, run_name=values["run_name"], provenance=values["provenance"],
        crossval=values["crossval"], origins=dict(sorted(origins.items())),
    )


def _resolve(upper, configs, grids, step_points, detect, allow_unverified, where):
    """One Config from the layers above the hardware yaml."""
    values, origins = default_values(), {}

    def pick(path):
        found = [layer[path] for _, layer in upper if path in layer]
        return found[-1] if found else values[path]

    backend = pick("hardware.backend")
    if backend == "auto":
        backend = detect_backend(detect.get("detect.tpu_name"))
    profile = pick("hardware.link_profile")
    if profile == "auto":
        profile = detect_link_profile(backend)
    if profile not in ("pcie", "nvlink"):
        raise ConfigError(f"{where}: unknown link profile {profile!r}; use pcie or nvlink")
    for origin, layer in [("hardware", _hardware_layer(configs, profile))] + upper:
        _apply(values, origins, layer, origin)
    values["hardware.backend"], values["hardware.link_profile"] = backend, profile
    _validate(values, origins, allow_unverified, where)
    return _build(values, origins, _grid(values, grids, step_points, where))


def load(run_yaml, overrides=(), env=None):
    """Configs for one run yaml: one, or one per point of its `matrix:`.

    env=None reads the process environment through env_map.read_env.
    The configs/ root is the run yaml's grandparent (configs/runs/<name>.yaml).
    """
    run_path = Path(run_yaml)
    configs = run_path.resolve().parents[1]
    raw = _read_yaml(run_path)
    _check_keys(raw, RUN_KEYS, run_path)
    model = _model_layer(configs / "models" / f"{raw['model']}.yaml") if raw.get("model") else {}
    run = {}
    for section in SECTIONS:
        run.update(_flatten(raw.get(section) or {}, f"{section}.", run_path))
    matrix = raw.get("matrix") or {}
    for path, choices in matrix.items():
        _check_path(path, f"{run_path} matrix")
        if not isinstance(choices, list) or not choices:
            raise ConfigError(f"{run_path} matrix: {path!r} needs a non-empty list")

    from_env = env_map.read_env(env)
    detect = {k: from_env.pop(k) for k in list(from_env) if k.startswith("detect.")}
    cli = dict(parse_set(item) for item in overrides)
    allow_unverified = bool(cli.pop("allow_unverified", False))
    for path in cli:
        _check_path(path, "--set")
    grids, step_points = _read_grids(configs)

    out = []
    for combo in itertools.product(*matrix.values()):
        point = dict(zip(matrix, combo))
        run_layer, cli_layer = {**run, **point}, dict(cli)
        named_by = cli_layer if "run_name" in cli else run_layer
        named_by["run_name"] = "-".join([str(cli.get("run_name", run_path.stem))] + [
            f"{p.rsplit('.', 1)[-1]}{v}" for p, v in point.items()])
        name = named_by["run_name"]
        upper = [("model", model), ("run", run_layer), ("env", from_env), ("cli", cli_layer)]
        out.append(_resolve(upper, configs, grids, step_points, detect, allow_unverified,
                            f"{run_path} [{name}]"))
    return out
