# The engine-agnostic run vocabulary: one frozen Config per run, built by
# loader.py. Field defaults here are the "default" precedence level.
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ConfigError(ValueError):
    """A config the loader refuses; the message names the file and key."""


# Allowed values per enumerated field, checked at load.
CHOICES = {
    "engine.name": ("vllm", "quettaserve", "sglang", "trtllm"),
    "engine.attention": ("dense", "mla", "hybrid"),
    "engine.instrumentation": ("none", "nsys", "torch"),
    "workload.kind": ("live", "grid", "sessions"),
    "workload.prompt_mode": ("distinct", "clone", "corpus", "synth", "trace"),
    "workload.pacing": ("per-session", "interleaved"),
    "mode.mode": ("serve", "probe"),
    "mode.serve_profile": (None, "stress-test", "single-turn", "multi-turn"),
    "mode.load": ("closed", "open"),
    "mode.returns": ("none", "text", "logits"),
    "hardware.backend": ("auto", "cuda", "rocm", "neuron", "tpu", "cpu"),
    "hardware.link_profile": ("auto", "pcie", "nvlink"),
}

# Keys each dict-valued field accepts; None means free-form.
DICT_KEYS = {
    "engine.extra": None,
    "hardware.collective": ("NCCL_ALGO", "NCCL_PROTO", "NCCL_P2P_LEVEL", "NCCL_IB_DISABLE",
                            "NCCL_SOCKET_IFNAME", "LLMSRV_NO_NVLS", "LLMSRV_TWOSHOT_MIN_BYTES"),
    # provision.py's inputs; the last six are usually per run (`--set <run>:...`)
    "hardware.provision": ("cache_root", "py", "vllm_spec", "qs_repo", "qs_commit", "nccl_lib",
                           "qs_dir", "bench_bin", "qs_ckpt", "qs_cfg", "weights", "hf_id"),
    "provenance": ("engine_sha", "engine_commit", "checkpoint"),
    "crossval": ("comm_bound_bs", "allow_method_mismatch"),
}


@dataclass(frozen=True)
class EngineCfg:
    name: str = "vllm"
    model: str | None = None       # weights dir or HF id: serve --grid --model
    model_name: str = ""              # display name: grid META model=, table.py crate match
    tp: int = 1
    pp: int = 1
    ep: bool = False                  # workloads.json expert_parallel, serve.py ENABLE_EP
    dtype: str = "auto"
    kv_dtype: str = "auto"
    quant: str | None = None
    max_model_len: int | None = None
    gpu_util: float = 0.9
    prefix_caching: bool = False
    serving_style: str = "aggregated"
    cudagraph: bool = True            # inverse of serve --grid --nograph
    port: int = 8000                  # launch_server.sh default
    instrumentation: str = "none"     # armed at launch: none | nsys | torch
    extra: Mapping[str, Any] = field(default_factory=dict)  # quettaserve bench block lives here
    # Facts the capacity gate needs (grid CAPACITY line).
    kv_bytes: int | None = None    # per token; None allowed only for dense
    weights_gib: float | None = None
    attention: str = "dense"
    num_experts: int | None = None
    verified: bool = False


@dataclass(frozen=True)
class WorkloadCfg:
    kind: str = "live"
    source: str = ""                  # trace path, profile name, or XVAL_CORPUS/XVAL_TRACE file
    # One arrival syntax (trace_serve's): steady | trace | poisson:<rate> |
    # ramp:<start>:<end> | burst:<n>x<size>@<gap_s>
    arrival: str = "steady"
    speed: float = 1.0
    limit: int | None = None
    prompt_mode: str = "distinct"     # grid runs only
    seed: int = 42
    num_requests: int = 100
    # kind sessions only (workloads/sessions.py TurnPacing)
    pacing: str = "per-session"       # interleaved: a barrier after every turn index
    tool_wait_ms: float = 0.0
    human_wait_ms: float = 0.0
    use_recorded_waits: bool = False
    use_recorded_arrivals: bool = False
    max_turn_index: int | None = None


@dataclass(frozen=True)
class ModeCfg:
    mode: str = "serve"
    serve_profile: str | None = None
    load: str = "closed"
    concurrency: int = 10
    warmup: int = 3
    warmup_concurrency: int | None = None
    discard_first: int = 0
    ignore_eos: bool = False
    temperature: float = 0.0
    sampling_seed: int | None = None
    exact_output_length: bool = False
    timeout: int = 120
    min_success_rate: float = 0.75
    reset_prefix_cache: bool = False  # POST the engine's prefix-cache reset after warmup
    allow_missing_usage: bool = False
    trace_request_ids: bool = False   # stable request ids for engine tracing
    prefix_cache_block_size: int | None = 16  # block-aligned cache estimates (multi-turn)
    sla_ttft_ms: float | None = None
    sla_tpot_ms: float | None = None
    goodput_sweep: bool = False
    returns: str = "none"             # serve --grid --dump-tokens => text
    grid: str | None = None        # name in grids.yaml; set => step metrics
    vmin_attempts: int = 4         # grid: fit passes per cell, best r2 kept
    fit_r2_stop: float = 0.999     # grid: a pass fitting at least this r2 ends the cell
    eval_pos: int | None = None    # None => hardware.timing_steps // 2
    cap_frac: float = 0.80
    probe: str | None = None       # mode=probe: module under quettabench/probes/
    probe_args: str = ""           # mode=probe: that probe's own flags, shell-quoted


@dataclass(frozen=True)
class GridCfg:
    cells: tuple = ()                 # ((ctx, bs), ...)
    step_points: tuple = ()
    # Cells where ctx + max(step_points) + 2 > max_model_len; serve SKIPs them.
    cells_over_maxlen: tuple = ()


@dataclass(frozen=True)
class HardwareCfg:
    backend: str = "auto"
    link_profile: str = "auto"
    world: int = 8
    timing_steps: int = 100
    max_seq_headroom: int = 28
    collective: Mapping[str, str] = field(default_factory=dict)
    provision: Mapping[str, str] = field(default_factory=dict)
    visible_devices: str | None = None  # CUDA_VISIBLE_DEVICES: trace_serve power sampler


@dataclass(frozen=True)
class Config:
    engine: EngineCfg
    workload: WorkloadCfg
    mode: ModeCfg
    hardware: HardwareCfg
    grid: GridCfg | None = None
    run_name: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)  # cache.py engine_sha
    crossval: Mapping[str, Any] = field(default_factory=dict)    # table.py comm_bound_bs
    origins: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self):
        """Plain nested dict of every value, origins included."""
        out = dataclasses.asdict(self)
        out["origins"] = dict(self.origins)
        return out

    def dump(self, origins=True):
        """One `path = value` line per leaf, with its origin, for --dry-run."""
        lines = []
        for path, value in flatten(self.to_dict(), skip=("origins", "grid")):
            tail = f"  ({self.origins.get(path, 'default')})" if origins else ""
            lines.append(f"{path} = {value!r}{tail}")
        if self.grid is not None:
            lines.append(f"grid.cells = {list(self.grid.cells)!r}")
            lines.append(f"grid.step_points = {list(self.grid.step_points)!r}")
            lines.append(f"grid.cells_over_maxlen = {list(self.grid.cells_over_maxlen)!r}")
        return "\n".join(lines)


SECTIONS = {"engine": EngineCfg, "workload": WorkloadCfg, "mode": ModeCfg, "hardware": HardwareCfg}


def flatten(d, prefix="", skip=()):
    """(dotted path, value) for each leaf; dicts nest one level under their field."""
    for k, v in d.items():
        if k in skip:
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v and path.count(".") < 2:
            yield from flatten(v, path + ".")
        else:
            yield path, v


def default_values():
    """Flat {path: default} for every scalar field and every dict field."""
    out = {}
    for section, cls in SECTIONS.items():
        for f in dataclasses.fields(cls):
            dv = f.default_factory() if f.default is dataclasses.MISSING else f.default
            out[f"{section}.{f.name}"] = dv
    out["run_name"] = ""
    out["provenance"] = {}
    out["crossval"] = {}
    return out


def dict_field(path):
    """The dict-valued field a path lands in, or None for a scalar field."""
    for df in DICT_KEYS:
        if path == df or path.startswith(df + "."):
            return df
    return None
