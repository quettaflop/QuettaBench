# Frozen run vocabulary; field defaults are the "default" precedence level.
import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Origins that state what the checkpoint or box is, not what this run asks for.
NOT_ASKED = ("default", "model", "hardware")


class ConfigError(ValueError):
    """The message names the file and key."""


PROMPT_MODES = ("distinct", "clone", "corpus", "trace")

# Allowed values per enumerated field, checked at load.
CHOICES = {
    # One per registered adapter (engines/): an engine without one is refused at load.
    "engine.name": ("vllm", "quettaserve", "reference"),
    "engine.attention": ("dense", "mla", "hybrid"),
    "engine.instrumentation": ("none", "nsys", "torch"),
    "workload.kind": ("live", "grid", "sessions"),
    "workload.prompt_mode": PROMPT_MODES,
    "workload.pacing": ("per-session", "interleaved"),
    "workload.salt": ("text", "cache_salt"),
    "mode.mode": ("serve", "probe"),
    "mode.serve_profile": (None, "stress-test", "single-turn", "multi-turn"),
    "mode.load": ("closed", "open"),
    "mode.return_outputs": ("none", "text", "logits"),
    "hardware.backend": ("auto", "cuda", "rocm", "neuron", "tpu", "cpu"),
    "hardware.link_profile": ("auto", "pcie", "nvlink"),
    "hardware.launcher": ("local", "docker", "attach"),
    "hardware.server_env": ("controlled", "inherit"),
}

# Accelerators qbench launches vLLM on besides cuda; cpu is the stub backend the tests run.
OTHER_ACCELERATORS = ("rocm", "neuron", "tpu")

# Keys each dict-valued field accepts; None means free-form.
DICT_KEYS = {
    "engine.extra": None,
    "engine.env": None,
    "hardware.collective": (
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "LLMSRV_NO_NVLS",
        "LLMSRV_TWOSHOT_MIN_BYTES",
    ),
    # provision.py's inputs; the last six are usually per run (`--set <run>:...`)
    "hardware.provision": (
        "cache_root",
        "py",
        "vllm_spec",
        "quettaserve_repo",
        "quettaserve_commit",
        "nccl_lib",
        "quettaserve_dir",
        "bench_bin",
        "quettaserve_checkpoint",
        "quettaserve_config",
        "weights",
        "hf_id",
    ),
    "provenance": ("engine_sha", "engine_commit", "checkpoint"),
    "compare": ("comm_bound_bs", "allow_method_mismatch"),
}


@dataclass(frozen=True)
class EngineCfg:
    name: str = "vllm"
    model: str | None = None  # weights dir or HF id
    model_name: str = ""  # display name: grid META model=, compare comm_bound_bs match
    tp: int = 1
    pp: int = 1
    ep: bool = False  # model yaml ep, ENABLE_EP
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quant: str | None = None
    max_model_len: int | None = None
    layers: int | None = None  # first N decoder layers only: a layer-truncated run (data/truncate_checkpoint.py)
    gpu_util: float = 0.9
    prefix_caching: bool = False
    serving_style: str = "aggregated"
    cudagraph: bool = True  # false: --enforce-eager
    port: int = 8000
    instrumentation: str = "none"  # armed at launch: none | nsys | torch
    image: str | None = None  # the server's docker image; required by hardware.launcher=docker
    extra: Mapping[str, Any] = field(default_factory=dict)  # quettaserve bench block lives here
    # The server's environment on top of the shell's: what a verify yaml used to ask for by hand (export ...).
    env: Mapping[str, Any] = field(default_factory=dict)
    # Facts the capacity gate needs (grid CAPACITY line).
    kv_bytes_per_token: int | None = None  # per token; None allowed only for dense
    weights_gib: float | None = None
    attention: str = "dense"
    num_experts: int | None = None
    verified: bool = False


@dataclass(frozen=True)
class WorkloadCfg:
    kind: str = "live"
    source: str = ""  # trace path, profile name, or QBENCH_CORPUS/QBENCH_TRACE file
    # steady | trace | poisson:<rate> | ramp:<start>:<end> | burst:<n>x<size>@<gap_s>
    arrival: str = "steady"
    speed: float = 1.0
    prompt_mode: str = "distinct"  # grid runs only
    seed: int = 42
    num_requests: int = 100
    # kind sessions only (workloads/sessions.py TurnPacing)
    pacing: str = "per-session"  # interleaved: a barrier after every turn index
    tool_wait_ms: float = 0.0
    human_wait_ms: float = 0.0
    use_recorded_waits: bool = False
    use_recorded_arrivals: bool = False
    max_turn_index: int | None = None
    num_sessions: int | None = None  # None: the profile's count; at least concurrency
    # Order the sample round-robin across strata of turn count and prompt size (and, with num_sessions, take each
    # stratum's share), so the sessions open at any moment mix like the whole sample.
    stratified: bool = False
    # How a recycled play misses earlier plays' cache: text puts a tag before the first user message (AgentX);
    # cache_salt sends vLLM's cache_salt field and leaves the prompt as recorded.
    salt: str = "text"


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
    # Recycled replay: the agents draw the sample in a cycle for this long, each play salted; 0 plays each once.
    duration_s: float = 0.0
    timeout: int = 120
    min_success_rate: float = 0.75
    reset_prefix_cache: bool = False  # POST the engine's prefix-cache reset after warmup
    allow_missing_usage: bool = False
    # Ask vLLM for cumulative usage on every chunk (continuous_usage_stats), so window tokens use the server's
    # per-chunk counts; off, request bodies match vllm bench serve's and AIPerf's.
    chunk_usage: bool = False
    trace_request_ids: bool = False  # stable request ids for engine tracing
    prefix_cache_block_size: int | None = 16  # block-aligned cache estimates (multi-turn)
    sla_ttft_ms: float | None = None
    sla_tpot_ms: float | None = None
    goodput_sweep: bool = False
    return_outputs: str = "none"  # none | text | logits
    grid: str | None = None  # name in grids.yaml; set => step metrics
    fit_attempts: int = 4  # grid: fit passes per cell, best r2 kept
    fit_r2_stop: float = 0.999  # grid: a pass fitting at least this r2 ends the cell
    eval_decode_pos: int | None = None  # None => hardware.timing_steps // 2
    max_kv_fraction: float = 0.80
    probe: str | None = None  # mode=probe: module under quettabench/probes/
    probe_args: str = ""  # mode=probe: that probe's own flags, shell-quoted


@dataclass(frozen=True)
class GridCfg:
    cells: tuple = ()  # ((ctx, bs), ...)
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
    visible_devices: str | None = None  # CUDA_VISIBLE_DEVICES; rocm/tpu/neuron runs must set it
    ready_timeout_s: int = 1800  # seconds the launcher waits for the server health check
    # local: a host process | docker: engine.image in a container | attach: measure the server already on
    # engine.port; qbench starts and stops nothing, so the same server can serve another harness too
    launcher: str = "local"
    # controlled: a local server starts from an empty environment (env -i) plus an allowlist from the shell
    # (config/env_map.py) and the launch spec's env; inherit: it inherits the whole shell
    server_env: str = "controlled"
    shm_size: str = "16g"  # docker --shm-size
    hf_home: str | None = None  # HF cache; docker mounts it at the same path


@dataclass(frozen=True)
class Config:
    engine: EngineCfg
    workload: WorkloadCfg
    mode: ModeCfg
    hardware: HardwareCfg
    grid: GridCfg | None = None
    run_name: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)  # engine_sha
    compare: Mapping[str, Any] = field(default_factory=dict)  # compare comm_bound_bs
    origins: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self):
        out = dataclasses.asdict(self)
        out["origins"] = dict(self.origins)
        return out

    def dump(self, origins=True, scrub=None):
        lines = []
        d = self.to_dict()
        for path, value in flatten(scrub(d) if scrub else d, skip=("origins", "grid")):
            tail = f"  ({self.origins.get(path, 'default')})" if origins else ""
            lines.append(f"{path} = {value!r}{tail}")
        if self.grid is not None:
            lines.append(f"grid.cells = {list(self.grid.cells)!r}")
            lines.append(f"grid.step_points = {list(self.grid.step_points)!r}")
            lines.append(f"grid.cells_over_maxlen = {list(self.grid.cells_over_maxlen)!r}")
        return "\n".join(lines)


SECTIONS = {"engine": EngineCfg, "workload": WorkloadCfg, "mode": ModeCfg, "hardware": HardwareCfg}


def flatten(d, prefix="", skip=()):
    for k, v in d.items():
        if k in skip:
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v and path.count(".") < 2:
            yield from flatten(v, path + ".")
        else:
            yield path, v


def default_values():
    out = {}
    for section, cls in SECTIONS.items():
        for f in dataclasses.fields(cls):
            # The stubs type default_factory as a union with MISSING.
            dv = f.default_factory() if f.default is dataclasses.MISSING else f.default  # type: ignore[misc]
            out[f"{section}.{f.name}"] = dv
    out["run_name"] = ""
    out["provenance"] = {}
    out["compare"] = {}
    return out


def dict_field(path):
    for df in DICT_KEYS:
        if path == df or path.startswith(df + "."):
            return df
    return None


# Serving-topology and expert-parallel rules: the loader refuses a Config that breaks them.
def parse_serving_style(spec):
    """ValueError if malformed."""
    if spec in (None, "", "aggregated"):
        return {"mode": "aggregated"}
    m = re.fullmatch(r"disagg:(\d+)p(\d+)d", spec)
    if m:
        p, d = int(m.group(1)), int(m.group(2))
        if p < 1 or d < 1:
            raise ValueError(f"disagg needs >=1 prefill and >=1 decode: {spec!r}")
        return {"mode": "disagg", "prefill": p, "decode": d}
    m = re.fullmatch(r"afd:(\d+)a(\d+)f", spec)
    if m:
        a, f = int(m.group(1)), int(m.group(2))
        if a < 1 or f < 1:
            raise ValueError(f"afd needs >=1 attention and >=1 FFN GPU: {spec!r}")
        return {"mode": "afd", "attn": a, "ffn": f}
    m = re.fullmatch(r"ep(\d+)dp(\d+)", spec)
    if m:
        ep, dp = int(m.group(1)), int(m.group(2))
        if ep < 1 or dp < 1:
            raise ValueError(f"ep_dp needs >=1 expert-parallel and >=1 data-parallel: {spec!r}")
        return {"mode": "ep_dp", "ep": ep, "dp": dp}
    raise ValueError(f"unknown serving_style {spec!r}; use aggregated, disagg:<P>p<D>d, afd:<A>a<F>f, or ep<N>dp<M>")


def device_count(engine):
    style = parse_serving_style(engine.serving_style)
    if style["mode"] == "afd":
        return style["attn"] + style["ffn"]
    if style["mode"] == "disagg":
        return style["prefill"] + style["decode"]
    return engine.tp * engine.pp


def check_ep_legal(ep, num_experts):
    """Fail here rather than deep in engine load."""
    if num_experts % ep != 0:
        raise ValueError(f"ep={ep} does not divide {num_experts} experts")


def serving_style_compatible(a, b):
    return parse_serving_style(a)["mode"] == parse_serving_style(b)["mode"]
