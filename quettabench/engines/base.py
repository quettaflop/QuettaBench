"""The translation boundary: an adapter turns one Config into one LaunchSpec, the only description of how an engine is started and driven."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from quettabench.config.schema import NOT_ASKED, EngineCfg

# {out} is the report path without extension
NSYS = ("nsys", "profile", "-t", "cuda,nvtx", "-o", "{out}", "--force-overwrite=true")


def nsys_argv(cfg, run_dir, name="nsys"):
    """NSYS writing <run dir>/<name>; the run dir is made absolute, since the engine may run in another cwd. run_dir None is results/<run_name>."""
    out = Path(run_dir if run_dir is not None else Path("results", cfg.run_name)).resolve() / name
    return [a.format(out=out) for a in NSYS]


class UnsupportedField(ValueError):
    """A Config value the engine has no flag for in any launch run today."""

    def __init__(self, engine, field_name, why=""):
        self.engine, self.field = engine, field_name
        super().__init__(f"{engine}: {field_name} has no launch flag in use today"
                         + (f" ({why})" if why else ""))


@dataclass(frozen=True)
class LaunchSpec:
    """Everything needed to start and drive one engine, as plain data.

    server_argv is empty when the adapter has no server, or when a field has no flag (named in server_unsupported; the launcher refuses such a spec, the in-process path does not need argv). env is added to the child's inherited environment; env_drop names inherited variables to remove.
    """
    adapter: str
    server_argv: tuple = ()
    engine_kwargs: Mapping = field(default_factory=dict)
    bench_argv: tuple = ()
    env: Mapping = field(default_factory=dict)
    env_drop: tuple = ()
    cwd: str | None = None
    health_path: str = "/health"
    port: int = 8000
    instrumentation: str = "none"
    image: str | None = None
    server_unsupported: tuple = ()
    ok_pattern: str | None = None     # bench stdout line that marks a finished cell
    log_header: str = ""              # the bench log's `>>> ctx= bs= ...` line for this cell
    cells: tuple = ()                 # ((ctx, bs, per-cell spec), ...): one bench run each
    timeout_s: int = 120              # client timeout per request, from mode.timeout

    def child_argv(self, argv):
        """argv run under `env`, so the child gets env and env_drop applied on top of what it inherits without this process reading os.environ."""
        unset = [a for name in self.env_drop for a in ("-u", name)]
        return ["env", *unset, *(f"{k}={v}" for k, v in self.env.items()), *argv]


# EngineCfg fields that describe the model or pick the adapter; no adapter carries them.
FACTS = frozenset({"name", "model_name", "kv_bytes", "weights_gib", "attention", "num_experts",
                   "verified"})


class EngineAdapter(ABC):
    """One engine: which Config fields it can express, and how."""
    name = ""
    supports: frozenset = frozenset()
    benchmarked = True   # False: a scoring side only (reference), not offered as an engine

    def refuse_unsupported(self, cfg):
        """UnsupportedField for an engine field the run, env or --set moves off its default when this adapter is not in its supports: no silent drop. A model yaml value describes the checkpoint and is not refused."""
        defaults = EngineCfg()
        for f in fields(EngineCfg):
            value = getattr(cfg.engine, f.name)
            if f.name in FACTS or f.name in self.supports or value == getattr(defaults, f.name):
                continue
            path = f"engine.{f.name}"
            asked = [o for p, o in cfg.origins.items()
                     if (p == path or p.startswith(path + ".")) and o not in NOT_ASKED]
            if asked:
                raise UnsupportedField(self.name, f"{f.name}={value}",
                                       f"set by {asked[0]}; this engine has no way to carry it")

    @abstractmethod
    def translate(self, cfg, run_dir=None) -> LaunchSpec:
        """The LaunchSpec for cfg; the single builder of this engine's arguments. run_dir is where the run writes (instrumentation output lands there)."""

    @abstractmethod
    def driver(self, spec, handle, trace=None):
        """The Driver that sends work to an engine started from spec; trace is the run's, for an adapter whose endpoint depends on it."""

    def needs_server(self, cfg):
        """Whether runner launches spec.server_argv before driving; by default only a serve run does."""
        return cfg.mode.grid is None


_ADAPTERS = {}


def register(adapter):
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get(name):
    """The registered adapter for an engine name."""
    if name not in _ADAPTERS:
        known = sorted(n for n, a in _ADAPTERS.items() if a.benchmarked)
        raise ValueError(f"unknown engine {name!r}; known: {', '.join(known)}")
    return _ADAPTERS[name]
