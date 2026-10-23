"""The boundary: an adapter turns one Config into one LaunchSpec."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from quettabench.config.schema import NOT_ASKED, EngineCfg

# {out} is the report path without extension.
NSYS = ("nsys", "profile", "-t", "cuda,nvtx", "-o", "{out}", "--force-overwrite=true")


def nsys_argv(cfg, run_dir, name="nsys"):
    out = Path(run_dir if run_dir is not None else Path("results", cfg.run_name)).resolve() / name
    return [a.format(out=out) for a in NSYS]


class UnsupportedField(ValueError):
    """A Config value the engine has no flag for."""

    def __init__(self, engine, field_name, why=""):
        self.engine, self.field = engine, field_name
        super().__init__(f"{engine}: {field_name} has no launch flag in use today" + (f" ({why})" if why else ""))


@dataclass(frozen=True)
class LaunchSpec:
    adapter: str
    server_argv: tuple = ()
    engine_kwargs: Mapping = field(default_factory=dict)
    bench_argv: tuple = ()
    env: Mapping = field(default_factory=dict)
    env_drop: tuple = ()
    cwd: str | None = None
    health_path: str = "/health"
    served_name: str = ""  # the model name requests send; set when the server can also answer to a launch token
    api_key_env: str = ""  # the env entry holding the server's bearer key, sent on discovery when attaching
    port: int = 8000
    instrumentation: str = "none"
    image: str | None = None
    ok_pattern: str | None = None  # bench stdout line that marks a finished cell
    log_header: str = ""  # the bench log's `>>> ctx= bs= ...` line for this cell
    cells: tuple = ()  # ((ctx, bs, per-cell spec), ...): one bench run each.
    timeout_s: int = 120  # client timeout per request, from mode.timeout
    # True: the child starts from an empty environment (env -i), so env is everything it gets.
    env_clear: bool = False

    def child_argv(self, argv):
        """Runs under `env` so env and env_drop apply without reading os.environ here."""
        unset = ["-i"] if self.env_clear else [a for name in self.env_drop for a in ("-u", name)]
        return ["env", *unset, *(f"{k}={v}" for k, v in self.env.items()), *argv]


# EngineCfg fields that describe the model or pick the adapter.
FACTS = frozenset({"name", "model_name", "kv_bytes_per_token", "weights_gib", "attention", "num_experts", "verified"})


class EngineAdapter(ABC):
    name = ""
    supports: frozenset = frozenset()
    benchmarked = True  # False: a scoring side only (reference), not an engine.

    def refuse_unsupported(self, cfg):
        defaults = EngineCfg()
        for f in fields(EngineCfg):
            value = getattr(cfg.engine, f.name)
            if f.name in FACTS or f.name in self.supports or value == getattr(defaults, f.name):
                continue
            path = f"engine.{f.name}"
            asked = [
                o for p, o in cfg.origins.items() if (p == path or p.startswith(path + ".")) and o not in NOT_ASKED
            ]
            if asked:
                raise UnsupportedField(
                    self.name, f"{f.name}={value}", f"set by {asked[0]}; this engine has no way to carry it"
                )

    @abstractmethod
    def translate(self, cfg, run_dir=None) -> LaunchSpec:
        """run_dir is where the run writes (instrumentation lands there)."""

    @abstractmethod
    def driver(self, spec, handle, trace=None):
        """trace is the run's, for adapters whose endpoint depends on it."""

    def needs_server(self, cfg):
        """By default only a serve run."""
        return cfg.mode.grid is None


_ADAPTERS: dict[str, Any] = {}


def register(adapter):
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get(name):
    if name not in _ADAPTERS:
        known = sorted(n for n, a in _ADAPTERS.items() if a.benchmarked)
        raise ValueError(f"unknown engine {name!r}; known: {', '.join(known)}")
    return _ADAPTERS[name]
