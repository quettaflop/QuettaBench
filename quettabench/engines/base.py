# The translation boundary: an adapter turns one Config into one LaunchSpec,
# the only description of how an engine is started and driven.
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

# scripts/crossval/prof.sh:16; {out} is the report path without extension.
NSYS = ("nsys", "profile", "-t", "cuda,nvtx", "-o", "{out}", "--force-overwrite=true")


class UnsupportedField(ValueError):
    """A Config value the engine has no flag for in any launch run today."""

    def __init__(self, engine, field_name, why=""):
        self.engine, self.field = engine, field_name
        super().__init__(f"{engine}: {field_name} has no launch flag in use today"
                         + (f" ({why})" if why else ""))


@dataclass(frozen=True)
class LaunchSpec:
    """Everything needed to start and drive one engine, as plain data.

    server_argv is empty when the adapter has no server, or when a field has
    no flag (named in server_unsupported; the launcher refuses such a spec,
    the in-process path does not need argv). env is added to the child's
    inherited environment; env_drop names inherited variables to remove.
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
    log_header: str = ""              # bench.sh's `>>> ctx= bs= ...` line for this cell

    def child_argv(self, argv):
        """argv run under `env`, so the child gets env and env_drop applied on
        top of what it inherits without this process reading os.environ."""
        unset = [a for name in self.env_drop for a in ("-u", name)]
        return ["env", *unset, *(f"{k}={v}" for k, v in self.env.items()), *argv]


class EngineAdapter(ABC):
    """One engine: which Config fields it can express, and how."""
    name = ""
    supports: frozenset = frozenset()

    @abstractmethod
    def translate(self, cfg) -> LaunchSpec:
        """The LaunchSpec for cfg; the single builder of this engine's arguments."""

    @abstractmethod
    def driver(self, spec, handle):
        """The Driver that sends work to an engine started from spec."""

    def profiler_hook(self, handle):
        """Attach point for the profiler; Phase 2. Returns None until then."""


_ADAPTERS = {}


def register(adapter):
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get(name):
    """The registered adapter for an engine name."""
    if name not in _ADAPTERS:
        raise ValueError(f"unknown engine {name!r}; known: {', '.join(sorted(_ADAPTERS))}")
    return _ADAPTERS[name]
