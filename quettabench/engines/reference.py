"""Reference adapter (engine.name: reference): the transformers side of the
FIDELITY gate. No server and no argv; the driver loads the weights itself
from engine_kwargs {model, dtype}. Grid runs only: it scores the grid's
prompts, there is nothing to serve."""

from quettabench.config.schema import ConfigError
from quettabench.engines.base import EngineAdapter, LaunchSpec, register
from quettabench.engines.drivers import ReferenceDriver


class ReferenceAdapter(EngineAdapter):
    name = "reference"
    supports = frozenset({"model", "dtype"})
    benchmarked = False

    def translate(self, cfg):
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("reference: engine.model is required")
        if engine.dtype == "auto":
            raise ConfigError("reference: engine.dtype must name a torch dtype, e.g. bfloat16")
        if cfg.mode.grid is None:
            raise ConfigError("reference: scores grid prompts only; set mode.grid")
        return LaunchSpec(adapter=self.name,
                          engine_kwargs={"model": engine.model, "dtype": engine.dtype})

    def needs_server(self, cfg):
        return False

    def driver(self, spec, handle):
        return ReferenceDriver(spec)


ADAPTER = register(ReferenceAdapter())
