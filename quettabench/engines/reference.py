"""Reference adapter (engine.name: reference): the transformers side of the
FIDELITY gate. No server and no argv; the driver loads the weights itself
from engine_kwargs {model, dtype, layers}. Grid runs only: it scores the
grid's prompts, there is nothing to serve. engine.layers keeps the first N
decoder layers, as slice_truncate.py cut the config for the slice runs."""

from quettabench.config.schema import ConfigError
from quettabench.engines.base import EngineAdapter, LaunchSpec, register
from quettabench.engines.drivers import ReferenceDriver


class ReferenceAdapter(EngineAdapter):
    name = "reference"
    supports = frozenset({"model", "dtype", "layers"})
    benchmarked = False

    def translate(self, cfg, run_dir=None):
        self.refuse_unsupported(cfg)
        engine = cfg.engine
        if engine.model is None:
            raise ConfigError("reference: engine.model is required")
        if engine.dtype == "auto":
            raise ConfigError("reference: engine.dtype must name a torch dtype, e.g. bfloat16")
        if cfg.mode.grid is None:
            raise ConfigError("reference: scores grid prompts only; set mode.grid")
        if engine.layers is not None and engine.layers < 1:
            raise ConfigError(f"reference: engine.layers must be >= 1, got {engine.layers}")
        kwargs = {"model": engine.model, "dtype": engine.dtype}
        if engine.layers is not None:
            kwargs["layers"] = engine.layers
        return LaunchSpec(adapter=self.name, engine_kwargs=kwargs)

    def needs_server(self, cfg):
        return False

    def driver(self, spec, handle, trace=None):
        return ReferenceDriver(spec)


ADAPTER = register(ReferenceAdapter())
