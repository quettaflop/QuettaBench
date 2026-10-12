"""Engine adapters: importing the package registers every one of them."""

from quettabench.engines import (  # noqa: F401  (registers the adapters)
    drivers,
    quettaserve,
    vllm,
)
