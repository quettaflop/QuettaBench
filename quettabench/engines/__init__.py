"""Importing the package registers every adapter."""

from quettabench.engines import (  # noqa: F401  (registers the adapters)
    drivers,
    quettaserve,
    vllm,
)
