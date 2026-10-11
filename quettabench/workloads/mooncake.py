"""The runner's old name for the Mooncake trace, kept because the golden
reader test imports it: MooncakeTraceDataset is live.MooncakeTrace."""

from .live import MooncakeTrace as MooncakeTraceDataset

__all__ = ["MooncakeTraceDataset"]
