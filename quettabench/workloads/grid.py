"""Grid prompts as a Trace: per (ctx, bs) cell and step point."""

from ..config.schema import PROMPT_MODES
from .base import Request, Trace, synth_ids
from .live import RealTrace


def take(reqs, count, ctx):
    """Cycles the trace if short; returns (ids, cycled)."""
    usable = [r for r in reqs if r["prompt_len"] >= ctx] or reqs
    out, cycled = [], False
    for j in range(count):
        if j >= len(usable):
            cycled = True
        r = usable[j % len(usable)]
        ids = r["prompt_token_ids"]
        ids = ids[:ctx] if len(ids) >= ctx else ids + synth_ids(ctx - len(ids), j)
        out.append(list(ids))
    return out, cycled


def distinct_ids(n, seed=10):
    """A different stream per seed, so slots route apart."""
    return [seed + (i % 1000) for i in range(n)]


def clone_ids(n):
    """Keep in sync with the Rust engine bench ids."""
    return [(i * 137 + 11) % 100_000 for i in range(n)]


def corpus_window(ids, n, slot):
    start = (slot * 997) % max(1, len(ids))
    out: list[int] = []
    while len(out) < n:
        out.extend(ids[start : start + (n - len(out))])
        start = 0
    return out


class GridTrace(Trace):
    """Cells over maxlen (ctx + max(step_points) + 2 > maxlen) carry no requests."""

    def __init__(self, cells, step_points, prompt_mode, maxlen, corpus_path=None, tokenizer=None, trace=None, seed=10):
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}; use {', '.join(PROMPT_MODES)}")
        if prompt_mode == "trace" and trace is None:
            raise ValueError("prompt_mode=trace needs a trace")
        self.cells = [tuple(c) for c in cells]
        self.step_points = list(step_points)
        self.prompt_mode = prompt_mode
        self.maxlen = maxlen
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.trace = trace
        self.seed = seed
        self._trace_rows = [r.to_dict() for r in trace.requests()] if trace else None
        self._corpus_ids = None

    @classmethod
    def from_config(cls, cfg, tokenizer=None):
        """workload.source is the corpus text under prompt_mode corpus, replay JSONL under trace."""
        w = cfg.workload
        trace = RealTrace.from_jsonl(w.source) if w.prompt_mode == "trace" and w.source else None
        corpus = w.source if w.prompt_mode == "corpus" and w.source else None
        return cls(
            cfg.grid.cells,
            cfg.grid.step_points,
            w.prompt_mode,
            cfg.engine.max_model_len,
            corpus_path=corpus,
            tokenizer=tokenizer,
            trace=trace,
        )

    @property
    def cells_over_maxlen(self):
        top = max(self.step_points)
        return [(ctx, bs) for ctx, bs in self.cells if ctx + top + 2 > self.maxlen]

    def _corpus(self):
        if self._corpus_ids is None:
            if not self.corpus_path:
                raise ValueError("prompt_mode=corpus needs a corpus file")
            with open(self.corpus_path) as f:
                self._corpus_ids = self.tokenizer()(f.read())["input_ids"]
        return self._corpus_ids

    def cell_prompts(self, ctx, bs):
        """cycled only in trace mode when fewer usable requests than bs."""
        mode = self.prompt_mode
        if mode == "clone":
            base = clone_ids(ctx)
            return [list(base) for _ in range(bs)], False
        if mode == "corpus":
            ids = self._corpus()
            return [corpus_window(ids, ctx, j) for j in range(bs)], False
        if mode == "trace":
            return take(self._trace_rows, bs, ctx)
        return [distinct_ids(ctx, seed=self.seed + j) for j in range(bs)], False

    def requests(self):
        skip = set(self.cells_over_maxlen)
        out: list[Request] = []
        for ctx, bs in self.cells:
            if (ctx, bs) in skip:
                continue
            prompts, _ = self.cell_prompts(ctx, bs)
            for step in self.step_points:
                for ids in prompts:
                    out.append(Request(ids, len(ids), order=len(out), output_length=step, seed=self.seed))
        return out

    def source_meta(self):
        meta = {
            "reader": "grid",
            "prompt_mode": self.prompt_mode,
            "cells": len(self.cells),
            "step_points": self.step_points,
        }
        if self.trace is not None:
            meta["trace"] = self.trace.source_meta()
        if self.corpus_path:
            meta["corpus"] = str(self.corpus_path)
        return meta
