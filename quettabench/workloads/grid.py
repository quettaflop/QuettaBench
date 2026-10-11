"""Grid prompts as a Trace: per (ctx, bs) cell and per step point, bs requests
of ctx prompt tokens asking for step-point output tokens.

The same bs prompts serve every step point of a cell, so the decode slope
over step points is measured on fixed inputs. Five prompt modes:
  distinct  slot j gets seed + j + (i % 1000), per-slot routing
  clone     every slot gets clone_ids(ctx), the engine bench's own prompt
  corpus    sliding windows over a tokenized text file, slots at distinct offsets
  synth     synth_requests streams (synth_ids)
  trace     requests of a given Trace trimmed or padded to ctx, cycled if short
"""

from .base import Request, Trace, synth_ids
from .live import RealTrace

PROMPT_MODES = ("distinct", "clone", "corpus", "synth", "trace")


def stamp_synthetic(reqs, seed):
    """Stamp seed, order and the SYNTHETIC label on generated requests; seed and
    order are hash inputs, so a stream is self-identifying and reproducible."""
    for i, r in enumerate(reqs):
        r["seed"] = seed
        r["order"] = i
        r["source"] = "SYNTHETIC"
    return reqs


def synth_requests(count, ctx, seed=0, length_jitter=0.0):
    """`count` requests of ~`ctx` tokens. length_jitter in [0,1) varies each length
    deterministically; 0.0 keeps them uniform for a clean grid cell."""
    reqs = []
    for j in range(count):
        s = seed * 1_000_003 + j
        if length_jitter > 0.0:
            span = int(ctx * length_jitter)
            length = max(1, ctx - span + (s * 2654435761) % (2 * span + 1))
        else:
            length = ctx
        reqs.append({"prompt_token_ids": synth_ids(length, s), "prompt_len": length})
    return stamp_synthetic(reqs, seed)


def take(reqs, count, ctx):
    """Pick `count` requests for a (ctx, bs) cell, trimming/padding ids to ctx and
    cycling the trace if short. Returns (ids, cycled)."""
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
    """seed + (i % 1000): a different stream per seed, so slots route apart."""
    return [seed + (i % 1000) for i in range(n)]


def clone_ids(n):
    """The synthetic prompt the Rust engine benches feed every slot. Must stay
    in sync with those bench ids, or clone mode no longer routes like the
    engine under greedy and the ratio compares different work."""
    return [(i * 137 + 11) % 100_000 for i in range(n)]


def corpus_window(ids, n, slot):
    """n ids from slot * 997 into the corpus, wrapping to the start when the
    window outruns the text."""
    start = (slot * 997) % max(1, len(ids))
    out = []
    while len(out) < n:
        out.extend(ids[start:start + (n - len(out))])
        start = 0
    return out


class GridTrace(Trace):
    """Cells over maxlen (ctx + max(step_points) + 2 > maxlen) carry no
    requests; they never run. tokenizer is a zero-arg callable returning a
    HF-style tokenizer, called once at the first corpus cell, because the
    engine's tokenizer exists only after the model loads."""

    def __init__(self, cells, step_points, prompt_mode, maxlen,
                 corpus_path=None, tokenizer=None, trace=None, seed=10):
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
        """The grid a Config names: cfg.grid's cells and step points at
        engine.max_model_len. workload.source is the corpus text under
        prompt_mode corpus and the replay JSONL under trace."""
        w = cfg.workload
        trace = RealTrace.from_jsonl(w.source) if w.prompt_mode == "trace" and w.source else None
        corpus = w.source if w.prompt_mode == "corpus" and w.source else None
        return cls(cfg.grid.cells, cfg.grid.step_points, w.prompt_mode, cfg.engine.max_model_len,
                   corpus_path=corpus, tokenizer=tokenizer, trace=trace)

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
        """(bs id lists of ctx tokens, cycled). cycled is True only in trace
        mode when the trace has fewer usable requests than bs."""
        mode = self.prompt_mode
        if mode == "clone":
            base = clone_ids(ctx)
            return [list(base) for _ in range(bs)], False
        if mode == "corpus":
            ids = self._corpus()
            return [corpus_window(ids, ctx, j) for j in range(bs)], False
        if mode == "synth":
            return take(synth_requests(bs, ctx, seed=ctx), bs, ctx)[0], False
        if mode == "trace":
            return take(self._trace_rows, bs, ctx)
        return [distinct_ids(ctx, seed=self.seed + j) for j in range(bs)], False

    def requests(self):
        skip = set(self.cells_over_maxlen)
        out = []
        for ctx, bs in self.cells:
            if (ctx, bs) in skip:
                continue
            prompts, _ = self.cell_prompts(ctx, bs)
            for step in self.step_points:
                for ids in prompts:
                    out.append(Request(ids, len(ids), order=len(out),
                                       output_length=step, seed=self.seed))
        return out

    def source_meta(self):
        meta = {"reader": "grid", "prompt_mode": self.prompt_mode,
                "cells": len(self.cells), "step_points": self.step_points}
        if self.trace is not None:
            meta["trace"] = self.trace.source_meta()
        if self.corpus_path:
            meta["corpus"] = str(self.corpus_path)
        return meta
