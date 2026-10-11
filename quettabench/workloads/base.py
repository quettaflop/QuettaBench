"""Request, the Trace base class, and the workload identity hash.

A Trace is a fixed list of Requests. workload_hash() names that list so two
engines can prove they ran identical work: serve gates on it, crossval joins
on it.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np

VOCAB = 100_000  # safe id ceiling for the checkpoints under test


def synth_ids(n, seed):
    """Distinct pseudo-random token stream; different seed gives different routing."""
    a = 1103515245 * (seed + 1) + 12345
    return [((a + i * (2 * seed + 7)) % (VOCAB - 1)) + 1 for i in range(n)]


@dataclass
class Request:
    """One request as the replay schema writes it. None means the field is
    absent from the line, which is not the same as zero for the hash."""

    prompt_token_ids: list
    prompt_len: int
    order: int | None = None
    output_length: int | None = None
    output_token_ids: list | None = None
    arrival_ts: float | None = None
    session_id: object = None
    turn: int | None = None
    seed: int | None = None
    image_tokens: int | None = None
    source: str | None = None
    # A workload profile's chat request: its messages and metadata objects,
    # shared with the dataset so a later turn sees replies written into them.
    # Not part of the replay schema, never hashed.
    messages: list | None = None
    metadata: dict | None = None

    @property
    def max_tokens(self):
        return self.output_length

    def to_dict(self):
        """The dict synth_bench.load_trace returns: set replay fields only."""
        plain = replace(self, messages=None, metadata=None)
        return {k: v for k, v in asdict(plain).items() if v is not None}


def request_digest(r):
    """Identity of one request over deterministic inputs only (seed, order, arrival,
    prompt content, requested output), never generated tokens or timings, so two
    engines doing identical work hash the same."""
    if isinstance(r, Request):
        r = r.to_dict()
    ids = r.get("prompt_token_ids") or []
    parts = [
        str(r.get("seed", "")),
        str(r.get("order", "")),
        f"{float(r.get('arrival_ts', 0.0)):.6f}",
        str(r.get("prompt_len", len(ids))),
        ",".join(map(str, ids)),
        str(r.get("output_length", "")),
        str(r.get("session_id", "")),
        str(r.get("turn", "")),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def workload_hash(reqs):
    """Order-independent identity of the whole workload: first 16 hex of sha256
    over the request count and the sorted per-request digests. Completion order
    does not move it; a dropped or altered request does. Takes Requests or the
    equivalent dicts."""
    digs = sorted(request_digest(r) for r in reqs)
    h = hashlib.sha256()
    h.update(str(len(digs)).encode())
    for d in digs:
        h.update(b"\n")
        h.update(d.encode())
    return h.hexdigest()[:16]


class Trace(ABC):
    """A fixed request list plus where it came from. supports_spec says whether
    requests carry real assistant tokens to score speculative acceptance against."""

    supports_spec: bool = False

    @abstractmethod
    def requests(self) -> list[Request]:
        """Every request, in submission order."""

    @abstractmethod
    def source_meta(self) -> dict:
        """Provenance for the results file: path, reader, sources."""

    def workload_hash(self) -> str:
        return workload_hash(self.requests())

    def warmup_requests(self, n) -> list[Request]:
        """n requests to warm the engine with: the trace's own, cycled."""
        reqs = self.requests()
        return [reqs[i % len(reqs)] for i in range(n)] if reqs else []


class DrawnTrace(Trace):
    """A trace drawn from a source that serves one request at a time and
    wraps around at its end, in the order the old runner drew its datasets.
    draw() fixes what a run sends: `warmup` draws, a rewind when the source
    replays a file in order (rewinds), then `num_requests` draws, each
    numbered from zero. The profile it was built from names it in source_meta."""

    rewinds = False

    def __init__(self):
        self._warmup, self._requests, self.profile = [], [], None

    @abstractmethod
    def get_next_request(self) -> Request:
        """The next request; the source wraps around at its end."""

    def reset(self):
        """Back to the first request; only sources that rewind override it."""

    def draw(self, num_requests, warmup=0, profile=None):
        self.profile = profile
        self._warmup = [replace(self.get_next_request(), order=i) for i in range(warmup)]
        if warmup and self.rewinds:
            self.reset()
        self._requests = [replace(self.get_next_request(), order=i) for i in range(num_requests)]
        return self

    def requests(self):
        return self._requests

    def warmup_requests(self, n):
        return self._warmup[:n]

    def source_meta(self):
        p = self.profile
        meta = ({"profile": p.name, "dataset": p.dataset, "path": p.file_path} if p
                else {"reader": type(self).__name__})
        return {**meta, "reqs": len(self._requests)}


def uniform_len_bounds(input_len, range_ratio):
    """InferenceX sample_uniform bounds."""
    return max(1, int(input_len * range_ratio)), input_len


def load_tokenizer(name):
    from transformers import AutoTokenizer  # optional dependency, only random-token traces need it
    return AutoTokenizer.from_pretrained(name)


class RandomTokenTrace(DrawnTrace):
    """InferenceX's random-token workload (SemiAnalysisAI/InferenceX
    utils/bench_serving/benchmark_serving.py) for cross-validation: prompt i
    is ids (offset_i + i + j) % vocab decoded to text, re-encoded and trimmed
    or padded to its target length, sent as one user message. Random tokens
    trigger EOS early, so output lengths are unreliable; not a production
    workload. The prompts cycle."""

    def __init__(self, tokenizer_name, input_len=1024, output_len=1024, num_prompts=500,
                 range_ratio=1.0, seed=0):
        super().__init__()
        self.output_len = output_len
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.prompts = self.make_prompts(input_len, num_prompts, range_ratio, seed)
        self._idx = 0

    def make_prompts(self, input_len, num_prompts, range_ratio, seed):
        tokenizer = self.tokenizer
        rng = np.random.default_rng(seed)
        lo, hi = uniform_len_bounds(input_len, range_ratio)
        input_lens = rng.integers(lo, hi + 1, size=num_prompts)
        offsets = rng.integers(0, tokenizer.vocab_size, size=num_prompts)
        prompts = []
        for i in range(num_prompts):
            tgt_len = int(input_lens[i])
            token_ids = [(int(offsets[i]) + i + j) % tokenizer.vocab_size for j in range(tgt_len)]
            re_encoded = tokenizer.encode(tokenizer.decode(token_ids), add_special_tokens=False)
            if len(re_encoded) > tgt_len:
                re_encoded = re_encoded[:tgt_len]
            elif len(re_encoded) < tgt_len:
                re_encoded.extend(rng.integers(0, tokenizer.vocab_size,
                                               size=tgt_len - len(re_encoded)).tolist())
            prompts.append(tokenizer.decode(re_encoded))
        return prompts

    def content(self, prompt):
        return prompt

    def get_next_request(self):
        prompt = self.prompts[self._idx % len(self.prompts)]
        self._idx += 1
        return Request([], 0, output_length=self.output_len,
                       messages=[{"role": "user", "content": self.content(prompt)}], metadata={})


class LegacyRandomTokenTrace(RandomTokenTrace):
    """The same prompts drawn the way InferenceX's sample_random_requests
    draws them: np.random.seed then legacy randint, an optional shared prefix,
    up to ten re-encode passes. Matching TTFT here and not with
    RandomTokenTrace pins a TTFT gap on RNG-driven prefix cache hits."""

    def __init__(self, tokenizer_name, input_len=1024, output_len=1024, num_prompts=500,
                 range_ratio=1.0, prefix_len=0, seed=0):
        self.prefix_len = prefix_len
        super().__init__(tokenizer_name, input_len, output_len, num_prompts, range_ratio, seed)

    def make_prompts(self, input_len, num_prompts, range_ratio, seed):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size
        np.random.seed(seed)
        prefix_token_ids = np.random.randint(0, vocab_size, size=self.prefix_len).tolist()
        lo, hi = uniform_len_bounds(input_len, range_ratio)
        input_lens = np.random.randint(lo, hi + 1, size=num_prompts).tolist()
        # Output lengths are drawn, unused, only to keep InferenceX's RNG sequence.
        np.random.randint(int(self.output_len * range_ratio), self.output_len + 1, size=num_prompts)
        offsets = np.random.randint(0, vocab_size, size=num_prompts)
        prompts = []
        for i in range(num_prompts):
            tgt_len = self.prefix_len + input_lens[i]
            token_ids = prefix_token_ids + [(int(offsets[i]) + i + j) % vocab_size
                                            for j in range(input_lens[i])]
            prompt = tokenizer.decode(token_ids)
            for _ in range(10):
                re_encoded = tokenizer.encode(prompt, add_special_tokens=False)
                if len(re_encoded) < tgt_len:
                    re_encoded.extend(np.random.randint(0, vocab_size,
                                                        size=tgt_len - len(re_encoded)).tolist())
                elif len(re_encoded) > tgt_len:
                    re_encoded = re_encoded[:tgt_len]
                else:
                    break
                prompt = tokenizer.decode(re_encoded)
            prompts.append(prompt)
        return prompts


class DoubleWrapRandomTokenTrace(RandomTokenTrace):
    """InferenceX's double chat template: with --backend openai-chat
    --use-chat-template it applies the template itself and sends the result
    as user content, which the server templates again. Kept to measure the
    TTFT that costs against the single wrap."""

    def content(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
