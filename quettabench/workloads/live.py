"""Recorded traces: the replay JSONL schema, the Mooncake production trace,
and the real-text profile sources (prompt JSONL, ShareGPT, a smoke prompt).

One reader for the replay schema and Mooncake. A line carries prompt_token_ids, or prompt_len, or the
Mooncake fields (input_length, output_length, timestamp in ms, hash_ids).
Mooncake hash_ids name 512-token prefix blocks; each id expands to the same
seeded block every time, so requests that share a hash id share those tokens
byte for byte and prefix sharing survives the replay. A Mooncake line with no
hash_ids falls back to synth_ids(input_length, line index).

Trace format reference: https://github.com/kvcache-ai/Mooncake
"""

import json
import random
from dataclasses import dataclass

from .base import DrawnTrace, Request, Trace, synth_ids

BLOCK_SIZE = 512

# Range avoids low special token ids and stays under small vocabs.
TOKEN_ID_LOW = 1000
TOKEN_ID_HIGH = 20000


@dataclass(frozen=True)
class MooncakeRecord:
    index: int  # line index in the file
    input_length: int
    output_length: int | None
    hash_ids: tuple[int, ...]
    timestamp: float | None = None  # ms, as the trace writes it


def _rows(path):
    """(line index, parsed dict) for every non-blank line."""
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def mooncake_record(row, index):
    """input_length is required; output_length, hash_ids and timestamp may be absent."""
    out, ts = row.get("output_length"), row.get("timestamp")
    return MooncakeRecord(
        index=index,
        input_length=int(row["input_length"]),
        output_length=None if out is None else int(out),
        hash_ids=tuple(int(h) for h in row.get("hash_ids") or ()),
        timestamp=None if ts is None else float(ts),
    )


def parse_mooncake_trace(path, limit=0):
    """Raw Mooncake records in file order, for tools that study the trace itself."""
    records = []
    for i, row in _rows(path):
        rec = mooncake_record(row, i)
        if rec.output_length is None:
            raise ValueError(f"{path}:{i + 1}: Mooncake line needs output_length")
        records.append(rec)
        if limit and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No records parsed from {path}")
    return records


def expand_block(hash_id, seed):
    """One hash id always yields the same BLOCK_SIZE token ids."""
    rng = random.Random(f"{seed}:{hash_id}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(BLOCK_SIZE)]


def fill_tail(record_index, length, seed):
    """Request-unique filler for input tokens not covered by hash blocks."""
    rng = random.Random(f"{seed}:tail:{record_index}")
    return [rng.randrange(TOKEN_ID_LOW, TOKEN_ID_HIGH) for _ in range(length)]


def mooncake_prompt_ids(record, seed=42, block_cache=None):
    """Exactly input_length ids: hash blocks in order, tail filled per request,
    truncated when the blocks overrun (the common case in the published trace).
    No hash_ids: synth_ids(input_length, line index)."""
    if not record.hash_ids:
        return synth_ids(record.input_length, record.index)
    tokens = []
    for hash_id in record.hash_ids:
        if len(tokens) >= record.input_length:
            break
        if block_cache is None:
            block = expand_block(hash_id, seed)
        else:
            block = block_cache.get(hash_id)
            if block is None:
                block = block_cache[hash_id] = expand_block(hash_id, seed)
        tokens.extend(block)
    if len(tokens) < record.input_length:
        tokens.extend(fill_tail(record.index, record.input_length - len(tokens), seed))
    return tokens[: record.input_length]


def _carry_meta(req, row):
    """Scheduling and provenance fields; Mooncake timestamp is ms."""
    if row.get("arrival_ts") is not None:
        req.arrival_ts = float(row["arrival_ts"])
    elif row.get("timestamp") is not None:
        req.arrival_ts = float(row["timestamp"]) / 1e3
    for k in ("session_id", "turn", "image_tokens", "seed", "source"):
        if row.get(k) is not None:
            setattr(req, k, row[k])
    return req


def _mooncake_request(row, index, seed, block_cache):
    """A Mooncake line keeps output_length 0: it is the trace's own number."""
    rec = mooncake_record(row, index)
    ids = mooncake_prompt_ids(rec, seed, block_cache)
    req = Request(ids, rec.input_length, order=index, output_length=rec.output_length)
    return _carry_meta(req, row)


def _line_request(path, row, index, seed, block_cache):
    """One replay-schema line; order is the line index, a hash input."""
    if row.get("prompt_token_ids"):
        ids = list(row["prompt_token_ids"])
        req = Request(ids, len(ids), order=index)
        if row.get("output_length"):
            req.output_length = int(row["output_length"])
        # Reference tokens for acceptance scoring (SPECSUM); must survive load
        # or the replay has nothing to compare the engine's output against.
        if row.get("output_token_ids"):
            req.output_token_ids = list(row["output_token_ids"])
        return _carry_meta(req, row)
    if not row.get("prompt_len") and row.get("input_length"):
        return _mooncake_request(row, index, seed, block_cache)
    length = row.get("prompt_len")
    if not length:
        raise ValueError(f"{path}:{index + 1}: need prompt_token_ids, prompt_len, or input_length")
    req = Request(synth_ids(int(length), index), int(length), order=index)
    if row.get("output_length"):
        req.output_length = int(row["output_length"])
    return _carry_meta(req, row)


class RealTrace(Trace):
    """Requests read from a file. supports_spec when any request carries the
    trace's real assistant tokens (output_token_ids)."""

    def __init__(self, requests, path, reader):
        if not requests:
            raise ValueError(f"{path}: no requests")
        self._requests = requests
        self.path = str(path)
        self.reader = reader
        self.supports_spec = any(r.output_token_ids for r in requests)

    @classmethod
    def from_jsonl(cls, path, seed=42):
        """The replay schema; seed only matters for Mooncake hash blocks."""
        cache = {}
        reqs = [_line_request(path, row, i, seed, cache) for i, row in _rows(path)]
        return cls(reqs, path, "jsonl")

    @classmethod
    def from_mooncake(cls, path, seed=42, limit=None):
        """Mooncake lines only (input_length required), first `limit` of them."""
        cache, reqs = {}, []
        for i, row in _rows(path):
            reqs.append(_mooncake_request(row, i, seed, cache))
            if limit and len(reqs) >= limit:
                break
        return cls(reqs, path, "mooncake")

    def requests(self):
        return self._requests

    def source_meta(self):
        return {
            "path": self.path,
            "reader": self.reader,
            "reqs": len(self._requests),
            "sources": sorted({"?" if r.source is None else str(r.source) for r in self._requests}),
            "text_mode": "real" if self.supports_spec else "synthetic",
        }


class MooncakeTrace(DrawnTrace):
    """Mooncake records in file order with expanded token ids, wrapping at
    the end and rewinding after warmup. Requests carry prompt_token_ids, so
    no chat template can disturb block boundaries; metadata names the record
    (request_index) and its block count (num_hash_blocks). arrival_ts is the
    record's timestamp in seconds, so workload.arrival=trace replays it; a
    wrapped draw's lap k is shifted by k * (span + mean gap) of the file's
    timestamps, so laps follow one another at the trace's own pace. With a
    span of 0 there is no pace; workloads.build refuses such a wrap under
    workload.arrival=trace."""

    rewinds = True

    def __init__(self, path, seed=42, limit=0):
        super().__init__()
        self.records = parse_mooncake_trace(path, limit)
        stamps = [r.timestamp for r in self.records if r.timestamp is not None]
        span = max(stamps) - min(stamps) if stamps else 0
        self.lap_ms = span + (span / (len(stamps) - 1) if len(stamps) > 1 else 0)
        self.seed = seed
        self._cursor = 0
        self._block_cache = {}

    def reset(self):
        self._cursor = 0

    def get_next_request(self):
        lap, i = divmod(self._cursor, len(self.records))
        record = self.records[i]
        self._cursor += 1
        ids = mooncake_prompt_ids(record, self.seed, self._block_cache)
        arrival = None if record.timestamp is None else (record.timestamp + lap * self.lap_ms) / 1e3
        return Request(ids, len(ids), output_length=record.output_length, arrival_ts=arrival,
                       messages=[], metadata={
                           "prompt_token_ids": ids,
                           "request_index": record.index,
                           "num_hash_blocks": len(record.hash_ids),
                       })


def reshuffled(samples, rng):
    """(messages, max_tokens) samples popped from the end; once used up they
    are refilled and reshuffled with rng."""
    available = list(samples)
    while True:
        if not available:
            available = list(samples)
            rng.shuffle(available)
        yield available.pop()


class SampledChatTrace(DrawnTrace):
    """Chat prompts sampled from a fixed set: shuffled once with the seed,
    then drawn by reshuffled() with a second generator on the same seed."""

    def __init__(self, samples, seed):
        super().__init__()
        self.samples = samples
        self._stream = reshuffled(samples, random.Random(seed))

    def get_next_request(self):
        messages, max_tokens = next(self._stream)
        return Request([], 0, output_length=max_tokens, messages=messages, metadata={})


class PromptJsonlTrace(SampledChatTrace):
    """Real prompts, one per line: {"system", "user", "osl_tokens"}; max
    tokens per request from osl_tokens."""

    def __init__(self, path, seed=42):
        samples = []
        for _, entry in _rows(path):
            messages = [{"role": "system", "content": entry["system"]},
                        {"role": "user", "content": entry["user"]}]
            samples.append((messages, entry["osl_tokens"]))
        random.Random(seed).shuffle(samples)
        super().__init__(samples, seed)


def est_tokens(text):
    """Token estimate for English: 1.35 tokens per word."""
    return int(len(text.split()) * 1.35)


def load_sharegpt():
    import datasets as hf_datasets  # optional dependency, only ShareGPT profiles need it
    return hf_datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")


def prompt_token_count(tokenizer, messages):
    try:
        return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
    except Exception:
        text = "\n".join(str(message.get("content", "")) for message in messages)
        return len(tokenizer.encode(text, add_special_tokens=True))


def load_chat_tokenizer(name):
    """The model's tokenizer for prompt counts; None without a name. A
    tokenizer that fails to load raises."""
    if not name:
        return None
    from transformers import AutoTokenizer  # optional dependency
    return AutoTokenizer.from_pretrained(name)


class ShareGPTTrace(SampledChatTrace):
    """The first user and assistant pair of ShareGPT conversations. max tokens
    is the estimated length of the recorded reply, so the server sees
    ShareGPT's own ISL/OSL spread. Pairs outside the ISL/OSL bounds are
    skipped, and with max_total_tokens so is a pair whose prompt (counted by
    tokenizer_name's tokenizer when given, else estimated), reply and safety
    margin overflow it. Loads
    up to 3x num_prompts pairs, shuffles, keeps num_prompts."""

    def __init__(self, num_prompts=1000, seed=42, system_prompt="You are a helpful assistant.",
                 max_isl_tokens=8192, max_osl_tokens=2048, min_osl_tokens=50,
                 max_total_tokens=None, context_safety_margin_tokens=0, tokenizer_name=""):
        ds = load_sharegpt()
        tokenizer = load_chat_tokenizer(tokenizer_name) if max_total_tokens is not None else None
        samples = []
        for item in ds:
            convs = item["conversations"]
            if len(convs) < 2:
                continue
            user_msg = assistant_msg = None
            for turn in convs:
                if turn.get("from") == "human" and user_msg is None:
                    user_msg = turn.get("value", "")
                elif turn.get("from") == "gpt" and user_msg is not None:
                    assistant_msg = turn.get("value", "")
                    break
            if not user_msg or not assistant_msg:
                continue
            isl_est, osl_est = est_tokens(user_msg), est_tokens(assistant_msg)
            if isl_est > max_isl_tokens or osl_est > max_osl_tokens or osl_est < min_osl_tokens:
                continue
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages.append({"role": "user", "content": user_msg})
            if max_total_tokens is not None:
                prompt_tokens = (prompt_token_count(tokenizer, messages) if tokenizer is not None
                                 else isl_est)
                if prompt_tokens + osl_est + context_safety_margin_tokens > max_total_tokens:
                    continue
            samples.append((messages, osl_est))
            if len(samples) >= num_prompts * 3:
                break
        random.Random(seed).shuffle(samples)
        super().__init__(samples[:num_prompts], seed)


class SmokeTrace(DrawnTrace):
    """One fixed prompt, 20 output tokens: a quick smoke test."""

    def __init__(self, prompt="Say hello in one word."):
        super().__init__()
        self.prompt = prompt

    def get_next_request(self):
        return Request([], 0, output_length=20,
                       messages=[{"role": "user", "content": self.prompt}], metadata={})
