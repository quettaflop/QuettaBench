"""
Dataset classes for benchmark workloads.

Ported from llm-bench/src/benchmark_dataset.py with improvements:
- Thread-safe with asyncio lock support
- Profile-aware (returns messages list, not just prompt string)
- ShareGPT loads full conversations, not just first message
- ShareGPT returns per-request max_tokens from the actual assistant reply length
"""

import asyncio
import threading
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BenchmarkRequest:
    messages: list[dict]
    max_tokens: int
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    """Base class for all benchmark datasets."""

    @abstractmethod
    def get_next_request(self) -> BenchmarkRequest:
        """Return the next request as a BenchmarkRequest (messages + max_tokens)."""
        pass

    def get_next_messages(self) -> list[dict]:
        """Deprecated shim — returns messages only. Use get_next_request()."""
        return self.get_next_request().messages


class TestDataset(BaseDataset):
    """Simple dataset for smoke testing."""

    def __init__(self, prompt: str = "Say hello in one word."):
        self.prompt = prompt

    def get_next_request(self) -> BenchmarkRequest:
        return BenchmarkRequest(
            messages=[{"role": "user", "content": self.prompt}],
            max_tokens=20,
        )


class FileDataset(BaseDataset):
    """Loads a single static prompt from a text file."""

    def __init__(
        self,
        filepath: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
    ):
        self.filepath = filepath
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._prompt: Optional[str] = None
        self._lock = threading.Lock()

    def _load(self):
        if self._prompt is None:
            with self._lock:
                if self._prompt is None:
                    with open(self.filepath, "r") as f:
                        self._prompt = f.read().strip()

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prompt})
        return BenchmarkRequest(messages=messages, max_tokens=self.max_tokens)


class JsonlDataset(BaseDataset):
    """
    Loads real prompts from a JSONL file.

    Each line: {"system": "...", "user": "...", "osl_tokens": N}
    Returns per-request max_tokens from osl_tokens field.
    Used for the coding-singleturn profile with real SWE-bench PLLM prompts.
    """

    def __init__(self, filepath: str, random_seed: int = 42):
        self.filepath = filepath
        self.random_seed = random_seed
        self._samples: Optional[list[tuple]] = None   # list of (messages, osl_tokens)
        self._available: Optional[list[tuple]] = None
        self._lock = threading.Lock()
        self._rng = random.Random(random_seed)

    def _load(self):
        if self._samples is not None:
            return
        with self._lock:
            if self._samples is not None:
                return
            import json
            samples = []
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    messages = [
                        {"role": "system", "content": entry["system"]},
                        {"role": "user", "content": entry["user"]},
                    ]
                    samples.append((messages, entry["osl_tokens"]))
            rng = random.Random(self.random_seed)
            rng.shuffle(samples)
            self._samples = samples
            self._available = list(self._samples)

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            if not self._available:
                self._available = list(self._samples)
                self._rng.shuffle(self._available)
            messages, osl_tokens = self._available.pop()
        return BenchmarkRequest(messages=messages, max_tokens=osl_tokens)


class ShareGPTDataset(BaseDataset):
    """
    Loads real conversations from ShareGPT dataset.

    Each request gets max_tokens = the actual assistant reply length for that
    conversation (estimated), filtered to be within realistic ISL/OSL bounds.
    This gives the server a natural ISL/OSL distribution rather than a fixed
    target for every request.

    Improvements over original version:
    - Returns BenchmarkRequest with per-request max_tokens
    - Filters by ISL and OSL bounds (not fixed targets)
    - Stores (messages, osl) tuples so max_tokens varies per request
    - Cycles through samples in random order each pass
    - Uses threading.Lock for thread safety
    """

    def __init__(
        self,
        num_prompts: int = 1000,
        random_seed: int = 42,
        system_prompt: str = "You are a helpful assistant.",
        max_isl_tokens: int = 8192,   # filter: skip conversations where user msg > this
        max_osl_tokens: int = 2048,   # filter: skip conversations where assistant reply > this
        min_osl_tokens: int = 50,     # filter: skip very short replies
        max_total_tokens: Optional[int] = None,
        context_safety_margin_tokens: int = 0,
        tokenizer_name: str = "",
    ):
        self.num_prompts = num_prompts
        self.random_seed = random_seed
        self.system_prompt = system_prompt
        self.max_isl_tokens = max_isl_tokens
        self.max_osl_tokens = max_osl_tokens
        self.min_osl_tokens = min_osl_tokens
        self.max_total_tokens = max_total_tokens
        self.context_safety_margin_tokens = context_safety_margin_tokens
        self.tokenizer_name = tokenizer_name
        self._samples: Optional[list[tuple]] = None   # list of (messages, osl)
        self._available: Optional[list[tuple]] = None
        self._lock = threading.Lock()
        self._rng = random.Random(random_seed)

    @staticmethod
    def _prompt_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
        try:
            return len(tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            ))
        except Exception:
            text = "\n".join(str(message.get("content", "")) for message in messages)
            return len(tokenizer.encode(text, add_special_tokens=True))

    def _load(self):
        if self._samples is not None:
            return
        with self._lock:
            if self._samples is not None:
                return
            import datasets as hf_datasets
            ds = hf_datasets.load_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                split="train",
            )
            tokenizer = None
            if self.tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                except Exception:
                    tokenizer = None
            samples = []
            for item in ds:
                convs = item["conversations"]
                if len(convs) < 2:
                    continue
                # Find first user+assistant pair
                user_msg = None
                assistant_msg = None
                for i, turn in enumerate(convs):
                    if turn.get("from") == "human" and user_msg is None:
                        user_msg = turn.get("value", "")
                    elif turn.get("from") == "gpt" and user_msg is not None:
                        assistant_msg = turn.get("value", "")
                        break
                if not user_msg or not assistant_msg:
                    continue
                # Estimate token counts (word-to-token ratio for English)
                isl_est = int(len(user_msg.split()) * 1.35)
                osl_est = int(len(assistant_msg.split()) * 1.35)
                if isl_est > self.max_isl_tokens:
                    continue
                if osl_est > self.max_osl_tokens:
                    continue
                if osl_est < self.min_osl_tokens:
                    continue
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": user_msg})
                if self.max_total_tokens is not None:
                    prompt_tokens = (
                        self._prompt_token_count(tokenizer, messages)
                        if tokenizer is not None
                        else isl_est
                    )
                    if (
                        prompt_tokens
                        + osl_est
                        + self.context_safety_margin_tokens
                        > self.max_total_tokens
                    ):
                        continue
                samples.append((messages, osl_est))
                if len(samples) >= self.num_prompts * 3:  # load 3x, shuffle, take num_prompts
                    break

            rng = random.Random(self.random_seed)
            rng.shuffle(samples)
            self._samples = samples[:self.num_prompts]
            self._available = list(self._samples)

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            if not self._available:
                self._available = list(self._samples)
                self._rng.shuffle(self._available)
            messages, osl = self._available.pop()
        return BenchmarkRequest(messages=messages, max_tokens=osl)


class RandomTokenDataset(BaseDataset):
    """
    Replicates InferenceX's random token workload for cross-validation.

    Generates random token IDs using the same algorithm as InferenceX
    (SemiAnalysisAI/InferenceX utils/bench_serving/benchmark_serving.py),
    decodes them to text via the tokenizer, and wraps in a chat message.

    Purpose: verify inference-benchmark produces the same TTFT/TPOT/E2EL
    as InferenceX when given identical inputs. Not for production benchmarking
    — random tokens trigger EOS early and give unreliable output lengths.
    """

    def __init__(
        self,
        tokenizer_name: str,
        input_len: int = 1024,
        output_len: int = 1024,
        num_prompts: int = 500,
        range_ratio: float = 1.0,
        seed: int = 0,
    ):
        self.output_len = output_len
        self._prompts: Optional[list[str]] = None
        self._idx = 0
        self._lock = threading.Lock()
        self._tokenizer_name = tokenizer_name
        self._input_len = input_len
        self._num_prompts = num_prompts
        self._range_ratio = range_ratio
        self._seed = seed

    def _load(self):
        if self._prompts is not None:
            return
        with self._lock:
            if self._prompts is not None:
                return
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            rng = np.random.default_rng(self._seed)

            lo = max(1, int(self._input_len * (1 - self._range_ratio / 2)))
            hi = int(self._input_len * (1 + self._range_ratio / 2))
            input_lens = rng.integers(lo, hi + 1, size=self._num_prompts)
            offsets = rng.integers(0, tokenizer.vocab_size, size=self._num_prompts)

            prompts = []
            for i in range(self._num_prompts):
                tgt_len = int(input_lens[i])
                token_ids = [(int(offsets[i]) + i + j) % tokenizer.vocab_size
                             for j in range(tgt_len)]
                prompt = tokenizer.decode(token_ids)
                # Re-encode and trim/pad to hit exact target length (InferenceX does this too)
                re_encoded = tokenizer.encode(prompt, add_special_tokens=False)
                if len(re_encoded) > tgt_len:
                    re_encoded = re_encoded[:tgt_len]
                elif len(re_encoded) < tgt_len:
                    extras = rng.integers(0, tokenizer.vocab_size,
                                         size=tgt_len - len(re_encoded)).tolist()
                    re_encoded.extend(extras)
                prompts.append(tokenizer.decode(re_encoded))

            self._prompts = prompts

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            prompt = self._prompts[self._idx % len(self._prompts)]
            self._idx += 1
        return BenchmarkRequest(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.output_len,
        )


class RandomTokenDatasetLegacy(BaseDataset):
    """
    Exact replication of InferenceX's random token generation using legacy numpy RNG.

    Uses np.random.seed(seed) + np.random.randint (same as InferenceX's
    sample_random_requests), with the identical token formula:
      token_ids[j] = (offsets[i] + i + j) % vocab_size

    If TTFT matches InferenceX with this dataset, it confirms the TTFT gap
    between inference-benchmark and InferenceX is purely due to RNG differences
    producing different prefix cache hit rates.
    """

    def __init__(
        self,
        tokenizer_name: str,
        input_len: int = 1024,
        output_len: int = 1024,
        num_prompts: int = 500,
        range_ratio: float = 1.0,
        prefix_len: int = 0,
        seed: int = 0,
    ):
        self.output_len = output_len
        self._tokenizer_name = tokenizer_name
        self._input_len = input_len
        self._num_prompts = num_prompts
        self._range_ratio = range_ratio
        self._prefix_len = prefix_len
        self._seed = seed
        self._prompts: Optional[list[str]] = None
        self._idx = 0
        self._lock = threading.Lock()

    def _load(self):
        if self._prompts is not None:
            return
        with self._lock:
            if self._prompts is not None:
                return
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            vocab_size = tokenizer.vocab_size

            # Match InferenceX exactly: np.random.seed then legacy randint
            np.random.seed(self._seed)
            prefix_token_ids = np.random.randint(0, vocab_size, size=self._prefix_len).tolist()

            lo = int(self._input_len * self._range_ratio)
            hi = self._input_len
            input_lens = np.random.randint(lo, hi + 1, size=self._num_prompts).tolist()
            _output_lens = np.random.randint(
                int(self.output_len * self._range_ratio),
                self.output_len + 1,
                size=self._num_prompts,
            ).tolist()
            offsets = np.random.randint(0, vocab_size, size=self._num_prompts)

            prompts = []
            for i in range(self._num_prompts):
                tgt_len = self._prefix_len + input_lens[i]
                token_ids = prefix_token_ids + [
                    (int(offsets[i]) + i + j) % vocab_size
                    for j in range(input_lens[i])
                ]
                prompt = tokenizer.decode(token_ids)
                # Re-encode and trim/pad (same as InferenceX)
                for _ in range(10):
                    re_encoded = tokenizer.encode(prompt, add_special_tokens=False)
                    if len(re_encoded) < tgt_len:
                        extras = np.random.randint(0, vocab_size, size=tgt_len - len(re_encoded)).tolist()
                        re_encoded.extend(extras)
                    elif len(re_encoded) > tgt_len:
                        re_encoded = re_encoded[:tgt_len]
                    else:
                        break
                    prompt = tokenizer.decode(re_encoded)
                prompts.append(prompt)

            self._prompts = prompts

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            prompt = self._prompts[self._idx % len(self._prompts)]
            self._idx += 1
        return BenchmarkRequest(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.output_len,
        )


class RandomTokenDatasetDoubleWrap(RandomTokenDataset):
    """
    Replicates InferenceX's double-chat-template bug for comparison.

    InferenceX with --backend openai-chat --use-chat-template pre-applies
    the chat template in sample_random_requests(), then sends the resulting
    formatted string as the 'content' of a user message to /v1/chat/completions.
    vLLM then applies the template a second time, making the effective prefill
    longer than intended.

    This class reproduces that exact behavior so we can measure the TTFT
    increase caused by the double-wrap vs our correct single-wrap approach.
    Use alongside random-1k profile to confirm the theory.
    """

    def _load(self):
        super()._load()
        # Cache tokenizer for chat template application
        if not hasattr(self, '_tokenizer_obj'):
            from transformers import AutoTokenizer
            self._tokenizer_obj = AutoTokenizer.from_pretrained(self._tokenizer_name)

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            prompt = self._prompts[self._idx % len(self._prompts)]
            self._idx += 1

        # Pre-apply chat template exactly as InferenceX does
        pre_formatted = self._tokenizer_obj.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        # Send pre-formatted string as user content → vLLM wraps it again
        return BenchmarkRequest(
            messages=[{"role": "user", "content": pre_formatted}],
            max_tokens=self.output_len,
        )


@dataclass
class MultiTurnSession:
    """A single multi-turn conversation: list of requests with growing history."""
    session_id: int
    turns: list[BenchmarkRequest]  # turn[0] = 1 msg, turn[1] = 3 msgs, turn[2] = 5 msgs, ...


class ShareGPTMultiTurnDataset(BaseDataset):
    """
    Loads multi-turn conversations from ShareGPT for multi-turn benchmarking.

    Instead of extracting only the first human+GPT pair, extracts full conversations
    and builds growing-history request sequences:
      Turn 1: [system, human1]                                        → max_tokens=osl1
      Turn 2: [system, human1, assistant1, human2]                    → max_tokens=osl2
      Turn 3: [system, human1, assistant1, human2, assistant2, human3] → max_tokens=osl3

    Assistant replies come from ShareGPT's pre-recorded GPT responses (Option B design),
    making requests deterministic and reproducible. The growing context tests prefix cache
    reuse — the server should recognize the shared prefix from earlier turns.

    Sessions are served round-robin for interleaved scheduling in the runner.
    """

    def __init__(
        self,
        min_turns: int = 3,
        max_turns: int = 10,
        num_sessions: int = 200,
        random_seed: int = 42,
        system_prompt: str = "You are a helpful assistant.",
        max_isl_tokens: int = 8192,   # max total context per turn (growing)
        max_osl_tokens: int = 2048,   # max assistant reply length per turn
        min_osl_tokens: int = 50,
    ):
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.num_sessions = num_sessions
        self.random_seed = random_seed
        self.system_prompt = system_prompt
        self.max_isl_tokens = max_isl_tokens
        self.max_osl_tokens = max_osl_tokens
        self.min_osl_tokens = min_osl_tokens
        self._sessions: Optional[list[MultiTurnSession]] = None
        self._lock = threading.Lock()
        # For get_next_request() fallback — flattened round-robin iterator
        self._flat_requests: Optional[list[BenchmarkRequest]] = None
        self._flat_available: Optional[list[BenchmarkRequest]] = None
        self._rng = random.Random(random_seed)

    def _load(self):
        if self._sessions is not None:
            return
        with self._lock:
            if self._sessions is not None:
                return
            import datasets as hf_datasets
            ds = hf_datasets.load_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                split="train",
            )
            sessions = []
            for item in ds:
                convs = item["conversations"]
                if len(convs) < self.min_turns * 2:
                    continue

                # Extract all human+assistant pairs
                pairs = []  # list of (human_msg, assistant_msg, osl_est)
                human_msg = None
                for turn in convs:
                    if turn.get("from") == "human":
                        human_msg = turn.get("value", "")
                    elif turn.get("from") == "gpt" and human_msg is not None:
                        assistant_msg = turn.get("value", "")
                        osl_est = int(len(assistant_msg.split()) * 1.35)
                        if osl_est > self.max_osl_tokens or osl_est < self.min_osl_tokens:
                            break  # stop at first bad turn
                        pairs.append((human_msg, assistant_msg, osl_est))
                        human_msg = None

                if len(pairs) < self.min_turns:
                    continue

                # Truncate to max_turns
                pairs = pairs[:self.max_turns]

                # Build growing-history requests
                turns = []
                messages_so_far = []
                if self.system_prompt:
                    messages_so_far.append({"role": "system", "content": self.system_prompt})

                total_est_tokens = 0
                for human_msg, assistant_msg, osl_est in pairs:
                    messages_so_far.append({"role": "user", "content": human_msg})
                    total_est_tokens += int(len(human_msg.split()) * 1.35)

                    if total_est_tokens > self.max_isl_tokens:
                        break  # context too long

                    turns.append(BenchmarkRequest(
                        messages=list(messages_so_far),  # snapshot
                        max_tokens=osl_est,
                    ))

                    # Append assistant reply for next turn's history
                    messages_so_far.append({"role": "assistant", "content": assistant_msg})
                    total_est_tokens += osl_est

                if len(turns) < self.min_turns:
                    continue

                sessions.append(MultiTurnSession(
                    session_id=len(sessions),
                    turns=turns,
                ))

                if len(sessions) >= self.num_sessions * 3:
                    break

            rng = random.Random(self.random_seed)
            rng.shuffle(sessions)
            self._sessions = sessions[:self.num_sessions]

            # Build flattened round-robin for get_next_request() compatibility
            self._build_flat_requests()

    def _build_flat_requests(self):
        """Build interleaved round-robin: [A1, B1, C1, A2, B2, C2, ...]"""
        if not self._sessions:
            self._flat_requests = []
            self._flat_available = []
            return
        max_num_turns = max(len(s.turns) for s in self._sessions)
        flat = []
        for turn_idx in range(max_num_turns):
            for session in self._sessions:
                if turn_idx < len(session.turns):
                    flat.append(session.turns[turn_idx])
        self._flat_requests = flat
        self._flat_available = list(flat)

    @property
    def sessions(self) -> list[MultiTurnSession]:
        self._load()
        return self._sessions

    def get_next_request(self) -> BenchmarkRequest:
        """Fallback for single-request dispatch — serves from flattened round-robin."""
        self._load()
        with self._lock:
            if not self._flat_available:
                self._flat_available = list(self._flat_requests)
            return self._flat_available.pop(0)


class TrajectoryMultiTurnDataset(BaseDataset):
    """
    Loads real multi-turn agent trajectories from pre-extracted JSONL files.

    Used for SWEBench and TerminalBench profiles. Each line in the JSONL is a
    complete session: {"session_id": ..., "turns": [{"messages": [...], "osl_tokens": N}, ...]}

    Filters sessions by turn count (min_turns, max_turns) and builds
    growing-history requests like ShareGPTMultiTurnDataset.
    """

    def __init__(
        self,
        filepath: str,
        min_turns: int = 3,
        max_turns: int = 30,
        num_sessions: int = 100,
        random_seed: int = 42,
        max_isl_tokens: int = 131072,
        max_osl_tokens: int = 2000,
        context_safety_margin_tokens: int = 0,
    ):
        self.filepath = filepath
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.num_sessions = num_sessions
        self.random_seed = random_seed
        self.max_isl_tokens = max_isl_tokens
        self.max_osl_tokens = max_osl_tokens
        self.context_safety_margin_tokens = context_safety_margin_tokens
        self._sessions: Optional[list[MultiTurnSession]] = None
        self._flat_requests: Optional[list[BenchmarkRequest]] = None
        self._flat_available: Optional[list[BenchmarkRequest]] = None
        self._lock = threading.Lock()
        self._rng = random.Random(random_seed)

    def _load(self):
        if self._sessions is not None:
            return
        with self._lock:
            if self._sessions is not None:
                return
            import json

            sessions = []
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    raw_turns = entry.get("turns", [])

                    if len(raw_turns) < self.min_turns:
                        continue

                    # Truncate to max_turns
                    raw_turns = raw_turns[:self.max_turns]

                    # Build BenchmarkRequest list, filtering by ISL/OSL bounds
                    turns = []
                    for t in raw_turns:
                        messages = t["messages"]
                        osl = min(t.get("osl_tokens", 500), self.max_osl_tokens)
                        # Estimate ISL from message content
                        isl_est = sum(
                            int(len(m.get("content", "").split()) * 1.35)
                            for m in messages
                        )
                        prompt_budget = (
                            self.max_isl_tokens
                            - osl
                            - self.context_safety_margin_tokens
                        )
                        if prompt_budget < 1 or isl_est > prompt_budget:
                            break
                        turn_index = t.get("turn_idx", t.get("turn_index", len(turns)))
                        turns.append(BenchmarkRequest(
                            messages=messages,
                            max_tokens=osl,
                            metadata={
                                "source_session_id": entry.get("session_id"),
                                "source_turn_index": turn_index,
                                "trace_content_source": entry.get("source"),
                                "planned_total_context_tokens": isl_est,
                                "planned_output_tokens": osl,
                                "planned_total_with_output_tokens": isl_est + osl,
                                "context_window_tokens": self.max_isl_tokens,
                                "context_safety_margin_tokens": (
                                    self.context_safety_margin_tokens
                                ),
                                "prompt_token_budget": prompt_budget,
                                "truncated_by_context_limit": False,
                            },
                        ))

                    if len(turns) < self.min_turns:
                        continue

                    sessions.append(MultiTurnSession(
                        session_id=len(sessions),
                        turns=turns,
                    ))

            rng = random.Random(self.random_seed)
            rng.shuffle(sessions)
            self._sessions = sessions[:self.num_sessions]
            self._build_flat_requests()

    def _build_flat_requests(self):
        """Build interleaved round-robin: [A1, B1, C1, A2, B2, C2, ...]"""
        if not self._sessions:
            self._flat_requests = []
            self._flat_available = []
            return
        max_num_turns = max(len(s.turns) for s in self._sessions)
        flat = []
        for turn_idx in range(max_num_turns):
            for session in self._sessions:
                if turn_idx < len(session.turns):
                    flat.append(session.turns[turn_idx])
        self._flat_requests = flat
        self._flat_available = list(flat)

    @property
    def sessions(self) -> list[MultiTurnSession]:
        self._load()
        return self._sessions

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            if not self._flat_available:
                self._flat_available = list(self._flat_requests)
            return self._flat_available.pop(0)


class DistributionalMultiTurnDataset(BaseDataset):
    """
    Builds synthetic growing-history sessions from compact trace distributions.

    This keeps the benchmark runner's existing multi-turn scheduling path while
    replacing expensive real-trace replay with sampled turn counts, input
    deltas, and output lengths.
    """

    def __init__(
        self,
        filepath: str,
        min_turns: int = 1,
        max_turns: Optional[int] = None,
        num_sessions: int = 10,
        random_seed: int = 42,
        max_context_tokens: Optional[int] = None,
        context_safety_margin_tokens: int = 256,
        system_prompt: str = "",
        tokenizer_name: str = "",
        source_session_ids: Optional[list[str]] = None,
    ):
        self.filepath = filepath
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.num_sessions = num_sessions
        self.random_seed = random_seed
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin_tokens = context_safety_margin_tokens
        self.system_prompt = system_prompt
        self.tokenizer_name = tokenizer_name
        self.source_session_ids = source_session_ids
        self._sessions: Optional[list[MultiTurnSession]] = None
        self._session_specs: Optional[dict[int, list]] = None
        self._flat_requests: Optional[list[BenchmarkRequest]] = None
        self._flat_available: Optional[list[BenchmarkRequest]] = None
        self._lock = threading.Lock()

    def _load(self):
        if self._sessions is not None:
            return
        with self._lock:
            if self._sessions is not None:
                return

            from .distributional import DistributionalSampler
            from .trace_distributions import load_trace_distribution

            distribution = load_trace_distribution(self.filepath)
            sampler = DistributionalSampler(
                distribution,
                seed=self.random_seed,
                min_turns=self.min_turns,
                max_turns=self.max_turns,
                max_context_tokens=self.max_context_tokens,
                context_safety_margin_tokens=self.context_safety_margin_tokens,
                system_prompt=self.system_prompt,
                tokenizer_name=self.tokenizer_name,
            )
            if self.source_session_ids:
                synthetic_sessions = sampler.sample_source_locked_sessions(self.source_session_ids)
            else:
                synthetic_sessions = sampler.sample_sessions(self.num_sessions)
            self._sessions = [
                MultiTurnSession(session_id=s.session_id, turns=s.turns)
                for s in synthetic_sessions
                if s.turns
            ]
            self._session_specs = {
                s.session_id: s.specs
                for s in synthetic_sessions
                if s.turns
            }
            self._build_flat_requests()

    def _build_flat_requests(self):
        """Build interleaved round-robin: [A1, B1, C1, A2, B2, C2, ...]"""
        if not self._sessions:
            self._flat_requests = []
            self._flat_available = []
            return
        max_num_turns = max(len(s.turns) for s in self._sessions)
        flat = []
        for turn_idx in range(max_num_turns):
            for session in self._sessions:
                if turn_idx < len(session.turns):
                    flat.append(session.turns[turn_idx])
        self._flat_requests = flat
        self._flat_available = list(flat)

    @property
    def sessions(self) -> list[MultiTurnSession]:
        self._load()
        return self._sessions

    @property
    def session_specs(self) -> dict[int, list]:
        self._load()
        return self._session_specs or {}

    def get_next_request(self) -> BenchmarkRequest:
        self._load()
        with self._lock:
            if not self._flat_available:
                self._flat_available = list(self._flat_requests)
            return self._flat_available.pop(0)


def make_dataset(
    profile,
    max_context_tokens: Optional[int] = None,
    random_seed: int = 42,
    context_safety_margin_tokens: int = 256,
    num_sessions: Optional[int] = None,
    tokenizer_name: str = "",
    source_session_ids: Optional[list[str]] = None,
) -> BaseDataset:
    """Factory: create the right dataset for a workload profile."""
    from .profiles import WorkloadProfile
    effective_num_sessions = profile.num_sessions if num_sessions is None else num_sessions

    if profile.dataset == "test":
        return TestDataset()
    elif profile.dataset == "file":
        return FileDataset(
            filepath=profile.file_path,
            system_prompt=profile.system_prompt,
            max_tokens=profile.osl_tokens,
        )
    elif profile.dataset == "sharegpt":
        return ShareGPTDataset(
            num_prompts=1000,
            random_seed=random_seed,
            system_prompt=profile.system_prompt,
            max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            max_total_tokens=max_context_tokens,
            context_safety_margin_tokens=context_safety_margin_tokens,
            tokenizer_name=tokenizer_name or profile.tokenizer_name,
        )
    elif profile.dataset == "random":
        return RandomTokenDataset(
            tokenizer_name=profile.tokenizer_name,
            input_len=profile.isl_tokens,
            output_len=profile.osl_tokens,
            num_prompts=500,
            seed=random_seed,
        )
    elif profile.dataset == "random-legacy":
        return RandomTokenDatasetLegacy(
            tokenizer_name=profile.tokenizer_name,
            input_len=profile.isl_tokens,
            output_len=profile.osl_tokens,
            num_prompts=500,
            seed=random_seed,
        )
    elif profile.dataset == "random-doublewrap":
        return RandomTokenDatasetDoubleWrap(
            tokenizer_name=profile.tokenizer_name,
            input_len=profile.isl_tokens,
            output_len=profile.osl_tokens,
            num_prompts=500,
            seed=random_seed,
        )
    elif profile.dataset == "sharegpt-multi-turn":
        return ShareGPTMultiTurnDataset(
            min_turns=profile.min_turns,
            max_turns=profile.max_turns,
            num_sessions=effective_num_sessions,
            random_seed=random_seed,
            system_prompt=profile.system_prompt,
            max_isl_tokens=profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
        )
    elif profile.dataset in ("swebench-multi-turn", "terminalbench-multi-turn", "osworld-multi-turn"):
        return TrajectoryMultiTurnDataset(
            filepath=profile.file_path,
            min_turns=profile.min_turns,
            max_turns=profile.max_turns,
            num_sessions=effective_num_sessions,
            random_seed=random_seed,
            max_isl_tokens=max_context_tokens or profile.isl_tokens,
            max_osl_tokens=profile.osl_tokens,
            context_safety_margin_tokens=context_safety_margin_tokens,
        )
    elif profile.dataset == "distributional-multi-turn":
        return DistributionalMultiTurnDataset(
            filepath=profile.file_path,
            min_turns=profile.min_turns,
            max_turns=profile.max_turns,
            num_sessions=effective_num_sessions,
            random_seed=random_seed,
            max_context_tokens=max_context_tokens or profile.isl_tokens,
            context_safety_margin_tokens=context_safety_margin_tokens,
            system_prompt=profile.system_prompt,
            tokenizer_name=tokenizer_name or profile.tokenizer_name,
            source_session_ids=source_session_ids,
        )
    elif profile.dataset == "jsonl":
        return JsonlDataset(
            filepath=profile.file_path,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown dataset type: {profile.dataset}")
