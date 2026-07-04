"""Distributional synthetic multi-turn workload sampling.

This module backs the canonical distributional multi-turn profiles. It turns
compact trace distributions into synthetic growing-history sessions whose
prompt deltas can be audited and recorded by the benchmark runner.
"""

from __future__ import annotations

import math
import os
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .dataset import BenchmarkRequest
from .trace_distributions import TraceDistribution, TraceTurnSample


DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 256
DEFAULT_SHARED_PREFIX_TOKENS = 1024
DEFAULT_PREFIX_CACHE_BLOCK_SIZE = 16

# English filler words that tokenize ~1:1 in Llama-family BPE tokenizers.
# Used when no real tokenizer is available. Avoids the code-label subword
# explosion that the old "s0_t0_user_42" pattern caused.
_FILLER_WORDS = [
    "benchmark", "filler", "text", "sequence", "data", "sample", "test",
    "measurement", "calibration", "workload", "throughput", "latency",
    "system", "performance", "evaluation", "profiling", "analysis",
    "kernel", "memory", "compute", "bandwidth", "capacity", "configuration",
    "parameter", "request", "response", "token", "generation", "inference",
    "serving", "engine", "model", "hardware", "software", "pipeline",
    "session", "conversation", "message", "output", "input", "context",
]

_CODELIKE_SINGLE_TOKEN_CANDIDATES = [
    " if", " in", " is", " as", " or", " to", " of", " id", " fn", " ok",
    " err", " tmp", " val", " ctx", " obj", " cls", " def", " for", " try",
    " ret", " log", " src", " dst", " res", " req", " out", " row", " col",
    " key", " str", " int", " len", " set", " get", " put", " run", " add",
    " == ", " != ", " <= ", " >= ", " = ", " + ", " - ", " / ", " .",
    " ,", " :", " {", " }", " [", " ]", " (", " )", " #", " //",
]

DEFAULT_CODELIKE_CHARS_PER_TOKEN = 3.8


def _stable_text_seed(label: str) -> int:
    """Return a process-stable RNG seed for synthetic filler text."""
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _get_tokenizer(tokenizer_name: str):
    """Lazy-load a HuggingFace tokenizer. Cached per name."""
    import threading
    _cache = getattr(_get_tokenizer, "_cache", None)
    _lock = getattr(_get_tokenizer, "_lock", None)
    if _cache is None:
        _cache = {}
        _lock = threading.Lock()
        _get_tokenizer._cache = _cache
        _get_tokenizer._lock = _lock
    if tokenizer_name not in _cache:
        with _lock:
            if tokenizer_name not in _cache:
                from transformers import AutoTokenizer
                _cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    return _cache[tokenizer_name]


@dataclass(frozen=True)
class SyntheticTurnSpec:
    turn_index: int
    sampled_new_prefill_tokens: int
    actual_new_prefill_tokens: int
    cached_context_tokens: int
    total_context_tokens: int
    new_user_tokens: int
    output_tokens: int
    cache_hit_rate: float
    context_window_tokens: int | None = None
    context_safety_margin_tokens: int = 0
    prompt_token_budget: int | None = None
    planned_total_with_output_tokens: int | None = None
    truncated_by_context_limit: bool = False


@dataclass
class SyntheticSession:
    session_id: int
    turns: list[BenchmarkRequest]
    specs: list[SyntheticTurnSpec]


class DistributionalSampler:
    """Sample synthetic sessions from empirical trace distributions."""

    def __init__(
        self,
        distribution: TraceDistribution,
        *,
        seed: int = 42,
        min_turns: int = 1,
        max_turns: int | None = None,
        max_context_tokens: int | None = None,
        context_safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
        system_prompt: str = "",
        tokenizer_name: str = "",
    ):
        if not distribution.turn_counts:
            raise ValueError("Distribution has no turn-count samples")
        if not distribution.turns:
            raise ValueError("Distribution has no turn samples")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive when provided")
        if min_turns <= 0:
            raise ValueError("min_turns must be positive")
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive when provided")
        if max_turns is not None and min_turns > max_turns:
            raise ValueError("min_turns must be <= max_turns")
        if context_safety_margin_tokens < 0:
            raise ValueError("context_safety_margin_tokens must be non-negative")
        if (
            max_context_tokens is not None
            and context_safety_margin_tokens >= max_context_tokens
        ):
            raise ValueError("context_safety_margin_tokens must be smaller than max_context_tokens")

        self.distribution = distribution
        self.rng = random.Random(seed)
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin_tokens = context_safety_margin_tokens
        self.system_prompt = system_prompt
        self._turns_by_index = distribution.turns_by_index
        self._turn_counts = self._eligible_turn_counts(distribution.turn_counts)
        if not self._turn_counts:
            raise ValueError(
                f"Distribution has no turn-count samples compatible with "
                f"min_turns={min_turns}, max_turns={max_turns}"
            )
        self._source_sessions_by_id = self._eligible_source_sessions_by_id(
            self._group_source_sessions_by_id(distribution.turns)
        )
        self._source_sessions = tuple(self._source_sessions_by_id.values())
        self._tokenizer_name = tokenizer_name
        self._tokenizer = None
        self.synthetic_filler_style = os.environ.get(
            "DISTRIBUTIONAL_SYNTHETIC_STYLE",
            "english",
        ).strip().lower()
        self.target_chars_per_token = _env_float(
            "DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN",
            DEFAULT_CODELIKE_CHARS_PER_TOKEN,
        )
        requested_shared_prefix_tokens = _env_int(
            "DISTRIBUTIONAL_SHARED_PREFIX_TOKENS",
            DEFAULT_SHARED_PREFIX_TOKENS,
        )
        self.prefix_aware_synthetic = _env_bool("DISTRIBUTIONAL_PREFIX_AWARE", False)
        self.prefix_cache_block_size = max(
            1,
            _env_int("DISTRIBUTIONAL_PREFIX_BLOCK_SIZE", DEFAULT_PREFIX_CACHE_BLOCK_SIZE),
        )
        self.shared_prefix_requested_tokens = (
            max(0, requested_shared_prefix_tokens)
            if self.prefix_aware_synthetic
            else 0
        )
        self.shared_prefix_target_tokens = self._shared_prefix_target_tokens()

    def sample_session(self, session_id: int = 0) -> SyntheticSession:
        if self._source_sessions:
            samples = self.rng.choice(self._source_sessions)
            return self._build_session_from_samples(session_id=session_id, samples=samples)
        turn_count = self.rng.choice(self._turn_counts)
        return self.sample_session_with_turn_count(session_id=session_id, turn_count=turn_count)

    def sample_sessions(self, num_sessions: int) -> list[SyntheticSession]:
        if num_sessions <= 0:
            raise ValueError("num_sessions must be positive")
        if self._source_sessions:
            samples_by_session = self._sample_source_sessions_without_replacement(num_sessions)
            return [
                self._build_session_from_samples(session_id=i, samples=samples)
                for i, samples in enumerate(samples_by_session)
            ]
        return [self.sample_session(session_id=i) for i in range(num_sessions)]

    def sample_source_locked_sessions(self, source_session_ids: list[str]) -> list[SyntheticSession]:
        if not source_session_ids:
            raise ValueError("source_session_ids must be non-empty")
        if not self._source_sessions_by_id:
            raise ValueError("Distribution does not include source_session_id samples")

        missing = [
            source_session_id
            for source_session_id in source_session_ids
            if source_session_id not in self._source_sessions_by_id
        ]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Unknown source_session_id values: {preview}{suffix}")

        return [
            self._build_session_from_samples(
                session_id=i,
                samples=self._source_sessions_by_id[source_session_id],
            )
            for i, source_session_id in enumerate(source_session_ids)
        ]

    def sample_session_with_turn_count(self, *, session_id: int, turn_count: int) -> SyntheticSession:
        if turn_count <= 0:
            raise ValueError("turn_count must be positive")
        if turn_count < self.min_turns:
            raise ValueError("turn_count must be >= min_turns")
        if self.max_turns is not None:
            turn_count = min(turn_count, self.max_turns)
        samples = [self._sample_turn(turn_index) for turn_index in range(turn_count)]
        return self._build_session_from_samples(session_id=session_id, samples=samples)

    def _build_session_from_samples(
        self,
        *,
        session_id: int,
        samples: list[TraceTurnSample] | tuple[TraceTurnSample, ...],
    ) -> SyntheticSession:
        messages, shared_prefix_actual_tokens = self._initial_messages()

        turns: list[BenchmarkRequest] = []
        specs: list[SyntheticTurnSpec] = []
        previous_prompt_context = sum(self._tokenize(str(m.get("content", ""))) for m in messages)
        previous_output_tokens = 0

        for synthetic_turn_index, sample in enumerate(samples):
            output_tokens = max(1, sample.output_tokens)
            context_before_user = previous_prompt_context + previous_output_tokens
            new_user_tokens = max(1, sample.total_context_tokens - context_before_user)
            desired_total_context = context_before_user + new_user_tokens
            prompt_token_budget = self._prompt_token_budget(output_tokens)
            truncated = False

            if prompt_token_budget is not None and desired_total_context > prompt_token_budget:
                remaining = prompt_token_budget - context_before_user
                if remaining <= 0:
                    break
                new_user_tokens = max(1, remaining)
                desired_total_context = context_before_user + new_user_tokens
                truncated = True

            user_text = self._synthetic_text(
                f"s{session_id}_t{synthetic_turn_index}_user",
                new_user_tokens,
            )
            messages.append({"role": "user", "content": user_text})

            actual_total_context = context_before_user + new_user_tokens
            cached_context = previous_prompt_context
            actual_new_prefill = actual_total_context - cached_context
            cache_hit_rate = cached_context / actual_total_context if actual_total_context > 0 else 0.0

            turns.append(
                BenchmarkRequest(
                    messages=list(messages),
                    max_tokens=sample.output_tokens,
                    metadata={
                        "synthetic_session_id": session_id,
                        "synthetic_turn_index": synthetic_turn_index,
                        "sampled_turn_index": sample.turn_index,
                        "sampled_source_session_id": sample.source_session_id,
                        "sampled_token_source": sample.token_source,
                        "sampled_total_context_tokens": sample.total_context_tokens,
                        "sampled_new_prefill_tokens": sample.new_prefill_tokens,
                        "planned_new_prefill_tokens": actual_new_prefill,
                        "planned_cached_context_tokens": cached_context,
                        "planned_total_context_tokens": actual_total_context,
                        "planned_cache_hit_rate": round(cache_hit_rate, 6),
                        "planned_new_user_tokens": new_user_tokens,
                        "planned_output_tokens": output_tokens,
                        "planned_total_with_output_tokens": actual_total_context + output_tokens,
                        "context_window_tokens": self.max_context_tokens,
                        "context_safety_margin_tokens": self.context_safety_margin_tokens,
                        "prompt_token_budget": prompt_token_budget,
                        "truncated_by_context_limit": truncated,
                        "synthetic_filler_style": self.synthetic_filler_style,
                        "synthetic_target_chars_per_token": self.target_chars_per_token,
                        "prefix_aware_synthetic": self.prefix_aware_synthetic,
                        "shared_prefix_requested_tokens": self.shared_prefix_requested_tokens,
                        "shared_prefix_target_tokens": self.shared_prefix_target_tokens,
                        "shared_prefix_actual_tokens": shared_prefix_actual_tokens,
                        "shared_prefix_block_size": self.prefix_cache_block_size,
                        "shared_prefix_block_aligned": (
                            shared_prefix_actual_tokens > 0
                            and shared_prefix_actual_tokens % self.prefix_cache_block_size == 0
                        ),
                    },
                )
            )
            specs.append(
                SyntheticTurnSpec(
                    turn_index=synthetic_turn_index,
                    sampled_new_prefill_tokens=sample.new_prefill_tokens,
                    actual_new_prefill_tokens=actual_new_prefill,
                    cached_context_tokens=cached_context,
                    total_context_tokens=actual_total_context,
                    new_user_tokens=new_user_tokens,
                    output_tokens=output_tokens,
                    cache_hit_rate=cache_hit_rate,
                    context_window_tokens=self.max_context_tokens,
                    context_safety_margin_tokens=self.context_safety_margin_tokens,
                    prompt_token_budget=prompt_token_budget,
                    planned_total_with_output_tokens=actual_total_context + output_tokens,
                    truncated_by_context_limit=truncated,
                )
            )

            assistant_text = self._synthetic_text(
                f"s{session_id}_t{synthetic_turn_index}_assistant",
                output_tokens,
            )
            messages.append({"role": "assistant", "content": assistant_text})
            previous_prompt_context = actual_total_context
            previous_output_tokens = output_tokens

            if truncated:
                break

        return SyntheticSession(session_id=session_id, turns=turns, specs=specs)

    def _prompt_token_budget(self, output_tokens: int) -> int | None:
        """Return max prompt tokens after reserving output and tokenizer headroom."""
        if self.max_context_tokens is None:
            return None
        return self.max_context_tokens - output_tokens - self.context_safety_margin_tokens

    def _initial_messages(self) -> tuple[list[dict], int]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.shared_prefix_target_tokens <= 0:
            return messages, 0

        shared_prefix = self._synthetic_text(
            f"{self.distribution.name}_shared_apc_prefix",
            self.shared_prefix_target_tokens,
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                f"{messages[0].get('content', '')}\n\n{shared_prefix}"
            )
        else:
            messages.insert(0, {"role": "system", "content": shared_prefix})

        # This is content-token accounting, matching the existing synthetic
        # distribution accounting. Chat-template exactness is validated by the
        # server-side run, not assumed here.
        return messages, self._tokenize(shared_prefix)

    def _shared_prefix_target_tokens(self) -> int:
        if self.shared_prefix_requested_tokens <= 0:
            return 0

        first_turn_contexts = [
            sample.total_context_tokens
            for sample in self.distribution.turns
            if sample.turn_index == 0
        ]
        if not first_turn_contexts:
            first_turn_contexts = [
                sample.total_context_tokens
                for sample in self.distribution.turns
            ]
        if not first_turn_contexts:
            return 0

        base_system_tokens = self._tokenize(self.system_prompt) if self.system_prompt else 0
        # Leave at least one token for a session-specific first user message so
        # sampled first-turn token totals remain achievable.
        max_shared = max(0, min(first_turn_contexts) - base_system_tokens - 1)
        target = min(self.shared_prefix_requested_tokens, max_shared)
        if self.prefix_cache_block_size > 1:
            target = (target // self.prefix_cache_block_size) * self.prefix_cache_block_size
        return max(0, target)

    def _tokenize(self, text: str) -> int:
        """Return token count for text, using real tokenizer if available."""
        if self._tokenizer is None and self._tokenizer_name:
            self._tokenizer = _get_tokenizer(self._tokenizer_name)
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # Fallback: English words average ~1.35 tokens each
        words = text.split()
        return max(1, int(len(words) * 1.35)) if words else 0

    def _synthetic_text(self, label: str, target_tokens: int) -> str:
        """Return deterministic filler text with accurately measured token count."""
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self._tokenizer is None and self._tokenizer_name:
            self._tokenizer = _get_tokenizer(self._tokenizer_name)

        rng = random.Random(_stable_text_seed(label))
        if self._tokenizer is not None:
            if self.synthetic_filler_style in {"code", "codelike", "morphology"}:
                return _calibrated_morphology_text(
                    target_tokens,
                    self._tokenizer,
                    rng,
                    self.target_chars_per_token,
                )
            return _calibrated_text(label, target_tokens, self._tokenizer, rng)

        # No tokenizer: use English filler words (much better than old code labels)
        word_count = max(1, math.ceil(target_tokens / 1.35))
        words = _sample_filler_words(rng, word_count)
        return " ".join(words)

    @property
    def _has_tokenizer(self) -> bool:
        if self._tokenizer is None and self._tokenizer_name:
            self._tokenizer = _get_tokenizer(self._tokenizer_name)
        return self._tokenizer is not None

    def _sample_turn(self, turn_index: int) -> TraceTurnSample:
        candidates = self._turns_by_index.get(turn_index)
        if not candidates:
            candidates = self.distribution.turns
        return self.rng.choice(candidates)

    def _sample_source_sessions_without_replacement(
        self,
        num_sessions: int,
    ) -> list[tuple[TraceTurnSample, ...]]:
        selected: list[tuple[TraceTurnSample, ...]] = []
        while len(selected) < num_sessions:
            pool = list(self._source_sessions)
            self.rng.shuffle(pool)
            needed = num_sessions - len(selected)
            selected.extend(pool[:needed])
        return selected

    @staticmethod
    def _group_source_sessions_by_id(
        turns: tuple[TraceTurnSample, ...],
    ) -> dict[str, tuple[TraceTurnSample, ...]]:
        grouped: dict[str, list[TraceTurnSample]] = defaultdict(list)
        for turn in turns:
            if turn.source_session_id:
                grouped[turn.source_session_id].append(turn)
        return {
            source_session_id: tuple(sorted(samples, key=lambda sample: sample.turn_index))
            for source_session_id, samples in grouped.items()
            if samples
        }

    def _eligible_turn_counts(self, turn_counts: tuple[int, ...]) -> tuple[int, ...]:
        eligible: list[int] = []
        for count in turn_counts:
            if count < self.min_turns:
                continue
            if self.max_turns is not None:
                count = min(count, self.max_turns)
            eligible.append(count)
        return tuple(eligible)

    def _eligible_source_sessions_by_id(
        self,
        sessions_by_id: dict[str, tuple[TraceTurnSample, ...]],
    ) -> dict[str, tuple[TraceTurnSample, ...]]:
        eligible: dict[str, tuple[TraceTurnSample, ...]] = {}
        for source_session_id, samples in sessions_by_id.items():
            if len(samples) < self.min_turns:
                continue
            if self.max_turns is not None:
                samples = samples[: self.max_turns]
            eligible[source_session_id] = samples
        return eligible


def _calibrated_text(label: str, target_tokens: int, tokenizer, rng: random.Random) -> str:
    """Generate filler text and calibrate word count to hit target_tokens exactly.

    Uses binary search over word count, measuring with the real tokenizer each
    iteration. Final adjustment pads or trims to within ±2 tokens.
    """
    # Initial guess: 1 English word ≈ 1.35 tokens
    lo, hi = 1, target_tokens + 5  # upper bound: worst case, each word is 1 token
    best_text = ""
    for _ in range(12):
        mid = (lo + hi) // 2
        text = " ".join(_sample_filler_words(rng, mid))
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual == target_tokens:
            return text
        if actual < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
        best_text = text

    # Fine-tune: the label tokens are coarse (~4-6 tokens each), so pad/trim
    # with single-token words to hit the target precisely
    actual = len(tokenizer.encode(best_text, add_special_tokens=False))
    # Pad with single-token words
    _SINGLE_TOKEN_FILLER = [" a", " the", " is", " at", " on", " in", " to", " of", " and"]
    while actual < target_tokens:
        best_text += rng.choice(_SINGLE_TOKEN_FILLER)
        actual = len(tokenizer.encode(best_text, add_special_tokens=False))
    # Trim if overshot
    while actual > target_tokens:
        words = best_text.rsplit(" ", 1)
        if len(words) == 1:
            break
        best_text = words[0]
        actual = len(tokenizer.encode(best_text, add_special_tokens=False))

    return best_text


def _calibrated_morphology_text(
    target_tokens: int,
    tokenizer,
    rng: random.Random,
    target_chars_per_token: float,
) -> str:
    """Generate short-fragment filler that also targets chars/token.

    This is intended for code-agent traces where token-count matching with
    natural-language words creates 2x+ too many characters.
    """
    if target_chars_per_token <= 0:
        target_chars_per_token = DEFAULT_CODELIKE_CHARS_PER_TOKEN

    candidates = _single_token_codelike_fragments(tokenizer)
    if not candidates:
        return _calibrated_text("morphology-fallback", target_tokens, tokenizer, rng)

    pieces: list[str] = []
    chars = 0
    target_chars = max(1, round(target_tokens * target_chars_per_token))
    for index in range(target_tokens):
        remaining_tokens = max(1, target_tokens - index)
        desired_len = (target_chars - chars) / remaining_tokens
        best_distance = min(abs(len(fragment) - desired_len) for fragment in candidates)
        nearest = [
            fragment
            for fragment in candidates
            if abs(len(fragment) - desired_len) == best_distance
        ]
        fragment = rng.choice(nearest)
        pieces.append(fragment)
        chars += len(fragment)

    text = "".join(pieces)
    actual = len(tokenizer.encode(text, add_special_tokens=False))

    # BPE can occasionally merge/split across fragment boundaries. Repair token
    # count with the same one-token fragment pool while keeping char ratio close.
    while actual < target_tokens:
        desired_len = target_chars_per_token
        best_distance = min(abs(len(fragment) - desired_len) for fragment in candidates)
        nearest = [
            fragment
            for fragment in candidates
            if abs(len(fragment) - desired_len) == best_distance
        ]
        text += rng.choice(nearest)
        actual = len(tokenizer.encode(text, add_special_tokens=False))

    while actual > target_tokens and pieces:
        pieces.pop()
        text = "".join(pieces)
        actual = len(tokenizer.encode(text, add_special_tokens=False))

    while actual < target_tokens:
        text += min(candidates, key=lambda fragment: abs(len(fragment) - target_chars_per_token))
        actual = len(tokenizer.encode(text, add_special_tokens=False))

    return text


def _single_token_codelike_fragments(tokenizer) -> tuple[str, ...]:
    cache = getattr(_single_token_codelike_fragments, "_cache", None)
    if cache is None:
        cache = {}
        _single_token_codelike_fragments._cache = cache
    key = id(tokenizer)
    if key not in cache:
        cache[key] = tuple(
            fragment
            for fragment in _CODELIKE_SINGLE_TOKEN_CANDIDATES
            if len(tokenizer.encode(fragment, add_special_tokens=False)) == 1
        )
    return cache[key]


def _sample_filler_words(rng: random.Random, word_count: int) -> list[str]:
    return [rng.choice(_FILLER_WORDS) for _ in range(word_count)]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
