"""CPU stand-in for the slice of vLLM the grid run touches.

Records every constructor's kwargs in CALLS and runs generate() on a fake
clock, so a test can drive a grid run (qbench run with mode.grid) end to end without a
GPU. The real code targets an unpinned vLLM (see tests/golden/REPORT.md);
only the names the InProcessDriver imports exist here.
"""

__version__ = "0.0.0+stub"

CALLS = []  # (constructor name, kwargs) in call order


class StubHalt(Exception):
    """Raised where a real engine would start serving; ends a capture run."""


class FakeClock:
    """perf_counter replacement that only moves when generate() spends time."""

    def __init__(self):
        self.now = 0.0
        self.generate_calls = 0

    def perf_counter(self):
        return self.now


CLOCK = FakeClock()


def generate_seconds(prompt_len, batch, max_tokens, call_no):
    """Seconds one generate() call costs on the fake clock; tests replace it."""
    return 0.0


def reset():
    CALLS.clear()
    CLOCK.now = 0.0
    CLOCK.generate_calls = 0


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_tokens = kwargs.get("max_tokens", 16)


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class Logprob:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def _prompt_logprobs(ids, k):
    """vLLM's shape: None at position 0, then per position the top-k tokens
    plus the prompt token, scored by torch.toy_logits of the prefix."""
    import torch
    out = [None]
    for pos in range(1, len(ids)):
        lps = torch.log_softmax_row(torch.toy_logits(ids)[pos - 1])
        ranked = sorted(range(len(lps)), key=lambda v: -lps[v])
        entry = {v: Logprob(lps[v], r + 1) for r, v in enumerate(ranked[:k])}
        entry.setdefault(ids[pos], Logprob(lps[ids[pos]], ranked.index(ids[pos]) + 1))
        out.append(entry)
    return out


class _RequestOutput:
    def __init__(self, token_ids, prompt_logprobs=None):
        self.outputs = [_Completion(token_ids)]
        self.prompt_logprobs = prompt_logprobs


def run_batch(prompts, sampling_params):
    """One batch on the fake clock: LLM.generate's body, which the golden
    harness's HTTP stand-in also runs for what a grid batch posts."""
    CLOCK.generate_calls += 1
    prompt_len = len(prompts[0]["prompt_token_ids"])
    n = sampling_params.max_tokens
    CLOCK.now += generate_seconds(prompt_len, len(prompts), n, CLOCK.generate_calls)
    k = sampling_params.kwargs.get("prompt_logprobs")
    return [_RequestOutput(list(range(n)), k and _prompt_logprobs(p["prompt_token_ids"], k))
            for p in prompts]


class LLM:
    def __init__(self, **kwargs):
        CALLS.append(("LLM", kwargs))

    def generate(self, prompts, sampling_params, use_tqdm=True):
        return run_batch(prompts, sampling_params)

    def get_tokenizer(self):
        raise StubHalt("corpus prompt mode needs a real tokenizer")
