"""CPU stand-in for the part of vLLM a grid run touches; generate() runs on a fake clock."""

__version__ = "0.0.0+stub"

CALLS = []


class StubHalt(Exception):
    """Ends a capture run."""


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.generate_calls = 0

    def perf_counter(self):
        return self.now


CLOCK = FakeClock()


def generate_seconds(prompt_len, batch, max_tokens, call_no):
    """Tests replace it."""
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
    """None at position 0, then the top-k tokens plus the prompt token per position."""
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
    """Also run by the golden harness's HTTP stand-in."""
    CLOCK.generate_calls += 1
    prompt_len = len(prompts[0]["prompt_token_ids"])
    n = sampling_params.max_tokens
    CLOCK.now += generate_seconds(prompt_len, len(prompts), n, CLOCK.generate_calls)
    k = sampling_params.kwargs.get("prompt_logprobs")
    return [_RequestOutput(list(range(n)), k and _prompt_logprobs(p["prompt_token_ids"], k)) for p in prompts]


class LLM:
    def __init__(self, **kwargs):
        CALLS.append(("LLM", kwargs))

    def generate(self, prompts, sampling_params, use_tqdm=True):
        return run_batch(prompts, sampling_params)

    def get_tokenizer(self):
        raise StubHalt("corpus prompt mode needs a real tokenizer")
