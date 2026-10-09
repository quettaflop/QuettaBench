"""CPU stand-in for the slice of vLLM the crossval scripts touch.

Records every constructor's kwargs in CALLS and runs generate() on a fake
clock, so a test can drive vmin_fit.main() end to end without a GPU. The
real code targets an unpinned vLLM (see tests/golden/REPORT.md); only the
names imported by vmin_fit.py and trace_serve.py exist here.
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


class _RequestOutput:
    def __init__(self, token_ids):
        self.outputs = [_Completion(token_ids)]


class LLM:
    def __init__(self, **kwargs):
        CALLS.append(("LLM", kwargs))

    def generate(self, prompts, sampling_params, use_tqdm=True):
        CLOCK.generate_calls += 1
        prompt_len = len(prompts[0]["prompt_token_ids"])
        n = sampling_params.max_tokens
        CLOCK.now += generate_seconds(prompt_len, len(prompts), n, CLOCK.generate_calls)
        return [_RequestOutput(list(range(n))) for _ in prompts]

    def get_tokenizer(self):
        raise StubHalt("corpus prompt mode needs a real tokenizer")
