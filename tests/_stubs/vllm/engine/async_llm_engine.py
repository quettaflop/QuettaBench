from vllm import StubHalt


class AsyncLLMEngine:
    @classmethod
    def from_engine_args(cls, engine_args):
        raise StubHalt("engine args captured; no engine on CPU")
