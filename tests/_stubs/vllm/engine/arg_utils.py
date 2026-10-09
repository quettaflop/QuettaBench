from vllm import CALLS


class AsyncEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        CALLS.append(("AsyncEngineArgs", kwargs))
