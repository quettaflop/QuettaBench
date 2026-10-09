"""CPU stand-in for the transformers calls ReferenceDriver makes; records
them in CALLS. The model's logits are torch.toy_logits of its input."""

import types

import torch

__version__ = "0.0.0+stub"
CALLS = []


class _Model:
    def to(self, device):
        CALLS.append(("to", device))
        return self

    def eval(self):
        CALLS.append(("eval",))
        return self

    def __call__(self, t):
        return types.SimpleNamespace(logits=torch.Tensor([torch.toy_logits(t.data[0])]))


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(path, **kwargs):
        CALLS.append(("AutoModelForCausalLM.from_pretrained", path, kwargs))
        return _Model()


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **kwargs):
        CALLS.append(("AutoTokenizer.from_pretrained", path, kwargs))
        return object()
