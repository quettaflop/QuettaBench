"""CPU stand-in for the two torch.cuda calls InProcessDriver.device makes,
and for the tensor ops ReferenceDriver.score runs, on nested lists.

toy_logits is the one model both the transformers and the vllm stubs
score with, so their summaries must agree position for position.
"""

import contextlib
import math
from collections import namedtuple

bfloat16, float16, float32 = "bfloat16", "float16", "float32"
_TopK = namedtuple("_TopK", "values indices")


def toy_logits(ids):
    """Row p scores the token after ids[p] over a vocab of max(ids) + 2:
    ids[p] + 1 wins by 0.5."""
    vocab = max(ids) + 2
    return [[-0.5 * (v - t - 1) ** 2 for v in range(vocab)] for t in ids]


def log_softmax_row(row):
    top = max(row)
    norm = top + math.log(sum(math.exp(x - top) for x in row))
    return [x - norm for x in row]


class Tensor:
    def __init__(self, data):
        self.data = data

    def float(self):
        return self

    def tolist(self):
        return self.data

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, col = index
            if isinstance(rows, slice):
                return Tensor([r[col] for r in self.data[rows]])
            return self.data[rows][col]
        return Tensor(self.data[index])

    def __sub__(self, other):
        return Tensor([a - b for a, b in zip(self.data, other.data)])

    def topk(self, k, dim=-1):
        order = [sorted(range(len(r)), key=lambda v, r=r: -r[v])[:k] for r in self.data]
        return _TopK(Tensor([[r[v] for v in o] for r, o in zip(self.data, order)]), Tensor(order))


def tensor(data, device=None):
    return Tensor(data)


def log_softmax(t, dim=-1):
    return Tensor([log_softmax_row(r) for r in t.data])


def no_grad():
    return contextlib.nullcontext()


GPU_NAME = "STUB-GPU-48GiB"
GPU_BYTES = 48 * (1 << 30)


class _Props:
    total_memory = GPU_BYTES


class cuda:
    @staticmethod
    def get_device_name(index):
        return GPU_NAME

    @staticmethod
    def get_device_properties(index):
        return _Props()
