"""Every runnable profile, built by workloads.build on fixed inputs and a fixed
seed, pinned by its workload_hash and by a digest of everything a run sends
or records: messages, max_tokens, metadata, warmup draws, replies, source_meta.

The data files are small fixtures written at each profile's file_path under
a temporary working directory; ShareGPT (datasets) and the tokenizer
(transformers) are fakes, so nothing touches the network.
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from quettabench import workloads
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.workloads.profiles import PROFILES, get_profile

SEED = 7
NUM_REQUESTS = 12
WARMUP = 3

# name: (workload_hash, content digest)
PINS = {
    "chat-medium": ("2c2c330fe937283f", "73ab40e8748e8f7e"),
    "chat-multiturn-long": ("274162becf17c102", "bef891c6940e60db"),
    "chat-multiturn-medium": ("489bcc4b486495ac", "1b53f1e6c546d1e3"),
    "chat-multiturn-short": ("57c82245406bcdf1", "5e2d31c9032a7cb8"),
    "chat-short": ("e0b677fed285920c", "79bb39e7e120588f"),
    "chat-singleturn": ("2c2c330fe937283f", "84b2b4875f8a177f"),
    "chat-singleturn-synth": ("2c2c330fe937283f", "325e87f900350ef6"),
    "coding-singleturn": ("03481570266bb2d4", "0359e7a0be1623a2"),
    "decode-heavy": ("f0719d3492f4fd06", "77345bbb26805e62"),
    "fixed-seq128": ("05a41401be097be3", "c0a7d2444b3654da"),
    "mooncake-trace": ("0e6f083e59f1f9dd", "d483cd1411833245"),
    "osworld-multiturn-long": ("f741501038aeb62c", "7c281ff0cc772773"),
    "osworld-multiturn-medium": ("5af8ab8b94a75466", "c8ecbee41859e40e"),
    "osworld-multiturn-short": ("5fa7da0de14a64fb", "387a35fabdd98d42"),
    "prefill-heavy": ("8c28ae1fb55fecea", "832fe89da55a65fc"),
    "random-1k": ("fdaa0aa7451df053", "e90e87f402815bd2"),
    "random-1k-random-doublewrap": ("fdaa0aa7451df053", "c4e5d1efa3c5ee45"),
    "random-1k-random-legacy": ("fdaa0aa7451df053", "8cc8f5f07ec127ef"),
    "swebench-multiturn-long": ("b94b837574393d16", "ec78ec95a027a8e7"),
    "swebench-multiturn-medium": ("8cb100a908e8d018", "d4ba4dd7703c762a"),
    "swebench-multiturn-short": ("d4dca931e18de175", "2e44a22e09c81ea1"),
    "terminalbench-multiturn-long": ("0c2fa9aec3717f8c", "06dd564c6c54ed56"),
    "terminalbench-multiturn-medium": ("80bf8305e3891345", "2f91359db9f0a799"),
    "terminalbench-multiturn-short": ("a28027e756a8a273", "fb564412a27e3079"),
    "test": ("c19e2bb3db6d4434", "a5c809ace7fb7a08"),
    "cut": ("c769b8ce9e3a8fe2", "00808563928dac5d"),
}


def words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


def sharegpt_rows():
    rows = [{"conversations": [{"from": "human", "value": "lonely"}]},
            {"conversations": [{"from": "gpt", "value": words(80, "g")},
                               {"from": "human", "value": words(30, "h")}]}]
    for k in range(40):
        convs = []
        for t in range(1 + k % 12):
            convs.append({"from": "human", "value": words(20 + (k * 37 + t * 11) % 600, f"h{k}.{t}.")})
            convs.append({"from": "gpt", "value": words(30 + (k * 53 + t * 7) % 900, f"g{k}.{t}.")})
        rows.append({"conversations": convs})
    return rows


def trajectory_rows(tag):
    rows = []
    for k, n in enumerate((2, 5, 12, 15, 25, 35, 70, 90, 160, 9, 22)):
        turns, history = [], []
        for t in range(n):
            history = history + [{"role": "user", "content": words(5 + (k + t) % 40, f"{tag}{k}.{t}.")}]
            turn = {"messages": list(history), "osl_tokens": 50 + (k * 13 + t) % 3000}
            if t % 3 == 0:
                turn["turn_idx"] = t
            if t % 4 == 1:
                turn["tool_ms"] = 10 * t
            if t % 5 == 2:
                turn["human_wait_after_ms"] = 3 * t
            turns.append(turn)
        row = {"session_id": f"{tag}-{k}", "source": tag, "turns": turns}
        if k % 2:
            row["arrival_time_ms"] = 250 * k
        rows.append(row)
    return rows


def write_fixtures(root):
    files = {
        "data/coding_agent_prompts.jsonl": [
            {"system": f"sys{i}", "user": words(10 + i, f"u{i}."), "osl_tokens": 100 + i}
            for i in range(5)],
        "data/mooncake/conversation_trace.jsonl": [
            {"timestamp": 10 * i, "input_length": 700 + 300 * i, "output_length": 5 + i,
             "hash_ids": list(range(i % 3, i % 3 + 1 + i))} for i in range(4)],
        "data/swebench_trajectories.jsonl": trajectory_rows("swe"),
        "data/terminalbench_trajectories.jsonl": trajectory_rows("tb"),
        "data/osworld_trajectories.jsonl": trajectory_rows("os"),
    }
    for rel, rows in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class FakeTokenizer:
    """Lossy on purpose: encode drops ids divisible by 7, so the random-token
    profiles pad with their rng."""

    vocab_size = 1000

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split() if t.isdigit() and int(t) % 7]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return [0] * len(text.split()) if tokenize else text


FAKE_MODULES = {
    "datasets": types.SimpleNamespace(load_dataset=lambda *a, **k: sharegpt_rows()),
    "transformers": types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())),
}

MULTI_TURN = ("sharegpt-multi-turn", "swebench-multi-turn", "terminalbench-multi-turn",
              "osworld-multi-turn")


def config(name, kind, max_turn_index=None):
    return Config(engine=EngineCfg(model="fake-model"), hardware=HardwareCfg(),
                  workload=WorkloadCfg(kind=kind, source=name, seed=SEED,
                                       num_requests=NUM_REQUESTS, max_turn_index=max_turn_index),
                  mode=ModeCfg(concurrency=4, warmup=WARMUP))


def row(r):
    return [r.prompt_token_ids, r.prompt_len, r.order, r.output_length, r.max_tokens,
            r.session_id, r.turn, r.arrival_ts, r.seed, r.source, r.messages, r.metadata]


def digest(trace):
    record = {
        "requests": [row(r) for r in trace.requests()],
        "warmup": [row(r) for r in trace.warmup_requests(WARMUP)],
        "source_meta": trace.source_meta(),
        "max_turns": getattr(trace, "max_turns", None),
        "replies": {str(k): v for k, v in getattr(trace, "replies", {}).items()},
        "sessions": [[r.order for r in s] for s in trace.sessions()]
        if hasattr(trace, "sessions") else None,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:16]


@contextlib.contextmanager
def fixture_cwd():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, FAKE_MODULES):
        write_fixtures(td)
        os.chdir(td)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.chdir(old)


def build(name, max_turn_index=None):
    kind = "sessions" if get_profile(name).dataset in MULTI_TURN else "live"
    trace = workloads.build(config(name, kind, max_turn_index))
    return trace.workload_hash(), digest(trace)


def runnable():
    return sorted(n for n, p in PROFILES.items() if p.dataset != "distributional-multi-turn")


def variants():
    """The two InferenceX random-token variants no shipped profile names."""
    base = get_profile("random-1k")
    return {f"random-1k-{d}": dataclasses.replace(base, name=f"random-1k-{d}", dataset=d)
            for d in ("random-legacy", "random-doublewrap")}


class ProfileHashPins(unittest.TestCase):
    def test_every_runnable_profile_is_pinned(self):
        self.assertEqual(sorted(PINS), sorted(runnable() + list(variants()) + ["cut"]))

    def test_profiles_build_the_pinned_traces(self):
        with fixture_cwd(), mock.patch.dict(PROFILES, variants()):
            got = {name: build(name) for name in runnable() + list(variants())}
            got["cut"] = build("swebench-multiturn-medium", max_turn_index=3)
        for name, pin in PINS.items():
            with self.subTest(profile=name):
                self.assertEqual(got[name], pin)


if __name__ == "__main__":
    unittest.main()
