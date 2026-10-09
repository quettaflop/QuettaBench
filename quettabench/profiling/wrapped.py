"""Wrapped profiler sources: board energy, dollar cost and gsm8k accuracy.

Each wraps a tool or a dataset outside the engine. The numbers are
legacy/trace_serve.py's functions moved unchanged; what changed is where
inputs come from: devices from engine.json or the Config, never os.environ,
and gsm8k answers from any `generate(prompt_ids, max_tokens) -> text`
callable instead of a vLLM engine object.
"""

import json
import re
import subprocess
import threading
import urllib.request

GSM8K_MAX_TOKENS = 256


def integrate_power(samples_w, interval_s):
    """Rectangle-rule joules from per-sample total board watts. A missed sample
    shrinks the integral rather than crashing the run, so joules is a floor."""
    return sum(samples_w) * interval_s


def energy_devices(engine_json, config):
    """CUDA_VISIBLE_DEVICES the engine was launched with: engine.json's
    launch_spec env first, then the Config's hardware.visible_devices; ""
    means every GPU on the box."""
    env = ((engine_json or {}).get("launch_spec") or {}).get("env") or {}
    return env.get("CUDA_VISIBLE_DEVICES") or \
        ((config or {}).get("hardware") or {}).get("visible_devices") or ""


def device_label(ids):
    """ENERGYSUM's devices= field: the id count, or `all` when none were named."""
    return len(ids.split(",")) if ids else "all"


class PowerSampler:
    """Total board power over `devices` every interval_s, on a daemon thread.

    One nvidia-smi call per tick, summed across the listed GPUs. Board scope
    includes idle draw and everything colocated on those devices, which is
    why the energy section is labeled gpu_board. A failed call or an
    unparsable reply (OSError, ValueError) skips that tick on purpose: a
    missing sample makes joules a floor instead of killing the run.
    """

    def __init__(self, devices, nvidia_smi="nvidia-smi", interval_s=1.0):
        self.devices, self.interval_s = devices or "", interval_s
        self.cmd = [nvidia_smi, "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        if self.devices:
            self.cmd += ["-i", self.devices]
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(self.cmd, capture_output=True, text=True, check=False).stdout
                vals = [float(x) for x in out.split()]
                if vals:
                    self.samples.append(sum(vals))
            except (OSError, ValueError):
                pass

    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout_s=2.0):
        """Stop sampling; returns the samples taken."""
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        return self.samples


def energy_section(samples_w, interval_s, out_tokens, devices):
    """profile.json `energy`: joules, joules per output token, scope, devices, samples."""
    joules = integrate_power(samples_w, interval_s)
    return {"joules": joules, "joules_per_token": joules / max(out_tokens, 1),
            "scope": "gpu_board", "devices": device_label(devices),
            "samples": len(samples_w), "interval_s": interval_s}


def cost_per_m_tokens(gpu_cost_hr, devices, wall_s, out_tokens):
    """(run USD, USD per million output tokens) at a flat per-GPU-hour rate.
    Wall time is the replay window, so model-load time is excluded; including
    it would price the benchmark harness, not the serving."""
    usd = gpu_cost_hr * devices * wall_s / 3600.0
    return usd, usd * 1e6 / max(out_tokens, 1)


def cost_section(gpu_cost_hr, devices, wall_s, out_tokens):
    """profile.json `cost`; devices is tp * pp, as COSTSUM always used."""
    usd, per_m = cost_per_m_tokens(gpu_cost_hr, devices, wall_s, out_tokens)
    return {"usd": usd, "usd_per_m_out_tokens": per_m, "gpu_cost_hr": float(gpu_cost_hr),
            "devices": devices, "wall_s": wall_s, "out_tokens": out_tokens}


def gsm8k_answer(text):
    """Final numeric answer: the last number in the text, comma and dollar
    stripped; gold answers sit after '####'. Returns None when no number
    appears, which scores as wrong rather than crashing mid-benchmark."""
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def load_gsm8k(path, tokenizer_name, n, tokenizer=None):
    """[(prompt_ids, gold_answer)] for the first n items of a GSM8k JSONL
    ({question, answer}). Gold is the number after #### in each answer.
    tokenizer, when given, replaces loading tokenizer_name."""
    if tokenizer is None:
        from transformers import (
            AutoTokenizer,  # optional dependency, as in workloads/tokenize.py
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    with open(path) as fh:
        items = [json.loads(line) for line in fh if line.strip()][:n]
    return [(tokenizer(it["question"])["input_ids"], gsm8k_answer(it["answer"])) for it in items]


def gsm8k_accuracy(generate, items, max_tokens=GSM8K_MAX_TOKENS):
    """Greedy answers through generate(prompt_ids, max_tokens) -> text, scored
    by exact match of gsm8k_answer against gold. Returns profile.json
    `accuracy`. Task-level accuracy beside the token-level FIDELITY gate is
    the accuracy story: either alone can hide a fast-but-wrong config."""
    correct = 0
    for ids, gold in items:
        got = gsm8k_answer(generate(ids, max_tokens))
        correct += got is not None and got == gold
    n = len(items)
    return {"dataset": "gsm8k", "correct": correct, "n": n,
            "exact_match": correct / max(n, 1), "max_tokens": max_tokens}


def http_generate(endpoint, model, timeout_s=600):
    """generate(prompt_ids, max_tokens) -> text over a server's /v1/completions,
    greedy and non-streaming: the gsm8k callable for an attached server."""
    def generate(prompt_ids, max_tokens):
        body = json.dumps({"model": model, "prompt": list(prompt_ids), "max_tokens": max_tokens,
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(f"{endpoint}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())["choices"][0]["text"]
    return generate
