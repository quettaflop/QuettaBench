"""Wrapped profiler sources: board energy, dollar cost and gsm8k accuracy."""

import json
import re
import subprocess
import threading
import urllib.request

from quettabench.engines import launcher

GSM8K_MAX_TOKENS = 256


def integrate_power(samples_w, interval_s):
    """A missed sample makes it a floor."""
    return sum(samples_w) * interval_s


def energy_devices(engine_json, config):
    """ "" means every GPU."""
    env = ((engine_json or {}).get("launch_spec") or {}).get("env") or {}
    return env.get("CUDA_VISIBLE_DEVICES") or ((config or {}).get("hardware") or {}).get("visible_devices") or ""


def device_label(ids):
    return len(ids.split(",")) if ids else "all"


class PowerSampler:
    """A failed or unparsable nvidia-smi call skips that tick."""

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
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        return self.samples


def energy_section(samples_w, interval_s, out_tokens, devices):
    joules = integrate_power(samples_w, interval_s)
    return {
        "joules": joules,
        "joules_per_token": joules / max(out_tokens, 1),
        "scope": "gpu_board",
        "devices": device_label(devices),
        "samples": len(samples_w),
        "interval_s": interval_s,
    }


def cost_per_m_tokens(gpu_cost_hr, devices, wall_s, out_tokens):
    """Flat per-GPU-hour rate over the replay window, excluding model load."""
    usd = gpu_cost_hr * devices * wall_s / 3600.0
    return usd, usd * 1e6 / max(out_tokens, 1)


def cost_section(gpu_cost_hr, devices, wall_s, out_tokens):
    """devices is tp * pp."""
    usd, per_m = cost_per_m_tokens(gpu_cost_hr, devices, wall_s, out_tokens)
    return {
        "usd": usd,
        "usd_per_m_out_tokens": per_m,
        "gpu_cost_hr": float(gpu_cost_hr),
        "devices": devices,
        "wall_s": wall_s,
        "out_tokens": out_tokens,
    }


def gsm8k_answer(text):
    """Gold answers sit after '####'; None when no number appears."""
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def load_gsm8k(path, tokenizer_name, n, tokenizer=None):
    """tokenizer replaces loading tokenizer_name."""
    if tokenizer is None:
        from transformers import AutoTokenizer  # imported here: slow to load

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    with open(path) as fh:
        items = [json.loads(line) for line in fh if line.strip()][:n]
    return [(tokenizer(it["question"])["input_ids"], gsm8k_answer(it["answer"])) for it in items]


def gsm8k_accuracy(generate, items, max_tokens=GSM8K_MAX_TOKENS):
    correct = 0
    for ids, gold in items:
        got = gsm8k_answer(generate(ids, max_tokens))
        correct += got is not None and got == gold
    n = len(items)
    return {
        "dataset": "gsm8k",
        "correct": correct,
        "n": n,
        "exact_match": correct / max(n, 1),
        "max_tokens": max_tokens,
    }


def http_generate(endpoint, model, timeout_s=600):
    """Greedy, non-streaming."""

    def generate(prompt_ids, max_tokens):
        body = json.dumps(
            {"model": model, "prompt": list(prompt_ids), "max_tokens": max_tokens, "temperature": 0.0}
        ).encode()
        req = urllib.request.Request(
            f"{endpoint}/v1/completions", data=body, headers={"Content-Type": "application/json"}
        )
        with launcher.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())["choices"][0]["text"]

    return generate
