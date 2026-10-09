"""profiling/wrapped.py gives trace_serve's numbers on the same inputs.

Every case is hand-computed and also checked against the legacy function,
so a drift in either shows up here. nvidia-smi is the fake script next to
this file, passed by path; no GPU, no network beyond a local test server.
"""

import http.server
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from legacy import trace_serve
from quettabench.profiling import wrapped

FAKE_SMI = str(Path(__file__).with_name("fake_nvidia_smi.py"))


class Tokenizer:
    """Stand-in tokenizer: one id per character."""

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}


class Energy(unittest.TestCase):
    def test_integrate_power_is_rectangle_rule(self):
        samples = [300.0, 310.5, 290.25]
        self.assertEqual(wrapped.integrate_power(samples, 1.0), 900.75)
        self.assertEqual(wrapped.integrate_power(samples, 0.5), 450.375)
        self.assertEqual(wrapped.integrate_power(samples, 1.0),
                         trace_serve.integrate_power(samples, 1.0))
        self.assertEqual(wrapped.integrate_power([], 1.0), 0)

    def test_energy_section(self):
        e = wrapped.energy_section([300.0, 300.0], 1.0, out_tokens=200, devices="0,1")
        self.assertEqual(e, {"joules": 600.0, "joules_per_token": 3.0, "scope": "gpu_board",
                             "devices": 2, "samples": 2, "interval_s": 1.0})
        self.assertEqual(wrapped.energy_section([5.0], 1.0, 0, "")["joules_per_token"], 5.0)
        self.assertEqual(wrapped.energy_section([], 1.0, 0, "")["devices"], "all")

    def test_devices_from_engine_json_then_config(self):
        engine = {"launch_spec": {"env": {"CUDA_VISIBLE_DEVICES": "2,3"}}}
        config = {"hardware": {"visible_devices": "0"}}
        self.assertEqual(wrapped.energy_devices(engine, config), "2,3")
        self.assertEqual(wrapped.energy_devices({"launch_spec": {"env": {}}}, config), "0")
        self.assertEqual(wrapped.energy_devices({}, {"hardware": {"visible_devices": None}}), "")
        self.assertEqual(wrapped.energy_devices(None, None), "")


class Sampler(unittest.TestCase):
    def sample(self, devices, smi=FAKE_SMI):
        sampler = wrapped.PowerSampler(devices, nvidia_smi=smi, interval_s=0.05).start()
        deadline = time.monotonic() + 5
        while not sampler.samples and time.monotonic() < deadline:
            time.sleep(0.02)
        return sampler, sampler.stop()

    def test_sums_the_listed_devices(self):
        sampler, samples = self.sample("0,2")
        self.assertEqual(sampler.cmd, [FAKE_SMI, "--query-gpu=power.draw",
                                       "--format=csv,noheader,nounits", "-i", "0,2"])
        self.assertTrue(samples)
        self.assertEqual(set(samples), {220.0})

    def test_no_devices_means_every_gpu(self):
        sampler, samples = self.sample("")
        self.assertNotIn("-i", sampler.cmd)
        self.assertEqual(set(samples), {460.0})

    def test_bad_reply_or_missing_binary_is_skipped(self):
        for smi in ("/bin/echo", "/nonexistent/nvidia-smi"):
            with self.subTest(smi):
                sampler = wrapped.PowerSampler("0", nvidia_smi=smi, interval_s=0.02).start()
                time.sleep(0.15)
                self.assertEqual(sampler.stop(), [])


class Cost(unittest.TestCase):
    def test_cost_per_m_tokens(self):
        # 2 GPUs x $3/h x 1800 s = $3.00; 3.00 * 1e6 / 1.5e6 tokens = $2.00 per M
        self.assertEqual(wrapped.cost_per_m_tokens(3.0, 2, 1800.0, 1_500_000), (3.0, 2.0))
        self.assertEqual(wrapped.cost_per_m_tokens(2.5, 4, 90.0, 0)[1], 0.25 * 1e6)
        for args in ((3.0, 2, 1800.0, 1_500_000), (2.49, 8, 123.4, 98765), (1, 1, 0.0, 0)):
            self.assertEqual(wrapped.cost_per_m_tokens(*args), trace_serve.cost_per_m_tokens(*args))

    def test_cost_section(self):
        c = wrapped.cost_section(3, 2, 1800.0, 1_500_000)
        self.assertEqual(c, {"usd": 3.0, "usd_per_m_out_tokens": 2.0, "gpu_cost_hr": 3.0,
                             "devices": 2, "wall_s": 1800.0, "out_tokens": 1_500_000})


class Gsm8k(unittest.TestCase):
    CASES = (("The answer is 42.", "42"), ("so $1,234.50 total", "1234.50"),
             ("step 1 ... #### 72", "72"), ("-3 then 7.", "7"), ("no digits", None),
             ("#### -5", "-5"), ("1,000,000 dollars", "1000000"))

    def test_answer(self):
        for text, want in self.CASES:
            with self.subTest(text):
                self.assertEqual(wrapped.gsm8k_answer(text), want)
                self.assertEqual(wrapped.gsm8k_answer(text), trace_serve.gsm8k_answer(text))

    def test_load_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in (
                {"question": "ab", "answer": "x #### 3"},
                {"question": "c", "answer": "y #### 4"},
                {"question": "d", "answer": "z #### 5"})) + "\n\n")
            items = wrapped.load_gsm8k(path, "unused", 2, tokenizer=Tokenizer())
        self.assertEqual(items, [([97, 98], "3"), ([99], "4")])

        calls = []

        def generate(ids, max_tokens):
            calls.append((ids, max_tokens))
            return "so 3" if ids == [97, 98] else "nothing"

        acc = wrapped.gsm8k_accuracy(generate, items)
        self.assertEqual(acc, {"dataset": "gsm8k", "correct": 1, "n": 2, "exact_match": 0.5,
                               "max_tokens": 256})
        self.assertEqual(calls, [([97, 98], 256), ([99], 256)])
        self.assertEqual(wrapped.gsm8k_accuracy(generate, [])["exact_match"], 0.0)


SEEN = []


class Completions(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        out = json.dumps({"choices": [{"text": f"#### {len(body['prompt'])}"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class HttpGenerate(unittest.TestCase):
    def test_posts_greedy_token_ids(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), Completions)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            gen = wrapped.http_generate(f"http://127.0.0.1:{server.server_port}", "m")
            self.assertEqual(gen([1, 2, 3], 256), "#### 3")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(SEEN[-1], ("/v1/completions", {
            "model": "m", "prompt": [1, 2, 3], "max_tokens": 256, "temperature": 0.0}))


if __name__ == "__main__":
    unittest.main()
