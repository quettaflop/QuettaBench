"""The rules config/loader.py enforces at load, one test each."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench.config import loader, recipe
from quettabench.config.loader import ConfigError, load, parse_set
from quettabench.config.schema import CHOICES
from quettabench.engines import base as engine_base

MODEL = {
    "name": "Toy-1B",
    "dtype": "bfloat16",
    "weights_gib": 2,
    "max_model_len": 4096,
    "attention": "dense",
    "verified": True,
}
HARDWARE = {
    "world": 8,
    "timing_steps": 100,
    "max_seq_headroom": 28,
    "collective": {"NCCL_ALGO": "ring", "NCCL_PROTO": "Simple"},
    "provision": {"cache_root": "/c"},
}
GRIDS = {"grids": {"toy": [[100, 1], [1024, 8], [4000, 1]]}, "step_points": [60, 230, 400]}


class LoaderRules(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "configs"
        self.write("models/toy.yaml", MODEL)
        self.write("hardware/pcie.yaml", HARDWARE)
        self.write("hardware/nvlink.yaml", {**HARDWARE, "collective": {"NCCL_ALGO": ""}})
        self.write("grids.yaml", GRIDS)

    def write(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data))
        return path

    def load(self, run=None, overrides=(), env=None, model=None):
        self.write("models/toy.yaml", {**MODEL, **(model or {})})
        path = self.write("runs/r.yaml", {"model": "toy", **(run or {})})
        return load(path, overrides, env={"QBENCH_LINK_PROFILE": "pcie", **(env or {})})

    def one(self, *args, **kwargs):
        configs = self.load(*args, **kwargs)
        self.assertEqual(len(configs), 1)
        return configs[0]

    def test_origin_recorded_for_each_level(self):
        cfg = self.one({"engine": {"tp": 2}}, ["mode.concurrency=4"], {"GPU_UTIL": "0.8"})
        o = cfg.origins
        self.assertEqual(o["mode.warmup"], "default")
        self.assertEqual(o["hardware.collective.NCCL_ALGO"], "hardware")
        self.assertEqual(o["engine.dtype"], "model")
        self.assertEqual(o["engine.tp"], "run")
        self.assertEqual(o["engine.gpu_util"], "env")
        self.assertEqual(o["mode.concurrency"], "cli")
        self.assertEqual((cfg.engine.tp, cfg.engine.gpu_util, cfg.mode.concurrency), (2, 0.8, 4))

    def test_visible_devices_is_a_string_even_when_yaml_reads_a_number(self):
        self.assertEqual(self.one(overrides=["hardware.visible_devices=0"]).hardware.visible_devices, "0")
        self.assertEqual(self.one({"hardware": {"visible_devices": 3}}).hardware.visible_devices, "3")
        self.assertIsNone(self.one().hardware.visible_devices)

    def test_cli_beats_env_beats_run_beats_model_beats_hardware(self):
        run = {"engine": {"pp": 2, "dtype": "float16"}, "hardware": {"collective": {"NCCL_PROTO": "LL"}}}
        cfg = self.one(run, ["engine.pp=4"], {"QBENCH_PP": "3", "NCCL_ALGO": "tree"})
        self.assertEqual(cfg.engine.pp, 4)
        self.assertEqual(cfg.engine.dtype, "float16")
        self.assertEqual(cfg.origins["engine.dtype"], "run")
        self.assertEqual(cfg.hardware.collective["NCCL_ALGO"], "tree")
        self.assertEqual(cfg.hardware.collective["NCCL_PROTO"], "LL")
        self.assertEqual(self.one(run, env={"QBENCH_PP": "3"}).engine.pp, 3)

    def test_caller_env_wins_for_each_collective_pin(self):
        pins = {
            "NCCL_P2P_LEVEL": ("SYS", "NVL"),
            "NCCL_IB_DISABLE": ("1", "0"),
            "NCCL_SOCKET_IFNAME": ("", "eth0"),
            "LLMSRV_NO_NVLS": ("1", "0"),
            "LLMSRV_TWOSHOT_MIN_BYTES": ("0", "65536"),
        }
        self.write(
            "hardware/pcie.yaml",
            {**HARDWARE, "collective": {**HARDWARE["collective"], **{k: yaml_v for k, (yaml_v, _) in pins.items()}}},
        )
        for key, (yaml_value, caller) in pins.items():
            with self.subTest(key):
                self.assertEqual(self.one().hardware.collective[key], yaml_value)
                cfg = self.one(env={key: caller})
                self.assertEqual(cfg.hardware.collective[key], caller)
                self.assertEqual(cfg.origins[f"hardware.collective.{key}"], "env")

    def test_link_profile_picks_the_hardware_yaml(self):
        cfg = self.one(env={"QBENCH_LINK_PROFILE": "nvlink"})
        self.assertEqual(cfg.hardware.collective, {"NCCL_ALGO": ""})

    def test_auto_link_profile_detects_like_captured_config(self):
        path = self.write("runs/r.yaml", {"model": "toy"})
        with mock.patch("shutil.which", return_value=None):
            cfg = load(path, env={})[0]
        self.assertEqual((cfg.hardware.backend, cfg.hardware.link_profile), ("cpu", "nvlink"))

    def test_unknown_key_names_file_and_key(self):
        cases = [
            ("runs/r.yaml", lambda: self.load({"engine": {"tpp": 2}}), "engine.tpp"),
            ("runs/r.yaml", lambda: self.load({"engines": {}}), "engines"),
            ("models/toy.yaml", lambda: self.load(model={"maxlen": 1}), "maxlen"),
            ("models/toy.yaml", lambda: self.load(model={"quettaserve": {"ok": "x"}}), "ok"),
            ("models/toy.yaml", lambda: self.load(model={"compare": {"x": 1}}), "compare.x"),
            (
                "runs/r.yaml",
                lambda: self.load({"hardware": {"collective": {"NCCL_X": 1}}}),
                "hardware.collective.NCCL_X",
            ),
        ]
        for rel, call, key in cases:
            with self.subTest(key=key), self.assertRaises(ConfigError) as ctx:
                call()
            self.assertIn(rel, str(ctx.exception))
            self.assertIn(repr(key), str(ctx.exception))

    def test_unknown_key_in_hardware_and_grids_yaml(self):
        self.write("hardware/pcie.yaml", {**HARDWARE, "nics": 2})
        with self.assertRaisesRegex(ConfigError, r"pcie\.yaml: unknown key 'nics'"):
            self.load()
        self.write("hardware/pcie.yaml", HARDWARE)
        self.write("grids.yaml", {**GRIDS, "cells": []})
        with self.assertRaisesRegex(ConfigError, r"grids\.yaml: unknown key 'cells'"):
            self.load()

    def test_unknown_set_key(self):
        with self.assertRaisesRegex(ConfigError, r"--set: unknown key 'engine.tensor_parallel'"):
            self.load(overrides=["engine.tensor_parallel=2"])

    def test_malformed_serving_style_keeps_parse_serving_style_message(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load({"engine": {"serving_style": "disagg:2p"}})
        self.assertEqual(
            str(ctx.exception),
            "unknown serving_style 'disagg:2p'; use aggregated, disagg:<P>p<D>d, afd:<A>a<F>f, or ep<N>dp<M>",
        )

    def test_zero_ep_or_dp_degree_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "ep_dp needs >=1 expert-parallel and >=1 data-parallel: 'ep0dp2'"):
            self.load({"engine": {"serving_style": "ep0dp2"}})

    def test_ep_degree_must_divide_known_expert_count(self):
        with self.assertRaisesRegex(ConfigError, "ep=3 does not divide 256 experts"):
            self.load({"engine": {"tp": 3}}, model={"ep": True, "num_experts": 256})
        self.one({"engine": {"tp": 8}}, model={"ep": True, "num_experts": 256})
        self.one({"engine": {"tp": 3}}, model={"ep": True})

    def test_unverified_needs_allow_unverified(self):
        with self.assertRaisesRegex(ConfigError, "is not validated yet; pass --allow-unverified"):
            self.load(model={"verified": False})
        cfg = self.one(model={"verified": False}, overrides=["allow_unverified=true"])
        self.assertFalse(cfg.engine.verified)

    def test_mla_and_hybrid_need_kv_bytes_per_token(self):
        for attention in ("mla", "hybrid"):
            with self.subTest(attention), self.assertRaisesRegex(ConfigError, "needs kv_bytes_per_token"):
                self.load(model={"attention": attention})
        self.assertEqual(
            self.one(model={"attention": "mla", "kv_bytes_per_token": 24768}).engine.kv_bytes_per_token, 24768
        )

    def test_bad_enum_value(self):
        with self.assertRaisesRegex(ConfigError, "unknown prompt_mode 'random'; use distinct, clone, corpus or trace"):
            self.load(env={"PROMPT_MODE": "random"})

    def test_synthetic_workload_source_is_refused(self):
        self.assertEqual(
            self.one({"workload": {"source": "random-traces/run.jsonl"}}).workload.source, "random-traces/run.jsonl"
        )
        for source in ("random-1k", "synthetic"):
            with self.subTest(source), self.assertRaisesRegex(ConfigError, "names a synthetic workload"):
                self.load(run={"workload": {"source": source}})

    def test_stress_test_forces_ignore_eos_and_no_prefix_caching(self):
        cfg = self.one({"mode": {"serve_profile": "stress-test"}})
        self.assertTrue(cfg.mode.ignore_eos)
        self.assertFalse(cfg.engine.prefix_caching)
        self.assertEqual(cfg.origins["mode.ignore_eos"], "run")

    def test_single_and_multi_turn_force_prefix_caching(self):
        for profile in ("single-turn", "multi-turn"):
            with self.subTest(profile):
                cfg = self.one(overrides=[f"mode.serve_profile={profile}"])
                self.assertTrue(cfg.engine.prefix_caching)
                self.assertEqual(cfg.origins["engine.prefix_caching"], "cli")

    def test_explicit_setting_against_serve_profile_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "engine.prefix_caching=True contradicts"):
            self.load({"mode": {"serve_profile": "stress-test"}, "engine": {"prefix_caching": True}})

    def test_workload_profile_turns_on_prefix_caching(self):
        cfg = self.one({"workload": {"source": "chat-singleturn"}})
        self.assertEqual((cfg.engine.prefix_caching, cfg.origins["engine.prefix_caching"]), (True, "profile"))
        with self.assertRaisesRegex(ConfigError, "contradicts workload profile chat-singleturn"):
            self.load({"workload": {"source": "chat-singleturn"}}, ["engine.prefix_caching=false"])
        self.assertFalse(
            self.one({"workload": {"source": "chat-singleturn"}, "mode": {"grid": "toy"}}).engine.prefix_caching
        )
        for origin in ("default", "model", "hardware"):
            values = {"workload.source": "chat-singleturn", "mode.grid": None, "engine.prefix_caching": False}
            origins = {"engine.prefix_caching": origin}
            loader._profile_prefix_caching(values, origins, "w")
            self.assertEqual((values["engine.prefix_caching"], origins["engine.prefix_caching"]), (True, "profile"))

    def test_open_loop_needs_rate_arrival_and_closed_loop_forbids_it(self):
        with self.assertRaisesRegex(ConfigError, "open load needs a rate-bearing arrival"):
            self.load({"mode": {"load": "open"}})
        with self.assertRaisesRegex(ConfigError, "is open-loop offered load"):
            self.load({"workload": {"arrival": "poisson:5"}})
        for arrival in ("poisson:5", "ramp:1:10", "burst:4x8@0.5", "trace"):
            self.one({"mode": {"load": "open"}, "workload": {"arrival": arrival}})
        self.one({"workload": {"arrival": "trace"}})

    def test_open_loop_sessions_need_recorded_arrivals_for_trace(self):
        open_trace = {"mode": {"load": "open"}, "workload": {"kind": "sessions", "arrival": "trace"}}
        with self.assertRaisesRegex(ConfigError, "set workload.use_recorded_arrivals=true"):
            self.load(open_trace)
        open_trace["workload"]["use_recorded_arrivals"] = True
        self.one(open_trace)

    def test_num_requests_is_refused_where_it_sizes_nothing(self):
        for workload, use in (
            ({"source": "/t/trace.jsonl"}, "a shorter source file"),
            ({"kind": "sessions", "source": "chat-multiturn-short"}, "workload.num_sessions"),
        ):
            with self.subTest(use=use), self.assertRaisesRegex(ConfigError, use):
                self.load({"workload": {**workload, "num_requests": 50}})
        self.one({"workload": {"source": "chat-singleturn", "num_requests": 50}})

    def test_num_sessions_must_be_a_positive_integer(self):
        for bad in (0, -2, 1.5, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ConfigError, "workload.num_sessions must be"):
                self.load({"workload": {"kind": "sessions", "num_sessions": bad}})
        self.one({"workload": {"kind": "sessions", "num_sessions": 3}})

    def test_malformed_arrival(self):
        for spec in (
            "poisson",
            "poisson:0",
            "ramp:1",
            "burst:4x8",
            "steady:3",
            "uniform",
            "poisson:nan",
            "poisson:inf",
            "ramp:-1:5",
            "burst:0x8@0.5",
            "burst:4x0@0.5",
            "burst:4x8@-1",
            "burst:4x8@nan",
            "burst:-2x8@0.5",
        ):
            with self.subTest(spec), self.assertRaisesRegex(ConfigError, "arrival"):
                self.load({"mode": {"load": "open"}, "workload": {"arrival": spec}})

    def test_a_burst_must_cover_the_drawn_requests(self):
        run = {"mode": {"load": "open"}, "workload": {"source": "test", "arrival": "burst:4x8@0.5", "num_requests": 33}}
        with self.assertRaisesRegex(ConfigError, "covers 32 requests but the trace has 33"):
            self.load(run)
        self.one({**run, "workload": {**run["workload"], "num_requests": 32}})

    def test_bad_values_are_config_errors(self):
        with self.assertRaisesRegex(ConfigError, "mode.concurrency must be a number, got 'x'"):
            self.load(overrides=["mode.concurrency=x"])
        for run in (
            {"mode": {"goodput_sweep": True, "sla_ttft_ms": 1.0}},
            {"mode": {"goodput_sweep": True, "load": "open"}, "workload": {"arrival": "poisson:2"}},
        ):
            with self.subTest(run), self.assertRaisesRegex(ConfigError, "goodput sweep needs"):
                self.load(run)
        with self.assertRaisesRegex(ConfigError, "a timing grid runs over HTTP with no tokenizer"):
            self.load({"mode": {"grid": "toy"}, "workload": {"prompt_mode": "corpus"}})
        for backend in ("rocm", "tpu", "neuron"):
            with self.subTest(backend), self.assertRaisesRegex(ConfigError, f"hardware.backend={backend} has no CUDA"):
                self.load(
                    {"mode": {"grid": "toy", "return_outputs": "text"}},
                    [f"hardware.backend={backend}", "engine.dtype=bfloat16"],
                )
        self.one({"mode": {"grid": "toy", "return_outputs": "text"}}, ["hardware.backend=cuda"])

    def test_quettaserve_is_cuda_only(self):
        for backend in ("rocm", "tpu", "neuron"):
            with (
                self.subTest(backend),
                self.assertRaisesRegex(ConfigError, f"quettaserve runs on CUDA only.*{backend}"),
            ):
                self.load(overrides=["engine.name=quettaserve", f"hardware.backend={backend}", "engine.dtype=bfloat16"])

    def test_empty_step_points_are_refused(self):
        self.write("grids.yaml", {**GRIDS, "step_points": []})
        with self.assertRaisesRegex(ConfigError, "step_points needs at least two distinct values"):
            self.load()

    def test_matrix_expands_cartesian_product_with_suffixed_names(self):
        configs = self.load({"matrix": {"engine.tp": [1, 2], "mode.concurrency": [4, 8]}})
        got = [(c.run_name, c.engine.tp, c.mode.concurrency) for c in configs]
        self.assertEqual(
            got,
            [
                ("r-tp1-concurrency4", 1, 4),
                ("r-tp1-concurrency8", 1, 8),
                ("r-tp2-concurrency4", 2, 4),
                ("r-tp2-concurrency8", 2, 8),
            ],
        )
        self.assertEqual(configs[0].origins["engine.tp"], "run")

    def test_matrix_points_with_one_run_name_are_refused(self):
        for matrix in (
            {"engine.tp": [1, 1]},
            {"engine.port": [1], "mode.load": ["closed"], "hardware.world": [8, "8"]},
        ):
            with self.subTest(matrix), self.assertRaisesRegex(ConfigError, "points share run_name"):
                self.load({"matrix": matrix})

    def test_matrix_key_is_checked(self):
        with self.assertRaisesRegex(ConfigError, "matrix: unknown key 'engine.tpp'"):
            self.load({"matrix": {"engine.tpp": [1]}})

    def test_grid_cells_over_maxlen_are_kept_and_flagged(self):
        cfg = self.one({"mode": {"grid": "toy"}})
        self.assertEqual(cfg.grid.cells, ((100, 1), (1024, 8), (4000, 1)))
        self.assertEqual(cfg.grid.step_points, (60, 230, 400))
        self.assertEqual(cfg.grid.cells_over_maxlen, ((4000, 1),))
        self.assertIsNone(self.one().grid)
        with self.assertRaisesRegex(ConfigError, "unknown grid 'nope'"):
            self.load({"mode": {"grid": "nope"}})

    def test_a_grid_run_needs_weights_gib(self):
        with self.assertRaisesRegex(ConfigError, "a grid run needs engine.weights_gib"):
            self.load({"mode": {"grid": "toy"}}, model={"weights_gib": None})
        self.assertIsNone(self.one(model={"weights_gib": None}).engine.weights_gib)

    def test_set_parses_scalars_the_yaml_way(self):
        self.assertEqual(parse_set("engine.tp=2"), ("engine.tp", 2))
        self.assertEqual(parse_set("engine.gpu_util=0.85"), ("engine.gpu_util", 0.85))
        self.assertEqual(parse_set("engine.ep=true"), ("engine.ep", True))
        self.assertEqual(parse_set("engine.quant=null"), ("engine.quant", None))
        self.assertEqual(parse_set("engine.serving_style=disagg:1p1d"), ("engine.serving_style", "disagg:1p1d"))
        with self.assertRaisesRegex(ConfigError, "expected key=value"):
            parse_set("engine.tp")

    def test_run_name_from_set(self):
        cfg = self.one(overrides=["run_name=mine"])
        self.assertEqual((cfg.run_name, cfg.origins["run_name"]), ("mine", "cli"))

    def test_a_run_name_that_is_no_plain_dir_name_is_refused(self):
        for name in (
            "..",
            ".",
            "/tmp/x",
            "a/../..",
            "probes",
            "compare",
            "baselines",
            "Probes",
            "",
            "-x",
            ".hidden",
            "a b",
            "a\\b",
            "trailing.",
            "x" * 201,
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ConfigError, "run_name"):
                self.load(overrides=[f"run_name={name}"])
        with self.assertRaisesRegex(ConfigError, "run_name"):
            self.write("runs/probes.yaml", {"model": "toy"})
            load(self.root / "runs" / "probes.yaml", env={"QBENCH_LINK_PROFILE": "pcie"})
        self.assertEqual(self.one(overrides=["run_name=a.b_c-1"]).run_name, "a.b_c-1")
        self.assertEqual(self.one(overrides=[f"run_name={'x' * 200}"]).run_name, "x" * 200)

    def test_matrix_suffixes_keep_to_the_run_name_charset(self):
        configs = self.load({"matrix": {"engine.model": ["/w/llama", "../x"]}})
        self.assertEqual([c.run_name for c in configs], ["r-model_w_llama", "r-model.._x"])

    def test_an_engine_without_an_adapter_is_refused_at_load(self):
        for name in ("sglang", "trtllm"):
            with (
                self.subTest(name),
                self.assertRaisesRegex(ConfigError, rf"unknown name '{name}'; .*vllm.*quettaserve"),
            ):
                self.load({"engine": {"name": name}})
        self.assertEqual(sorted(CHOICES["engine.name"]), sorted(engine_base._ADAPTERS))

    def test_env_lands_in_provenance_and_hardware(self):
        cfg = self.one(env={"QS_SHA": "1234567", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(cfg.provenance["engine_sha"], "1234567")
        self.assertEqual(cfg.origins["provenance.engine_sha"], "env")
        self.assertEqual(cfg.hardware.visible_devices, "0")

    def test_dump_lists_every_leaf_with_origin(self):
        text = self.one({"engine": {"tp": 2}}).dump()
        self.assertIn("engine.tp = 2  (run)", text)
        self.assertIn("hardware.collective.NCCL_ALGO = 'ring'  (hardware)", text)
        self.assertNotIn("(", self.one().dump(origins=False))

    def recipe(self, text, engine=None, **kwargs):
        """The loaded Config of a serve run whose engine.recipe is `vllm serve toy/m` plus text."""
        run = {"engine": {"model": "toy/m", "recipe": f"vllm serve toy/m {text}", **(engine or {})}}
        return self.one(run, **kwargs)

    def test_recipe_layout_supersedes_the_model_yaml(self):
        cfg = self.recipe("--tensor-parallel-size 2", model={"tp": 4})
        e = cfg.engine
        self.assertEqual((e.tp, e.pp, e.dp, e.ep, e.dtype, e.kv_cache_dtype), (2, 1, 1, False, "auto", "auto"))
        self.assertEqual((e.max_model_len, e.gpu_util, e.prefix_caching, e.cudagraph), (None, None, None, True))
        for field in ("tp", "dtype", "max_model_len", "gpu_util", "prefix_caching"):
            self.assertEqual(cfg.origins[f"engine.{field}"], "recipe")

    def test_recipe_values_are_recorded_as_set(self):
        cfg = self.recipe("--gpu-memory-utilization 0.9 --max-model-len 1000 --enable-prefix-caching --enforce-eager")
        e = cfg.engine
        self.assertEqual((e.gpu_util, e.max_model_len, e.prefix_caching, e.cudagraph), (0.9, 1000, True, False))

    def test_flag_spellings_read_as_vllm_reads_them(self):
        cases = {
            "-tp 2 -pp 2 -dp 2 -ep": {"tp": 2, "pp": 2, "dp": 2, "ep": True},
            "--tensor_parallel_size=2 --data-parallel-size=4": {"tp": 2, "dp": 4},
            "--enable-expert-parallel=true": {"ep": True},
            "--enable-expert-parallel=false": {"ep": False},
            "--enable-expert-parallel --no-enable-expert-parallel": {"ep": False},
            "--no-enable-expert-parallel --enable-expert-parallel": {"ep": True},
            "--no-enable-prefix-caching": {"prefix_caching": False},
            "--enforce-eager": {"cudagraph": False},
            "--kv_cache_dtype fp8 --dtype=bfloat16": {"kv_cache_dtype": "fp8", "dtype": "bfloat16"},
            "--gpu-memory-utilization=0.8 --port 1": {"gpu_util": 0.8},
        }
        for text, want in cases.items():
            with self.subTest(text):
                engine = self.recipe(text).engine
                self.assertEqual({k: getattr(engine, k) for k in want}, want)

    def test_recipe_text_is_read_as_a_shell_would(self):
        text = (
            "# B300\r\nexport A=1 B='x y'  # env\r\n\r\n"
            "vllm serve toy/m \\\r\n  --tensor-parallel-size 2 \\\r\n  # a comment\r\n"
            '  --speculative-config \'{"a": "#b"}\' # note\r\n'
        )
        run = {"engine": {"model": "toy/m", "recipe": text.replace("  # a comment\r\n", "")}}
        cfg = self.one(run)
        self.assertEqual(cfg.engine.tp, 2)
        exports, model, flags = recipe.parse_recipe(cfg.engine.recipe)
        self.assertEqual((exports, model), ({"A": "1", "B": "x y"}, "toy/m"))
        self.assertEqual(flags, ["--tensor-parallel-size", "2", "--speculative-config", '{"a": "#b"}'])

    def test_a_recipe_that_is_not_one_serve_command_is_refused(self):
        bad = {
            "vllm serve toy/m\nvllm serve toy/m": "nothing may follow",
            "vllm serve toy/m\nexport A=1": "nothing may follow",
            "vllm serve toy/m && rm x": "shell operator '&&'",
            "vllm serve toy/m; rm x": "shell operator ';'",
            "vllm serve toy/m | tee x": "shell operator '|'",
            "export A": "export wants K=V",
            "export A=1": "no `vllm serve",
            "python x.py": "must be `export",
        }
        for text, why in bad.items():
            with self.subTest(text), self.assertRaisesRegex(ConfigError, why):
                self.one({"engine": {"model": "toy/m", "recipe": text}})

    def test_a_hash_begins_a_comment_only_at_the_start_of_a_word(self):
        cases = {
            "vllm serve toy/m --chat-template a#b": ["--chat-template", "a#b"],
            "# whole line\nvllm serve toy/m --x 1": ["--x", "1"],
            "vllm serve toy/m --x 1  # trailing": ["--x", "1"],
            "vllm serve toy/m --x '#q' \"#r\" a\\#b": ["--x", "#q", "#r", "a#b"],
        }
        for text, flags in cases.items():
            with self.subTest(text):
                self.assertEqual(recipe.parse_recipe(text)[2], flags)

    def test_flags_qbench_cannot_read_are_refused(self):
        bad = {
            "--data-parallel-size-local 2": "one node",
            "--data_parallel_size_local=2": "one node",
            "--tensor-parallel 2": "abbreviates",
            "--max-model 5": "abbreviates",
            "--config c.yaml": "--config is refused",
            "--api-key k": "VLLM_API_KEY",
            "--max-model-len 1M": "not a plain int",
            "--max-model-len=1k": "not a plain int",
            "--enable-expert-parallel=maybe": "must be true or false",
            "--no-enable-expert-parallel=true": "takes no value",
        }
        for text, why in bad.items():
            with self.subTest(text), self.assertRaisesRegex(ConfigError, why):
                self.recipe(text)

    def test_the_model_is_positional(self):
        with self.assertRaisesRegex(ConfigError, "the model is positional"):
            self.one({"engine": {"model": "toy/m", "recipe": "vllm serve --model toy/m"}})

    def test_a_recipe_needs_a_plain_vllm_serve_run(self):
        cases = [
            ({"engine": {"name": "quettaserve"}}, "needs engine.name=vllm"),
            ({"mode": {"return_outputs": "text"}}, "needs engine.name=vllm over HTTP"),
            ({"mode": {"grid": "toy"}, "workload": {"kind": "grid"}}, "mode.grid does not run with engine.recipe"),
            ({"engine": {"extra": {"vllm": {"enforce_eager": True}}}}, "engine.extra.vllm does not apply"),
            ({"engine": {"serving_style": "disagg:1p1d"}}, "serving_style=disagg:1p1d does not apply"),
            ({"engine": {"model": "other/m"}}, "serves 'toy/m' but engine.model='other/m'"),
        ]
        for run, why in cases:
            with self.subTest(why), self.assertRaisesRegex(ConfigError, why):
                self.one({**run, "engine": {"model": "toy/m", "recipe": "vllm serve toy/m", **run.get("engine", {})}})

    def test_an_explicit_field_that_differs_from_the_recipe_is_refused(self):
        cases = [
            (
                {"gpu_util": 0.8},
                [],
                {},
                "--gpu-memory-utilization 0.9",
                "engine.gpu_util=0.8 but engine.recipe launches 0.9",
            ),
            ({}, ["engine.tp=4"], {}, "--tensor-parallel-size 2", "engine.tp=4 but engine.recipe launches 2"),
            ({}, [], {"GPU_UTIL": "0.9"}, "", "engine.gpu_util=0.9 but engine.recipe launches None"),
            ({"dp": 2}, [], {}, "", "engine.dp=2 but engine.recipe launches 1"),
        ]
        for engine, sets, env, text, why in cases:
            with self.subTest(why), self.assertRaisesRegex(ConfigError, why):
                self.recipe(text, engine, overrides=sets, env=env)
        self.assertEqual(self.recipe("--gpu-memory-utilization 0.9", {"gpu_util": 0.9}).engine.gpu_util, 0.9)

    def test_a_recipe_silent_on_prefix_caching_is_left_to_the_live_server(self):
        for run in ({"mode": {"serve_profile": "single-turn"}}, {"workload": {"source": "chat-singleturn"}}):
            cfg = self.one({**run, "engine": {"model": "toy/m", "recipe": "vllm serve toy/m"}})
            self.assertIsNone(cfg.engine.prefix_caching)
            self.assertEqual(cfg.origins["engine.prefix_caching"], "recipe")

    def test_a_recipe_that_turns_off_what_a_profile_needs_is_refused(self):
        run = {"mode": {"serve_profile": "single-turn"}}
        run["engine"] = {"model": "toy/m", "recipe": "vllm serve toy/m --no-enable-prefix-caching"}
        with self.assertRaisesRegex(ConfigError, "engine.prefix_caching=False contradicts mode.serve_profile"):
            self.one(run)
        run["engine"] = {"model": "toy/m", "recipe": "vllm serve toy/m --enable-prefix-caching"}
        self.assertTrue(self.one(run).engine.prefix_caching)


if __name__ == "__main__":
    unittest.main()
