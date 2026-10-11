"""The rules config/loader.py enforces at load, one test each, on a throwaway
configs/ tree so the committed yaml cannot mask a rule."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from quettabench.config import loader
from quettabench.config.loader import ConfigError, load, parse_set

MODEL = {"name": "Toy-1B", "dtype": "bfloat16", "weights_gib": 2, "max_model_len": 4096,
         "attention": "dense", "verified": True}
HARDWARE = {"world": 8, "timing_steps": 100, "max_seq_headroom": 28,
            "collective": {"NCCL_ALGO": "ring", "NCCL_PROTO": "Simple"},
            "provision": {"cache_root": "/c"}}
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
        return load(path, overrides, env={"XVAL_LINK_PROFILE": "pcie", **(env or {})})

    def one(self, *args, **kwargs):
        configs = self.load(*args, **kwargs)
        self.assertEqual(len(configs), 1)
        return configs[0]

    # precedence and origins

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
        run = {"engine": {"pp": 2, "dtype": "float16"},
               "hardware": {"collective": {"NCCL_PROTO": "LL"}}}
        cfg = self.one(run, ["engine.pp=4"], {"XVAL_PP": "3", "NCCL_ALGO": "tree"})
        self.assertEqual(cfg.engine.pp, 4)
        self.assertEqual(cfg.engine.dtype, "float16")
        self.assertEqual(cfg.origins["engine.dtype"], "run")
        self.assertEqual(cfg.hardware.collective["NCCL_ALGO"], "tree")
        self.assertEqual(cfg.hardware.collective["NCCL_PROTO"], "LL")
        self.assertEqual(self.one(run, env={"XVAL_PP": "3"}).engine.pp, 3)

    def test_caller_env_wins_for_each_collective_pin(self):
        pins = {"NCCL_P2P_LEVEL": ("SYS", "NVL"), "NCCL_IB_DISABLE": ("1", "0"),
                "NCCL_SOCKET_IFNAME": ("", "eth0"), "LLMSRV_NO_NVLS": ("1", "0"),
                "LLMSRV_TWOSHOT_MIN_BYTES": ("0", "65536")}
        self.write("hardware/pcie.yaml", {**HARDWARE, "collective": {
            **HARDWARE["collective"], **{k: yaml_v for k, (yaml_v, _) in pins.items()}}})
        for key, (yaml_value, caller) in pins.items():
            with self.subTest(key):
                self.assertEqual(self.one().hardware.collective[key], yaml_value)
                cfg = self.one(env={key: caller})
                self.assertEqual(cfg.hardware.collective[key], caller)
                self.assertEqual(cfg.origins[f"hardware.collective.{key}"], "env")

    def test_link_profile_picks_the_hardware_yaml(self):
        cfg = self.one(env={"XVAL_LINK_PROFILE": "nvlink"})
        self.assertEqual(cfg.hardware.collective, {"NCCL_ALGO": ""})

    def test_auto_link_profile_detects_like_xval_config(self):
        path = self.write("runs/r.yaml", {"model": "toy"})
        with mock.patch("shutil.which", return_value=None):
            cfg = load(path, env={})[0]
        self.assertEqual((cfg.hardware.backend, cfg.hardware.link_profile), ("cpu", "nvlink"))

    # unknown keys

    def test_unknown_key_names_file_and_key(self):
        cases = [
            ("runs/r.yaml", lambda: self.load({"engine": {"tpp": 2}}), "engine.tpp"),
            ("runs/r.yaml", lambda: self.load({"engines": {}}), "engines"),
            ("models/toy.yaml", lambda: self.load(model={"maxlen": 1}), "maxlen"),
            ("models/toy.yaml", lambda: self.load(model={"quettaserve": {"ok": "x"}}), "ok"),
            ("models/toy.yaml", lambda: self.load(model={"crossval": {"x": 1}}), "crossval.x"),
            ("runs/r.yaml", lambda: self.load({"hardware": {"collective": {"NCCL_X": 1}}}),
             "hardware.collective.NCCL_X"),
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

    # parse-time refusals

    def test_malformed_serving_style_keeps_parse_serving_style_message(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load({"engine": {"serving_style": "disagg:2p"}})
        self.assertEqual(str(ctx.exception), "unknown serving_style 'disagg:2p'; use aggregated, "
                         "disagg:<P>p<D>d, afd:<A>a<F>f, or ep<N>dp<M>")

    def test_zero_ep_or_dp_degree_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "ep_dp needs >=1 expert-parallel and >=1 "
                                                 "data-parallel: 'ep0dp2'"):
            self.load({"engine": {"serving_style": "ep0dp2"}})

    def test_ep_degree_must_divide_known_expert_count(self):
        with self.assertRaisesRegex(ConfigError, "ep=3 does not divide 256 experts"):
            self.load({"engine": {"tp": 3}}, model={"ep": True, "num_experts": 256})
        self.one({"engine": {"tp": 8}}, model={"ep": True, "num_experts": 256})
        self.one({"engine": {"tp": 3}}, model={"ep": True})  # num_experts unknown: skipped

    def test_unverified_needs_allow_unverified(self):
        with self.assertRaisesRegex(ConfigError, "is not validated yet; pass --allow-unverified"):
            self.load(model={"verified": False})
        cfg = self.one(model={"verified": False}, overrides=["allow_unverified=true"])
        self.assertFalse(cfg.engine.verified)

    def test_mla_and_hybrid_need_kv_bytes(self):
        for attention in ("mla", "hybrid"):
            with self.subTest(attention), self.assertRaisesRegex(ConfigError, "needs kv_bytes"):
                self.load(model={"attention": attention})
        self.assertEqual(self.one(model={"attention": "mla", "kv_bytes": 24768}).engine.kv_bytes,
                         24768)

    def test_bad_enum_value(self):
        with self.assertRaisesRegex(ConfigError, "unknown prompt_mode 'random'; use distinct, clone, corpus, synth or trace"):
            self.load(env={"PROMPT_MODE": "random"})

    # serve profile and load rules

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
            self.load({"mode": {"serve_profile": "stress-test"},
                       "engine": {"prefix_caching": True}})

    def test_workload_profile_turns_on_prefix_caching(self):
        cfg = self.one({"workload": {"source": "chat-short"}})
        self.assertEqual((cfg.engine.prefix_caching, cfg.origins["engine.prefix_caching"]),
                         (True, "profile"))
        with self.assertRaisesRegex(ConfigError, "contradicts workload profile chat-short"):
            self.load({"workload": {"source": "chat-short"}}, ["engine.prefix_caching=false"])
        self.assertFalse(self.one({"workload": {"source": "chat-short"},
                                   "mode": {"grid": "toy"}}).engine.prefix_caching)
        # a false from a level that states the checkpoint or box, not the run, is turned on
        for origin in ("default", "model", "hardware"):
            values = {"workload.source": "chat-short", "mode.grid": None,
                      "engine.prefix_caching": False}
            origins = {"engine.prefix_caching": origin}
            loader._profile_prefix_caching(values, origins, "w")
            self.assertEqual((values["engine.prefix_caching"], origins["engine.prefix_caching"]),
                             (True, "profile"))

    def test_open_loop_needs_rate_arrival_and_closed_loop_forbids_it(self):
        with self.assertRaisesRegex(ConfigError, "open load needs a rate-bearing arrival"):
            self.load({"mode": {"load": "open"}})
        with self.assertRaisesRegex(ConfigError, "is open-loop offered load"):
            self.load({"workload": {"arrival": "poisson:5"}})
        for arrival in ("poisson:5", "ramp:1:10", "burst:4x8@0.5", "trace"):
            self.one({"mode": {"load": "open"}, "workload": {"arrival": arrival}})
        self.one({"workload": {"arrival": "trace"}})

    def test_malformed_arrival(self):
        for spec in ("poisson", "poisson:0", "ramp:1", "burst:4x8", "steady:3", "uniform",
                     "poisson:nan", "poisson:inf", "ramp:-1:5", "burst:0x8@0.5", "burst:4x0@0.5",
                     "burst:4x8@-1", "burst:4x8@nan", "burst:-2x8@0.5"):
            with self.subTest(spec), self.assertRaisesRegex(ConfigError, "arrival"):
                self.load({"mode": {"load": "open"}, "workload": {"arrival": spec}})

    def test_a_burst_must_cover_the_drawn_requests(self):
        run = {"mode": {"load": "open"},
               "workload": {"source": "test", "arrival": "burst:4x8@0.5", "num_requests": 33}}
        with self.assertRaisesRegex(ConfigError, "covers 32 requests but the trace has 33"):
            self.load(run)
        self.one({**run, "workload": {**run["workload"], "num_requests": 32}})

    def test_bad_values_are_config_errors(self):
        with self.assertRaisesRegex(ConfigError, "mode.concurrency must be a number, got 'x'"):
            self.load(overrides=["mode.concurrency=x"])
        for run in ({"mode": {"goodput_sweep": True, "sla_ttft_ms": 1.0}},
                    {"mode": {"goodput_sweep": True, "load": "open"},
                     "workload": {"arrival": "poisson:2"}}):
            with self.subTest(run), self.assertRaisesRegex(ConfigError, "goodput sweep needs"):
                self.load(run)
        with self.assertRaisesRegex(ConfigError, "engine.driver=http has no tokenizer"):
            self.load({"mode": {"grid": "toy"}, "engine": {"driver": "http"},
                       "workload": {"prompt_mode": "corpus"}})

    def test_empty_step_points_are_refused(self):
        self.write("grids.yaml", {**GRIDS, "step_points": []})
        with self.assertRaisesRegex(ConfigError, "step_points must be a non-empty list"):
            self.load()

    # matrix, grid, --set

    def test_matrix_expands_cartesian_product_with_suffixed_names(self):
        configs = self.load({"matrix": {"engine.tp": [1, 2], "mode.concurrency": [4, 8]}})
        got = [(c.run_name, c.engine.tp, c.mode.concurrency) for c in configs]
        self.assertEqual(got, [("r-tp1-concurrency4", 1, 4), ("r-tp1-concurrency8", 1, 8),
                               ("r-tp2-concurrency4", 2, 4), ("r-tp2-concurrency8", 2, 8)])
        self.assertEqual(configs[0].origins["engine.tp"], "run")

    def test_matrix_points_with_one_run_name_are_refused(self):
        for matrix in ({"engine.tp": [1, 1]}, {"engine.port": [1], "mode.load": ["closed"],
                                               "hardware.world": [8, "8"]}):
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

    def test_set_parses_scalars_the_yaml_way(self):
        self.assertEqual(parse_set("engine.tp=2"), ("engine.tp", 2))
        self.assertEqual(parse_set("engine.gpu_util=0.85"), ("engine.gpu_util", 0.85))
        self.assertEqual(parse_set("engine.ep=true"), ("engine.ep", True))
        self.assertEqual(parse_set("engine.quant=null"), ("engine.quant", None))
        self.assertEqual(parse_set("engine.serving_style=disagg:1p1d"),
                         ("engine.serving_style", "disagg:1p1d"))
        with self.assertRaisesRegex(ConfigError, "expected key=value"):
            parse_set("engine.tp")

    def test_run_name_from_set(self):
        cfg = self.one(overrides=["run_name=mine"])
        self.assertEqual((cfg.run_name, cfg.origins["run_name"]), ("mine", "cli"))

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


if __name__ == "__main__":
    unittest.main()
