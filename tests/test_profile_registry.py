import tempfile
import unittest
from pathlib import Path

from quettabench import workloads
from quettabench.config.schema import Config, EngineCfg, HardwareCfg, ModeCfg, WorkloadCfg
from quettabench.workloads.profiles import (
    PROFILES,
    active_profiles,
    get_profile,
    load_profiles,
)
from quettabench.workloads.sessions import resolve_multi_turn_num_sessions


class ProfileRegistryTests(unittest.TestCase):
    def test_profiles_are_the_yaml_files(self):
        self.assertEqual(len(PROFILES), 39)
        self.assertEqual(sum(1 for p in PROFILES.values() if p.sampler), 16)
        self.assertTrue(all(p.sampler == "analysis.synthetic" for p in PROFILES.values()
                            if p.dataset == "distributional-multi-turn"))

    def test_unknown_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "x.yaml").write_text("isl_tokens: 1\nosl_tokens: 1\nisl_stddev: 0.0\n"
                                             "description: d\ndataset: test\nisl: 3\n")
            with self.assertRaisesRegex(ValueError, r"x.yaml: unknown profile key\(s\) \['isl'\]"):
                load_profiles(td)

    def test_listings_hide_inactive_profiles(self):
        self.assertEqual(active_profiles(), ["chat-singleturn", "chat-singleturn-synth",
                                             "coding-singleturn", "mooncake-trace"])
        with self.assertRaises(ValueError) as e:
            get_profile("nope")
        self.assertIn(str(active_profiles()), str(e.exception))
        self.assertNotIn("chat-short", str(e.exception))
        self.assertFalse(any(PROFILES[n].sampler for n in active_profiles()))

    def test_an_inactive_sampler_profile_refuses_to_build_with_its_sampler_message(self):
        cfg = Config(engine=EngineCfg(), workload=WorkloadCfg(kind="sessions",
                                                              source="swebench-multiturn-synth"),
                     mode=ModeCfg(), hardware=HardwareCfg())
        self.assertFalse(PROFILES["swebench-multiturn-synth"].active)
        with self.assertRaisesRegex(ValueError, "samples synthetic sessions .* analysis/synthetic/"):
            workloads.build(cfg)

    def test_coding_agent_is_alias_for_coding_singleturn(self):
        self.assertEqual(get_profile("coding-agent").name, "coding-singleturn")
        self.assertEqual(get_profile("coding-singleturn").dataset, "jsonl")

    def test_legacy_multiturn_profiles_remain_explicitly_runnable(self):
        self.assertEqual(get_profile("chat-multiturn-long").dataset, "sharegpt-multi-turn")
        self.assertEqual(get_profile("swebench-multiturn-short").dataset, "swebench-multi-turn")
        self.assertEqual(get_profile("terminalbench-multiturn-medium").dataset, "terminalbench-multi-turn")
        self.assertEqual(get_profile("osworld-multiturn-long").dataset, "osworld-multi-turn")

    def test_multi_turn_num_sessions_uses_concurrency_floor(self):
        profile = get_profile("osworld-multiturn")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            profile.num_sessions + 1,
        )

        self.assertEqual(effective, profile.num_sessions + 1)
        self.assertEqual(source, "concurrency_floor")

    def test_multi_turn_num_sessions_keeps_profile_default_when_large_enough(self):
        profile = get_profile("chat-multiturn")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            max(1, profile.num_sessions - 1),
        )

        self.assertEqual(effective, profile.num_sessions)
        self.assertEqual(source, "profile_default")


if __name__ == "__main__":
    unittest.main()
