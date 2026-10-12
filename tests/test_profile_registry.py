import tempfile
import unittest
from pathlib import Path

from quettabench.workloads.profiles import (
    PROFILES,
    active_profiles,
    get_profile,
    load_profiles,
)
from quettabench.workloads.sessions import resolve_multi_turn_num_sessions


class ProfileRegistryTests(unittest.TestCase):
    def test_profiles_are_the_yaml_entries(self):
        self.assertEqual(sorted(PROFILES), ["chat-multiturn-short", "chat-singleturn",
                                            "coding-singleturn", "mooncake-trace", "random-1k",
                                            "swebench-multiturn-short", "test"])

    def test_unknown_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "w.yaml"
            path.write_text("x:\n  isl_tokens: 1\n  osl_tokens: 1\n  isl_stddev: 0.0\n"
                            "  description: d\n  dataset: test\n  isl: 3\n")
            with self.assertRaisesRegex(ValueError, r"w.yaml: x: unknown profile key\(s\) \['isl'\]"):
                load_profiles(path)

    def test_listings_hide_inactive_profiles(self):
        self.assertEqual(active_profiles(), ["chat-singleturn", "coding-singleturn",
                                             "mooncake-trace"])
        with self.assertRaises(ValueError) as e:
            get_profile("nope")
        self.assertIn(str(active_profiles()), str(e.exception))
        self.assertNotIn("random-1k", str(e.exception))

    def test_coding_agent_is_alias_for_coding_singleturn(self):
        self.assertEqual(get_profile("coding-agent").name, "coding-singleturn")
        self.assertEqual(get_profile("coding-singleturn").dataset, "jsonl")

    def test_legacy_multiturn_profiles_remain_explicitly_runnable(self):
        self.assertEqual(get_profile("multi-turn-short").dataset, "sharegpt-multi-turn")
        self.assertEqual(get_profile("swebench-multiturn-short").dataset, "swebench-multi-turn")

    def test_multi_turn_num_sessions_uses_concurrency_floor(self):
        profile = get_profile("swebench-multiturn-short")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            profile.num_sessions + 1,
        )

        self.assertEqual(effective, profile.num_sessions + 1)
        self.assertEqual(source, "concurrency_floor")

    def test_multi_turn_num_sessions_keeps_profile_default_when_large_enough(self):
        profile = get_profile("chat-multiturn-short")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            max(1, profile.num_sessions - 1),
        )

        self.assertEqual(effective, profile.num_sessions)
        self.assertEqual(source, "profile_default")


if __name__ == "__main__":
    unittest.main()
