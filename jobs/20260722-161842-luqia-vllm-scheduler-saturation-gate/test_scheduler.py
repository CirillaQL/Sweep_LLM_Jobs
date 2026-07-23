#!/usr/bin/env python3
"""Regression tests for saturation-gated scheduling and overload fallback."""

import unittest
from pathlib import Path

from scheduler import PDPlacementScheduler


HERE = Path(__file__).resolve().parent


class SchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler = PDPlacementScheduler(
            HERE / "model_bundle.json",
            HERE / "saturation_bundle.json",
        )

    def recommend(self, il, ol, rate, overload_action="min-slo-violation"):
        return self.scheduler.recommend(
            il,
            ol,
            rate,
            500,
            200,
            1,
            "latency_plus_saturation",
            "auto",
            overload_action,
        )

    def test_safe_workload_keeps_normal_power_objective(self):
        result = self.recommend(32, 32, 1)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["decision_mode"], "safe_min_power")
        self.assertTrue(result["recommended"]["is_safe"])

    def test_no_safe_config_uses_explicit_overload_fallback(self):
        result = self.recommend(2, 64, 50)
        self.assertEqual(result["status"], "OVERLOAD_FALLBACK")
        self.assertEqual(result["decision_mode"], "overload_min_slo_violation")
        self.assertEqual(result["num_safe"], 0)
        self.assertFalse(result["recommended"]["is_safe"])
        selected = result["recommended"]["predicted_overload_violation_probability"]
        self.assertTrue(all(
            selected <= alternative["predicted_overload_violation_probability"]
            for alternative in result["alternatives"]
        ))
        self.assertGreater(
            result["recommended"]["prefill"]["p99_ttft_ms"],
            result["slos"]["ttft_ms"],
        )

    def test_reject_mode_remains_available_for_gate_only_experiments(self):
        result = self.recommend(2, 64, 50, overload_action="reject")
        self.assertEqual(result["status"], "NO_SAFE_CONFIG")
        self.assertEqual(result["decision_mode"], "reject")
        self.assertEqual(result["num_safe"], 0)
        self.assertNotIn("recommended", result)


if __name__ == "__main__":
    unittest.main()
