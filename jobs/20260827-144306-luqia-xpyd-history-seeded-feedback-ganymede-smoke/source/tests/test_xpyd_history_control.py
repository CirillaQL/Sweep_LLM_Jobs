"""CPU-only tests for the history-seeded controller."""

import copy
import unittest

from xpyd.history_seeded_control import (
    HistoricalExperienceStore,
    HistoryControlError,
    workload_distance,
)


WORKLOADS = [
    {"id": "light", "input_len": 128, "output_len": 128, "rate_rps": 0.5},
    {"id": "heavy", "input_len": 2048, "output_len": 256, "rate_rps": 2.0},
]


def summary():
    rows = []
    for workload in WORKLOADS:
        rows.extend([
            {
                "workload": workload["id"], "config_id": "safe-low",
                "prefill_endpoint_id": "P1", "decode_endpoint_id": "D0",
                "oracle_eligible": True,
                "P0_freq_mhz": 1260, "P1_freq_mhz": 1890,
                "D0_freq_mhz": 1125, "D1_freq_mhz": 750,
                "joules_per_request_mean": 90.0,
                "joules_per_request_ci95_half_width": 2.0,
                "mean_ttft_ms_mean": 600.0,
                "mean_ttft_ms_ci95_half_width": 20.0,
                "mean_tpot_ms_mean": 55.0,
                "mean_tpot_ms_ci95_half_width": 2.0,
            },
            {
                "workload": workload["id"], "config_id": "unsafe-cheap",
                "prefill_endpoint_id": "P0", "decode_endpoint_id": "D1",
                "oracle_eligible": True,
                "P0_freq_mhz": 1260, "P1_freq_mhz": 1260,
                "D0_freq_mhz": 750, "D1_freq_mhz": 750,
                "joules_per_request_mean": 80.0,
                "joules_per_request_ci95_half_width": 1.0,
                "mean_ttft_ms_mean": 890.0,
                "mean_ttft_ms_ci95_half_width": 20.0,
                "mean_tpot_ms_mean": 60.0,
                "mean_tpot_ms_ci95_half_width": 3.0,
            },
        ])
    return {
        "valid": True, "ready_for_phase4b": True,
        "models_trained_or_used": [],
        "slo": {"ttft_ms": 1000.0, "tpot_ms": 80.0},
        "configuration_aggregates": rows,
    }


class HistorySelectionTests(unittest.TestCase):
    def test_exact_context_selects_safe_conservative_energy_candidate(self):
        store = HistoricalExperienceStore(
            summary(), WORKLOADS, safety_fraction=0.9, max_context_distance=0.0
        )
        decision, candidates = store.choose(WORKLOADS[0])
        self.assertEqual(decision.config_id, "safe-low")
        self.assertEqual(decision.route, ("P1", "D0"))
        self.assertEqual(decision.frequencies_mhz["P1"], 1890)
        self.assertEqual(decision.candidate_count, 1)
        self.assertEqual(len(candidates), 4)

    def test_context_distance_is_symmetric_and_zero_for_same_workload(self):
        self.assertEqual(workload_distance(WORKLOADS[0], WORKLOADS[0]), 0.0)
        self.assertAlmostEqual(
            workload_distance(WORKLOADS[0], WORKLOADS[1]),
            workload_distance(WORKLOADS[1], WORKLOADS[0]),
        )

    def test_missing_safe_history_fails_closed(self):
        value = summary()
        for row in value["configuration_aggregates"]:
            row["mean_ttft_ms_mean"] = 950.0
        store = HistoricalExperienceStore(
            value, WORKLOADS, safety_fraction=0.9, max_context_distance=0.0
        )
        with self.assertRaisesRegex(HistoryControlError, "no historical candidate"):
            store.choose(WORKLOADS[0])

    def test_rejects_history_that_used_predictive_models(self):
        value = copy.deepcopy(summary())
        value["models_trained_or_used"] = ["forbidden"]
        with self.assertRaisesRegex(HistoryControlError, "used a model"):
            HistoricalExperienceStore(
                value, WORKLOADS, safety_fraction=0.9, max_context_distance=0.0
            )


if __name__ == "__main__":
    unittest.main()
