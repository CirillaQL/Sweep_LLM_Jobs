import random
import unittest

from xpyd.energy_validation import clock_within_tolerance, integrate_intervals
from xpyd.mixed_variable_canary_energy import matched_pair_plan, paired_summary


PANELS = {
    "small_light": [(128, 48), (160, 64), (200, 80)],
    "prefill_medium": [(768, 48), (1024, 64), (1280, 80)],
    "prefill_heavy": [(1792, 48), (2048, 64), (2304, 80)],
    "decode_medium": [(96, 96), (128, 128), (200, 160)],
    "decode_heavy": [(96, 224), (128, 256), (200, 320)],
    "balanced_medium": [(384, 96), (512, 128), (640, 160)],
    "both_heavy": [(1792, 224), (2048, 256), (2304, 320)],
}


class MixedVariableCanaryEnergyTests(unittest.TestCase):
    def test_pair_plan_is_balanced_exact_shape_and_reproducible(self):
        first = matched_pair_plan(PANELS, 12, random.Random(20260911))
        second = matched_pair_plan(PANELS, 12, random.Random(20260911))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 84)
        for workload_id, shapes in PANELS.items():
            rows = [row for row in first if row["workload_id"] == workload_id]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(set(row["arms"]) == {"safe_high", "table"} for row in rows))
            self.assertTrue(all((row["input_len"], row["output_len"]) in shapes for row in rows))
            self.assertEqual({(row["input_len"], row["output_len"]) for row in rows}, set(shapes))

    def test_paired_summary_uses_within_shape_energy_deltas(self):
        rows = []
        for pair_id, shape in (("a", (128, 48)), ("b", (200, 80))):
            for source, active, inclusive in (
                ("safe_high", 100.0, 105.0), ("table", 80.0, 90.0)
            ):
                rows.append({
                    "workload_id": "small_light", "pair_id": pair_id,
                    "frequency_source": source, "input_len": shape[0], "output_len": shape[1],
                    "energy_j": active, "control_inclusive": {"energy_j": inclusive},
                    "ttft_ms": 100.0, "tpot_ms": 20.0, "p_mhz": 900, "d_mhz": 450,
                })
        result = paired_summary(rows)[0]
        self.assertEqual(result["matched_pairs"], 2)
        self.assertAlmostEqual(result["active_mean_saving_j"], 20.0)
        self.assertAlmostEqual(result["active_energy_reduction_percent"], 20.0)
        self.assertAlmostEqual(result["control_inclusive_mean_saving_j"], 15.0)

    def test_clock_tolerance_accepts_normal_readback_jitter(self):
        self.assertTrue(clock_within_tolerance(1489, 1500))
        self.assertTrue(clock_within_tolerance(2510, 2520))
        self.assertFalse(clock_within_tolerance(1450, 1500))
        samples = [
            {"t": index * 0.05, "energy_j": index * 5.0,
             "clock_mhz": 1489, "power_w": 100.0}
            for index in range(41)
        ]
        value = integrate_intervals(samples, [(0.25, 1.75)], target=1500)
        self.assertEqual(value["clock_match_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
