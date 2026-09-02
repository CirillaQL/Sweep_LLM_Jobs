"""CPU-only tests for fixed-table loading and comparison reporting."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from xpyd.compare_fixed_table_power import compare
from xpyd.disagg_proxy import _build_multi_core
from xpyd.phase3c_substrate import _workload_slo_met


class FixedPowerCompareTests(unittest.TestCase):
    @staticmethod
    def _write_fixture_run(run_dir, pair_energy, p_frequency, d_frequency):
        (run_dir / "client").mkdir(parents=True)
        with (run_dir / "client/trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "request_id", "workload_id", "arrival_time_s", "input_len", "output_len"
            ))
            writer.writeheader()
            writer.writerow({"request_id": "fixture", "workload_id": "small_light",
                             "arrival_time_s": 0.0, "input_len": 128, "output_len": 64})
        fields = (
            "workload_id", "scope", "role", "duration_s", "gross_energy_j",
            "mean_power_w", "logical_requests", "valid", "prefill_frequency_mhz",
            "decode_frequency_mhz", "ttft_p90_ms", "tpot_p90_ms",
        )
        with (run_dir / "workload_energy_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for scope, role, energy in (("P1", "prefill", pair_energy * 0.6),
                                        ("D1", "decode", pair_energy * 0.4),
                                        ("service_PD_pair", "all", pair_energy)):
                writer.writerow({
                    "workload_id": "small_light", "scope": scope, "role": role,
                    "duration_s": 10.0, "gross_energy_j": energy,
                    "mean_power_w": energy / 10.0, "logical_requests": 1,
                    "valid": True, "prefill_frequency_mhz": p_frequency,
                    "decode_frequency_mhz": d_frequency, "ttft_p90_ms": 200.0,
                    "tpot_p90_ms": 60.0,
                })

    def test_fixed_table_is_complete_before_first_request(self):
        root = Path(__file__).resolve().parent
        config = json.loads((
            root / "source/paper/configs/xpyd_power_compare_fixed_table.json"
        ).read_text(encoding="utf-8"))
        config.pop("frequency_table_path", None)
        core = _build_multi_core(config, lambda: None)
        observed = {
            key: (
                core.frequency_table.read(key).value.prefill_frequency_mhz,
                core.frequency_table.read(key).value.decode_frequency_mhz,
            )
            for key in core.frequency_table.keys
        }
        self.assertEqual(observed, {
            "small_light": (900, 720),
            "prefill_medium": (900, 975),
            "prefill_heavy": (1980, 975),
            "decode_medium": (900, 975),
            "decode_heavy": (900, 975),
            "balanced_medium": (900, 975),
        })

    def test_service_slo_uses_strict_ttft_p90_not_single_outlier(self):
        requests = [
            {"workload_id": "small_light", "ttft_ms": 400.0, "tpot_ms": 60.0}
            for _ in range(19)
        ] + [{"workload_id": "small_light", "ttft_ms": 900.0, "tpot_ms": 60.0}]
        self.assertTrue(_workload_slo_met(requests, "small_light", 500.0, 200.0))
        for row in requests:
            row["ttft_ms"] = 500.0
        self.assertFalse(_workload_slo_met(requests, "small_light", 500.0, 200.0))

    def test_comparison_rejects_different_request_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            optimized = root / "optimized"
            for run_dir, arrival in ((baseline, "0.0"), (optimized, "1.0")):
                (run_dir / "client").mkdir(parents=True)
                with (run_dir / "client/trace.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=(
                        "request_id", "workload_id", "arrival_time_s", "input_len", "output_len"
                    ))
                    writer.writeheader()
                    writer.writerow({"request_id": "x", "workload_id": "small_light",
                                     "arrival_time_s": arrival, "input_len": 128, "output_len": 64})
            with self.assertRaisesRegex(ValueError, "request traces differ"):
                compare(baseline, optimized, root / "output")

    def test_comparison_reports_pair_energy_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            optimized = root / "optimized"
            self._write_fixture_run(baseline, 100.0, 2520, 1500)
            self._write_fixture_run(optimized, 75.0, 900, 720)
            result = compare(baseline, optimized, root / "output")
            self.assertTrue(result["valid"])
            self.assertEqual(result["aggregate"]["energy_saved_percent"], 25.0)
            self.assertTrue((root / "output/power_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
