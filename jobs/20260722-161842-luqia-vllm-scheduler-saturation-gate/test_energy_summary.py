#!/usr/bin/env python3
"""Regression tests for allocation-wide multi-GPU energy accounting."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("energy_summary.py")
MONITOR_SCRIPT = Path(__file__).with_name("monitor_gpu_power.py")


class MultiGpuEnergySummaryTest(unittest.TestCase):
    def test_twelve_gpu_streams_are_summed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            powers = {
                "neptune": [100.0, 110.0, 120.0, 130.0],
                "ganymede": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            }
            for host, host_powers in powers.items():
                with (out_dir / f"allocation_{host}_power.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["unix_ts", "host", "gpu_index", "gpu_uuid", "gpu_power_w"]
                    )
                    for unix_ts in (1000.0, 1005.0, 1010.0):
                        for index, power_w in enumerate(host_powers):
                            writer.writerow(
                                [unix_ts, host, index, f"GPU-{host}-{index}", power_w]
                            )

            with (out_dir / "events.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["unix_ts", "event", "seq", "workload_id", "policy"])
                writer.writerow([1000.0, "workload_start", 1, "multi_gpu", "test"])
                writer.writerow([1010.0, "workload_end", 1, "multi_gpu", "test"])
            (out_dir / "bench_1_multi_gpu.txt").write_text(
                "Successful requests: 10\n", encoding="utf-8"
            )
            # Allocation-wide files take precedence, so the per-instance
            # telemetry must not be counted a second time.
            with (out_dir / "prefill_neptune_telemetry.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["unix_ts", "gpu_power_w"])
                writer.writerow([1000.0, 9999.0])
                writer.writerow([1010.0, 9999.0])

            output_json = out_dir / "summary.json"
            output_csv = out_dir / "summary.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--out-dir",
                    str(out_dir),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))
            workload = result["workloads"][0]

            self.assertEqual(result["power_source"], "allocation_wide_multi_gpu_monitor")
            self.assertEqual(result["gpu_stream_count"], 12)
            self.assertEqual(workload["covered_gpu_stream_count"], 12)
            self.assertAlmostEqual(workload["min_gpu_coverage_ratio"], 1.0)
            self.assertAlmostEqual(workload["host_energy_j"]["neptune"], 4600.0)
            self.assertAlmostEqual(workload["host_energy_j"]["ganymede"], 6000.0)
            self.assertAlmostEqual(workload["combined_energy_j"], 10600.0)
            self.assertAlmostEqual(workload["energy_per_request_j"], 1060.0)

    def test_asymmetric_host_gpu_counts_are_parsed(self):
        spec = importlib.util.spec_from_file_location("monitor_gpu_power", MONITOR_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.parse_expected_host_gpus("ganymede=8,neptune=4"),
            {"ganymede": 8, "neptune": 4},
        )


if __name__ == "__main__":
    unittest.main()
