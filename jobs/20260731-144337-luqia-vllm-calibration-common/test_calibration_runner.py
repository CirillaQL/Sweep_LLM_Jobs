#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).with_name("calibration_runner.py")
SPEC = importlib.util.spec_from_file_location("calibration_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def gpu_row() -> dict[str, str]:
    return {
        "index": "0",
        "name": "NVIDIA Test GPU",
        "uuid": "GPU-test",
        "driver_version": "test",
        "pstate": "P0",
        "clocks.sm": "1200",
        "clocks.mem": "5000",
        "utilization.gpu": "50",
        "utilization.memory": "25",
        "memory.used": "1000",
        "memory.total": "2000",
        "power.draw": "50",
        "power.limit": "100",
        "temperature.gpu": "45",
    }


class NetworkCounterTests(unittest.TestCase):
    def test_empty_interface_returns_zeroes(self) -> None:
        self.assertEqual(runner.network_counters(""), (0, 0))

    def test_reads_rx_and_tx_for_valid_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = root / "eth-test" / "statistics"
            statistics.mkdir(parents=True)
            (statistics / "rx_bytes").write_text("12345\n", encoding="utf-8")
            (statistics / "tx_bytes").write_text("67890\n", encoding="utf-8")
            self.assertEqual(
                runner.network_counters("eth-test", root),
                (12345, 67890),
            )

    def test_missing_or_invalid_counters_return_zeroes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = root / "eth-test" / "statistics"
            statistics.mkdir(parents=True)
            (statistics / "rx_bytes").write_text("invalid\n", encoding="utf-8")
            self.assertEqual(runner.network_counters("eth-test", root), (0, 0))

    def test_strict_mode_rejects_missing_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "cannot read network counters"):
                runner.network_counters(
                    "eth-test",
                    Path(directory),
                    strict=True,
                )


class GPUMonitorTests(unittest.TestCase):
    def test_monitor_must_produce_a_real_sample_before_ready(self) -> None:
        with (
            mock.patch.object(runner, "default_network_interface", return_value="eth0"),
            mock.patch.object(runner, "network_counters", return_value=(100, 200)),
            mock.patch.object(runner, "gpu_query", return_value=[gpu_row()]),
        ):
            monitor = runner.GPUMonitor(1, interval_ms=10)
            monitor.start()
            monitor.wait_until_ready(0.5)
            samples = monitor.stop()
        self.assertGreaterEqual(len(samples), 1)
        self.assertEqual(samples[0]["rx_bytes"], 100)
        self.assertEqual(samples[0]["tx_bytes"], 200)
        self.assertFalse(monitor.failed_event.is_set())

    def test_monitor_startup_error_is_fail_fast(self) -> None:
        with (
            mock.patch.object(runner, "default_network_interface", return_value="eth0"),
            mock.patch.object(
                runner,
                "network_counters",
                side_effect=TypeError("cannot unpack telemetry"),
            ),
        ):
            monitor = runner.GPUMonitor(1, interval_ms=10)
            monitor.start()
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "GPU monitor startup failed"):
                monitor.wait_until_ready(1.0)
            elapsed = time.monotonic() - started
            monitor.stop()
        self.assertLess(elapsed, 0.5)
        self.assertTrue(monitor.failed_event.is_set())
        self.assertIn("TypeError", monitor.error)

    def test_monitor_requires_every_requested_gpu(self) -> None:
        with (
            mock.patch.object(runner, "default_network_interface", return_value="eth0"),
            mock.patch.object(runner, "network_counters", return_value=(100, 200)),
            mock.patch.object(runner, "gpu_query", return_value=[gpu_row()]),
        ):
            monitor = runner.GPUMonitor(2, interval_ms=10)
            monitor.start()
            with self.assertRaisesRegex(RuntimeError, "GPU monitor startup failed"):
                monitor.wait_until_ready(0.5)
            monitor.stop()
        self.assertIn("expected 2 telemetry GPUs", monitor.error)

    def test_benchmark_process_is_stopped_on_monitor_failure(self) -> None:
        failed_event = threading.Event()
        failed_event.set()
        monitor = SimpleNamespace(
            failed_event=failed_event,
            error="RuntimeError: telemetry stopped",
        )
        with tempfile.TemporaryFile(mode="w+") as output:
            started = time.monotonic()
            with self.assertRaisesRegex(
                RuntimeError,
                "GPU monitor failed during benchmark",
            ):
                runner.run_benchmark_with_monitor(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    output,
                    30,
                    monitor,
                )
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
