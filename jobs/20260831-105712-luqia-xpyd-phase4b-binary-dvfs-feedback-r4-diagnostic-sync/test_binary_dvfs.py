#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest


JOB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JOB_DIR / "source" / "paper" / "scripts"))

from xpyd.binary_dvfs_evaluation import (  # noqa: E402
    BinaryDVFSEvaluationHarness,
    BinaryDVFSError,
    balanced_frozen_routes_valid,
    binary_update,
    load_client_request_results,
    peak_observed_concurrency,
    select_frequency_grid,
)
from xpyd.phase3c_substrate import (  # noqa: E402
    Phase3CError,
    _wait_for_proxy_diagnostics,
)


class BinaryDVFSTest(unittest.TestCase):
    def test_grid_is_supported_ordered_and_includes_bounds(self) -> None:
        supported = list(range(450, 1501, 15))
        grid = select_frequency_grid(supported, 450, 1500, 15)
        self.assertEqual(len(grid), 15)
        self.assertEqual(grid[0], 450)
        self.assertEqual(grid[-1], 1500)
        self.assertEqual(tuple(sorted(set(grid))), grid)
        self.assertTrue(set(grid).issubset(supported))

    def test_grid_rejects_fewer_than_ten_levels(self) -> None:
        with self.assertRaises(BinaryDVFSError):
            select_frequency_grid(range(100, 1001, 100), 100, 1000, 9)

    def test_binary_update_searches_lower_on_pass(self) -> None:
        self.assertEqual(binary_update(0, 16, 8, True), (0, 8))

    def test_binary_update_searches_upper_on_failure(self) -> None:
        self.assertEqual(binary_update(0, 16, 8, False), (9, 16))

    def test_binary_update_converges_to_lowest_feasible_index(self) -> None:
        low, high = 0, 16
        first_feasible = 6
        while low < high:
            candidate = (low + high) // 2
            low, high = binary_update(
                low, high, candidate, candidate >= first_feasible
            )
        self.assertEqual(low, first_feasible)

    def test_peak_observed_concurrency_uses_real_overlap(self) -> None:
        requests = [
            {"ok": True, "send_unix_s": 0.0, "complete_unix_s": 3.0},
            {"ok": True, "send_unix_s": 1.0, "complete_unix_s": 4.0},
            {"ok": True, "send_unix_s": 2.0, "complete_unix_s": 5.0},
            {"ok": True, "send_unix_s": 6.0, "complete_unix_s": 7.0},
        ]
        self.assertEqual(peak_observed_concurrency(requests), 3)

    def test_touching_requests_do_not_count_as_overlap(self) -> None:
        requests = [
            {"ok": True, "send_unix_s": 0.0, "complete_unix_s": 1.0},
            {"ok": True, "send_unix_s": 1.0, "complete_unix_s": 2.0},
        ]
        self.assertEqual(peak_observed_concurrency(requests), 1)

    def test_concurrency_evidence_loads_from_requests_jsonl(self) -> None:
        requests = [
            {"ok": True, "send_unix_s": 0.0, "complete_unix_s": 3.0},
            {"ok": True, "send_unix_s": 1.0, "complete_unix_s": 4.0},
            {"ok": True, "send_unix_s": 2.0, "complete_unix_s": 5.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            window_dir = Path(directory)
            client_dir = window_dir / "client"
            client_dir.mkdir()
            (client_dir / "summary.json").write_text(
                json.dumps({"max_concurrency": 4}), encoding="utf-8"
            )
            (client_dir / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in requests),
                encoding="utf-8",
            )
            loaded = load_client_request_results(window_dir)
        self.assertEqual(peak_observed_concurrency(loaded), 3)

    def test_missing_request_timing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BinaryDVFSError):
                load_client_request_results(Path(directory))

    def test_frozen_routes_require_exact_balance(self) -> None:
        self.assertTrue(balanced_frozen_routes_valid({
            "P0->D0": 8, "P1->D1": 8, "P0->D1": 0, "P1->D0": 0,
        }, 16))
        self.assertFalse(balanced_frozen_routes_valid({
            "P0->D0": 12, "P1->D1": 4,
        }, 16))
        self.assertFalse(balanced_frozen_routes_valid({
            "P0->D0": 8, "P1->D1": 7, "P0->D1": 1,
        }, 16))

    def test_valid_high_frequency_evidence_may_be_slo_infeasible(self) -> None:
        rows = [{
            "measurement_valid": True,
            "route_valid": True,
            "clocks_valid": True,
            "concurrency_valid": True,
            "client_queue_valid": True,
            "slo_pass": False,
        }]
        self.assertTrue(BinaryDVFSEvaluationHarness._evidence_valid(rows))
        self.assertFalse(BinaryDVFSEvaluationHarness._probe_feasible(rows))

    def test_client_queue_is_a_feasibility_gate(self) -> None:
        rows = [{
            "measurement_valid": True,
            "route_valid": True,
            "clocks_valid": True,
            "concurrency_valid": True,
            "client_queue_valid": False,
            "slo_pass": True,
        }]
        self.assertFalse(BinaryDVFSEvaluationHarness._evidence_valid(rows))
        self.assertFalse(BinaryDVFSEvaluationHarness._probe_feasible(rows))

    def test_no_queue_and_processing_ttft_contract_is_vendored(self) -> None:
        config = json.loads((
            JOB_DIR / "source/paper/configs/"
            "xpyd_phase4b_binary_dvfs_uranus_ganymede.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["client"]["dispatch_mode"], "closed_loop_batches")
        self.assertEqual(config["client"]["closed_loop_batch_size"], 2)
        self.assertEqual(config["binary_dvfs"]["maximum_observed_concurrency"], 2)
        proxy = (JOB_DIR / "source/paper/scripts/xpyd/disagg_proxy.py").read_text()
        replay = (JOB_DIR / "source/paper/scripts/replay_synthetic_trace.py").read_text()
        self.assertIn('"processing_ttft"', proxy)
        self.assertIn('dispatch_mode == "closed_loop_batches"', replay)
        self.assertIn('result["client_queue_delay_ms"]', replay)

    def test_proxy_diagnostic_sync_waits_for_last_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.jsonl"
            path.write_text(
                json.dumps({"logical_request_id": "req-0"}) + "\n",
                encoding="utf-8",
            )

            def append_last() -> None:
                time.sleep(0.03)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"logical_request_id": "req-1"}) + "\n")

            thread = threading.Thread(target=append_last)
            thread.start()
            evidence = _wait_for_proxy_diagnostics(
                path, ["req-0", "req-1"], timeout_s=0.5, poll_interval_s=0.01
            )
            thread.join()
        self.assertTrue(evidence["valid"])
        self.assertEqual(evidence["observed_expected_count"], 2)
        self.assertGreaterEqual(evidence["wait_s"], 0.02)

    def test_proxy_diagnostic_sync_times_out_with_missing_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(Phase3CError, "missing=.*req-0"):
                _wait_for_proxy_diagnostics(
                    path, ["req-0"], timeout_s=0.02, poll_interval_s=0.005
                )


if __name__ == "__main__":
    unittest.main()
