"""CPU-only Phase 3C configuration and audit primitives."""

import csv
from pathlib import Path
import tempfile
import unittest

from xpyd.phase3c_substrate import (
    Phase3CError,
    _online_inference_latency,
    _throttle_audit,
    _write_trace,
    build_registry_and_compatibility,
    load_config,
)


class Phase3CSubstrateTests(unittest.TestCase):
    def test_windowed_trace_finishes_each_workload_before_next(self):
        workloads = [
            {"id": "small", "input_len": 8, "output_len": 2,
             "count": 3, "rate_rps": 1.0},
            {"id": "large", "input_len": 16, "output_len": 4,
             "count": 2, "rate_rps": 1.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.assertEqual(_write_trace(
                path, workloads, "window-test", ordering="windowed"
            ), 5)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(
            [row["workload_id"] for row in rows],
            ["small", "small", "small", "large", "large"],
        )

    def config(self):
        endpoints = []
        for endpoint_id, role, node, gpu in (
            ("P0", "prefill", "pnode", 0), ("P1", "prefill", "pnode", 1),
            ("D0", "decode", "dnode", 0), ("D1", "decode", "dnode", 1),
        ):
            endpoints.append({
                "endpoint_id": endpoint_id, "role": role, "node": node,
                "gpu_type": "fixture", "gpu_ids": [gpu], "tp_degree": 1,
                "http_uri": "http://%s:%d" % (node, 8000 + gpu),
                "kv_connector": "test-kv",
            })
        return {
            "endpoints": endpoints,
            "fixed_clocks": {item["endpoint_id"]: {"graphics_mhz": 1500} for item in endpoints},
            "compatible_pairs": [
                {"prefill_endpoint_id": p, "decode_endpoint_id": d,
                 "connector": "test-kv", "supported": True, "reason": "fixture"}
                for p in ("P0", "P1") for d in ("D0", "D1")
            ],
        }

    def test_registry_rejects_overlap_and_exact_pairs_are_enumerated(self):
        registry, table, pairs = build_registry_and_compatibility(self.config())
        self.assertEqual(len(registry.healthy_active()), 4)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(table.is_compatible(registry.get_spec("P1"), registry.get_spec("D0")))

    def test_online_metrics_exclude_wait_but_include_full_prefill(self):
        request = {"completion_tokens": 4, "ttft_ms": 10700.0}
        diagnostic = {
            "timestamps_monotonic_s": {
                "request_received": 20.0,
                "decode_first_real_chunk_received": 20.7,
            },
            "durations_ms": {"full_decode_stream": 600.0},
        }
        ttft_ms, tpot_ms = _online_inference_latency(request, diagnostic)
        self.assertAlmostEqual(ttft_ms, 700.0)
        self.assertAlmostEqual(tpot_ms, 200.0)

    def test_throttle_audit_requires_available_noninvalidating_samples(self):
        records = [{
            "status": "success", "actual_start_wall_s": 2.0,
            "invalidating_thermal_or_hw_slowdown": False,
            "clock_throttle_reasons": ["sw_power_cap"],
        }]
        audit = _throttle_audit(records, 1.0, 3.0)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["observed_reasons"], ["sw_power_cap"])
        records[0]["invalidating_thermal_or_hw_slowdown"] = True
        self.assertFalse(_throttle_audit(records, 1.0, 3.0)["valid"])

    def test_default_phase3c_config_keeps_full_coverage_policy(self):
        config = self.config()
        self.assertNotIn("coverage_policy", config)
        _, _, pairs = build_registry_and_compatibility(config)
        self.assertEqual(set(pairs), {
            ("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"),
        })

    def test_4p4d_random_job_declares_all_sixteen_pairs(self):
        root = Path(__file__).resolve().parent / "source"
        config = load_config(
            root / "paper/configs/xpyd_phase3c_4p4d_random_l40s_l4.json"
        )
        registry, _, pairs = build_registry_and_compatibility(config)

        self.assertEqual(config["routing_policy"], "random")
        self.assertEqual(len(registry.healthy_active()), 8)
        self.assertEqual(len(pairs), 16)
        self.assertEqual(set(pairs), {
            (prefill, decode)
            for prefill in ("P0", "P1", "P2", "P3")
            for decode in ("D0", "D1", "D2", "D3")
        })
        self.assertGreaterEqual(
            sum(int(item["count"]) for item in config["workloads"]),
            len(pairs),
        )
        self.assertEqual(
            [
                (item["id"], item["input_len"], item["output_len"], item["count"])
                for item in config["workloads"]
            ],
            [
                ("small_light", 128, 64, 16),
                ("prefill_medium", 1024, 64, 16),
                ("prefill_heavy", 2048, 64, 16),
                ("decode_medium", 128, 128, 16),
                ("decode_heavy", 128, 256, 16),
                ("balanced_medium", 512, 128, 16),
                ("both_heavy", 2048, 256, 16),
            ],
        )

    def test_dynamic_multi_endpoint_launch_disables_endpoint_local_prefix_cache(self):
        root = Path(__file__).resolve().parent / "source"
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        phase3c_launch = source.split("run_phase3c_substrate()", 1)[1].split(
            "# ===========================================================================\n"
            "run_experiment_E()",
            1,
        )[0]
        self.assertIn('no_local_prefix_cache="--no-enable-prefix-caching"', phase3c_launch)
        self.assertIn(
            'start_vllm_server "${L40S_NODE}" "${http_port}"', phase3c_launch
        )
        self.assertIn(
            'start_vllm_server "${L4_NODE}" "${http_port}"', phase3c_launch
        )
        self.assertEqual(phase3c_launch.count("${no_local_prefix_cache}"), 2)

    def test_dynamic_multi_endpoint_launch_disables_unvalidated_chunked_prefill(self):
        root = Path(__file__).resolve().parent / "source"
        source = (root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
        phase3c_launch = source.split("run_phase3c_substrate()", 1)[1].split(
            "# ===========================================================================\n"
            "run_experiment_E()",
            1,
        )[0]
        self.assertIn('no_chunked_prefill="--no-enable-chunked-prefill"', phase3c_launch)
        self.assertGreaterEqual(
            phase3c_launch.count("endpoint_index<XPYD_ENDPOINTS_PER_ROLE"), 4
        )
        self.assertEqual(phase3c_launch.count("${no_chunked_prefill}"), 2)

    def test_persistent_windows_use_distinct_request_id_namespaces(self):
        workloads = [{
            "id": "fixture", "input_len": 128, "output_len": 16,
            "count": 2, "rate_rps": 1.0,
        }]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            _write_trace(first, workloads, "phase3c-light-before")
            _write_trace(second, workloads, "phase3c-moderate")
            with first.open(newline="", encoding="utf-8") as stream:
                first_rows = list(csv.DictReader(stream))
                first_ids = {row["request_id"] for row in first_rows}
            with second.open(newline="", encoding="utf-8") as stream:
                second_ids = {row["request_id"] for row in csv.DictReader(stream)}
        self.assertEqual(first_ids, {
            "phase3c-light-before-0000", "phase3c-light-before-0001",
        })
        self.assertEqual(second_ids, {
            "phase3c-moderate-0000", "phase3c-moderate-0001",
        })
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertEqual({row["workload_id"] for row in first_rows}, {"fixture"})

    def test_request_id_namespace_rejects_unsafe_or_empty_values(self):
        workloads = [{
            "id": "fixture", "input_len": 1, "output_len": 1,
            "count": 1, "rate_rps": 1.0,
        }]
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(("", "has spaces", "unsafe/slash")):
                with self.assertRaises(Phase3CError):
                    _write_trace(Path(directory) / ("%d.csv" % index), workloads, value)

    def test_4p4d_workloads_are_interleaved_by_logical_class(self):
        root = Path(__file__).resolve().parent / "source"
        config = load_config(
            root / "paper/configs/xpyd_phase3c_4p4d_random_l40s_l4.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            count = _write_trace(trace, config["workloads"], "phase3c-4p4d")
            with trace.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(count, 112)
        self.assertEqual(
            [row["workload_id"] for row in rows[:7]],
            [item["id"] for item in config["workloads"]],
        )
        self.assertEqual(
            {workload_id: sum(row["workload_id"] == workload_id for row in rows)
             for workload_id in {row["workload_id"] for row in rows}},
            {item["id"]: 16 for item in config["workloads"]},
        )


if __name__ == "__main__":
    unittest.main()
