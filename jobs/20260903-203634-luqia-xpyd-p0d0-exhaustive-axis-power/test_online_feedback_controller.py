"""CPU-only tests for 2P2D online feedback routing and exploration."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from xpyd.online_feedback_controller import (
    DEFAULT_AXIS_SEARCH_LEVELS,
    OnlineFeedbackController,
    OnlineFeedbackError,
    ProbeResult,
    aggregate_probe_results,
    classify_workload,
    pd_inference_metrics,
    select_grid,
)
from xpyd.workload_frequency_table import WorkloadFrequencyTable


class FakeCore:
    def __init__(self):
        self.calls = []

    async def prepare(self, body, request_id):
        self.calls.append((body, request_id))
        return {"request_id": request_id}


class OnlineFeedbackControllerTests(unittest.TestCase):
    def test_default_axis_search_uses_seventeen_prefill_and_fifteen_decode_levels(self):
        self.assertEqual(DEFAULT_AXIS_SEARCH_LEVELS, {"prefill": 17, "decode": 15})

    def test_exploration_ttft_boundary_is_strict(self):
        result = ProbeResult(900, 450, 100.0, 50.0, 500.0, 60.0)
        self.assertFalse(result.feasible(500.0, 200.0))
        self.assertTrue(ProbeResult(
            900, 450, 100.0, 50.0, 499.999, 60.0
        ).feasible(500.0, 200.0))

    def test_candidate_uses_p95_latency_and_mean_energy(self):
        result = aggregate_probe_results([
            ProbeResult(900, 450, 100.0, 40.0, 400.0, 60.0),
            ProbeResult(900, 450, 110.0, 50.0, 450.0, 70.0),
            ProbeResult(900, 450, 120.0, 60.0, 550.0, 80.0),
        ])
        self.assertAlmostEqual(result.measured_energy_j, 50.0)
        self.assertAlmostEqual(result.ttft_ms, 540.0)
        self.assertAlmostEqual(result.tpot_ms, 79.0)
        self.assertEqual(result.sample_count, 3)
        self.assertFalse(result.feasible(500.0, 200.0))

    def test_preferred_grid_must_be_supported_and_include_safe_high(self):
        self.assertEqual(
            select_grid((450, 720, 975, 1230, 1500), 450, 1500, 4,
                        (450, 720, 975, 1500)),
            (450, 720, 975, 1500),
        )
        with self.assertRaises(OnlineFeedbackError):
            select_grid((450, 720, 975, 1500), 450, 1500, 3,
                        (450, 720, 975))

    def test_stable_candidate_stops_at_three_but_confirmation_uses_five(self):
        async def scenario():
            calls = []

            async def actuate(*_):
                return False

            async def probe(body, workload, p_mhz, d_mhz, probe_id):
                calls.append(probe_id)
                return ProbeResult(p_mhz, d_mhz, 100.0, 50.0, 300.0, 60.0)

            controller = OnlineFeedbackController(
                WorkloadFrequencyTable(), FakeCore(), actuate, probe,
                (900, 2520), (450, 1500), probe_interval_s=0.0,
                probe_samples_per_candidate=5, minimum_probe_samples=3,
            )
            body = {"max_tokens": 64, "xpyd_output_len": 64}
            adaptive = await controller._probe_candidate(
                body, "small_light", 900, 450, "adaptive"
            )
            confirmation = await controller._probe_candidate(
                body, "small_light", 900, 450, "confirmation", adaptive=False
            )
            self.assertEqual(adaptive.sample_count, 3)
            self.assertEqual(confirmation.sample_count, 5)
            self.assertEqual(len(calls), 8)

        asyncio.run(scenario())

    def test_classifies_exact_shape_and_rejects_mismatch(self):
        self.assertEqual(classify_workload({
            "xpyd_input_len": 128, "xpyd_output_len": 64,
            "xpyd_workload_id": "small_light", "max_tokens": 64,
        }), "small_light")
        with self.assertRaises(OnlineFeedbackError):
            classify_workload({
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "both_heavy", "max_tokens": 64,
            })

    def test_miss_serves_high_and_serial_exploration_populates_table(self):
        async def scenario():
            table = WorkloadFrequencyTable()
            core = FakeCore()
            actions = []
            probes = []
            sleeps = []

            async def actuate(group, p_mhz, d_mhz):
                actions.append((group, p_mhz, d_mhz))
                return False

            async def probe(body, workload, p_mhz, d_mhz, probe_id):
                probes.append((workload, p_mhz, d_mhz, probe_id))
                safe = p_mhz >= 200 and d_mhz >= 20
                p_energy = {200: 30.0, 300: 10.0, 400: 20.0}
                d_energy = {20: 15.0, 30: 5.0, 40: 10.0}
                p_value = p_energy.get(p_mhz, 100.0)
                d_value = d_energy.get(d_mhz, 100.0)
                energy = p_value + d_value
                return ProbeResult(
                    p_mhz, d_mhz, float(p_mhz + d_mhz), energy,
                    400.0 if safe else 600.0,
                    150.0 if safe else 250.0,
                    prefill_power_w=p_value,
                    decode_power_w=d_value,
                    prefill_energy_j=p_value,
                    decode_energy_j=d_value,
                )

            async def no_sleep(seconds):
                sleeps.append(seconds)

            controller = OnlineFeedbackController(
                table, core, actuate, probe,
                (100, 200, 300, 400), (10, 20, 30, 40),
                probe_interval_s=0.0, probe_samples_per_candidate=3,
                sleep=no_sleep,
            )
            await controller.start()
            body = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "small_light",
            }
            await asyncio.gather(*[
                controller.handle(body, "request-%d" % index)
                for index in range(8)
            ])
            await controller._queue.join()
            entry = table.read("small_light")
            self.assertIsNotNone(entry.value)
            self.assertEqual(entry.value.prefill_frequency_mhz, 300)
            self.assertEqual(entry.value.decode_frequency_mhz, 30)
            self.assertEqual(entry.value.measured_energy_j, 15.0)
            self.assertEqual(entry.value.sample_count, 3)
            self.assertFalse(any("confirm" in item[3] for item in probes))
            self.assertEqual(len(probes), 24)
            self.assertEqual(sleeps, [])
            self.assertTrue(all(call[0].get("xpyd_workload_id") is None
                                for call in core.calls))
            self.assertIn(("service", 400, 40), actions)
            await controller.handle(body, "table-hit")
            self.assertEqual(actions[-1], ("service", 300, 30))
            await controller.stop()

        asyncio.run(scenario())

    def test_pd_ttft_includes_prefill_but_excludes_frequency_wait(self):
        ttft_ms, tpot_ms = pd_inference_metrics({
            "request_received": 10.0,
            "decode_request_started": 10.5,
            "decode_first_real_chunk_received": 10.7,
            "decode_last_chunk_received": 11.3,
        }, output_len=4)
        self.assertAlmostEqual(ttft_ms, 700.0)
        self.assertAlmostEqual(tpot_ms, 200.0)

    def test_changed_service_frequency_settles_before_inference(self):
        async def scenario():
            table = WorkloadFrequencyTable()
            core = FakeCore()
            order = []

            async def actuate(group, p_mhz, d_mhz):
                order.append(("actuate", group, p_mhz, d_mhz))
                return True

            async def probe(*_):
                raise AssertionError("exploration must not run in this test")

            async def sleep(seconds):
                order.append(("sleep", seconds))

            original_prepare = core.prepare

            async def prepare(body, request_id):
                order.append(("prepare", request_id))
                return await original_prepare(body, request_id)

            core.prepare = prepare
            controller = OnlineFeedbackController(
                table, core, actuate, probe, (900, 2520), (450, 1500),
                probe_interval_s=0.0, service_settle_s=10.0, sleep=sleep,
            )
            body = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "small_light",
            }
            await controller.handle(body, "settled-request")
            self.assertEqual(order[:3], [
                ("actuate", "service", 2520, 1500),
                ("sleep", 10.0),
                ("prepare", "settled-request"),
            ])

        asyncio.run(scenario())

    def test_experiment_warmup_runs_once_before_measured_search(self):
        async def scenario():
            table = WorkloadFrequencyTable()
            core = FakeCore()
            probe_ids = []

            async def actuate(*_):
                return False

            async def probe(body, workload, p_mhz, d_mhz, probe_id):
                probe_ids.append(probe_id)
                return ProbeResult(
                    p_mhz, d_mhz, 100.0, 50.0, 400.0, 100.0,
                    prefill_power_w=float(p_mhz),
                    decode_power_w=float(d_mhz),
                    prefill_energy_j=25.0,
                    decode_energy_j=25.0,
                )

            async def no_sleep(_):
                return None

            controller = OnlineFeedbackController(
                table, core, actuate, probe,
                (100, 200, 300, 400), (10, 20, 30, 40),
                probe_interval_s=0.0,
                experiment_warmup_requests=2,
                probe_samples_per_candidate=2,
                minimum_probe_samples=2,
                sleep=no_sleep,
            )
            first = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "small_light",
            }
            second = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 1024, "xpyd_output_len": 64,
                "xpyd_workload_id": "prefill_medium",
            }
            await controller._explore("small_light", first, "first")
            first_search_start = next(
                index for index, value in enumerate(probe_ids)
                if "explore-" in value
            )
            self.assertEqual(probe_ids[:first_search_start], [
                "first-warmup-1", "first-warmup-2",
            ])
            self.assertIsNotNone(table.read("small_light").value)
            await controller._explore("prefill_medium", second, "second")
            self.assertFalse(any("second-warmup" in value for value in probe_ids))
            self.assertIsNotNone(table.read("prefill_medium").value)

        asyncio.run(scenario())

    def test_service_warmup_runs_once_before_first_measured_request(self):
        async def scenario():
            table = WorkloadFrequencyTable()
            core = FakeCore()

            async def actuate(*_):
                return False

            async def probe(*_):
                raise AssertionError("exploration worker is not started")

            controller = OnlineFeedbackController(
                table, core, actuate, probe, (900, 2520), (450, 1500),
                probe_interval_s=0.0,
                service_warmup_requests=2,
            )
            body = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "small_light",
            }
            await controller.handle(body, "first")
            await controller.handle(body, "second")
            self.assertEqual([request_id for _, request_id in core.calls], [
                "first-service-warmup-1",
                "first-service-warmup-2",
                "first",
                "second",
            ])
            self.assertTrue(all(
                call_body.get("xpyd_workload_id") is None
                for call_body, _ in core.calls
            ))

        asyncio.run(scenario())

    def test_formal_service_dispatch_records_frequency_context(self):
        async def scenario(path):
            table = WorkloadFrequencyTable()
            core = FakeCore()

            async def actuate(*_):
                return True

            async def probe(*_):
                raise AssertionError("exploration worker is not started")

            controller = OnlineFeedbackController(
                table, core, actuate, probe, (900, 2520), (450, 1500),
                probe_interval_s=0.0,
                service_settle_s=0.0,
                service_request_log=str(path),
            )
            body = {
                "prompt": "x", "stream": True, "max_tokens": 64,
                "xpyd_input_len": 128, "xpyd_output_len": 64,
                "xpyd_workload_id": "small_light",
            }
            await controller.handle(body, "formal-1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.jsonl"
            asyncio.run(scenario(path))
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["request_id"], "formal-1")
        self.assertEqual(row["service_sequence"], 1)
        self.assertFalse(row["table_hit"])
        self.assertEqual(row["frequency_source"], "safe_high")
        self.assertEqual(row["prefill_frequency_mhz"], 2520)
        self.assertEqual(row["decode_frequency_mhz"], 1500)


if __name__ == "__main__":
    unittest.main()
