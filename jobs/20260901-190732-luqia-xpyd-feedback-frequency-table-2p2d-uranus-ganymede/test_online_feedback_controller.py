"""CPU-only tests for 2P2D online feedback routing and exploration."""

import asyncio
import unittest

from xpyd.online_feedback_controller import (
    OnlineFeedbackController,
    OnlineFeedbackError,
    ProbeResult,
    classify_workload,
)
from xpyd.workload_frequency_table import WorkloadFrequencyTable


class FakeCore:
    def __init__(self):
        self.calls = []

    async def prepare(self, body, request_id):
        self.calls.append((body, request_id))
        return {"request_id": request_id}


class OnlineFeedbackControllerTests(unittest.TestCase):
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

            async def actuate(group, p_mhz, d_mhz):
                actions.append((group, p_mhz, d_mhz))

            async def probe(body, workload, p_mhz, d_mhz, probe_id):
                probes.append((workload, p_mhz, d_mhz, probe_id))
                safe = p_mhz >= 200 and d_mhz >= 20
                return ProbeResult(
                    p_mhz, d_mhz, float(p_mhz + d_mhz),
                    400.0 if safe else 600.0,
                    150.0 if safe else 250.0,
                )

            async def no_sleep(_):
                return None

            controller = OnlineFeedbackController(
                table, core, actuate, probe,
                (100, 200, 300, 400), (10, 20, 30, 40),
                probe_interval_s=20.0, sleep=no_sleep,
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
            self.assertEqual(entry.value.prefill_frequency_mhz, 200)
            self.assertEqual(entry.value.decode_frequency_mhz, 20)
            self.assertEqual(sum("confirm" in item[3] for item in probes), 1)
            self.assertTrue(all(call[0].get("xpyd_workload_id") is None
                                for call in core.calls))
            self.assertIn(("service", 400, 40), actions)
            await controller.handle(body, "table-hit")
            self.assertEqual(actions[-1], ("service", 200, 20))
            await controller.stop()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
