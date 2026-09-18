import unittest
from unittest.mock import patch

from xpyd.nodegroup_telemetry import NodeGroupTelemetry


CONFIG = {
    "telemetry_contract": {
        "arrival_rate_window_s": 60,
        "prefill_capacity_input_tokens_per_s": 2000,
        "decode_capacity_output_tokens_per_s": 1000,
    },
    "endpoints": [
        {"endpoint_id": "P0", "role": "prefill", "gpu_type": "NVIDIA L40S", "node": "neptune", "gpu_ids": [0], "tp_degree": 1, "http_uri": "http://p:8100", "kv_connector": "P2pNcclConnector"},
        {"endpoint_id": "D0", "role": "decode", "gpu_type": "NVIDIA L4", "node": "ganymede", "gpu_ids": [0], "tp_degree": 1, "http_uri": "http://d:8200", "kv_connector": "P2pNcclConnector"},
    ],
}


class NodeGroupTelemetryTests(unittest.TestCase):
    def test_complete_record_and_rho_use_measured_rates(self):
        telemetry = NodeGroupTelemetry(CONFIG)
        states = iter((
            ({"running": 2, "waiting": 1, "kv_usage": 0.1, "prompt_rate": 1000, "generation_rate": 20, "missing_metrics": [], "window_reason": "ok"}, []),
            ({"running": 3, "waiting": 4, "kv_usage": 0.5, "prompt_rate": 10, "generation_rate": 500, "missing_metrics": [], "window_reason": "ok"}, []),
        ))
        with patch.object(telemetry, "_scrape", side_effect=states), patch("xpyd.nodegroup_telemetry.time.time", return_value=100.0):
            row = telemetry.build(source="canary", sent_timestamp_s=99.0, pair=("P0", "D0"), input_tokens=128, max_output_tokens=64, actual_output_tokens=61, targets_mhz=(900, 450), endpoint_energy={"P0": {"energy_j": 10, "observed_clock_min_mhz": 897, "observed_clock_max_mhz": 901}, "D0": {"energy_j": 5, "observed_clock_min_mhz": 449, "observed_clock_max_mhz": 451}}, ttft_ms=120, tpot_ms=30, slo_met=True)
        self.assertEqual(row["source"], "canary")
        self.assertEqual(row["batch_size"], 3)
        self.assertEqual(row["P_rho"], 0.5)
        self.assertEqual(row["D_rho"], 0.5)
        self.assertEqual(row["total_energy"], 15)
        self.assertEqual(row["actual_output_tokens"], row["output_tokens"])
        self.assertEqual(row["P_freq"]["target_mhz"], 900)

    def test_unavailable_metrics_and_uncalibrated_capacity_remain_null(self):
        config = dict(CONFIG)
        config["telemetry_contract"] = dict(CONFIG["telemetry_contract"], prefill_capacity_input_tokens_per_s=None, decode_capacity_output_tokens_per_s=None)
        telemetry = NodeGroupTelemetry(config)
        with patch.object(telemetry, "_scrape", return_value=({}, ["P0:VLLMMetricsConnectionError:down"])), patch("xpyd.nodegroup_telemetry.time.time", return_value=100.0):
            row = telemetry.build(source="production", sent_timestamp_s=99.0, pair=("P0", "D0"), input_tokens=128, max_output_tokens=64, actual_output_tokens=64, targets_mhz=(2520, 1500), endpoint_energy={"P0": {"energy_j": 10}, "D0": {"energy_j": 5}}, ttft_ms=120, tpot_ms=30, slo_met=True)
        self.assertIsNone(row["P_rho"])
        self.assertIsNone(row["D_rho"])
        self.assertTrue(row["telemetry_errors"])


if __name__ == "__main__":
    unittest.main()
