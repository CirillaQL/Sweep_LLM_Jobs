"""CPU-only tests for P1/D1 steady-state service aggregation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from xpyd.summarize_service_steady_state import summarize


class ServiceSteadyStateTests(unittest.TestCase):
    def test_discards_transient_prefix_and_summarizes_remaining_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dispatch = root / "dispatch.jsonl"
            requests = root / "requests.csv"
            output = root / "steady"
            dispatch.write_text("".join(
                json.dumps({
                    "request_id": f"req-{index}",
                    "service_sequence": index + 1,
                    "workload_id": "small_light",
                    "table_hit": True,
                    "table_revision": 1,
                    "frequency_source": "table",
                    "prefill_frequency_mhz": 900,
                    "decode_frequency_mhz": 450,
                    "frequency_changed": index == 0,
                    "settle_wait_s": 10.0 if index == 0 else 0.0,
                }) + "\n"
                for index in range(6)
            ), encoding="utf-8")
            with requests.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "request_id", "prefill_endpoint_id", "decode_endpoint_id",
                    "input_len", "requested_output_len", "ttft_ms", "tpot_ms",
                    "client_observed_ttft_ms", "client_observed_tpot_ms",
                ))
                writer.writeheader()
                for index, ttft in enumerate((900, 98, 102, 100, 99, 101)):
                    writer.writerow({
                        "request_id": f"req-{index}",
                        "prefill_endpoint_id": "P1", "decode_endpoint_id": "D1",
                        "input_len": 128, "requested_output_len": 64,
                        "ttft_ms": ttft, "tpot_ms": 60,
                        "client_observed_ttft_ms": ttft,
                        "client_observed_tpot_ms": 60,
                    })
            result = summarize(
                dispatch, requests, output,
                discard_first=1, minimum_samples=5, max_cv=0.10,
            )
            with (output / "service_requests_joined.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(result["stable_window_count"], 1)
        self.assertEqual(result["stable_workload_count"], 1)
        self.assertEqual(
            result["selected_steady_state_by_workload"][0]["workload_id"],
            "small_light",
        )
        self.assertEqual(result["windows"][0]["steady_samples"], 5)
        self.assertEqual(result["windows"][0]["ttft_ms"]["median"], 100.0)
        self.assertEqual(rows[0]["steady_state_included"], "False")
        self.assertEqual(rows[1]["steady_state_included"], "True")


if __name__ == "__main__":
    unittest.main()
