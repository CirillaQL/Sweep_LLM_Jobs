import asyncio
import math
import unittest
from types import SimpleNamespace

from xpyd.load_frequency_energy import (
    best_valid,
    binary_energy_minimize,
    choose_binary_half,
    consume_timed,
    planned_request_count,
)
from xpyd.energy_validation import (
    attribute_interval_energy,
    integrate_intervals,
    merge_intervals,
)


def row(frequency, energy, valid=True):
    return {
        "candidate_id": f"f-{frequency}",
        "axis_frequency_mhz": frequency,
        "joules_per_completed_request": energy,
        "measurement_valid": valid,
    }


class LoadFrequencyEnergyTests(unittest.TestCase):
    def test_planned_requests_preserve_low_rate_and_bound_high_rate(self):
        self.assertEqual(planned_request_count(0.1, 20, 12, 60), 12)
        self.assertEqual(planned_request_count(1.0, 20, 12, 60), 20)
        self.assertEqual(planned_request_count(3.0, 20, 12, 60), 60)
        self.assertEqual(planned_request_count(4.0, 20, 12, 60), 60)

    def test_invalid_pair_moves_toward_higher_frequency(self):
        invalid = row(900, None, valid=False)
        self.assertEqual(choose_binary_half(invalid, invalid), "right")
        self.assertEqual(choose_binary_half(invalid, row(1000, 12.0)), "right")
        self.assertEqual(choose_binary_half(row(900, 11.0), invalid), "left")

    def test_best_valid_ignores_failed_measurements(self):
        selected = best_valid([
            row(900, 1.0, valid=False),
            row(1000, 8.0),
            row(1100, 7.0),
        ])
        self.assertEqual(selected["axis_frequency_mhz"], 1100)

    def test_binary_energy_search_finds_unimodal_minimum(self):
        grid = [900, 1000, 1100, 1200, 1300]
        energies = [9.0, 5.0, 2.0, 3.0, 8.0]
        evaluated = []

        async def evaluate(index, frequency):
            evaluated.append(index)
            return row(frequency, energies[index])

        result = asyncio.run(binary_energy_minimize(grid, evaluate))
        self.assertEqual(result["selected_frequency_mhz"], 1100)
        self.assertFalse(result["global_oracle_proven"])
        self.assertIn(0, evaluated)
        self.assertIn(len(grid) - 1, evaluated)
        self.assertEqual(len(evaluated), len(set(evaluated)))

    def test_objective_does_not_treat_invalid_as_low_energy(self):
        invalid = row(900, 0.1, valid=False)
        valid = row(1000, 10.0, valid=True)
        self.assertEqual(choose_binary_half(invalid, valid), "right")
        with self.assertRaises(RuntimeError):
            best_valid([invalid, row(1100, math.nan, valid=True)])

    def test_request_interval_union_excludes_idle_gap(self):
        self.assertEqual(
            merge_intervals([(1.0, 2.0), (1.5, 2.5), (4.0, 5.0)]),
            [(1.0, 2.5), (4.0, 5.0)],
        )
        samples = [
            {"t": index * 0.1, "energy_j": index * 10.0,
             "clock_mhz": 900, "power_w": 100.0}
            for index in range(61)
        ]
        result = integrate_intervals(
            samples, [(1.0, 2.0), (4.0, 5.0)], target=900)
        self.assertAlmostEqual(result["duration_s"], 2.0)
        self.assertAlmostEqual(result["energy_j"], 200.0)
        self.assertEqual(result["interval_count"], 2)

    def test_concurrent_attribution_conserves_union_energy(self):
        samples = [
            {"t": index * 0.1, "energy_j": index * 10.0,
             "clock_mhz": 900, "power_w": 100.0}
            for index in range(51)
        ]
        intervals = [("a", 1.0, 3.0), ("b", 2.0, 4.0)]
        attributed = attribute_interval_energy(samples, intervals)
        union = integrate_intervals(samples, [(1.0, 4.0)])
        self.assertAlmostEqual(attributed["a"], 150.0)
        self.assertAlmostEqual(attributed["b"], 150.0)
        self.assertAlmostEqual(sum(attributed.values()), union["energy_j"])

    def test_stream_consumer_timestamps_first_nonempty_token(self):
        async def stream():
            yield b'data: {"choices":[{"text":"x"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'

        result = asyncio.run(consume_timed(
            SimpleNamespace(status_code=200, stream=stream()), 2))
        self.assertEqual(result["observed_output_tokens"], 2)
        self.assertIsInstance(result["client_first_token_wall_ns"], int)
        self.assertIsInstance(result["client_first_token_mono_ns"], int)
        self.assertEqual(result["client_stream_chunk_count"], 3)


if __name__ == "__main__":
    unittest.main()
