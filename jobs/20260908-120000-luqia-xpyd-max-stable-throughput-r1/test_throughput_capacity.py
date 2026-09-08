import unittest

from xpyd.throughput_capacity import candidate_is_stable, late_latency_stable


class CapacityLogicTests(unittest.TestCase):
    def rows(self, values):
        return [{"arrival_wall_s": index, "ttft_ms": value}
                for index, value in enumerate(values)]

    def test_flat_latency_is_stable(self):
        stable, early, late = late_latency_stable(
            self.rows([100, 101, 99, 100, 102, 101, 103, 100, 102]), 1.25, 50)
        self.assertTrue(stable)
        self.assertLess(abs(late - early), 10)

    def test_sustained_late_growth_is_unstable(self):
        stable, early, late = late_latency_stable(
            self.rows([100, 101, 99, 120, 130, 140, 220, 230, 240]), 1.25, 50)
        self.assertFalse(stable)
        self.assertGreater(late, early)

    def test_rows_are_ordered_by_arrival(self):
        rows = self.rows([100, 100, 100, 200, 200, 200])
        rows.reverse()
        stable, early, late = late_latency_stable(rows, 1.25, 50)
        self.assertFalse(stable)
        self.assertEqual(early, 100)
        self.assertEqual(late, 200)

    def test_candidate_requires_every_stability_gate(self):
        protocol = {
            "minimum_success_ratio": 0.99,
            "minimum_achieved_to_offered_ratio": 0.95,
        }
        slo = {"ttft_ms": 500, "tpot_ms": 200}
        values = dict(
            inflight_limit_hit=False,
            drain_timed_out=False,
            issued=30,
            planned=30,
            success_ratio=1.0,
            completion_ratio=0.98,
            ttft_p95_ms=499.9,
            tpot_p95_ms=200.0,
            trend_ok=True,
            protocol=protocol,
            slo=slo,
        )
        self.assertTrue(candidate_is_stable(**values))
        for key, bad in (
            ("inflight_limit_hit", True),
            ("drain_timed_out", True),
            ("issued", 29),
            ("success_ratio", 0.98),
            ("completion_ratio", 0.94),
            ("ttft_p95_ms", 500.0),
            ("tpot_p95_ms", 200.1),
            ("trend_ok", False),
        ):
            changed = dict(values)
            changed[key] = bad
            self.assertFalse(candidate_is_stable(**changed), key)


if __name__ == "__main__":
    unittest.main()
