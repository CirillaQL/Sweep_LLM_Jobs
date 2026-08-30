#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest


JOB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(JOB_DIR / "source" / "paper" / "scripts"))

from xpyd.binary_dvfs_evaluation import (  # noqa: E402
    BinaryDVFSError,
    balanced_frozen_routes_valid,
    binary_update,
    peak_observed_concurrency,
    select_frequency_grid,
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


if __name__ == "__main__":
    unittest.main()
