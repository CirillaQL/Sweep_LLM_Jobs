"""CPU-only concurrency tests for the proxy workload-frequency table."""

from concurrent.futures import ThreadPoolExecutor
import unittest

from xpyd.workload_frequency_table import (
    FrequencyTableError,
    FrequencyTableValue,
    WORKLOAD_KEYS,
    WorkloadFrequencyTable,
)


def value(
    energy: float, p_mhz: int = 1500, d_mhz: int = 900,
    power: float = 120.0,
) -> FrequencyTableValue:
    return FrequencyTableValue(
        prefill_frequency_mhz=p_mhz,
        decode_frequency_mhz=d_mhz,
        measured_power_w=power,
        measured_energy_j=energy,
        ttft_ms=400.0,
        tpot_ms=150.0,
        prefill_endpoint_id="P0",
        decode_endpoint_id="D0",
        sample_count=16,
        updated_unix_s=1.0,
        source="test",
    )


class WorkloadFrequencyTableTests(unittest.TestCase):
    def test_has_exactly_seven_empty_logical_keys(self):
        table = WorkloadFrequencyTable()
        self.assertEqual(table.keys, WORKLOAD_KEYS)
        self.assertTrue(all(item["value"] is None for item in table.snapshot().values()))

    def test_read_write_and_revision_conflict(self):
        table = WorkloadFrequencyTable()
        written = table.write("small_light", value(120.0), expected_revision=0)
        self.assertEqual(written.revision, 1)
        self.assertEqual(table.read("small_light"), written)
        with self.assertRaisesRegex(FrequencyTableError, "revision conflict"):
            table.write("small_light", value(110.0), expected_revision=0)

    def test_rejects_non_slo_safe_value(self):
        table = WorkloadFrequencyTable()
        unsafe = value(100.0).__dict__ | {"slo_met": False}
        with self.assertRaisesRegex(FrequencyTableError, "SLO-safe"):
            table.write("small_light", unsafe)
        violation = value(100.0).__dict__ | {"ttft_ms": 501.0}
        with self.assertRaisesRegex(FrequencyTableError, "exceeds"):
            table.write("small_light", violation)
        strict_boundary = value(100.0).__dict__ | {"ttft_ms": 500.0}
        with self.assertRaisesRegex(FrequencyTableError, "exceeds"):
            table.write("small_light", strict_boundary)

    def test_write_ranks_by_energy_not_power(self):
        table = WorkloadFrequencyTable()
        table.write("small_light", value(100.0, power=90.0))
        table.write("small_light", value(80.0, power=150.0))
        selected = table.read("small_light").value
        self.assertEqual(selected.measured_energy_j, 80.0)
        self.assertEqual(selected.measured_power_w, 150.0)

    def test_concurrent_writes_keep_lowest_energy_observation(self):
        table = WorkloadFrequencyTable()
        energies = [180.0, 140.0, 175.0, 95.0, 120.0, 101.0, 160.0]
        with ThreadPoolExecutor(max_workers=len(energies)) as executor:
            list(executor.map(
                lambda energy: table.write("both_heavy", value(energy)),
                energies,
            ))
        entry = table.read("both_heavy")
        self.assertEqual(entry.value.measured_energy_j, min(energies))
        self.assertGreaterEqual(entry.revision, 1)

    def test_concurrent_reads_return_immutable_consistent_entries(self):
        table = WorkloadFrequencyTable()
        table.write("decode_heavy", value(130.0))
        with ThreadPoolExecutor(max_workers=16) as executor:
            entries = list(executor.map(
                lambda _: table.read("decode_heavy"), range(200)
            ))
        self.assertTrue(all(item == entries[0] for item in entries))


if __name__ == "__main__":
    unittest.main()
