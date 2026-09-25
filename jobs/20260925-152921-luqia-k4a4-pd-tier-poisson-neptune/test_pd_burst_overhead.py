"""Local tests for the K1/K1b/K4a driver against a fake P/D proxy core (no GPU, no vLLM)."""

import asyncio
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "source" / "paper" / "scripts"))

from xpyd.online_feedback_controller import pd_inference_metrics, strip_feedback_metadata  # noqa: E402
from xpyd import pd_burst_overhead as k1  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "source/paper/configs/xpyd_p0d0_k4a_tier_poisson.json"


class FakeCore:
    """Batches concurrent prefills: completion time grows with requests in flight."""

    def __init__(self) -> None:
        self.inflight = 0
        self.bodies = []

    async def prepare(self, body, request_id):
        assert not any(k.startswith("xpyd_") for k in body), "metadata must be stripped"
        self.bodies.append(body)
        stamps = {"request_received": time.monotonic()}
        self.inflight += 1
        stamps["prefill_started"] = time.monotonic()
        await asyncio.sleep(0.002 * self.inflight)
        stamps["prefill_completed"] = stamps["kv_handoff_completed"] = time.monotonic()
        self.inflight -= 1
        stamps["decode_request_started"] = time.monotonic()
        n = int(body["max_tokens"])

        async def stream():
            for i in range(n):
                await asyncio.sleep(0.0005)
                now = time.monotonic()
                if i == 0:
                    stamps["decode_first_real_chunk_received"] = now
                stamps["decode_last_chunk_received"] = now
                yield b"x"
            stamps["response_completed"] = time.monotonic()

        return SimpleNamespace(stream=stream(), diagnostics=SimpleNamespace(timestamps_monotonic_s=stamps))


class FakeRuntime:
    def __init__(self) -> None:
        self.clocks = []

    async def actuate(self, group, p, d):
        assert group == "experiment"
        changed = (p, d) != (self.clocks[-1] if self.clocks else None)
        self.clocks.append((p, d))
        return changed

    def _energy_mj(self, endpoint):
        return time.monotonic() * 1000.0


def small_config():
    cfg = json.loads(CONFIG.read_text())
    s = cfg["burst_overhead"]
    s.update(burst_reps=1, background_levels=[0, 2], background_output_len=60, background_warmup_s=0.01,
             busy_probe_reps=1, busy_probe_gap_s=0.0, freq_burst_reps=1, idle_gap_s=0.0,
             frequency_settle_s=0.0, reserve_s=0.0,
             phase_order=["bursts_reference", "decode_busy_1500", "bursts_p1305", "tier_poisson", "park_idle"])
    s["tier_poisson"].update(rates_rps=[150, 250], duration_s=0.3, drain_allowance_s=0.0, settle_s=0.0, output_len=4)
    s["park_idle"].update(settle_s=0.0, measure_s=0.05)
    cfg["model"] = "m"
    return cfg


class DriverTest(unittest.TestCase):
    def run_driver(self, cfg, deadline):
        out = Path(tempfile.mkdtemp())
        runtime = FakeRuntime()
        prompts = {l: "p" * l for l in (128, 512, 1024, 2048)}
        drv = k1.Driver(cfg, FakeCore(), runtime, prompts, out, deadline,
                        pd_inference_metrics, strip_feedback_metadata)
        asyncio.run(drv.run())
        return drv, runtime, out

    def test_all_phases_and_columns(self):
        drv, runtime, out = self.run_driver(small_config(), time.time() + 60)
        rows = list(csv.DictReader((out / "pd_requests.csv").open()))
        phases = {r["phase"] for r in rows}
        self.assertTrue({"bursts_reference", "decode_busy", "background", "bursts_p1305",
                         "tier_1305", "tier_1815", "tier_2520"} <= phases)
        self.assertTrue(all(r["status"] == "ok" for r in rows), [r["error"] for r in rows if r["status"] != "ok"])
        self.assertEqual(sum(r["phase"] == "bursts_reference" for r in rows), 2 * (1 + 2 + 4 + 8))
        self.assertTrue(all(float(r["ttft_ms"]) > 0 for r in rows))
        self.assertIn((2520, 1050), runtime.clocks)
        pois = [r for r in rows if r["phase"].startswith("tier_")]
        self.assertGreater(len(pois), 20)
        self.assertTrue(any(int(r["awaiting_first_requests"]) > 0 for r in pois), "state at arrival never non-zero")
        self.assertTrue(all(int(r["awaiting_first_tokens"]) >= 0 for r in pois))
        self.assertTrue(any(p.get("phase") == "decode_busy" and p.get("completed") for p in drv.phase_log))
        # the same arrival sequence is replayed at every P clock for a given rate
        for rate in ("150.0", "250.0"):
            seqs = {}
            for r in rows:
                if r["phase"].startswith("tier_") and r["rate_rps"] == rate:
                    seqs.setdefault(r["phase"], []).append((round(float(r["arrival_s"]), 6), r["input_len"]))
            self.assertEqual(len(seqs), 3)
            self.assertEqual(len({tuple(sorted(v)) for v in seqs.values()}), 1, "arrivals differ between P clocks")
        tiers = [p for p in drv.phase_log if str(p.get("phase", "")).startswith("tier_")]
        self.assertEqual([p["p_mhz"] for p in tiers], [1305, 1815, 2520, 1815, 2520, 1305])  # rotated order
        self.assertIn((1305, 1050), runtime.clocks)
        parks = [p for p in drv.phase_log if p.get("phase") == "park_idle"]
        self.assertEqual([(p["p_mhz"], p["d_mhz"]) for p in parks], [(900, 1050), (1305, 1050), (1815, 1050)])
        busy = [r for r in rows if r["phase"] == "decode_busy" and r["background_target"] == "2"]
        self.assertTrue(busy and all(int(r["background_active"]) >= 1 for r in busy))
        summary = k1.summarize(drv.sink.rows)
        self.assertTrue(any(g["phase"] == "bursts_reference" and g["burst_size_or_rate"] == 8 for g in summary))
        self.assertIsNotNone(summary[0]["prefill_ms_p50"])

    def test_deadline_stops_cleanly(self):
        drv, _, out = self.run_driver(small_config(), time.time() - 1)
        self.assertEqual(len(drv.sink.rows), 0)
        self.assertTrue(all(p.get("skipped") == "deadline" for p in drv.phase_log))

    def test_config_values(self):
        s = json.loads(CONFIG.read_text())["burst_overhead"]
        self.assertEqual(s["phase_order"], ["tier_poisson", "park_idle"])
        self.assertLessEqual(s["request_timeout_s"], 120)
        self.assertLess(max(s["busy_probe_lengths"]), 4096)


if __name__ == "__main__":
    unittest.main()
