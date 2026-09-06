import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

from xpyd.energy_validation import consume, integrate, observed_summary, probe_metadata


class EnergyTests(unittest.TestCase):
    def test_probe_metadata_preserves_axis_index_repeat(self):
        result = probe_metadata("production-1-attempt-2-binary-slo-D-7-sample-3")
        self.assertEqual((result["axis"], result["grid_index"], result["repeat_index"]), ("D", 7, 3))
        self.assertEqual(probe_metadata("x-confirm-sample-2")["search_stage"], "confirmation")

    def rows(self):
        return [dict(t=i/10, energy_j=100+i*10, clock_mhz=900) for i in range(31)]

    def test_interpolation_100_watts(self):
        result = integrate(self.rows(), .15, 2.65, 900)
        self.assertAlmostEqual(result["energy_j"], 250)
        self.assertEqual(result["clock_match_fraction"], 1)

    def test_missing_boundary_gap_reset_and_wrong_clock_rejected(self):
        for rows, start, end, target in (
            (self.rows(), -.1, 2, 900),
            (self.rows()[::6], .1, 2, 900),
            ([dict(r, energy_j=-1) if i == 9 else r for i,r in enumerate(self.rows())], .1, 2, 900),
            (self.rows(), .1, 2, 1005),
        ):
            with self.assertRaises(ValueError):
                integrate(rows, start, end, target)

    def test_observational_energy_and_power_use_actual_duration(self):
        rows = []
        for source, energy, duration in (("safe_high", 100, 1), ("table", 200, 2)):
            rows.append(dict(workload_id="small", frequency_source=source, energy_j=energy,
                             duration_s=duration, inference=dict(energy_j=energy, duration_s=duration),
                             ttft_ms=508, tpot_ms=60, control_inclusive_ttft_ms=600,
                             start=0, end=duration, p_mhz=900, d_mhz=900))
        row = observed_summary(rows)[0]
        self.assertEqual(row["observed_power_reduction_percent"], 0)
        self.assertEqual(row["observed_energy_reduction_percent"], -100)

    def test_chunked_usage_and_truncation(self):
        async def chunks(valid=True):
            yield b'data: {"usage": {"completion_'
            yield b'tokens": 64}}\n\n'
            if valid:
                yield b'data: [DONE]\n\n'
        self.assertEqual(asyncio.run(consume(SimpleNamespace(status_code=200, stream=chunks()), 64)), 64)
        with self.assertRaises(RuntimeError):
            asyncio.run(consume(SimpleNamespace(status_code=200, stream=chunks(False)), 64))

    def test_complete_driver_with_fake_hardware(self):
        from xpyd import energy_validation as ev
        from xpyd.workload_frequency_table import WorkloadFrequencyTable
        from xpyd.online_feedback_controller import WORKLOAD_SHAPES
        clock = [1000.0]
        real_sleep = asyncio.sleep
        async def sleep(seconds):
            clock[0] += seconds
            await real_sleep(0)
        class FakeMeter:
            def __init__(self, root): pass
            def start(self): pass
            def stop(self): pass
            async def ready(self, end=None): pass
            async def energy(self, pair, start, end, targets=None):
                return dict(start=start, end=end, duration_s=end-start,
                            energy_j=100*(end-start), endpoints={})
        class FakeRuntime:
            def __init__(self, config, core):
                self.prefill_grid = tuple(900+100*i for i in range(17))
                self.decode_grid = tuple(450+75*i for i in range(15))
            async def actuate(self, *args): return True
        def build(config, factory, diagnostic_sink):
            class Core:
                async def prepare(self, body, rid):
                    clock[0] += 1
                    received = clock[0]
                    async def stream():
                        clock[0] += 1
                        yield ('data: '+json.dumps({"usage": {"completion_tokens": body["max_tokens"]}})+'\n\n').encode()
                        yield b'data: [DONE]\n\n'
                        diagnostic_sink({"request_id": rid})
                    return SimpleNamespace(status_code=200, stream=stream(),
                        diagnostics=SimpleNamespace(timestamps_monotonic_s={
                            "request_received": 0., "decode_first_real_chunk_received": .2,
                            "decode_last_chunk_received": 4.0}, timestamps_wall_s={
                            "request_received": received, "decode_first_real_chunk_received": received+.2}))
            return SimpleNamespace(cores={p:Core() for p in (("P0","D0"),("P1","D1"))},
                frequency_table=WorkloadFrequencyTable(list(WORKLOAD_SHAPES),
                                                       persistence_path=config["frequency_table_path"]))
        modules = {
            "aiohttp": SimpleNamespace(ClientTimeout=lambda **kw: None, ClientSession=lambda **kw: None),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k: None)),
            "replay_synthetic_trace": SimpleNamespace(build_prompt_cache=lambda _,lens:{n:"x"*n for n in lens}),
            "xpyd.disagg_proxy": SimpleNamespace(_build_multi_core=build),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"model":"fake", "tokenizer_model":"fake",
                "frequency_table_path": str(root/"table.json"), "online_feedback": {},
                "workloads": [dict(id=w,input_len=s[0],output_len=s[1]) for w,s in WORKLOAD_SHAPES.items()]}
            with patch.dict("sys.modules", modules), patch.object(ev, "Meter", FakeMeter), \
                 patch("xpyd.online_feedback_controller.PhysicalFeedbackRuntime", FakeRuntime), \
                 patch.object(ev.time, "time", lambda:clock[0]), patch.object(ev.asyncio, "sleep", sleep):
                asyncio.run(ev.run(config, root))
            summary = json.loads((root/"summary.json").read_text())
            audit = json.loads((root/"audit.json").read_text())
            self.assertTrue(audit["valid"])
            self.assertTrue(all(n >= 30 for n in audit["table_requests_per_class"].values()))
            self.assertEqual(len(summary["comparison"]), 7)
            for row in summary["comparison"]:
                self.assertGreater(row["safe_high"]["n"], 0)
                self.assertGreaterEqual(row["table"]["n"], 30)
            production = [json.loads(line) for line in (root/"production_requests.jsonl").read_text().splitlines()]
            # Each workload moves once from natural high to learned Table, never AB/BA.
            for w in WORKLOAD_SHAPES:
                flags = [r["table_hit"] for r in production if r["workload_id"] == w]
                self.assertEqual(flags, sorted(flags))
            probes = [json.loads(line) for line in (root/"canary_probes.jsonl").read_text().splitlines()]
            self.assertEqual(len({p["request_id"] for p in probes}), len(probes))
            self.assertTrue(all("including_control" in p and "search_stage" in p for p in probes))
            self.assertEqual(len(summary["canary_overhead"]["per_class"]), 7)
            self.assertTrue((root/"summary.md").exists())


if __name__ == "__main__":
    unittest.main()
