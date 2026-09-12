import random
import unittest
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from xpyd.energy_validation import Meter as RealMeter, clock_within_tolerance, integrate_intervals
from xpyd.mixed_variable_canary_energy import (
    FallbackFeedbackController, current_policy, matched_pair_plan, paired_summary,
    randomized_evaluation_requests, burst_evaluation_plan, allocate_burst_energy,
)
from xpyd.online_feedback_controller import OnlineFeedbackError
from xpyd.workload_frequency_table import WorkloadFrequencyTable, FrequencyTableError


PANELS = {
    "small_light": [(128, 48), (160, 64), (200, 80)],
    "prefill_medium": [(768, 48), (1024, 64), (1280, 80)],
    "prefill_heavy": [(1792, 48), (2048, 64), (2304, 80)],
    "decode_medium": [(96, 96), (128, 128), (200, 160)],
    "decode_heavy": [(96, 224), (128, 256), (200, 320)],
    "balanced_medium": [(384, 96), (512, 128), (640, 160)],
    "both_heavy": [(1792, 224), (2048, 256), (2304, 320)],
}


class MixedVariableCanaryEnergyTests(unittest.TestCase):
    def test_readback_only_mismatch_nonfatal_service_but_invalid_canary_candidate(self):
        from xpyd.online_feedback_controller import PhysicalFeedbackRuntime
        from xpyd.energy_validation import ClockQualityError
        runtime=PhysicalFeedbackRuntime.__new__(PhysicalFeedbackRuntime)
        status=['success']
        runtime.actuator=SimpleNamespace(requested={},actuate=lambda *a:dict(
            command_status=status[0],readback_valid=False))
        runtime._log_actuation=lambda *a:None
        self.assertTrue(runtime._actuate_sync('service',900,900))
        with self.assertRaises(ClockQualityError):
            runtime._actuate_sync('experiment',900,900)
        status[0]='failed'
        with self.assertRaises(OnlineFeedbackError):
            runtime._actuate_sync('service',900,900)

    def test_canary_clock_failures_publish_fallback_not_terminal_error(self):
        from xpyd.energy_validation import ClockQualityError
        async def scenario():
            async def no_op(*args):return False
            class Controller(FallbackFeedbackController):
                async def _explore(self,*args):
                    raise ClockQualityError('actual GPU clock mismatch')
            table=WorkloadFrequencyTable()
            c=Controller(table,None,no_op,no_op,(900,2520),(450,1500),exploration_retry_backoff_s=0)
            await c.start()
            await c._enqueue('small_light',{},'trigger')
            await asyncio.wait_for(c._queue.join(),2)
            self.assertFalse(c.terminal_errors)
            value=table.read('small_light').value
            self.assertEqual(value.source,'safe_high_after_measurement_exhausted')
            self.assertFalse(value.slo_met)
            self.assertIsNone(value.measured_energy_j)
            self.assertEqual(current_policy(table,'small_light',(2520,1500))[1],'safe_high_fallback')
            await c.stop()
        asyncio.run(scenario())

    def test_real_meter_subsample_plateau_partition(self):
        # Two simultaneous clients differ by 1ms, inside a flat counter sample.
        meter=RealMeter(Path('/tmp'))
        for ep in meter.samples:
            meter.samples[ep]=[dict(t=i*.05,energy_j=100+(i//2)*10,
                                    clock_mhz=900) for i in range(11)]
        rows=[dict(request_id=str(i),request_send_wall_ns=int(a*1e9),
            request_complete_wall_ns=int(b*1e9),energy_j=1,duration_s=b-a,
            endpoints={},control_inclusive=dict(duration_s=b,energy_j=1))
            for i,(a,b) in enumerate([(.01,.401),(.011,.402)])]
        audit=asyncio.run(allocate_burst_energy(meter,rows,(900,900),0))
        self.assertEqual(audit['active_energy_j'],80)
        self.assertEqual(audit['segments'][0]['energy_j'],0)
        self.assertAlmostEqual(sum(r['energy_j'] for r in rows),80)
        self.assertAlmostEqual(sum(r['control_inclusive']['energy_j'] for r in rows),80)
        with self.assertRaisesRegex(ValueError,'clock'):
            asyncio.run(allocate_burst_energy(meter,rows,(1500,1500),0))

    def test_concurrent_cohorts_match_both_arms(self):
        plan=matched_pair_plan(PANELS,12,random.Random(14))
        batches=burst_evaluation_plan(plan,random.Random(14))
        self.assertEqual(sum(map(len,batches)),168)
        cohorts={}
        for batch in batches:
            self.assertEqual(len(batch),4)
            self.assertEqual(len({r['workload_id'] for r in batch}),1)
            self.assertEqual(len({r['arm'] for r in batch}),1)
            key=tuple(r['pair_id'] for r in batch)
            cohorts.setdefault(key,{})[batch[0]['arm']]=[(r['input_len'],r['output_len']) for r in batch]
        self.assertTrue(all(v['safe_high']==v['table'] for v in cohorts.values()))

    def test_overlap_energy_is_conserved_not_double_counted(self):
        class Meter:
            async def energy_intervals(self,pair,intervals,targets,*,allow_zero=False,enforce_clock=True):
                from xpyd.energy_validation import merge_intervals
                e=10*sum(b-a for a,b in merge_intervals(intervals))
                return dict(energy_j=e,endpoints={'P1':dict(energy_j=e)})
            async def energy(self,pair,a,b):return dict(energy_j=10*(b-a))
        def row(a,b):
            return dict(request_send_wall_ns=int(a*1e9),request_complete_wall_ns=int(b*1e9),
                energy_j=10*(b-a),duration_s=b-a,endpoints={},
                control_inclusive=dict(duration_s=b+.5,energy_j=10*(b+.5),endpoints={}))
        rows=[row(0,2),row(1,3)]
        audit=asyncio.run(allocate_burst_energy(Meter(),rows,(900,900),-.5))
        self.assertEqual(audit['active_energy_j'],30)
        self.assertEqual([r['energy_j'] for r in rows],[15,15])
        self.assertEqual(sum(r['control_inclusive']['energy_j'] for r in rows),35)
        self.assertEqual(sum(r['endpoints']['P1']['energy_j'] for r in rows),30)

    def test_zero_inflight_gap_is_excluded(self):
        class Meter:
            async def energy_intervals(self,pair,intervals,targets,*,allow_zero=False,enforce_clock=True):
                return dict(energy_j=10*sum(b-a for a,b in intervals),endpoints={})
            async def energy(self,pair,a,b):return dict(energy_j=10*(b-a))
        rows=[dict(request_send_wall_ns=int(a*1e9),request_complete_wall_ns=int(b*1e9),
            energy_j=10,duration_s=1,endpoints={},control_inclusive=dict(duration_s=3,energy_j=30))
            for a,b in [(0,1),(2,3)]]
        audit=asyncio.run(allocate_burst_energy(Meter(),rows,(900,900),0))
        self.assertEqual(audit['active_energy_j'],20)
        self.assertEqual(sum(r['energy_j'] for r in rows),20)
        self.assertEqual(sum(r['control_inclusive']['energy_j'] for r in rows),30)

    def test_three_slo_failures_publish_high_and_stop_retrying(self):
        async def scenario():
            calls = []
            class Controller(FallbackFeedbackController):
                async def _explore(self, workload_id, body, request_id):
                    calls.append(workload_id)
                    raise OnlineFeedbackError("no SLO-safe D frequency")
            async def no_op(*args): return False
            table = WorkloadFrequencyTable()
            controller = Controller(table, None, no_op, no_op, (900, 2520),
                                    (450, 1500), exploration_retry_backoff_s=0)
            await controller.start()
            await controller._enqueue("both_heavy", {}, "trigger")
            await asyncio.wait_for(controller._queue.join(), timeout=2)
            self.assertEqual(len(calls), 3)
            value = table.read("both_heavy").value
            self.assertEqual((value.prefill_frequency_mhz, value.decode_frequency_mhz), (2520, 1500))
            self.assertFalse(value.slo_met)
            self.assertIsNone(value.measured_energy_j)
            await controller._enqueue("both_heavy", {}, "again")
            self.assertEqual(controller._queue.qsize(), 0)
            await controller.stop()
        asyncio.run(scenario())

    def test_one_table_entry_applies_while_others_missing(self):
        table = WorkloadFrequencyTable()
        table.write("small_light", {
            "prefill_frequency_mhz": 900, "decode_frequency_mhz": 1050,
            "measured_power_w": 120, "measured_energy_j": 450,
            "ttft_ms": 200, "tpot_ms": 60, "sample_count": 3,
            "prefill_endpoint_id": "P0", "decode_endpoint_id": "D0", "slo_met": True,
        })
        self.assertEqual(current_policy(table, "small_light", (2520, 1500))[:2],
                         ((900, 1050), "table"))
        self.assertEqual(current_policy(table, "both_heavy", (2520, 1500))[:2],
                         ((2520, 1500), "safe_high"))

    def test_fallback_cannot_claim_measured_slo_safety(self):
        value = dict(prefill_frequency_mhz=2520,decode_frequency_mhz=1500,
                     measured_power_w=None,measured_energy_j=None,ttft_ms=None,tpot_ms=None,
                     prefill_endpoint_id="P0",decode_endpoint_id="D0",sample_count=0,
                     source="safe_high_after_slo_exhausted",fallback_reason="no safe P",slo_met=True)
        with self.assertRaises(FrequencyTableError):
            WorkloadFrequencyTable().write("both_heavy",value)

    def test_mixed_infrastructure_and_slo_failures_do_not_claim_three_slo_failures(self):
        async def scenario():
            async def no_op(*args): return False
            controller=FallbackFeedbackController(WorkloadFrequencyTable(),None,
                no_op,no_op,(900,2520),(450,1500))
            errors=[ValueError("gap"),OnlineFeedbackError("no SLO-safe P frequency"),
                    OnlineFeedbackError("no SLO-safe P frequency")]
            for attempt,error in enumerate(errors,1):
                await controller._on_exploration_failure("both_heavy",attempt,error)
            await controller._on_exploration_exhausted("both_heavy",3,errors[-1])
            self.assertIn("both_heavy",controller.terminal_errors)
            self.assertIsNone(controller.table.read("both_heavy").value)
        asyncio.run(scenario())

    def test_infrastructure_failure_is_not_disguised_as_slo_fallback(self):
        async def scenario():
            async def no_op(*args): return False
            controller = FallbackFeedbackController(WorkloadFrequencyTable(), None,
                no_op, no_op, (900, 2520), (450, 1500))
            for attempt in range(1, 4):
                await controller._on_exploration_failure("small_light", attempt, ValueError("sample gap"))
            await controller._on_exploration_exhausted("small_light", 3, ValueError("sample gap"))
            self.assertIn("small_light", controller.terminal_errors)
            self.assertIsNone(controller.table.read("small_light").value)
        asyncio.run(scenario())

    def test_evaluation_randomizes_individual_requests_not_pair_blocks(self):
        plan = matched_pair_plan(PANELS, 12, random.Random(20260912))
        requests = randomized_evaluation_requests(plan, random.Random(13))
        self.assertEqual(len(requests), 168)
        self.assertTrue(any(requests[i]["pair_id"] != requests[i + 1]["pair_id"]
                            for i in range(0, 168, 2)))
        for pair in plan:
            arms = [row["arm"] for row in requests if row["pair_id"] == pair["pair_id"]]
            self.assertEqual(set(arms), {"safe_high", "table"})

    def test_complete_r13_driver_with_five_learned_and_two_fallback_classes(self):
        from xpyd import mixed_variable_canary_energy as ev
        real_sleep = asyncio.sleep
        async def sleep(seconds): await real_sleep(0)
        class FakeMeter:
            def __init__(self, root): self.samples = {}
            def start(self): pass
            def stop(self): pass
            async def ready(self, end=None): pass
            def sampler_health(self): return {}
            async def energy(self, pair, start, end, targets=None):
                return dict(start=start, end=end, duration_s=end-start,
                            energy_j=120*(end-start), mean_power_w=120, endpoints={})
            async def energy_intervals(self, pair, intervals, targets=None, *, allow_zero=False,enforce_clock=True):
                from xpyd.energy_validation import merge_intervals
                duration = sum(end-start for start,end in merge_intervals(intervals))
                return dict(intervals=intervals, duration_s=duration,
                            energy_j=120*duration, mean_power_w=120, endpoints={
                                ep:dict(clock_target_met=False,clock_match_fraction=.2,energy_j=60*duration)
                                for ep in pair} if pair==('P1','D1') else {})
        class FakeRuntime:
            def __init__(self, config, core):
                self.prefill_grid = (900, 2520)
                self.decode_grid = (450, 1500)
                self.current = {}
            async def actuate(self, group, p, d):
                changed = self.current.get(group) != (p,d)
                self.current[group] = (p,d)
                await real_sleep(0)
                return changed
        def build(config, factory, diagnostic_sink):
            class Core:
                async def prepare(self, body, rid):
                    ttft = .6 if "prefill_heavy" in rid or "both_heavy" in rid else .2
                    if rid.startswith(('online-production','evaluation-production')): ttft=.8
                    async def stream():
                        await real_sleep(0)
                        yield b'data: {"choices":[{"text":"x"}]}\n\n'
                        yield ('data: '+json.dumps({"usage":{"completion_tokens":body["max_tokens"]}})+'\n\n').encode()
                        diagnostic_sink({"request_id":rid})
                        yield b'data: [DONE]\n\n'
                    return SimpleNamespace(status_code=200, stream=stream(),
                        diagnostics=SimpleNamespace(timestamps_wall_s={}, timestamps_monotonic_s={
                            "request_received":0, "decode_first_real_chunk_received":ttft,
                            "decode_last_chunk_received":ttft+4}))
            return SimpleNamespace(cores={pair:Core() for pair in (("P0","D0"),("P1","D1"))},
                frequency_table=WorkloadFrequencyTable(persistence_path=config["frequency_table_path"]))
        modules = {
            "aiohttp":SimpleNamespace(ClientTimeout=lambda **kw:None,ClientSession=lambda **kw:None),
            "transformers":SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:None)),
            "replay_synthetic_trace":SimpleNamespace(build_prompt_cache=lambda _,lens:{n:'x'*n for n in lens}),
            "xpyd.disagg_proxy":SimpleNamespace(_build_multi_core=build),
        }
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=json.loads((Path(__file__).parent/'source/paper/configs/xpyd_mixed_variable_canary_energy_2p2d.json').read_text())
            config['frequency_table_path']=str(root/'table.json')
            config['online_feedback']['exploration_retry_backoff_s']=0
            with patch.dict('sys.modules', modules), patch.object(ev,'Meter',FakeMeter), \
                 patch('xpyd.online_feedback_controller.PhysicalFeedbackRuntime',FakeRuntime), \
                 patch.object(ev.asyncio,'sleep',sleep):
                asyncio.run(asyncio.wait_for(ev.run(config,root),timeout=5))
            summary=json.loads((root/'summary.json').read_text())
            self.assertTrue(json.loads((root/'audit.json').read_text())['valid'])
            self.assertEqual(set(summary['fallback_workloads']), {'prefill_heavy','both_heavy'})
            self.assertEqual(summary['matched_evaluation_request_count'],168)
            self.assertGreater(summary['production_clock_noncompliant_requests'],0)
            self.assertEqual(summary['production_slo_noncompliant_requests'],summary['production_request_count'])
            self.assertTrue(all(n>=12 for n in summary['online_table_policy_requests_per_class'].values()))
            production=[json.loads(line) for line in (root/'production_requests.jsonl').read_text().splitlines()]
            bursts=[json.loads(line) for line in (root/'production_bursts.jsonl').read_text().splitlines()]
            for burst in bursts:
                rr=[r for r in production if r.get('burst_id')==burst['burst_id']]
                self.assertAlmostEqual(sum(r['energy_j'] for r in rr),burst['active_energy_j'])
                self.assertAlmostEqual(sum(r['control_inclusive']['energy_j'] for r in rr),burst['control_energy_j'])
                self.assertEqual(len({(r['p_mhz'],r['d_mhz']) for r in rr}),1)
            self.assertTrue(any(max(seg['inflight'] for seg in b['segments'])>=5 for b in bursts))
            attempts=[json.loads(line) for line in (root/'canary_attempts.jsonl').read_text().splitlines()]
            last_exploration=max(row['end'] for row in attempts)
            self.assertTrue(any(row['phase']=='production_online_feedback' and
                row['applied_policy']=='table' and row['request_send_wall_ns']/1e9<last_exploration
                for row in production))
            online=[row for row in production if row['phase']=='production_online_feedback']
            self.assertLess(len({row['workload_id'] for row in online[:7]}),7)
    def test_pair_plan_is_balanced_exact_shape_and_reproducible(self):
        first = matched_pair_plan(PANELS, 12, random.Random(20260911))
        second = matched_pair_plan(PANELS, 12, random.Random(20260911))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 84)
        for workload_id, shapes in PANELS.items():
            rows = [row for row in first if row["workload_id"] == workload_id]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(set(row["arms"]) == {"safe_high", "table"} for row in rows))
            self.assertTrue(all((row["input_len"], row["output_len"]) in shapes for row in rows))
            self.assertEqual({(row["input_len"], row["output_len"]) for row in rows}, set(shapes))

    def test_paired_summary_uses_within_shape_energy_deltas(self):
        rows = []
        for pair_id, shape in (("a", (128, 48)), ("b", (200, 80))):
            for source, active, inclusive in (
                ("safe_high", 100.0, 105.0), ("table", 80.0, 90.0)
            ):
                rows.append({
                    "workload_id": "small_light", "pair_id": pair_id,
                    "frequency_source": source, "input_len": shape[0], "output_len": shape[1],
                    "energy_j": active, "control_inclusive": {"energy_j": inclusive},
                    "ttft_ms": 100.0, "tpot_ms": 20.0, "p_mhz": 900, "d_mhz": 450,
                })
        result = paired_summary(rows)[0]
        self.assertEqual(result["matched_pairs"], 2)
        self.assertAlmostEqual(result["active_mean_saving_j"], 20.0)
        self.assertAlmostEqual(result["active_energy_reduction_percent"], 20.0)
        self.assertAlmostEqual(result["control_inclusive_mean_saving_j"], 15.0)

    def test_clock_tolerance_accepts_normal_readback_jitter(self):
        self.assertTrue(clock_within_tolerance(1489, 1500))
        self.assertTrue(clock_within_tolerance(2510, 2520))
        self.assertFalse(clock_within_tolerance(1450, 1500))
        samples = [
            {"t": index * 0.05, "energy_j": index * 5.0,
             "clock_mhz": 1489, "power_w": 100.0}
            for index in range(41)
        ]
        value = integrate_intervals(samples, [(0.25, 1.75)], target=1500)
        self.assertEqual(value["clock_match_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
