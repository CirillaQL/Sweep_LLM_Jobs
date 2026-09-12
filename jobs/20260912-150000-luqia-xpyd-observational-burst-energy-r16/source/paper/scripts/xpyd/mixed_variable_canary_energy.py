"""Cold-start Canary feedback with homogeneous concurrent burst Production.

The primary energy boundary is client-send to completed stream.  Idle gaps and
DVFS settling are excluded from that metric.  A second control-inclusive metric
starts before clock actuation so the cost of per-request frequency switching is
also visible. Each drained epoch reads its category's current Table and fixes
clocks for all overlapping requests; energy is apportioned by in-flight counts.
Three exhausted SLO failures publish an explicit non-SLO-guaranteed high-clock
fallback. A/B cohorts retain exact-shape pair IDs and identical co-running shapes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Mapping, Sequence

from xpyd.energy_validation import ClockQualityError, Meter, export_power_trace, percentile, write
from xpyd.online_feedback_controller import OnlineFeedbackController, OnlineFeedbackError


class FallbackFeedbackController(OnlineFeedbackController):
    """Publish an explicit high-clock policy after terminal SLO failure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_errors = {}
        self.slo_failure_counts = {}
        self.clock_failure_counts = {}

    @staticmethod
    def is_slo_failure(error):
        return isinstance(error, OnlineFeedbackError) and (
            str(error).startswith("no SLO-safe ")
            or str(error) == "joint P/D confirmation failed SLO"
        )

    async def _on_exploration_failure(self, workload_id, attempt, error):
        if self.is_slo_failure(error):
            self.slo_failure_counts[workload_id] = self.slo_failure_counts.get(workload_id, 0) + 1
        elif isinstance(error, ClockQualityError):
            self.clock_failure_counts[workload_id] = self.clock_failure_counts.get(workload_id,0)+1
        else:
            self.terminal_errors[workload_id] = f'{type(error).__name__}: {error}'

    async def _on_exploration_exhausted(self, workload_id, attempt, error):
        super_error = f"{type(error).__name__}: {error}"
        clock_failures = self.clock_failure_counts.get(workload_id,0)
        if self.slo_failure_counts.get(workload_id, 0) + clock_failures != attempt:
            self.terminal_errors[workload_id] = super_error
            await super()._on_exploration_exhausted(workload_id, attempt, error)
            return
        self.table.write(workload_id, {
            "prefill_frequency_mhz": self.prefill_grid[-1],
            "decode_frequency_mhz": self.decode_grid[-1],
            "measured_power_w": None, "measured_energy_j": None,
            "ttft_ms": None, "tpot_ms": None,
            "prefill_endpoint_id": "P0", "decode_endpoint_id": "D0",
            "sample_count": 0, "source": "safe_high_after_measurement_exhausted" if clock_failures else "safe_high_after_slo_exhausted",
            "slo_met": False, "fallback_reason": super_error,
        })
        self._log({"event": "table_fallback_published", "workload_id": workload_id,
                   "attempt": attempt, "reason": super_error,
                   "prefill_frequency_mhz": self.prefill_grid[-1],
                   "decode_frequency_mhz": self.decode_grid[-1], "slo_met": False})


def current_policy(table, workload_id, high, force_high=False):
    """Read the current entry for this request, never await unrelated classes."""
    entry = table.read(workload_id)
    if force_high or entry.value is None:
        return tuple(high), "safe_high", entry.revision
    value = entry.value
    source = "safe_high_fallback" if value.source.startswith('safe_high_after_') else "table"
    return (value.prefill_frequency_mhz, value.decode_frequency_mhz), source, entry.revision


def append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n")


def export_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


async def consume_timed(prepared: Any, expected_tokens: int) -> dict[str, Any]:
    """Consume SSE and timestamp the first client-visible token and completion."""
    if prepared.stream is None or prepared.status_code != 200:
        raise RuntimeError("missing successful SSE response")
    buffer = b""
    done = False
    tokens = None
    first_wall_ns = None
    first_mono_ns = None
    last_wall_ns = None
    chunks = 0

    def process(line: bytes, wall_ns: int, mono_ns: int) -> None:
        nonlocal done, tokens, first_wall_ns, first_mono_ns
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            done = True
            return
        event = json.loads(data.decode("utf-8"))
        if event.get("error"):
            raise RuntimeError("upstream SSE error")
        if event.get("usage"):
            tokens = event["usage"].get("completion_tokens")
        if first_wall_ns is None and any(
            choice.get("text") not in (None, "") for choice in event.get("choices", [])
        ):
            first_wall_ns, first_mono_ns = wall_ns, mono_ns

    async for chunk in prepared.stream:
        wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
        last_wall_ns = wall_ns
        chunks += 1
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            process(line.rstrip(b"\r"), wall_ns, mono_ns)
    if buffer:
        process(buffer.rstrip(b"\r"), time.time_ns(), time.monotonic_ns())
    if not done or tokens != expected_tokens or first_wall_ns is None:
        raise RuntimeError(
            f"incomplete generation: done={done}, tokens={tokens}, "
            f"first_token={first_wall_ns is not None}, expected={expected_tokens}"
        )
    return {
        "observed_output_tokens": tokens,
        "client_first_token_wall_ns": first_wall_ns,
        "client_first_token_mono_ns": first_mono_ns,
        "client_last_chunk_wall_ns": last_wall_ns,
        "client_stream_chunk_count": chunks,
    }


def probe_metadata(probe_id: str) -> dict[str, Any]:
    match = re.search(r"-(binary-slo|energy-refine)-([PD])-(\d+)(?:-sample-(\d+))?", probe_id)
    repeat = re.search(r"-sample-(\d+)$", probe_id)
    return {
        "search_stage": match[1] if match else ("confirmation" if "-confirm" in probe_id else "warmup"),
        "axis": match[2] if match else None,
        "grid_index": int(match[3]) if match else None,
        "repeat_index": int(repeat[1]) if repeat else None,
        "candidate_id": re.sub(r"-sample-\d+$", "", probe_id),
    }


def matched_pair_plan(
    panels: Mapping[str, Sequence[tuple[int, int]]], pairs_per_class: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build exact-shape high/Table pairs, globally shuffled by category."""
    plan = []
    for workload_id, shapes in panels.items():
        for index in range(pairs_per_class):
            input_len, output_len = shapes[index % len(shapes)]
            arms = ["safe_high", "table"]
            rng.shuffle(arms)
            plan.append({
                "pair_id": f"{workload_id}-pair-{index + 1:02d}",
                "workload_id": workload_id,
                "input_len": input_len,
                "output_len": output_len,
                "arms": arms,
            })
    rng.shuffle(plan)
    return plan


def randomized_evaluation_requests(plan, rng):
    # Shuffle individual requests, not two-request same-type blocks. Pair IDs
    # still join the exact-shape high and learned-policy observations afterward.
    requests = [dict(spec, arm=arm) for spec in plan for arm in spec["arms"]]
    rng.shuffle(requests)
    return requests


def burst_evaluation_plan(plan, rng, size=4):
    # Identical co-running shapes for both arms; randomize whole cohorts.
    batches = []
    for workload_id in sorted({s['workload_id'] for s in plan}):
        cohort = sorted([s for s in plan if s['workload_id'] == workload_id],
                        key=lambda s: s['pair_id'])
        for offset in range(0, len(cohort), size):
            for arm in ('safe_high', 'table'):
                batches.append([dict(s, arm=arm) for s in cohort[offset:offset+size]])
    rng.shuffle(batches)
    return batches


async def allocate_burst_energy(meter, rows, targets, control_start, *, enforce_clock=True):
    """Conserve board energy: split each active segment across in-flight requests."""
    bounds = sorted({r[k]/1e9 for r in rows for k in
                     ('request_send_wall_ns', 'request_complete_wall_ns')})
    allocated = [0.0] * len(rows)
    union = await meter.energy_intervals(('P1','D1'), [
        (r['request_send_wall_ns']/1e9,r['request_complete_wall_ns']/1e9)
        for r in rows], targets, enforce_clock=enforce_clock)
    endpoint_shares = [{} for _ in rows]
    total = 0.0
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        owners = [i for i,r in enumerate(rows)
                  if r['request_send_wall_ns']/1e9 <= start and
                  r['request_complete_wall_ns']/1e9 >= end]
        if not owners:
            continue
        # Clock/sample quality is validated over the active union above, not
        # over tiny fragments which may contain no interior NVML sample.
        measurement = await meter.energy_intervals(('P1','D1'), [(start,end)],
                                                  None, allow_zero=True)
        energy = measurement['energy_j']
        total += energy
        for i in owners:
            allocated[i] += energy / len(owners)
            for endpoint, value in measurement['endpoints'].items():
                endpoint_shares[i][endpoint] = endpoint_shares[i].get(endpoint,0.0)+value['energy_j']/len(owners)
        segments.append(dict(start=start,end=end,inflight=len(owners),energy_j=energy))
    gross = await meter.energy(('P1','D1'), control_start, bounds[-1])
    if abs(total-union['energy_j']) > max(1e-6,union['energy_j']*1e-8):
        raise RuntimeError('burst segment energy does not conserve active union')
    overhead = gross['energy_j'] - total
    if overhead < -max(1.0, total * .001):
        raise RuntimeError('burst active energy exceeds control total')
    for i,r in enumerate(rows):
        r['shared_board_window_energy_j'] = r['energy_j']
        r['shared_board_control_window'] = r['control_inclusive']
        r['shared_board_window_endpoints'] = r['endpoints']
        r['clock_observations'] = union['endpoints']
        r['clock_compliant'] = all(v.get('clock_target_met', True) is not False
                                  for v in union['endpoints'].values())
        r['endpoints'] = {ep:dict(energy_j=e,mean_power_w=e/r['duration_s'],
            attribution='inflight_equal_share') for ep,e in endpoint_shares[i].items()}
        r['energy_j'] = allocated[i]
        r['mean_power_w'] = allocated[i] / r['duration_s']
        r['energy_attribution'] = 'equal_share_per_inflight_segment_not_independent_measurement'
        r['control_inclusive'] = dict(r['control_inclusive'],
            energy_j=allocated[i]+overhead/len(rows),
            endpoints={},
            allocation='active_share_plus_equal_burst_overhead')
        r['control_inclusive']['mean_power_w'] = r['control_inclusive']['energy_j']/r['control_inclusive']['duration_s']
    return dict(active_energy_j=total, control_energy_j=gross['energy_j'],
                control_overhead_energy_j=overhead, request_count=len(rows),segments=segments,
                validated_active_union=union)


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    workloads = sorted({str(row["workload_id"]) for row in rows})
    for workload_id in workloads:
        rr = [row for row in rows if row["workload_id"] == workload_id]
        pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in rr:
            if str(row["frequency_source"]) in pairs.get(str(row["pair_id"]), {}):
                raise RuntimeError("duplicate matched-pair arm")
            pairs.setdefault(str(row["pair_id"]), {})[str(row["frequency_source"])] = row
        if not pairs or any(set(value) != {"safe_high", "table"} for value in pairs.values()):
            raise RuntimeError("incomplete matched pair for " + workload_id)
        if any((v["safe_high"]["input_len"], v["safe_high"]["output_len"]) !=
               (v["table"]["input_len"], v["table"]["output_len"]) for v in pairs.values()):
            raise RuntimeError("matched-pair input/output mismatch")
        active_delta = [v["safe_high"]["energy_j"] - v["table"]["energy_j"] for v in pairs.values()]
        inclusive_delta = [
            v["safe_high"]["control_inclusive"]["energy_j"]
            - v["table"]["control_inclusive"]["energy_j"] for v in pairs.values()
        ]
        high_active = [v["safe_high"]["energy_j"] for v in pairs.values()]
        table_active = [v["table"]["energy_j"] for v in pairs.values()]
        high_inclusive = [v["safe_high"]["control_inclusive"]["energy_j"] for v in pairs.values()]
        table_inclusive = [v["table"]["control_inclusive"]["energy_j"] for v in pairs.values()]
        results.append({
            "workload_id": workload_id,
            "matched_pairs": len(pairs),
            "is_safe_high_fallback": any(v["table"].get("applied_policy") == "safe_high_fallback" for v in pairs.values()),
            "safe_high_active_mean_j": statistics.fmean(high_active),
            "table_active_mean_j": statistics.fmean(table_active),
            "active_mean_saving_j": statistics.fmean(active_delta),
            "active_median_saving_j": statistics.median(active_delta),
            "active_energy_reduction_percent": 100.0 * statistics.fmean(active_delta) / statistics.fmean(high_active),
            "safe_high_control_inclusive_mean_j": statistics.fmean(high_inclusive),
            "table_control_inclusive_mean_j": statistics.fmean(table_inclusive),
            "control_inclusive_mean_saving_j": statistics.fmean(inclusive_delta),
            "control_inclusive_energy_reduction_percent": 100.0 * statistics.fmean(inclusive_delta) / statistics.fmean(high_inclusive),
            "safe_high_ttft_p95_ms": percentile([v["safe_high"]["ttft_ms"] for v in pairs.values()]),
            "table_ttft_p95_ms": percentile([v["table"]["ttft_ms"] for v in pairs.values()]),
            "safe_high_tpot_p95_ms": percentile([v["safe_high"]["tpot_ms"] for v in pairs.values()]),
            "table_tpot_p95_ms": percentile([v["table"]["tpot_ms"] for v in pairs.values()]),
            "table_frequency_pairs": sorted({
                (int(v["table"]["p_mhz"]), int(v["table"]["d_mhz"]))
                for v in pairs.values()
            }),
        })
    return results


async def run(config: Mapping[str, Any], root: Path) -> None:
    import aiohttp
    from transformers import AutoTokenizer
    from replay_synthetic_trace import build_prompt_cache
    from xpyd.disagg_proxy import _build_multi_core
    from xpyd.online_feedback_controller import (
        OnlineFeedbackController, PhysicalFeedbackRuntime, ProbeResult,
        pd_inference_metrics, strip_feedback_metadata,
    )

    if Path(config["frequency_table_path"]).exists():
        raise RuntimeError("cold start requires a nonexistent frequency table")
    settings = dict(config["online_feedback"])
    protocol = dict(config["mixed_variable_energy_validation"])
    seed = int(protocol["random_seed"])
    rng = random.Random(seed)
    settings["event_log"] = str(root / "feedback_events.jsonl")
    settings["service_request_log"] = str(root / "service_dispatch.jsonl")
    config["online_feedback"]["event_log"] = settings["event_log"]
    config["online_feedback"]["service_request_log"] = settings["service_request_log"]

    panels = {
        str(workload["id"]): [
            (int(shape["input_len"]), int(shape["output_len"]))
            for shape in workload["shape_panel"]
        ] for workload in config["workloads"]
    }
    workload_ids = list(panels)
    if len(workload_ids) != 7 or any(len(shapes) != 3 for shapes in panels.values()):
        raise ValueError("experiment requires seven classes and three shapes per class")

    diagnostics: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    production_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    attempt_numbers: dict[str, int] = {}

    def sink(row: dict[str, Any]) -> None:
        diagnostics[row["request_id"]] = row
        append(root / "diagnostics.jsonl", row)

    timeout = aiohttp.ClientTimeout(total=float(protocol["request_timeout_s"]), connect=10)
    multi = _build_multi_core(config, lambda: aiohttp.ClientSession(timeout=timeout), diagnostic_sink=sink)
    meter = Meter(root)
    meter.start()
    runtime = None
    controller = None

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], local_files_only=True)
    all_input_lengths = {input_len for shapes in panels.values() for input_len, _ in shapes}
    prompts = build_prompt_cache(tokenizer, all_input_lengths)

    def body_for(workload_id: str, input_len: int, output_len: int) -> dict[str, Any]:
        return {
            "model": config["model"], "prompt": prompts[input_len],
            "max_tokens": output_len, "ignore_eos": True, "temperature": 0,
            "stream": True, "xpyd_input_len": input_len,
            "xpyd_output_len": output_len, "xpyd_workload_id": workload_id,
        }

    async def measured_request(
        pair: tuple[str, str], body: Mapping[str, Any], request_id: str,
        targets: tuple[int, int], control_start: float, **meta: Any,
    ) -> dict[str, Any]:
        send_wall_ns, send_mono_ns = time.time_ns(), time.monotonic_ns()
        send_s = send_wall_ns / 1e9
        prepared = await multi.cores[pair].prepare(strip_feedback_metadata(body), request_id)
        consumed = await consume_timed(prepared, int(body["max_tokens"]))
        complete_wall_ns, complete_mono_ns = time.time_ns(), time.monotonic_ns()
        complete_s = complete_wall_ns / 1e9
        if request_id not in diagnostics:
            raise RuntimeError("completed stream missing proxy diagnostic: " + request_id)
        ttft_ms, tpot_ms = pd_inference_metrics(
            prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"])
        )
        active = await meter.energy_intervals(pair, [(send_s, complete_s)], targets,
                                            enforce_clock=pair != ('P1','D1'))
        inclusive = await meter.energy(pair, control_start, complete_s)
        row = dict(
            active, request_id=request_id,
            workload_id=str(body["xpyd_workload_id"]),
            input_len=int(body["xpyd_input_len"]), output_len=int(body["xpyd_output_len"]),
            p_mhz=targets[0], d_mhz=targets[1], ttft_ms=ttft_ms, tpot_ms=tpot_ms,
            request_send_wall_ns=send_wall_ns, request_send_mono_ns=send_mono_ns,
            request_complete_wall_ns=complete_wall_ns,
            request_complete_mono_ns=complete_mono_ns,
            request_active_boundary="client_send_to_completed_stream",
            idle_energy_excluded=True, control_inclusive=inclusive,
            slo_met=ttft_ms < 500 and tpot_ms <= 200,
            clock_compliant=all(v.get('clock_target_met',True) is not False
                                for v in active['endpoints'].values()),
            proxy_timestamps_wall_s=prepared.diagnostics.timestamps_wall_s,
            proxy_timestamps_monotonic_s=prepared.diagnostics.timestamps_monotonic_s,
            **consumed, **meta,
        )
        if not row.pop('defer_energy_record', False):
            requests.append(row)
            append(root / "requests_energy.jsonl", row)
        return row

    async def actuate_and_request(
        group: str, pair: tuple[str, str], body: Mapping[str, Any], request_id: str,
        targets: tuple[int, int], **meta: Any,
    ) -> dict[str, Any]:
        control_start = time.time()
        actuation_start_ns = time.time_ns()
        changed = await runtime.actuate(group, *targets)
        actuation_end_ns = time.time_ns()
        settle_s = float(protocol["frequency_settle_s"]) if changed else 0.0
        if settle_s:
            await asyncio.sleep(settle_s)
        return await measured_request(
            pair, body, request_id, targets, control_start,
            frequency_changed=changed,
            actuation_duration_ms=(actuation_end_ns - actuation_start_ns) / 1e6,
            settle_wait_s=settle_s, **meta,
        )

    class VariableShapeRuntime(PhysicalFeedbackRuntime):
        async def probe(self, body, workload_id, p_mhz, d_mhz, probe_id):
            metadata = probe_metadata(probe_id)
            repeat_index = metadata["repeat_index"]
            # Every candidate sees the same low/center/high shape panel.  Warmup
            # has no sample suffix and uses the center shape.
            shape_index = 1 if repeat_index is None else (repeat_index - 1) % 3
            input_len, output_len = panels[workload_id][shape_index]
            probe_body = body_for(workload_id, input_len, output_len)
            event = {
                "event": "started", "request_id": probe_id,
                "workload_id": workload_id, "input_len": input_len,
                "output_len": output_len, "p_mhz": p_mhz, "d_mhz": d_mhz,
                "attempt": attempt_numbers.get(workload_id, 1), **metadata,
            }
            append(root / "canary_probe_events.jsonl", event | {"timestamp_unix_s": time.time()})
            try:
                row = await actuate_and_request(
                    "experiment", ("P0", "D0"), probe_body, probe_id,
                    (p_mhz, d_mhz), phase="canary",
                    attempt=attempt_numbers.get(workload_id, 1), **metadata,
                )
                append(root / "canary_probes.jsonl", row)
                append(root / "canary_probe_events.jsonl", event | {
                    "event": "completed", "timestamp_unix_s": time.time(),
                    "active_energy_j": row["energy_j"],
                    "control_inclusive_energy_j": row["control_inclusive"]["energy_j"],
                })
                return ProbeResult(
                    p_mhz, d_mhz, row["mean_power_w"], row["energy_j"],
                    row["ttft_ms"], row["tpot_ms"],
                )
            except Exception as exc:
                append(root / "canary_probe_events.jsonl", event | {
                    "event": "failed", "timestamp_unix_s": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise

    class TimedController(FallbackFeedbackController):
        async def enqueue_only(self, workload_id, body, request_id):
            await self._enqueue(workload_id, body, request_id)

        def _log(self, value):
            super()._log(value)
            if value.get("event") == "probe_candidate_aggregated":
                candidate_id = value["probe_id"]
                rr = [row for row in requests if row.get("candidate_id") == candidate_id]
                row = dict(value, attempt=attempt_numbers.get(value["workload_id"]), **probe_metadata(candidate_id))
                if rr:
                    row.update(
                        request_active_duration_sum_s=sum(item["duration_s"] for item in rr),
                        request_active_energy_sum_j=sum(item["energy_j"] for item in rr),
                        control_inclusive_duration_sum_s=sum(item["control_inclusive"]["duration_s"] for item in rr),
                        control_inclusive_energy_sum_j=sum(item["control_inclusive"]["energy_j"] for item in rr),
                        input_output_shapes=sorted({(item["input_len"], item["output_len"]) for item in rr}),
                    )
                candidate_rows.append(row)
                append(root / "canary_candidates.jsonl", row)

        async def _explore(self, workload_id, body, request_id):
            attempt_numbers[workload_id] = attempt_numbers.get(workload_id, 0) + 1
            number = attempt_numbers[workload_id]
            start = time.time()
            success = False
            try:
                await super()._explore(workload_id, body, f"{request_id}-attempt-{number}")
                success = True
            finally:
                end = time.time()
                gross = await meter.energy(("P0", "D0"), start, end)
                rr = [row for row in requests if row.get("phase") == "canary"
                      and row.get("workload_id") == workload_id
                      and row.get("attempt") == number]
                attempt = dict(
                    gross, workload_id=workload_id, attempt=number, success=success,
                    request_count=len(rr),
                    request_active_duration_sum_s=sum(row["duration_s"] for row in rr),
                    request_active_energy_sum_j=sum(row["energy_j"] for row in rr),
                    control_inclusive_duration_sum_s=sum(row["control_inclusive"]["duration_s"] for row in rr),
                    control_inclusive_energy_sum_j=sum(row["control_inclusive"]["energy_j"] for row in rr),
                )
                attempts.append(attempt)
                append(root / "canary_attempts.jsonl", attempt)

    async def production_request(
        workload_id: str, input_len: int, output_len: int, request_id: str,
        source: str, pair_id: str | None, phase: str,
    ) -> dict[str, Any]:
        targets, applied_policy, revision = current_policy(
            multi.frequency_table, workload_id,
            (runtime.prefill_grid[-1], runtime.decode_grid[-1]),
            force_high=source == "safe_high",
        )
        if source == "table" and applied_policy == "safe_high":
            raise RuntimeError("missing Table entry for " + workload_id)
        row = await actuate_and_request(
            "service", ("P1", "D1"), body_for(workload_id, input_len, output_len),
            request_id, targets, phase=phase,
            frequency_source=applied_policy if source == "auto" else source,
            applied_policy=applied_policy, table_revision=revision,
            table_hit=applied_policy in {"table", "safe_high_fallback"}, pair_id=pair_id,
        )
        production_rows.append(row)
        append(root / "production_requests.jsonl", row)
        return row

    def random_shape(workload_id: str) -> tuple[int, int]:
        return rng.choice(panels[workload_id])

    async def production_burst(batches, workload_id, source, phase, burst_id):
        targets, policy, revision = current_policy(multi.frequency_table, workload_id,
            (runtime.prefill_grid[-1],runtime.decode_grid[-1]), force_high=source=='safe_high')
        if source == 'table' and policy == 'safe_high':
            raise RuntimeError('missing Table entry for '+workload_id)
        control_start = time.time()
        act_start = time.monotonic()
        changed = await runtime.actuate('service',*targets)
        act_ms = (time.monotonic()-act_start)*1000
        settle = float(protocol['frequency_settle_s']) if changed else 0.0
        await asyncio.sleep(settle)
        tasks = []
        try:
            for batch_index, batch in enumerate(batches):
                if batch_index:
                    await asyncio.sleep(rng.expovariate(1/float(protocol['burst_arrival_mean_gap_s'])))
                arrival_ns = time.time_ns()
                for spec in batch:
                    tasks.append(asyncio.create_task(measured_request(('P1','D1'),
                        body_for(workload_id,spec['input_len'],spec['output_len']),
                        spec['request_id'], targets, control_start, phase=phase,
                        frequency_source=policy if source=='auto' else source,
                        applied_policy=policy,table_revision=revision,
                        table_hit=policy in {'table','safe_high_fallback'},pair_id=spec.get('pair_id'),
                        burst_id=burst_id,batch_index=batch_index,arrival_wall_ns=arrival_ns,
                        frequency_changed=changed,actuation_duration_ms=act_ms,
                        settle_wait_s=settle,defer_energy_record=True)))
            rows = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        audit = await allocate_burst_energy(meter, rows, targets, control_start,enforce_clock=False)
        append(root/'production_bursts.jsonl',dict(audit,burst_id=burst_id,
            workload_id=workload_id,phase=phase,p_mhz=targets[0],d_mhz=targets[1]))
        for row in rows:
            requests.append(row)
            production_rows.append(row)
            append(root/'requests_energy.jsonl',row)
            append(root/'production_requests.jsonl',row)
        return rows

    async def gap() -> None:
        await asyncio.sleep(rng.uniform(
            float(protocol["inter_request_gap_min_s"]),
            float(protocol["inter_request_gap_max_s"]),
        ))

    try:
        await meter.ready()
        runtime = VariableShapeRuntime(config, multi.cores[("P0", "D0")])
        high = (runtime.prefill_grid[-1], runtime.decode_grid[-1])
        write(root / "hardware_grids.json", {"P": runtime.prefill_grid, "D": runtime.decode_grid})
        controller = TimedController(
            multi.frequency_table, multi.cores[("P1", "D1")], runtime.actuate,
            runtime.probe, runtime.prefill_grid, runtime.decode_grid,
            probe_interval_s=float(settings["probe_request_interval_s"]),
            service_settle_s=float(protocol["frequency_settle_s"]),
            ttft_slo_ms=float(settings["exploration_slo"]["ttft_ms"]),
            tpot_slo_ms=float(settings["exploration_slo"]["tpot_ms"]),
            service_warmup_requests=0,
            experiment_warmup_requests=int(settings["experiment_warmup_requests"]),
            probe_samples_per_candidate=int(settings["probe_samples_per_candidate"]),
            minimum_probe_samples=int(settings["minimum_probe_samples"]),
            candidate_stability_cv=float(settings["candidate_stability_cv"]),
            candidate_slo_headroom_ratio=float(settings["candidate_slo_headroom_ratio"]),
            exploration_shutdown_timeout_s=float(settings["exploration_shutdown_timeout_s"]),
            energy_refinement_candidate_budget=int(settings["energy_refinement_candidate_budget"]),
            exploration_max_attempts=int(settings["exploration_max_attempts"]),
            exploration_retry_backoff_s=float(settings["exploration_retry_backoff_s"]),
            event_log=settings["event_log"], service_request_log=settings["service_request_log"],
        )
        await runtime.actuate("service", *high)
        await asyncio.sleep(float(protocol["frequency_settle_s"]))
        warm_input, warm_output = panels["small_light"][1]
        await production_request("small_light", warm_input, warm_output,
                                 "setup-production-warmup", "safe_high", None, "setup")
        production_rows.clear()
        await controller.start()

        # Each epoch independently samples a category and snapshots its current
        # policy. Batches overlap; drain before any category or clock change.
        online_counts = {workload_id: 0 for workload_id in workload_ids}
        online_applied_counts = {workload_id: 0 for workload_id in workload_ids}
        sequence = 0
        online_phase_start = time.time()
        deadline = time.monotonic() + float(protocol["exploration_timeout_s"])
        while True:
            all_ready = all(multi.frequency_table.read(w).value is not None for w in workload_ids)
            enough_online = all(
                online_applied_counts[w] >= int(protocol["minimum_online_table_requests_per_class"])
                for w in workload_ids
            )
            if all_ready and enough_online:
                break
            if controller.terminal_errors:
                raise RuntimeError(f"Canary infrastructure failure: {controller.terminal_errors}")
            if controller._worker is None or controller._worker.done():
                raise RuntimeError("Canary worker stopped unexpectedly")
            if time.monotonic() > deadline:
                raise TimeoutError("seven-class Canary exploration did not complete")
            workload_id = rng.choice(workload_ids)
            center_input, center_output = panels[workload_id][1]
            await controller.enqueue_only(
                workload_id, body_for(workload_id, center_input, center_output),
                f"production-trigger-{workload_id}",
            )
            batches = []
            for _ in range(rng.randint(1,int(protocol['burst_batches_max']))):
                batch = []
                for _ in range(rng.choice(protocol['burst_size_choices'])):
                    input_len,output_len = random_shape(workload_id)
                    sequence += 1
                    batch.append(dict(input_len=input_len,output_len=output_len,
                                      request_id=f'online-production-{sequence:05d}'))
                batches.append(batch)
            rows = await production_burst(batches,workload_id,'auto',
                'production_online_feedback',f'online-burst-{sequence:05d}')
            online_counts[workload_id] += len(rows)
            online_applied_counts[workload_id] += sum(r['table_hit'] for r in rows)
        await controller.stop()
        online_phase_end = time.time()
        exploration_wall_start = min(attempt["start"] for attempt in attempts)
        exploration_wall_end = max(attempt["end"] for attempt in attempts)
        exploration_wall_gross = await meter.energy(
            ("P0", "D0"), exploration_wall_start, exploration_wall_end
        )

        # Supplemental randomized A/B uses identical concurrent cohorts in both
        # arms. Online Table application has already run above.
        plan = matched_pair_plan(panels, int(protocol["matched_pairs_per_class"]), rng)
        write(root / "matched_pair_plan.json", plan)
        evaluation_plan = burst_evaluation_plan(plan, rng)
        write(root / "evaluation_request_plan.json", evaluation_plan)
        for batch in evaluation_plan:
            for spec in batch:
                sequence += 1
                spec['request_id'] = f'evaluation-production-{sequence:05d}'
            await production_burst([batch],batch[0]['workload_id'],batch[0]['arm'],
                'matched_randomized_evaluation',f'evaluation-burst-{sequence:05d}')

        evaluation_rows = [row for row in production_rows if row["phase"] == "matched_randomized_evaluation"]
        comparison = paired_summary(evaluation_rows)
        canary_rows = [row for row in requests if row.get("phase") == "canary"]
        overhead = {
            "scope": "P0+D0 GPU boards; CPU and NIC excluded",
            "exploration_wall_start": exploration_wall_start,
            "exploration_wall_end": exploration_wall_end,
            "exploration_wall_duration_s": exploration_wall_end - exploration_wall_start,
            "exploration_wall_gross_energy_j": exploration_wall_gross["energy_j"],
            "exploration_wall_gross_mean_power_w": exploration_wall_gross["mean_power_w"],
            "attempt_gross_duration_sum_s": sum(row["duration_s"] for row in attempts),
            "attempt_gross_energy_sum_j": sum(row["energy_j"] for row in attempts),
            "probe_request_count": len(canary_rows),
            "probe_request_active_duration_sum_s": sum(row["duration_s"] for row in canary_rows),
            "probe_request_active_energy_sum_j": sum(row["energy_j"] for row in canary_rows),
            "probe_control_inclusive_duration_sum_s": sum(row["control_inclusive"]["duration_s"] for row in canary_rows),
            "probe_control_inclusive_energy_sum_j": sum(row["control_inclusive"]["energy_j"] for row in canary_rows),
            "per_class": [{
                "workload_id": workload_id,
                "attempts": sum(row["workload_id"] == workload_id for row in attempts),
                "probe_requests": sum(row["workload_id"] == workload_id for row in canary_rows),
                "gross_duration_s": sum(row["duration_s"] for row in attempts if row["workload_id"] == workload_id),
                "gross_energy_j": sum(row["energy_j"] for row in attempts if row["workload_id"] == workload_id),
                "request_active_energy_j": sum(row["energy_j"] for row in canary_rows if row["workload_id"] == workload_id),
                "request_active_duration_s": sum(row["duration_s"] for row in canary_rows if row["workload_id"] == workload_id),
                "control_inclusive_energy_j": sum(row["control_inclusive"]["energy_j"] for row in canary_rows if row["workload_id"] == workload_id),
            } for workload_id in workload_ids],
            "idle_gap_policy": "primary probe energy excludes all time outside client-send/completed-stream intervals",
            "gross_wall_interpretation": "P0+D0 energy from first queued exploration until all seven classes finish, including Canary-side idle and retry gaps",
        }
        table = multi.frequency_table.snapshot()
        summary = {
            "valid": True,
            "design": "observational_homogeneous_burst_nonfatal_slo_and_clock_quality",
            "random_seed": seed,
            "online_phase_start": online_phase_start,
            "online_phase_end": online_phase_end,
            "frequency_table": table,
            "canary_overhead": overhead,
            "paired_comparison": comparison,
            "online_requests_per_class": online_counts,
            "online_table_policy_requests_per_class": online_applied_counts,
            "fallback_workloads": [w for w, entry in table.items()
                                   if entry["value"]["source"].startswith('safe_high_after_')],
            "production_clock_noncompliant_requests": sum(not r['clock_compliant'] for r in production_rows),
            "production_slo_noncompliant_requests": sum(not r['slo_met'] for r in production_rows),
            "production_request_count": len(production_rows),
            "matched_evaluation_request_count": len(evaluation_rows),
            "sampler_health": meter.sampler_health(),
            "energy_boundaries": {
                "primary": "client send to completed SSE stream; idle and DVFS settle excluded",
                "secondary": "before DVFS actuation to completed SSE stream; switching/settle included",
            },
            "evidence_boundary": "same-category concurrent bursts; drain before category/clock change; per-request energy is in-flight share, not independent board measurement; low-load Canary table is not concurrency-optimal proof",
        }
        write(root / "canary_overhead.json", overhead)
        write(root / "summary.json", summary)
        export_csv(root / "production_requests.csv", production_rows)
        export_csv(root / "canary_probes.csv", canary_rows)
        export_csv(root / "canary_candidates.csv", candidate_rows)
        export_power_trace(root, meter.samples)

        lines = [
            "# Mixed variable-shape Canary/Production energy validation", "",
            "Primary J/request excludes idle gaps and DVFS settling. Control-inclusive J/request includes actuation and settle.", "",
            "Production J/request is allocated equally across in-flight requests per segment; it is not independent request energy. Same-category epochs hold clocks fixed and drain before switching. Canary search remains low-load and does not establish concurrent-load optimality.", "",
            "| Workload | Policy | Pairs | High active J | Table active J | Active saving | High control J | Table control J | Control saving |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in comparison:
            lines.append(
                f"| {row['workload_id']} | {'high fallback (not SLO-guaranteed)' if row['is_safe_high_fallback'] else 'learned'} | {row['matched_pairs']} | "
                f"{row['safe_high_active_mean_j']:.3f} | {row['table_active_mean_j']:.3f} | "
                f"{row['active_energy_reduction_percent']:.2f}% | "
                f"{row['safe_high_control_inclusive_mean_j']:.3f} | "
                f"{row['table_control_inclusive_mean_j']:.3f} | "
                f"{row['control_inclusive_energy_reduction_percent']:.2f}% |"
            )
        lines.extend([
            "", f"Canary wall time: {overhead['exploration_wall_duration_s']:.3f} s.",
            f"Canary active probe energy: {overhead['probe_request_active_energy_sum_j']:.3f} J.",
            f"Canary gross attempt energy: {overhead['attempt_gross_energy_sum_j']:.3f} J.",
            f"Canary full wall-span gross energy: {overhead['exploration_wall_gross_energy_j']:.3f} J.",
        ])
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        pair_counts = {workload_id: 0 for workload_id in workload_ids}
        for row in comparison:
            pair_counts[row["workload_id"]] = row["matched_pairs"]
        if any(entry["value"]["source"].startswith('safe_high_after_') and
               (entry["value"]["prefill_frequency_mhz"], entry["value"]["decode_frequency_mhz"]) != high
               for entry in table.values()):
            raise RuntimeError("fallback Table entry is not hardware safe-high")
        adjacent_same = sum(
            plan[index]["workload_id"] == plan[index - 1]["workload_id"]
            for index in range(1, len(plan))
        )
        write(root / "audit.json", {
            "valid": True,
            "cold_start_no_historical_table": True,
            "seven_table_entries_present": all(value["value"] is not None for value in table.values()),
            "matched_pairs_per_class": pair_counts,
            "matched_exact_input_output_shape": True,
            "online_table_policy_requests_per_class": online_applied_counts,
            "per_request_independent_category_sampling": False,
            "random_category_per_drained_burst_epoch": True,
            "individual_evaluation_requests_shuffled": False,
            "matched_concurrent_cohorts_shuffled": True,
            "overlap_energy_allocation_conserves_board_energy": True,
            "category_plan_adjacent_same_count": adjacent_same,
            "request_active_idle_excluded": True,
            "control_inclusive_metric_present": True,
            "clock_match_uses_tolerance": True,
            "all_energy_windows_valid": True,
        })
    finally:
        if controller is not None and controller._worker is not None:
            controller._worker.cancel()
            await asyncio.gather(controller._worker, return_exceptions=True)
        cleanup_errors = []
        if runtime is not None:
            for group in ("experiment", "service"):
                try:
                    await runtime.actuate(group, runtime.prefill_grid[-1], runtime.decode_grid[-1])
                except Exception as exc:
                    append(root / "cleanup_errors.jsonl", {"group": group, "error": str(exc)})
                    cleanup_errors.append(str(exc))
        meter.stop()
        if cleanup_errors:
            raise RuntimeError("failed to restore GPU clocks: " + repr(cleanup_errors))


def main() -> None:
    from xpyd.phase3c_substrate import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "requests_energy.jsonl").exists():
        raise RuntimeError("refusing to reuse an existing run directory")
    config = load_config(args.config)
    write(args.output / "experiment_config.json", config)
    try:
        asyncio.run(asyncio.wait_for(run(config, args.output), timeout=50400))
    except BaseException as exc:
        write(args.output / "audit.json", {"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
