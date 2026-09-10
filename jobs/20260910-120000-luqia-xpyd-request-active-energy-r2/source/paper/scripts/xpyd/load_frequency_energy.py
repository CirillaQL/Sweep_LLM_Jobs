"""Explore whether the energy-minimizing P/D clocks change with offered load.

The experiment is intentionally predictor-free and imports no historical table.
For every workload/load cell it minimizes the P axis with D held high, then the
D axis with P held at the selected value.  Each one-dimensional minimization
uses a binary search over adjacent energy slopes, which assumes a unimodal
discrete energy curve.  Results are therefore an axis-factorized empirical
search, not a full P-by-D oracle.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from xpyd.energy_validation import Meter, export_power_trace, percentile, write


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
    """Consume SSE while timestamping the first client-visible token."""
    if prepared.stream is None or prepared.status_code != 200:
        raise RuntimeError("missing successful SSE response")
    buffer = b""
    done = False
    tokens = None
    first_token_wall_ns = None
    first_token_mono_ns = None
    last_chunk_wall_ns = None
    chunk_count = 0

    def process_line(line: bytes, received_wall_ns: int, received_mono_ns: int) -> None:
        nonlocal done, tokens, first_token_wall_ns, first_token_mono_ns
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
        choices = event.get("choices") or []
        if (first_token_wall_ns is None and any(
                choice.get("text") not in (None, "") for choice in choices)):
            first_token_wall_ns = received_wall_ns
            first_token_mono_ns = received_mono_ns

    async for chunk in prepared.stream:
        received_wall_ns = time.time_ns()
        received_mono_ns = time.monotonic_ns()
        last_chunk_wall_ns = received_wall_ns
        chunk_count += 1
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            process_line(line.rstrip(b"\r"), received_wall_ns, received_mono_ns)
    if buffer:
        now_wall_ns = time.time_ns()
        process_line(buffer.rstrip(b"\r"), now_wall_ns, time.monotonic_ns())
    if not done or tokens != expected_tokens or first_token_wall_ns is None:
        raise RuntimeError(
            f"incomplete generation: done={done}, tokens={tokens}, "
            f"first_token={first_token_wall_ns is not None}, expected={expected_tokens}")
    return {
        "observed_output_tokens": tokens,
        "client_first_token_wall_ns": first_token_wall_ns,
        "client_first_token_mono_ns": first_token_mono_ns,
        "client_last_chunk_wall_ns": last_chunk_wall_ns,
        "client_stream_chunk_count": chunk_count,
    }


def planned_request_count(
    rate_rps: float, target_arrival_window_s: float,
    minimum_requests: int, maximum_requests: int,
) -> int:
    if rate_rps <= 0:
        raise ValueError("rate_rps must be positive")
    return min(
        int(maximum_requests),
        max(int(minimum_requests), int(math.ceil(rate_rps * target_arrival_window_s))),
    )


def objective_energy(row: Mapping[str, Any]) -> float:
    value = row.get("joules_per_completed_request")
    if not row.get("measurement_valid") or value is None:
        return math.inf
    number = float(value)
    return number if math.isfinite(number) and number > 0 else math.inf


def choose_binary_half(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """Choose the half containing a discrete unimodal minimum.

    Adjacent samples at indices mid and mid+1 define the local slope.  Invalid
    points have infinite objective, so the search naturally moves toward a
    neighboring valid point.  A double-invalid comparison conservatively moves
    upward in frequency, where completion is more likely.
    """
    left_value = objective_energy(left)
    right_value = objective_energy(right)
    if math.isinf(left_value) and math.isinf(right_value):
        return "right"
    return "left" if left_value <= right_value else "right"


def best_valid(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [row for row in rows if math.isfinite(objective_energy(row))]
    if not valid:
        raise RuntimeError("frequency search produced no valid energy measurement")
    return min(valid, key=objective_energy)


async def binary_energy_minimize(
    grid: Sequence[int], evaluate: Callable[[int, int], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Minimize a one-dimensional discrete energy curve by binary slope search."""
    if len(grid) < 2 or tuple(grid) != tuple(sorted(set(int(v) for v in grid))):
        raise ValueError("frequency grid must contain at least two increasing values")
    cache: dict[int, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []

    async def at(index: int) -> dict[str, Any]:
        if index not in cache:
            cache[index] = await evaluate(index, int(grid[index]))
        return cache[index]

    # Explicit endpoint measurements make the approximation auditable and keep
    # a noisy interior comparison from hiding an endpoint optimum.
    await at(0)
    await at(len(grid) - 1)
    low = 0
    high = len(grid) - 1
    while low < high:
        midpoint = (low + high) // 2
        left = await at(midpoint)
        right = await at(midpoint + 1)
        direction = choose_binary_half(left, right)
        steps.append({
            "low_index_before": low,
            "high_index_before": high,
            "mid_index": midpoint,
            "mid_frequency_mhz": int(grid[midpoint]),
            "next_index": midpoint + 1,
            "next_frequency_mhz": int(grid[midpoint + 1]),
            "mid_joules_per_request": left.get("joules_per_completed_request"),
            "next_joules_per_request": right.get("joules_per_completed_request"),
            "mid_measurement_valid": bool(left.get("measurement_valid")),
            "next_measurement_valid": bool(right.get("measurement_valid")),
            "decision": direction,
        })
        if direction == "left":
            high = midpoint
        else:
            low = midpoint + 1

    # Measure the immediate neighborhood and choose the lowest observed energy
    # among every point actually visited by the search.
    for index in range(max(0, low - 1), min(len(grid), low + 2)):
        await at(index)
    selected = best_valid(list(cache.values()))
    return {
        "assumption": "discrete_unimodal_axis_energy",
        "global_oracle_proven": False,
        "binary_terminal_index": low,
        "binary_terminal_frequency_mhz": int(grid[low]),
        "selected_frequency_mhz": int(selected["axis_frequency_mhz"]),
        "selected_candidate_id": selected["candidate_id"],
        "selected_joules_per_completed_request": objective_energy(selected),
        "tested_frequency_mhz": sorted(
            int(row["axis_frequency_mhz"]) for row in cache.values()),
        "steps": steps,
        "candidates": sorted(cache.values(), key=lambda row: int(row["axis_frequency_mhz"])),
    }


async def run(config: Mapping[str, Any], root: Path) -> None:
    import aiohttp
    from transformers import AutoTokenizer
    from replay_synthetic_trace import build_prompt_cache
    from xpyd.disagg_proxy import _build_multi_core
    from xpyd.online_feedback_controller import (
        PhysicalFeedbackRuntime,
        pd_inference_metrics,
        strip_feedback_metadata,
    )

    protocol = dict(config["load_frequency_energy"])
    pair = tuple(protocol.get("experiment_pair", ["P0", "D0"]))
    reference_ttft_ms = float(protocol.get("reference_ttft_ms", 500.0))
    request_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}

    def sink(row: dict[str, Any]) -> None:
        diagnostics[row["request_id"]] = row
        append(root / "diagnostics.jsonl", row)

    timeout = aiohttp.ClientTimeout(
        total=float(protocol.get("request_timeout_s", 900.0)), connect=10)
    multi = _build_multi_core(
        config, lambda: aiohttp.ClientSession(timeout=timeout), diagnostic_sink=sink)
    experiment_core = multi.cores[pair]
    meter = Meter(root)
    meter.start()
    runtime = None
    sequence = 0

    async def one_request(
        body: Mapping[str, Any], request_id: str, workload_id: str,
        candidate_id: str, offered_rate_rps: float, request_index: int,
        scheduled_arrival_wall_s: float,
    ) -> dict[str, Any]:
        arrival_wall_ns = time.time_ns()
        arrival_mono_ns = time.monotonic_ns()
        arrival_wall = arrival_wall_ns / 1e9
        prepared = None
        try:
            prepared = await experiment_core.prepare(
                strip_feedback_metadata(body), request_id)
            stream_timing = await consume_timed(prepared, int(body["max_tokens"]))
            completion_wall_ns = time.time_ns()
            completion_mono_ns = time.monotonic_ns()
            completion_wall = completion_wall_ns / 1e9
            ttft_ms, _ = pd_inference_metrics(
                prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"]))
            proxy_wall = dict(prepared.diagnostics.timestamps_wall_s)
            proxy_mono = dict(prepared.diagnostics.timestamps_monotonic_s)
            row = {
                "request_id": request_id,
                "workload_id": workload_id,
                "candidate_id": candidate_id,
                "offered_rate_rps": offered_rate_rps,
                "request_index": request_index,
                "scheduled_arrival_wall_s": scheduled_arrival_wall_s,
                "actual_arrival_wall_s": arrival_wall,
                "client_send_wall_ns": arrival_wall_ns,
                "client_send_mono_ns": arrival_mono_ns,
                "client_schedule_slip_ms": max(
                    0.0, (arrival_wall - scheduled_arrival_wall_s) * 1000.0),
                "completion_wall_s": completion_wall,
                "client_complete_wall_ns": completion_wall_ns,
                "client_complete_mono_ns": completion_mono_ns,
                "client_latency_s": (completion_mono_ns - arrival_mono_ns) / 1e9,
                "client_ttft_ms": (
                    stream_timing["client_first_token_mono_ns"] - arrival_mono_ns
                ) / 1e6,
                "ttft_ms": ttft_ms,
                "ttft_below_reference": ttft_ms < reference_ttft_ms,
                **stream_timing,
                "proxy_timestamps_wall_s": proxy_wall,
                "proxy_timestamps_monotonic_s": proxy_mono,
                "proxy_receive_after_client_send_ms": (
                    float(proxy_wall["request_received"]) - arrival_wall
                ) * 1000.0,
                "active_interval_start_wall_s": arrival_wall,
                "active_interval_end_wall_s": completion_wall,
                "success": True,
                "error": None,
            }
            if request_id not in diagnostics:
                raise RuntimeError("completed stream missing diagnostic")
        except Exception as exc:
            completion_wall_ns = time.time_ns()
            completion_mono_ns = time.monotonic_ns()
            row = {
                "request_id": request_id,
                "workload_id": workload_id,
                "candidate_id": candidate_id,
                "offered_rate_rps": offered_rate_rps,
                "request_index": request_index,
                "scheduled_arrival_wall_s": scheduled_arrival_wall_s,
                "actual_arrival_wall_s": arrival_wall,
                "client_send_wall_ns": arrival_wall_ns,
                "client_send_mono_ns": arrival_mono_ns,
                "client_schedule_slip_ms": max(
                    0.0, (arrival_wall - scheduled_arrival_wall_s) * 1000.0),
                "completion_wall_s": completion_wall_ns / 1e9,
                "client_complete_wall_ns": completion_wall_ns,
                "client_complete_mono_ns": completion_mono_ns,
                "client_latency_s": (completion_mono_ns - arrival_mono_ns) / 1e9,
                "client_ttft_ms": None,
                "ttft_ms": None,
                "ttft_below_reference": None,
                "observed_output_tokens": None,
                "success": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        return row

    async def warmup(body: Mapping[str, Any], workload_id: str, count: int) -> None:
        nonlocal sequence
        for index in range(count):
            sequence += 1
            now = time.time()
            row = await one_request(
                body, f"warmup-{sequence:07d}", workload_id, "warmup", 0.0,
                index + 1, now)
            if not row["success"]:
                raise RuntimeError(f"warmup failed for {workload_id}: {row['error']}")
            row["energy_scope"] = "warmup_not_measured"
            request_rows.append(row)
            append(root / "requests.jsonl", row)

    async def candidate(
        body: Mapping[str, Any], workload_id: str, rate_rps: float,
        p_mhz: int, d_mhz: int, axis: str, axis_frequency_mhz: int,
        phase: str, ordinal: int,
    ) -> dict[str, Any]:
        nonlocal sequence
        label = (
            f"{workload_id}-rps-{rate_rps:.3f}-{phase}-{ordinal:02d}-"
            f"p-{p_mhz}-d-{d_mhz}")
        changed = await runtime.actuate("experiment", p_mhz, d_mhz)
        await asyncio.sleep(float(protocol["frequency_settle_s"]))
        planned = planned_request_count(
            rate_rps, float(protocol["target_arrival_window_s"]),
            int(protocol["minimum_requests_per_candidate"]),
            int(protocol["maximum_requests_per_candidate"]),
        )
        interval = 1.0 / rate_rps
        max_inflight = int(protocol["maximum_inflight_requests"])
        start_wall = time.time()
        start_mono = time.monotonic()
        tasks: list[asyncio.Task] = []
        issued = 0
        inflight_limit_hit = False
        for index in range(planned):
            target_mono = start_mono + index * interval
            await asyncio.sleep(max(0.0, target_mono - time.monotonic()))
            if sum(not task.done() for task in tasks) >= max_inflight:
                inflight_limit_hit = True
                break
            sequence += 1
            issued += 1
            scheduled_wall = start_wall + index * interval
            tasks.append(asyncio.create_task(one_request(
                body, f"loadfreq-{sequence:07d}", workload_id, label,
                rate_rps, issued, scheduled_wall)))
        arrival_stop_wall = time.time()
        backlog_at_arrival_stop = sum(not task.done() for task in tasks)
        drain_timed_out = False
        try:
            rows = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=float(protocol["drain_timeout_s"]))
        except asyncio.TimeoutError:
            drain_timed_out = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            rows = [row for row in gathered if isinstance(row, dict)]
        end_wall = time.time()
        successes = [row for row in rows if row.get("success")]
        request_intervals = [
            (row["request_id"], row["active_interval_start_wall_s"],
             row["active_interval_end_wall_s"])
            for row in successes
        ]
        try:
            active_energy = await meter.energy_intervals(
                pair,
                [(start, end) for _, start, end in request_intervals],
                targets=(int(p_mhz), int(d_mhz)),
            )
            gross_energy = await meter.energy(
                pair, start_wall, end_wall, targets=(int(p_mhz), int(d_mhz)))
            attributed = await meter.attribute_energy(pair, request_intervals)

            p_intervals = []
            d_intervals = []
            for row in successes:
                stamps = row["proxy_timestamps_wall_s"]
                if stamps.get("prefill_started") is not None and stamps.get("prefill_completed") is not None:
                    p_intervals.append((row["request_id"], stamps["prefill_started"],
                                        stamps["prefill_completed"]))
                if stamps.get("decode_request_started") is not None and stamps.get("response_completed") is not None:
                    d_intervals.append((row["request_id"], stamps["decode_request_started"],
                                        stamps["response_completed"]))
            p_attributed = (
                await meter.attribute_energy((pair[0],), p_intervals)
                if p_intervals else {})
            d_attributed = (
                await meter.attribute_energy((pair[1],), d_intervals)
                if d_intervals else {})

            for row in rows:
                allocation = attributed.get(row["request_id"])
                row["energy_scope"] = "client_send_to_client_complete_active_union"
                row["active_shared_energy_j"] = (
                    allocation["total_j"] if allocation else None)
                row["active_shared_endpoint_energy_j"] = (
                    allocation["endpoints_j"] if allocation else None)
                row["prefill_phase_shared_p0_energy_j"] = (
                    p_attributed.get(row["request_id"], {}).get("total_j"))
                row["decode_phase_shared_d0_energy_j"] = (
                    d_attributed.get(row["request_id"], {}).get("total_j"))
                request_rows.append(row)
                append(root / "requests.jsonl", row)
        except BaseException:
            for row in rows:
                row["energy_scope"] = "measurement_failed"
                row["active_shared_energy_j"] = None
                request_rows.append(row)
                append(root / "requests.jsonl", row)
            raise
        ttfts = [float(row["ttft_ms"]) for row in successes]
        duration = max(end_wall - start_wall, 1e-9)
        allocated_energy = sum(
            value["total_j"] for value in attributed.values())
        conservation_error_j = allocated_energy - active_energy["energy_j"]
        timeline_keys = (
            "request_received", "route_selected", "prefill_started",
            "prefill_completed", "kv_handoff_completed", "decode_request_started",
            "decode_response_headers_received", "decode_first_real_chunk_received",
            "decode_first_real_chunk_forwarded", "decode_last_chunk_received",
            "response_completed",
        )
        timeline_complete = all(
            all(row["proxy_timestamps_wall_s"].get(key) is not None
                for key in timeline_keys)
            for row in successes
        )
        measurement_valid = all((
            not inflight_limit_hit,
            not drain_timed_out,
            issued == planned,
            len(successes) == planned,
            timeline_complete,
            active_energy["energy_j"] > 0,
            abs(conservation_error_j) <= max(1e-6, active_energy["energy_j"] * 1e-9),
        ))
        result = {
            "candidate_id": label,
            "phase": phase,
            "axis": axis,
            "axis_frequency_mhz": int(axis_frequency_mhz),
            "workload_id": workload_id,
            "offered_rate_rps": rate_rps,
            "p_mhz": int(p_mhz),
            "d_mhz": int(d_mhz),
            "frequency_changed": changed,
            "planned_requests": planned,
            "issued_requests": issued,
            "successful_requests": len(successes),
            "failed_requests": issued - len(successes),
            "window_start_s": start_wall,
            "arrival_stop_s": arrival_stop_wall,
            "window_end_s": end_wall,
            "window_duration_s": duration,
            "request_active_union_duration_s": active_energy["duration_s"],
            "excluded_no_request_inflight_s": duration - active_energy["duration_s"],
            "request_active_interval_count_after_merge": len(active_energy["intervals"]),
            "request_active_time_sum_s": sum(end - start for _, start, end in request_intervals),
            "mean_inflight_during_active_union": (
                sum(end - start for _, start, end in request_intervals)
                / active_energy["duration_s"]),
            "backlog_at_arrival_stop": backlog_at_arrival_stop,
            "inflight_limit_hit": inflight_limit_hit,
            "drain_timed_out": drain_timed_out,
            "achieved_throughput_rps_including_drain": len(successes) / duration,
            "achieved_to_offered_ratio_including_drain": (
                len(successes) / duration / rate_rps),
            "ttft_count": len(ttfts),
            "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else None,
            "ttft_p50_ms": percentile(ttfts, 0.50) if ttfts else None,
            "ttft_p95_ms": percentile(ttfts, 0.95) if ttfts else None,
            "ttft_max_ms": max(ttfts) if ttfts else None,
            "reference_ttft_ms": reference_ttft_ms,
            "ttft_reference_pass_fraction": (
                sum(value < reference_ttft_ms for value in ttfts) / len(ttfts)
                if ttfts else None),
            "energy_scope": "client_send_to_client_complete_active_union",
            "energy_j": active_energy["energy_j"],
            "active_energy_j": active_energy["energy_j"],
            "active_mean_power_w": active_energy["mean_power_w"],
            "gross_window_energy_j": gross_energy["energy_j"],
            "gross_window_mean_power_w": gross_energy["mean_power_w"],
            "gross_window_joules_per_completed_request": (
                gross_energy["energy_j"] / len(successes) if successes else None),
            "attributed_energy_sum_j": allocated_energy,
            "attribution_conservation_error_j": conservation_error_j,
            "joules_per_completed_request": (
                active_energy["energy_j"] / len(successes) if successes else None),
            "active_energy_endpoints": active_energy["endpoints"],
            "gross_energy_endpoints": gross_energy["endpoints"],
            "timeline_complete": timeline_complete,
            "measurement_valid": measurement_valid,
            "slo_used_as_selection_gate": False,
        }
        candidate_rows.append(result)
        append(root / "load_frequency_candidates.jsonl", result)
        await asyncio.sleep(float(protocol["candidate_cooldown_s"]))
        return result

    async def explore_cell(
        body: Mapping[str, Any], workload_id: str, rate_rps: float,
    ) -> dict[str, Any]:
        high_p = int(runtime.prefill_grid[-1])
        high_d = int(runtime.decode_grid[-1])
        ordinal = 0

        async def evaluate_p(index: int, frequency: int) -> dict[str, Any]:
            nonlocal ordinal
            ordinal += 1
            return await candidate(
                body, workload_id, rate_rps, frequency, high_d,
                "prefill", frequency, "p-axis", ordinal)

        p_search = await binary_energy_minimize(runtime.prefill_grid, evaluate_p)
        selected_p = int(p_search["selected_frequency_mhz"])

        async def evaluate_d(index: int, frequency: int) -> dict[str, Any]:
            nonlocal ordinal
            ordinal += 1
            return await candidate(
                body, workload_id, rate_rps, selected_p, frequency,
                "decode", frequency, "d-axis", ordinal)

        d_search = await binary_energy_minimize(runtime.decode_grid, evaluate_d)
        selected_d = int(d_search["selected_frequency_mhz"])
        confirmations = []
        required_confirmations = int(protocol["confirmation_windows"])
        maximum_attempts = int(protocol["confirmation_max_attempts"])
        attempt = 0
        while (sum(row["measurement_valid"] for row in confirmations)
               < required_confirmations and attempt < maximum_attempts):
            attempt += 1
            ordinal += 1
            confirmations.append(await candidate(
                body, workload_id, rate_rps, selected_p, selected_d,
                "joint", selected_d, f"confirmation-{attempt}", ordinal))
        valid_confirmations = [row for row in confirmations if row["measurement_valid"]]
        if len(valid_confirmations) < required_confirmations:
            raise RuntimeError(
                f"invalid confirmation for {workload_id} at {rate_rps} RPS")
        pooled_ttft = []
        confirmation_ids = {row["candidate_id"] for row in valid_confirmations}
        for row in request_rows:
            if row["candidate_id"] in confirmation_ids and row.get("success"):
                pooled_ttft.append(float(row["ttft_ms"]))
        total_energy = sum(float(row["energy_j"]) for row in valid_confirmations)
        total_gross_energy = sum(
            float(row["gross_window_energy_j"]) for row in valid_confirmations)
        total_requests = sum(int(row["successful_requests"]) for row in valid_confirmations)
        return {
            "workload_id": workload_id,
            "offered_rate_rps": rate_rps,
            "selected_p_mhz": selected_p,
            "selected_d_mhz": selected_d,
            "selection_method": "P-axis binary energy minimum at D-high, then D-axis binary energy minimum at selected P",
            "factorized_global_optimum_assumption": True,
            "full_p_by_d_oracle_proven": False,
            "slo_used_as_selection_gate": False,
            "p_search": p_search,
            "d_search": d_search,
            "confirmation_candidate_ids": sorted(confirmation_ids),
            "confirmation_windows": len(valid_confirmations),
            "confirmation_attempts": len(confirmations),
            "confirmation_requests": total_requests,
            "confirmation_energy_j": total_energy,
            "confirmation_joules_per_request": total_energy / total_requests,
            "confirmation_gross_window_energy_j": total_gross_energy,
            "confirmation_gross_window_joules_per_request": (
                total_gross_energy / total_requests),
            "confirmation_ttft_mean_ms": statistics.fmean(pooled_ttft),
            "confirmation_ttft_p50_ms": percentile(pooled_ttft, 0.50),
            "confirmation_ttft_p95_ms": percentile(pooled_ttft, 0.95),
            "confirmation_ttft_max_ms": max(pooled_ttft),
            "reference_ttft_ms": reference_ttft_ms,
            "reference_ttft_p95_pass": percentile(pooled_ttft, 0.95) < reference_ttft_ms,
        }

    try:
        await meter.ready()
        runtime = PhysicalFeedbackRuntime(config, experiment_core)
        if len(runtime.prefill_grid) != 17 or len(runtime.decode_grid) != 15:
            raise RuntimeError(
                f"unexpected grids P={runtime.prefill_grid}, D={runtime.decode_grid}")
        high = (int(runtime.prefill_grid[-1]), int(runtime.decode_grid[-1]))
        if high != (2520, 1500):
            raise RuntimeError(f"unexpected safe-high clocks: {high}")
        await runtime.actuate("experiment", *high)
        await runtime.actuate("service", *high)
        await asyncio.sleep(1.0)
        write(root / "hardware_grids.json", {
            "P": runtime.prefill_grid,
            "D": runtime.decode_grid,
            "selected_safe_high": {"P": high[0], "D": high[1]},
        })
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_model"], local_files_only=True)
        prompts = build_prompt_cache(
            tokenizer, {int(item["input_len"]) for item in config["workloads"]})
        bodies = {
            item["id"]: {
                "model": config["model"],
                "prompt": prompts[item["input_len"]],
                "max_tokens": item["output_len"],
                "ignore_eos": True,
                "temperature": 0,
                "stream": True,
                "xpyd_input_len": item["input_len"],
                "xpyd_output_len": item["output_len"],
                "xpyd_workload_id": item["id"],
            }
            for item in config["workloads"]
        }
        results = []
        deadline = time.monotonic() + float(protocol["experiment_timeout_s"])
        for workload_id, body in bodies.items():
            await runtime.actuate("experiment", *high)
            await warmup(
                body, workload_id, int(protocol["warmup_requests_per_workload"]))
            for rate_rps in protocol["load_rps"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("load-frequency experiment exceeded internal deadline")
                results.append(await explore_cell(body, workload_id, float(rate_rps)))

        expected_cells = len(config["workloads"]) * len(protocol["load_rps"])
        valid = len(results) == expected_cells
        table = {
            workload_id: {
                f"{float(row['offered_rate_rps']):.3f}": {
                    "P_mhz": row["selected_p_mhz"],
                    "D_mhz": row["selected_d_mhz"],
                    "active_joules_per_request": row["confirmation_joules_per_request"],
                    "gross_window_joules_per_request": (
                        row["confirmation_gross_window_joules_per_request"]),
                    "ttft_p95_ms": row["confirmation_ttft_p95_ms"],
                }
                for row in results if row["workload_id"] == workload_id
            }
            for workload_id in bodies
        }
        write(root / "load_frequency_table.json", table)
        write(root / "summary.json", {
            "valid": valid,
            "design": "open_loop_load_conditioned_axis_factorized_energy_search",
            "primary_energy_boundary": "union of client-send to client-complete intervals",
            "idle_gap_policy": "exclude intervals with zero requests in flight",
            "concurrent_energy_attribution": "equal share among in-flight requests per interval",
            "experiment_pair": list(pair),
            "load_rps": protocol["load_rps"],
            "workloads": list(bodies),
            "slo_used_as_selection_gate": False,
            "reference_ttft_ms_for_reporting_only": reference_ttft_ms,
            "frequency_grids": {
                "P": list(runtime.prefill_grid), "D": list(runtime.decode_grid)},
            "cells": results,
            "candidate_windows": len(candidate_rows),
            "request_attempts": len(request_rows),
            "interpretation": (
                "best observed axis-factorized joint active-interval GPU "
                "joules/request under each open-loop load; zero-inflight gaps "
                "are excluded; not a full P-by-D global oracle"),
        })
        write(root / "audit.json", {
            "valid": valid,
            "historical_prior_used": False,
            "workload_ids": list(bodies),
            "only_requested_workloads": set(bodies) == {
                "small_light", "prefill_medium"},
            "load_count": len(protocol["load_rps"]),
            "all_load_cells_present": len(results) == expected_cells,
            "open_loop_arrivals": True,
            "slo_used_as_selection_gate": False,
            "ttft_recorded": all(
                row["confirmation_ttft_p95_ms"] is not None for row in results),
            "energy_recorded": all(
                row["confirmation_joules_per_request"] > 0 for row in results),
            "request_timeline_complete": all(
                row["timeline_complete"] for row in candidate_rows),
            "active_energy_attribution_conserved": all(
                abs(row["attribution_conservation_error_j"])
                <= max(1e-6, row["active_energy_j"] * 1e-9)
                for row in candidate_rows),
            "p_grid_levels": len(runtime.prefill_grid),
            "d_grid_levels": len(runtime.decode_grid),
            "axis_factorized_search": True,
            "full_p_by_d_oracle_proven": False,
        })
        export_csv(root / "load_frequency_candidates.csv", candidate_rows)
        export_csv(root / "requests.csv", request_rows)
        lines = [
            "# Load-conditioned P/D energy search", "",
            "SLO is not a selection gate; TTFT is reported descriptively.", "",
            "| Workload | Offered RPS | P MHz | D MHz | Active J/request | Gross-window J/request | TTFT P95 ms | TTFT<500 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            lines.append(
                "| {workload_id} | {offered_rate_rps:.3f} | {selected_p_mhz} | "
                "{selected_d_mhz} | {confirmation_joules_per_request:.3f} | "
                "{confirmation_gross_window_joules_per_request:.3f} | "
                "{confirmation_ttft_p95_ms:.3f} | {reference_ttft_p95_pass} |".format(**row))
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not valid:
            raise RuntimeError("one or more load-frequency cells are missing")
    finally:
        cleanup_errors = []
        if runtime is not None:
            for group in ("experiment", "service"):
                try:
                    await runtime.actuate(
                        group, runtime.prefill_grid[-1], runtime.decode_grid[-1])
                except Exception as exc:
                    append(root / "cleanup_errors.jsonl", {
                        "group": group, "error": str(exc)})
                    cleanup_errors.append(str(exc))
        meter.stop()
        write(root / "sampler_health.json", meter.sampler_health())
        try:
            export_power_trace(root, meter.samples)
        except Exception as exc:
            append(root / "cleanup_errors.jsonl", {
                "group": "power_trace_export", "error": str(exc)})
        if cleanup_errors:
            raise RuntimeError("failed to restore GPU clocks: " + repr(cleanup_errors))


def main() -> None:
    from xpyd.phase3c_substrate import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "load_frequency_candidates.jsonl").exists():
        raise RuntimeError("refusing to reuse an existing run directory")
    config = load_config(args.config)
    write(args.output / "experiment_config.json", config)
    try:
        asyncio.run(asyncio.wait_for(
            run(config, args.output),
            timeout=float(config["load_frequency_energy"]["experiment_timeout_s"]) + 600.0,
        ))
    except BaseException as exc:
        write(args.output / "audit.json", {
            "valid": False,
            "error": type(exc).__name__ + ": " + str(exc),
        })
        raise


if __name__ == "__main__":
    main()
