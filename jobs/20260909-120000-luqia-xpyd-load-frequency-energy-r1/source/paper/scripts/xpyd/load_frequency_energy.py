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

from xpyd.energy_validation import Meter, consume, export_power_trace, percentile, write


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
        arrival_wall = time.time()
        arrival_mono = time.monotonic()
        try:
            prepared = await experiment_core.prepare(
                strip_feedback_metadata(body), request_id)
            observed_tokens = await consume(prepared, int(body["max_tokens"]))
            completion_wall = time.time()
            ttft_ms, _ = pd_inference_metrics(
                prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"]))
            row = {
                "request_id": request_id,
                "workload_id": workload_id,
                "candidate_id": candidate_id,
                "offered_rate_rps": offered_rate_rps,
                "request_index": request_index,
                "scheduled_arrival_wall_s": scheduled_arrival_wall_s,
                "actual_arrival_wall_s": arrival_wall,
                "client_schedule_slip_ms": max(
                    0.0, (arrival_wall - scheduled_arrival_wall_s) * 1000.0),
                "completion_wall_s": completion_wall,
                "client_latency_s": time.monotonic() - arrival_mono,
                "ttft_ms": ttft_ms,
                "ttft_below_reference": ttft_ms < reference_ttft_ms,
                "observed_output_tokens": observed_tokens,
                "success": True,
                "error": None,
            }
            if request_id not in diagnostics:
                raise RuntimeError("completed stream missing diagnostic")
        except Exception as exc:
            row = {
                "request_id": request_id,
                "workload_id": workload_id,
                "candidate_id": candidate_id,
                "offered_rate_rps": offered_rate_rps,
                "request_index": request_index,
                "scheduled_arrival_wall_s": scheduled_arrival_wall_s,
                "actual_arrival_wall_s": arrival_wall,
                "client_schedule_slip_ms": max(
                    0.0, (arrival_wall - scheduled_arrival_wall_s) * 1000.0),
                "completion_wall_s": time.time(),
                "client_latency_s": time.monotonic() - arrival_mono,
                "ttft_ms": None,
                "ttft_below_reference": None,
                "observed_output_tokens": None,
                "success": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        request_rows.append(row)
        append(root / "requests.jsonl", row)
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
        energy = await meter.energy(
            pair, start_wall, end_wall, targets=(int(p_mhz), int(d_mhz)))
        successes = [row for row in rows if row.get("success")]
        ttfts = [float(row["ttft_ms"]) for row in successes]
        duration = max(end_wall - start_wall, 1e-9)
        measurement_valid = all((
            not inflight_limit_hit,
            not drain_timed_out,
            issued == planned,
            len(successes) == planned,
            energy["energy_j"] > 0,
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
            "energy_j": energy["energy_j"],
            "mean_power_w": energy["mean_power_w"],
            "joules_per_completed_request": (
                energy["energy_j"] / len(successes) if successes else None),
            "energy_endpoints": energy["endpoints"],
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
                    "joules_per_request": row["confirmation_joules_per_request"],
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
                "best observed axis-factorized joint GPU joules/request under each "
                "open-loop load; not a full P-by-D global oracle"),
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
            "| Workload | Offered RPS | P MHz | D MHz | Confirm J/request | TTFT P95 ms | TTFT<500 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            lines.append(
                "| {workload_id} | {offered_rate_rps:.3f} | {selected_p_mhz} | "
                "{selected_d_mhz} | {confirmation_joules_per_request:.3f} | "
                "{confirmation_ttft_p95_ms:.3f} | {reference_ttft_p95_pass} |".format(**row))
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        export_power_trace(root, meter.samples)
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
