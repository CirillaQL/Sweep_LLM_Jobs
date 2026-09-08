"""Measure per-workload maximum stable open-loop throughput at safe-high clocks.

The driver intentionally starts from no historical capacity or frequency table.
For each request shape it brackets the stable/unstable offered rate, refines the
boundary, and requires repeated confirmation of the selected conservative rate.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

from xpyd.energy_validation import Meter, consume, export_power_trace, percentile, write


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def export_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def late_latency_stable(rows: list[dict], ratio: float, allowance_ms: float) -> tuple[bool, float, float]:
    ordered = sorted(rows, key=lambda row: row["arrival_wall_s"])
    width = max(1, len(ordered) // 3)
    early = statistics.median(row["ttft_ms"] for row in ordered[:width])
    late = statistics.median(row["ttft_ms"] for row in ordered[-width:])
    return late <= min(early * ratio, early + allowance_ms), early, late


def candidate_is_stable(*, inflight_limit_hit: bool, drain_timed_out: bool,
                        issued: int, planned: int, success_ratio: float,
                        completion_ratio: float, ttft_p95_ms: float | None,
                        tpot_p95_ms: float | None, trend_ok: bool,
                        protocol: dict, slo: dict) -> bool:
    return all((
        not inflight_limit_hit,
        not drain_timed_out,
        issued == planned,
        success_ratio >= float(protocol["minimum_success_ratio"]),
        completion_ratio >= float(protocol["minimum_achieved_to_offered_ratio"]),
        ttft_p95_ms is not None and ttft_p95_ms < float(slo["ttft_ms"]),
        tpot_p95_ms is not None and tpot_p95_ms <= float(slo["tpot_ms"]),
        trend_ok,
    ))


async def run(config: dict, root: Path) -> None:
    import aiohttp
    from transformers import AutoTokenizer
    from replay_synthetic_trace import build_prompt_cache
    from xpyd.disagg_proxy import _build_multi_core
    from xpyd.online_feedback_controller import (
        PhysicalFeedbackRuntime,
        pd_inference_metrics,
        strip_feedback_metadata,
    )

    protocol = config["throughput_capacity"]
    pair = tuple(protocol.get("service_pair", ["P1", "D1"]))
    slo = protocol["slo"]
    diagnostics: dict[str, dict] = {}
    request_rows: list[dict] = []
    candidate_rows: list[dict] = []

    def sink(row: dict) -> None:
        diagnostics[row["request_id"]] = row
        append(root / "diagnostics.jsonl", row)

    timeout = aiohttp.ClientTimeout(
        total=float(protocol.get("request_timeout_s", 900)), connect=10)
    multi = _build_multi_core(config, lambda: aiohttp.ClientSession(timeout=timeout),
                              diagnostic_sink=sink)
    service_core = multi.cores[pair]
    meter = Meter(root)
    meter.start()
    runtime = None
    sequence = 0

    async def one_request(body: dict, request_id: str, workload_id: str,
                          candidate_id: str, offered_rate_rps: float,
                          request_index: int) -> dict:
        arrival_wall = time.time()
        arrival_mono = time.monotonic()
        try:
            prepared = await service_core.prepare(strip_feedback_metadata(body), request_id)
            tokens = await consume(prepared, int(body["max_tokens"]))
            completion_wall = time.time()
            ttft, tpot = pd_inference_metrics(
                prepared.diagnostics.timestamps_monotonic_s, int(body["max_tokens"]))
            row = {
                "request_id": request_id,
                "workload_id": workload_id,
                "candidate_id": candidate_id,
                "offered_rate_rps": offered_rate_rps,
                "request_index": request_index,
                "arrival_wall_s": arrival_wall,
                "completion_wall_s": completion_wall,
                "client_latency_s": time.monotonic() - arrival_mono,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "observed_output_tokens": tokens,
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
                "arrival_wall_s": arrival_wall,
                "completion_wall_s": time.time(),
                "client_latency_s": time.monotonic() - arrival_mono,
                "ttft_ms": None,
                "tpot_ms": None,
                "observed_output_tokens": None,
                "success": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        request_rows.append(row)
        append(root / "requests.jsonl", row)
        return row

    async def warmup(body: dict, workload_id: str, count: int) -> None:
        nonlocal sequence
        for index in range(count):
            sequence += 1
            row = await one_request(body, f"warmup-{sequence:06d}", workload_id,
                                    "warmup", 0.0, index + 1)
            if not row["success"]:
                raise RuntimeError(f"warmup failed for {workload_id}: {row['error']}")

    async def candidate(body: dict, workload_id: str, rate: float,
                        label: str, phase: str = "search") -> dict:
        nonlocal sequence
        min_requests = int(protocol["minimum_requests_per_candidate"])
        target_seconds = float(protocol["target_arrival_window_s"])
        max_requests = int(protocol["maximum_requests_per_candidate"])
        planned = min(max_requests, max(min_requests, int(math.ceil(rate * target_seconds))))
        max_inflight = int(protocol["maximum_inflight_requests"])
        interval = 1.0 / rate
        start_wall = time.time()
        start_mono = time.monotonic()
        tasks: list[asyncio.Task] = []
        issued = 0
        inflight_limit_hit = False
        for index in range(planned):
            target = start_mono + index * interval
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            active = sum(not task.done() for task in tasks)
            if active >= max_inflight:
                inflight_limit_hit = True
                break
            sequence += 1
            issued += 1
            tasks.append(asyncio.create_task(one_request(
                body, f"capacity-{sequence:06d}", workload_id, label, rate, issued)))
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
            rows = await asyncio.gather(*tasks, return_exceptions=True)
            rows = [row for row in rows if isinstance(row, dict)]
        end_wall = time.time()
        energy = await meter.energy(pair, start_wall, end_wall)
        successes = [row for row in rows if row.get("success")]
        failures = issued - len(successes)
        duration = max(end_wall - start_wall, 1e-9)
        achieved = len(successes) / duration
        completion_ratio = achieved / rate
        success_ratio = len(successes) / issued if issued else 0.0
        ttft_p95 = percentile([row["ttft_ms"] for row in successes]) if successes else None
        tpot_p95 = percentile([row["tpot_ms"] for row in successes]) if successes else None
        trend_ok, early_ttft, late_ttft = (late_latency_stable(
            successes, float(protocol["late_ttft_growth_ratio"]),
            float(protocol["late_ttft_growth_allowance_ms"]))
            if len(successes) >= 6 else (False, None, None))
        stable = candidate_is_stable(
            inflight_limit_hit=inflight_limit_hit,
            drain_timed_out=drain_timed_out,
            issued=issued,
            planned=planned,
            success_ratio=success_ratio,
            completion_ratio=completion_ratio,
            ttft_p95_ms=ttft_p95,
            tpot_p95_ms=tpot_p95,
            trend_ok=trend_ok,
            protocol=protocol,
            slo=slo,
        )
        result = {
            "candidate_id": label,
            "phase": phase,
            "workload_id": workload_id,
            "offered_rate_rps": rate,
            "planned_requests": planned,
            "issued_requests": issued,
            "successful_requests": len(successes),
            "failed_requests": failures,
            "success_ratio": success_ratio,
            "window_start_s": start_wall,
            "arrival_stop_s": arrival_stop_wall,
            "window_end_s": end_wall,
            "window_duration_s": duration,
            "backlog_at_arrival_stop": backlog_at_arrival_stop,
            "inflight_limit_hit": inflight_limit_hit,
            "drain_timed_out": drain_timed_out,
            "achieved_throughput_rps": achieved,
            "achieved_to_offered_ratio": completion_ratio,
            "ttft_p95_ms": ttft_p95,
            "tpot_p95_ms": tpot_p95,
            "early_ttft_median_ms": early_ttft,
            "late_ttft_median_ms": late_ttft,
            "late_latency_stable": trend_ok,
            "energy_j": energy["energy_j"],
            "mean_power_w": energy["mean_power_w"],
            "joules_per_completed_request": (
                energy["energy_j"] / len(successes) if successes else None),
            "stable": stable,
            "stability_gates": {
                "ttft_p95_lt_ms": float(slo["ttft_ms"]),
                "tpot_p95_le_ms": float(slo["tpot_ms"]),
                "minimum_success_ratio": float(protocol["minimum_success_ratio"]),
                "minimum_achieved_to_offered_ratio": float(
                    protocol["minimum_achieved_to_offered_ratio"]),
                "late_ttft_growth_ratio": float(protocol["late_ttft_growth_ratio"]),
                "late_ttft_growth_allowance_ms": float(
                    protocol["late_ttft_growth_allowance_ms"]),
            },
        }
        candidate_rows.append(result)
        append(root / "capacity_candidates.jsonl", result)
        await asyncio.sleep(float(protocol["cooldown_s"]))
        return result

    async def characterize(body: dict, workload_id: str) -> dict:
        await warmup(body, workload_id, int(protocol["warmup_requests_per_workload"]))
        initial = float(protocol["initial_rate_rps"])
        minimum = float(protocol["minimum_rate_rps"])
        maximum = float(protocol["maximum_rate_rps"])
        growth = float(protocol["bracket_growth_factor"])
        tolerance = float(protocol["relative_boundary_tolerance"])
        max_refinements = int(protocol["maximum_binary_refinements"])
        tested: list[dict] = []

        async def test(rate: float, phase: str) -> dict:
            label = f"{workload_id}-{phase}-{len(tested)+1:02d}-rps-{rate:.6f}"
            row = await candidate(body, workload_id, rate, label, phase)
            tested.append(row)
            return row

        first = await test(initial, "bracket")
        low = first["offered_rate_rps"] if first["stable"] else None
        high = None if first["stable"] else first["offered_rate_rps"]
        if first["stable"]:
            rate = initial
            while rate < maximum:
                next_rate = min(maximum, rate * growth)
                row = await test(next_rate, "bracket")
                if row["stable"]:
                    low = next_rate
                    rate = next_rate
                    if next_rate == maximum:
                        break
                else:
                    high = next_rate
                    break
        else:
            rate = initial
            while rate > minimum:
                next_rate = max(minimum, rate / growth)
                row = await test(next_rate, "bracket")
                if row["stable"]:
                    low = next_rate
                    break
                high = next_rate
                rate = next_rate
                if next_rate == minimum:
                    break

        if low is not None and high is not None:
            for _ in range(max_refinements):
                if (high - low) / low <= tolerance:
                    break
                midpoint = (low + high) / 2.0
                row = await test(midpoint, "refine")
                if row["stable"]:
                    low = midpoint
                else:
                    high = midpoint

        confirmed_rate = None
        confirmations: list[dict] = []
        if low is not None:
            candidates = sorted({row["offered_rate_rps"] for row in tested if row["stable"]},
                                reverse=True)
            for proposed in candidates[:int(protocol["maximum_confirmation_fallbacks"])]:
                confirmations = []
                for repeat in range(1, int(protocol["confirmation_windows"]) + 1):
                    label = f"{workload_id}-confirm-rps-{proposed:.6f}-repeat-{repeat}"
                    confirmations.append(await candidate(
                        body, workload_id, proposed, label, "confirmation"))
                if all(row["stable"] for row in confirmations):
                    confirmed_rate = proposed
                    break

        result = {
            "workload_id": workload_id,
            "maximum_stable_throughput_rps": confirmed_rate,
            "search_safe_lower_bound_rps": low,
            "search_unstable_upper_bound_rps": high,
            "capacity_is_lower_bound_only": low == maximum and high is None,
            "confirmed": confirmed_rate is not None,
            "search_candidate_count": len(tested),
            "confirmation_window_count": len(confirmations),
            "tested_rates_rps": [row["offered_rate_rps"] for row in tested],
            "high_frequency_mhz": {"P": runtime.prefill_grid[-1],
                                   "D": runtime.decode_grid[-1]},
        }
        append(root / "capacity_by_workload.jsonl", result)
        return result

    try:
        await meter.ready()
        runtime = PhysicalFeedbackRuntime(config, service_core)
        high = (runtime.prefill_grid[-1], runtime.decode_grid[-1])
        if high != (2520, 1500):
            raise RuntimeError(f"unexpected discovered safe-high clocks: {high}")
        await runtime.actuate("experiment", *high)
        await runtime.actuate("service", *high)
        await asyncio.sleep(1.0)
        write(root / "hardware_grids.json", {"P": runtime.prefill_grid,
                                             "D": runtime.decode_grid,
                                             "selected_high": high})
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], local_files_only=True)
        prompts = build_prompt_cache(tokenizer, {int(w["input_len"]) for w in config["workloads"]})
        bodies = {w["id"]: {
            "model": config["model"],
            "prompt": prompts[w["input_len"]],
            "max_tokens": w["output_len"],
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "xpyd_input_len": w["input_len"],
            "xpyd_output_len": w["output_len"],
            "xpyd_workload_id": w["id"],
        } for w in config["workloads"]}
        results = []
        deadline = time.monotonic() + float(protocol["experiment_timeout_s"])
        for workload_id, body in bodies.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("throughput characterization exceeded internal deadline")
            results.append(await characterize(body, workload_id))
        valid = all(row["confirmed"] for row in results)
        summary = {
            "valid": valid,
            "design": "open_loop_per_workload_capacity_search",
            "frequency_mhz": {"P": high[0], "D": high[1]},
            "service_pair": list(pair),
            "slo": slo,
            "stability_definition": {
                "success_ratio": protocol["minimum_success_ratio"],
                "achieved_to_offered_ratio": protocol["minimum_achieved_to_offered_ratio"],
                "late_ttft_growth_ratio": protocol["late_ttft_growth_ratio"],
                "late_ttft_growth_allowance_ms": protocol["late_ttft_growth_allowance_ms"],
            },
            "results": results,
            "candidate_windows": len(candidate_rows),
            "request_attempts": len(request_rows),
            "interpretation": "maximum confirmed SLO-safe open-loop offered rate on one P1-D1 pair at safe-high clocks; no historical capacity or Table prior",
        }
        write(root / "summary.json", summary)
        write(root / "audit.json", {
            "valid": valid,
            "all_seven_workloads_present": len(results) == 7,
            "all_workloads_confirmed": valid,
            "open_loop_arrivals": True,
            "concurrency_greater_than_one_allowed": True,
            "historical_prior_used": False,
            "fixed_high_frequency_verified": high == (2520, 1500),
        })
        export_csv(root / "capacity_candidates.csv", candidate_rows)
        export_csv(root / "requests.csv", request_rows)
        lines = [
            "# Maximum stable throughput at safe-high clocks", "",
            "P1-D1 frequency: 2520/1500 MHz. Arrival process: open loop.", "",
            "| Workload | Confirmed max stable RPS | Safe lower bound | Unstable upper bound | Confirmed |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in results:
            value = row["maximum_stable_throughput_rps"]
            lines.append("| {workload_id} | {value} | {low} | {high} | {confirmed} |".format(
                workload_id=row["workload_id"],
                value="" if value is None else f"{value:.6f}",
                low="" if row["search_safe_lower_bound_rps"] is None else f"{row['search_safe_lower_bound_rps']:.6f}",
                high="" if row["search_unstable_upper_bound_rps"] is None else f"{row['search_unstable_upper_bound_rps']:.6f}",
                confirmed=row["confirmed"]))
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        export_power_trace(root, meter.samples)
        if not valid:
            raise RuntimeError("one or more workload capacities were not confirmed")
    finally:
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
    if (args.output / "capacity_candidates.jsonl").exists():
        raise RuntimeError("refusing to reuse an existing run directory")
    config = load_config(args.config)
    write(args.output / "experiment_config.json", config)
    try:
        asyncio.run(asyncio.wait_for(
            run(config, args.output),
            timeout=float(config["throughput_capacity"]["experiment_timeout_s"]) + 600.0))
    except BaseException as exc:
        write(args.output / "audit.json", {
            "valid": False,
            "error": type(exc).__name__ + ": " + str(exc),
        })
        raise


if __name__ == "__main__":
    main()
