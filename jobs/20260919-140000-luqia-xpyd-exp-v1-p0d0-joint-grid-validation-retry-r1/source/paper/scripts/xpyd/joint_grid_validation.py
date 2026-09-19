"""Measure every P0/D0 frequency pair and test the Canary factorization."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Mapping

from xpyd.disagg_proxy import _build_multi_core
from xpyd.online_feedback_controller import (
    PhysicalFeedbackRuntime,
    pd_inference_metrics,
    strip_feedback_metadata,
)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate percentile from no values")
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot calculate mean from no values")
    return statistics.fmean(values)


def _pair(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["prefill_frequency_mhz"]), int(row["decode_frequency_mhz"])


async def _probe(
    runtime: PhysicalFeedbackRuntime,
    core: Any,
    body: Mapping[str, Any],
    workload_id: str,
    p_mhz: int,
    d_mhz: int,
    request_id: str,
) -> dict[str, float]:
    """One active-only request boundary with independent P and D counters."""
    changed = await runtime.actuate("experiment", p_mhz, d_mhz)
    settle_s = float(runtime.config["joint_grid_validation"]["frequency_settle_s"])
    if changed and settle_s:
        await asyncio.sleep(settle_s)
    before_p, before_d = await asyncio.gather(
        asyncio.to_thread(runtime._energy_mj, "P0"),
        asyncio.to_thread(runtime._energy_mj, "D0"),
    )
    started = time.monotonic()
    prepared = await core.prepare(
        strip_feedback_metadata(dict(body) | {"stream": True}), request_id,
    )
    if prepared.stream is None:
        raise RuntimeError("joint-grid clone did not return a streaming response")
    async for _ in prepared.stream:
        pass
    duration_s = time.monotonic() - started
    after_p, after_d = await asyncio.gather(
        asyncio.to_thread(runtime._energy_mj, "P0"),
        asyncio.to_thread(runtime._energy_mj, "D0"),
    )
    ttft_ms, tpot_ms = pd_inference_metrics(
        prepared.diagnostics.timestamps_monotonic_s,
        int(body["xpyd_output_len"]),
    )
    if ttft_ms is None or tpot_ms is None:
        raise RuntimeError("request did not yield TTFT/TPOT evidence")
    p_energy_j = (after_p - before_p) / 1000.0
    d_energy_j = (after_d - before_d) / 1000.0
    return {
        "prefill_energy_j": p_energy_j,
        "decode_energy_j": d_energy_j,
        "total_energy_j": p_energy_j + d_energy_j,
        "duration_s": duration_s,
        "ttft_ms": float(ttft_ms),
        "tpot_ms": float(tpot_ms),
    }


def _candidate_rows(
    raw_rows: list[dict[str, Any]], slo: Mapping[str, float], samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["status"] == "ok":
            grouped.setdefault((row["workload_id"], *_pair(row)), []).append(row)
    candidates = []
    for (workload_id, p_mhz, d_mhz), rows in grouped.items():
        complete = len(rows) == samples
        ttft_p95 = _p95(row["ttft_ms"] for row in rows)
        tpot_p95 = _p95(row["tpot_ms"] for row in rows)
        candidates.append({
            "workload_id": workload_id,
            "prefill_frequency_mhz": p_mhz,
            "decode_frequency_mhz": d_mhz,
            "sample_count": len(rows),
            "complete": complete,
            "slo_met": complete and ttft_p95 <= float(slo["ttft_ms"]) and tpot_p95 <= float(slo["tpot_ms"]),
            "mean_prefill_energy_j": _mean(row["prefill_energy_j"] for row in rows),
            "mean_decode_energy_j": _mean(row["decode_energy_j"] for row in rows),
            "mean_total_energy_j": _mean(row["total_energy_j"] for row in rows),
            "std_total_energy_j": statistics.stdev([row["total_energy_j"] for row in rows]) if len(rows) > 1 else 0.0,
            "ttft_p95_ms": ttft_p95,
            "tpot_p95_ms": tpot_p95,
        })
    return sorted(candidates, key=lambda row: (row["workload_id"], *_pair(row)))


def _best(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["slo_met"] is True]
    return min(candidates, key=lambda row: (float(row[key]), *_pair(row))) if candidates else None


def _strategy_comparison(
    workload_id: str,
    candidates: list[dict[str, Any]],
    d_high: int,
) -> dict[str, Any]:
    rows = [row for row in candidates if row["workload_id"] == workload_id]
    global_best = _best(rows, "mean_total_energy_j")
    p_axis = _best((row for row in rows if int(row["decode_frequency_mhz"]) == d_high), "mean_total_energy_j")
    coordinate = _best(
        (row for row in rows if p_axis is not None and int(row["prefill_frequency_mhz"]) == int(p_axis["prefill_frequency_mhz"])),
        "mean_total_energy_j",
    )
    p_component = _best(rows, "mean_prefill_energy_j")
    d_component = _best(rows, "mean_decode_energy_j")
    component_pair = None
    if p_component and d_component:
        component_pair = next(
            row for row in rows
            if _pair(row) == (int(p_component["prefill_frequency_mhz"]), int(d_component["decode_frequency_mhz"]))
        )
    comparison: dict[str, Any] = {
        "workload_id": workload_id,
        "global_joint_energy_best": global_best,
        "canary_coordinate_reconstruction": coordinate,
        "component_additive_reconstruction": component_pair,
        "coordinate_p_axis_at_D_high": p_axis,
        "component_prefill_argmin": p_component,
        "component_decode_argmin": d_component,
    }
    for name, predicted in (
        ("coordinate", coordinate),
        ("component_additive", component_pair),
    ):
        comparison[name + "_matches_global"] = bool(
            predicted and global_best and _pair(predicted) == _pair(global_best)
        )
        comparison[name + "_energy_gap_j"] = (
            float(predicted["mean_total_energy_j"]) - float(global_best["mean_total_energy_j"])
            if predicted and global_best else None
        )
        comparison[name + "_energy_gap_pct"] = (
            100.0 * comparison[name + "_energy_gap_j"] / float(global_best["mean_total_energy_j"])
            if comparison[name + "_energy_gap_j"] is not None and float(global_best["mean_total_energy_j"]) else None
        )
    return comparison


async def run(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import aiohttp
    from replay_synthetic_trace import build_prompt_cache
    from transformers import AutoTokenizer

    settings = config["joint_grid_validation"]
    samples = int(settings["samples_per_pair"])
    slo = settings["slo"]
    output.mkdir(parents=True, exist_ok=True)
    core = _build_multi_core(config, lambda: aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900.0)))
    experiment_core = core.cores[("P0", "D0")]
    runtime = PhysicalFeedbackRuntime(config, experiment_core)
    p_grid, d_grid = runtime.prefill_grid, runtime.decode_grid
    if len(p_grid) != 17 or len(d_grid) != 15:
        raise RuntimeError("supported frequency grids must be exactly 17 P and 15 D levels")

    tokenizer = AutoTokenizer.from_pretrained(str(config["tokenizer_model"]))
    prompts = build_prompt_cache(tokenizer, {int(row["input_len"]) for row in config["workloads"]})
    raw_rows: list[dict[str, Any]] = []
    pairs = [(int(p_mhz), int(d_mhz)) for p_mhz in p_grid for d_mhz in d_grid]
    for workload_index, workload in enumerate(config["workloads"]):
        workload_id = str(workload["id"])
        body = {
            "model": str(config["model"]), "prompt": prompts[int(workload["input_len"])],
            "max_tokens": int(workload["output_len"]), "temperature": 0.0,
            "top_p": 1.0, "ignore_eos": True, "stream": True,
            "xpyd_input_len": int(workload["input_len"]),
            "xpyd_output_len": int(workload["output_len"]), "xpyd_workload_id": workload_id,
        }
        for repeat in range(samples):
            pass_pairs = list(pairs)
            random.Random(int(settings["random_seed"]) + workload_index * 100 + repeat).shuffle(pass_pairs)
            for ordinal, (p_mhz, d_mhz) in enumerate(pass_pairs):
                row: dict[str, Any] = {
                    "timestamp": time.time(), "workload_id": workload_id, "repeat": repeat + 1,
                    "sequence_in_repeat": ordinal + 1, "prefill_frequency_mhz": p_mhz,
                    "decode_frequency_mhz": d_mhz, "status": "ok", "error": None,
                }
                try:
                    row.update(await _probe(
                        runtime, experiment_core, body, workload_id, p_mhz, d_mhz,
                        "%s-p%d-d%d-r%d" % (workload_id, p_mhz, d_mhz, repeat + 1),
                    ))
                except Exception as exc:  # Retain failed candidates; do not abort the remaining grid.
                    row.update({"status": "error", "error": "%s: %s" % (type(exc).__name__, exc)})
                raw_rows.append(row)

    candidates = _candidate_rows(raw_rows, slo, samples)
    comparisons = [_strategy_comparison(str(row["id"]), candidates, int(d_grid[-1])) for row in config["workloads"]]
    expected_raw = len(config["workloads"]) * len(pairs) * samples
    expected_candidates = len(config["workloads"]) * len(pairs)
    valid = len(raw_rows) == expected_raw and len(candidates) == expected_candidates and all(
        row["sample_count"] == samples for row in candidates
    ) and all(
        item["global_joint_energy_best"] is not None
        and item["canary_coordinate_reconstruction"] is not None
        for item in comparisons
    )
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "design": settings["design"],
        "endpoint_pair": ["P0", "D0"],
        "workloads": [str(row["id"]) for row in config["workloads"]],
        "prefill_grid_mhz": list(p_grid), "decode_grid_mhz": list(d_grid),
        "pairs_per_workload": len(pairs), "samples_per_pair": samples,
        "expected_raw_measurements": expected_raw, "recorded_raw_measurements": len(raw_rows),
        "expected_candidate_rows": expected_candidates, "recorded_candidate_rows": len(candidates),
        "error_measurements": sum(row["status"] != "ok" for row in raw_rows),
        "slo": slo, "comparisons": comparisons,
    }
    raw_fields = ["timestamp", "workload_id", "repeat", "sequence_in_repeat", "prefill_frequency_mhz", "decode_frequency_mhz", "status", "error", "prefill_energy_j", "decode_energy_j", "total_energy_j", "duration_s", "ttft_ms", "tpot_ms"]
    with (output / "joint_grid_request_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(raw_rows)
    candidate_fields = list(candidates[0]) if candidates else ["workload_id"]
    with (output / "joint_grid_candidate_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields); writer.writeheader(); writer.writerows(candidates)
    _write_json(output / "joint_grid_strategy_validation.json", summary)
    _write_json(output / "audit.json", {"valid": valid, "endpoint_pair": ["P0", "D0"], "raw_measurements": len(raw_rows), "candidate_rows": len(candidates), "failed_measurements": summary["error_measurements"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = _expand(json.loads(args.config.read_text(encoding="utf-8")))
    summary = asyncio.run(run(config, args.output))
    print(json.dumps({"valid": summary["valid"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
