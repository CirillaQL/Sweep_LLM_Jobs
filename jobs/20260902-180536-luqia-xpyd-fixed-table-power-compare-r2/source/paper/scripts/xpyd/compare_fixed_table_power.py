#!/usr/bin/env python3
"""Compare full-frequency and fixed-table Phase 3C energy windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCOPES = ("P1", "D1", "service_PD_pair")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _normalized_trace(path: Path) -> list[tuple[str, str, str, str]]:
    return [
        (row["workload_id"], row["arrival_time_s"], row["input_len"], row["output_len"])
        for row in _read_csv(path)
    ]


def _index_energy(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(path)
    indexed = {
        (row["workload_id"], row["scope"]): row
        for row in rows if row["scope"] in SCOPES
    }
    if not indexed or any(row.get("valid", "").lower() != "true" for row in indexed.values()):
        raise ValueError("invalid or empty workload energy file: %s" % path)
    return indexed


def compare(baseline_dir: Path, optimized_dir: Path, output_dir: Path) -> dict[str, Any]:
    baseline_trace = _normalized_trace(baseline_dir / "client" / "trace.csv")
    optimized_trace = _normalized_trace(optimized_dir / "client" / "trace.csv")
    if baseline_trace != optimized_trace:
        raise ValueError("baseline and optimized request traces differ")

    baseline = _index_energy(baseline_dir / "workload_energy_summary.csv")
    optimized = _index_energy(optimized_dir / "workload_energy_summary.csv")
    workload_ids = list(dict.fromkeys(row[0] for row in baseline_trace))
    expected_keys = {(workload_id, scope) for workload_id in workload_ids for scope in SCOPES}
    if set(baseline) != expected_keys or set(optimized) != expected_keys:
        raise ValueError("energy rows do not exactly cover the shared trace")

    rows = []
    for workload_id in workload_ids:
        baseline_pair = baseline[(workload_id, "service_PD_pair")]
        optimized_pair = optimized[(workload_id, "service_PD_pair")]
        baseline_energy = _float(baseline_pair, "gross_energy_j")
        optimized_energy = _float(optimized_pair, "gross_energy_j")
        rows.append({
            "workload_id": workload_id,
            "requests": int(baseline_pair["logical_requests"]),
            "baseline_prefill_mhz": int(float(baseline_pair["prefill_frequency_mhz"])),
            "baseline_decode_mhz": int(float(baseline_pair["decode_frequency_mhz"])),
            "optimized_prefill_mhz": int(float(optimized_pair["prefill_frequency_mhz"])),
            "optimized_decode_mhz": int(float(optimized_pair["decode_frequency_mhz"])),
            "baseline_p_energy_j": _float(baseline[(workload_id, "P1")], "gross_energy_j"),
            "optimized_p_energy_j": _float(optimized[(workload_id, "P1")], "gross_energy_j"),
            "baseline_d_energy_j": _float(baseline[(workload_id, "D1")], "gross_energy_j"),
            "optimized_d_energy_j": _float(optimized[(workload_id, "D1")], "gross_energy_j"),
            "baseline_total_energy_j": baseline_energy,
            "optimized_total_energy_j": optimized_energy,
            "energy_saved_j": baseline_energy - optimized_energy,
            "energy_saved_percent": (baseline_energy - optimized_energy) / baseline_energy * 100.0,
            "baseline_mean_power_w": _float(baseline_pair, "mean_power_w"),
            "optimized_mean_power_w": _float(optimized_pair, "mean_power_w"),
            "baseline_ttft_p90_ms": _float(baseline_pair, "ttft_p90_ms"),
            "optimized_ttft_p90_ms": _float(optimized_pair, "ttft_p90_ms"),
            "baseline_tpot_p90_ms": _float(baseline_pair, "tpot_p90_ms"),
            "optimized_tpot_p90_ms": _float(optimized_pair, "tpot_p90_ms"),
        })

    total_requests = sum(int(row["requests"]) for row in rows)
    baseline_energy = sum(float(row["baseline_total_energy_j"]) for row in rows)
    optimized_energy = sum(float(row["optimized_total_energy_j"]) for row in rows)
    baseline_duration = sum(
        _float(baseline[(workload_id, "service_PD_pair")], "duration_s")
        for workload_id in workload_ids
    )
    optimized_duration = sum(
        _float(optimized[(workload_id, "service_PD_pair")], "duration_s")
        for workload_id in workload_ids
    )
    aggregate = {
        "workloads": len(workload_ids),
        "requests": total_requests,
        "baseline_total_energy_j": baseline_energy,
        "optimized_total_energy_j": optimized_energy,
        "energy_saved_j": baseline_energy - optimized_energy,
        "energy_saved_percent": (baseline_energy - optimized_energy) / baseline_energy * 100.0,
        "baseline_mean_power_w": baseline_energy / baseline_duration,
        "optimized_mean_power_w": optimized_energy / optimized_duration,
        "baseline_joules_per_request": baseline_energy / total_requests,
        "optimized_joules_per_request": optimized_energy / total_requests,
    }
    result = {
        "schema_version": 1,
        "valid": True,
        "measurement_scope": "P1+D1 only",
        "measurement_window": "first route_selected through last response_completed per workload",
        "trace_identical": True,
        "both_heavy_excluded": "both_heavy" not in workload_ids,
        "aggregate": aggregate,
        "workloads": rows,
    }
    if not result["both_heavy_excluded"]:
        raise ValueError("both_heavy must be excluded from this comparison")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "power_comparison.json", result)
    with (output_dir / "power_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fixed-table versus full-frequency power comparison", "",
        "Measurement scope: P1+D1; tuning, settling, and warmup are outside each route-bounded window.", "",
        "| Workload | Baseline P/D MHz | Fixed-table P/D MHz | Baseline energy (J) | Fixed-table energy (J) | Saved |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {workload_id} | {baseline_prefill_mhz}/{baseline_decode_mhz} | "
            "{optimized_prefill_mhz}/{optimized_decode_mhz} | "
            "{baseline_total_energy_j:.3f} | {optimized_total_energy_j:.3f} | "
            "{energy_saved_percent:.2f}% |".format(**row)
        )
    lines.extend([
        "", "Aggregate: baseline %.3f J, fixed-table %.3f J, saved %.2f%% over %d requests."
        % (baseline_energy, optimized_energy, aggregate["energy_saved_percent"], total_requests), "",
    ])
    (output_dir / "power_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--optimized-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.baseline_dir, args.optimized_dir, args.output_dir)
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
