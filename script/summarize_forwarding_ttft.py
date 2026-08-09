#!/usr/bin/env python3
"""Summarize exact post-DVFS forwarding TTFT from proxy request records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p90_ms": round(percentile(values, 0.90), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--slo-ttft-ms", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    if not args.decisions.is_file():
        records = []
        errors.append(f"missing decisions file: {args.decisions}")
    else:
        records = load_records(args.decisions)
    if len(records) != args.expected_requests:
        errors.append(
            f"decision count {len(records)} != expected {args.expected_requests}"
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        actual = record.get("actual") or {}
        prediction = record.get("prediction") or {}
        recommended = prediction.get("recommended") or {}
        prefill_prediction = recommended.get("prefill") or {}
        forwarding = actual.get("proxy_forwarding_ttft_ms")
        if actual.get("success") is not True or forwarding is None:
            errors.append(
                f"request {record.get('request_index')} has no successful forwarding TTFT"
            )
            continue
        forwarding = float(forwarding)
        predicted = prefill_prediction.get("p99_ttft_ms")
        raw = actual.get("proxy_ttft_raw_ms")
        wait_excluded = actual.get("proxy_ttft_ms")
        row = {
            "request_index": record.get("request_index"),
            "request_id": record.get("request_id"),
            "route": record.get("route"),
            "prefill_tp": record.get("prefill_tp"),
            "decode_tp": record.get("decode_tp"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens_requested"),
            "prefill_mhz": (record.get("clock") or {}).get("prefill_target_mhz"),
            "decode_mhz": (record.get("clock") or {}).get("decode_target_mhz"),
            "predicted_p99_ttft_ms": predicted,
            "forwarding_ttft_ms": forwarding,
            "forwarding_ttft_slo_met": forwarding <= args.slo_ttft_ms,
            "wait_excluded_ttft_ms": wait_excluded,
            "raw_ttft_ms": raw,
            "pre_forward_overhead_ms": (
                round(float(wait_excluded) - forwarding, 3)
                if wait_excluded is not None else None
            ),
            "prefill_http_ms": actual.get("prefill_http_ms"),
            "prefill_to_decode_dispatch_ms": actual.get(
                "prefill_to_decode_dispatch_ms"
            ),
            "decode_to_first_chunk_ms": actual.get("decode_to_first_chunk_ms"),
            "actual_minus_predicted_ms": (
                round(forwarding - float(predicted), 3)
                if predicted is not None else None
            ),
        }
        rows.append(row)

    forwarding_values = [row["forwarding_ttft_ms"] for row in rows]
    wait_excluded_values = [
        float(row["wait_excluded_ttft_ms"])
        for row in rows
        if row["wait_excluded_ttft_ms"] is not None
    ]
    raw_values = [
        float(row["raw_ttft_ms"]) for row in rows if row["raw_ttft_ms"] is not None
    ]
    predicted_values = [
        float(row["predicted_p99_ttft_ms"])
        for row in rows
        if row["predicted_p99_ttft_ms"] is not None
    ]

    route_groups: dict[str, list[float]] = defaultdict(list)
    topology_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        route_groups[str(row["route"])].append(row["forwarding_ttft_ms"])
        topology_groups[
            f"P_TP{row['prefill_tp']}-D_TP{row['decode_tp']}"
        ].append(row["forwarding_ttft_ms"])

    result = {
        "ok": not errors,
        "timer_definition": "after both clock ACKs to first Decode response chunk",
        "subtraction_used": False,
        "expected_requests": args.expected_requests,
        "record_count": len(records),
        "successful_forwarding_records": len(rows),
        "slo_ttft_ms": args.slo_ttft_ms,
        "forwarding_ttft_slo_met": sum(
            row["forwarding_ttft_slo_met"] for row in rows
        ),
        "forwarding_ttft": metrics(forwarding_values),
        "wait_excluded_ttft_fixed_subtraction": metrics(wait_excluded_values),
        "raw_client_visible_ttft": metrics(raw_values),
        "predicted_p99_ttft": metrics(predicted_values),
        "actual_forwarding_exceeds_prediction": sum(
            row["actual_minus_predicted_ms"] is not None
            and row["actual_minus_predicted_ms"] > 0
            for row in rows
        ),
        "by_route": {
            key: metrics(values) for key, values in sorted(route_groups.items())
        },
        "by_topology": {
            key: metrics(values) for key, values in sorted(topology_groups.items())
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["request_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
