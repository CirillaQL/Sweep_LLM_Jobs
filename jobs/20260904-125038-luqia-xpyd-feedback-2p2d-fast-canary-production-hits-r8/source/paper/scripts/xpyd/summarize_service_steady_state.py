#!/usr/bin/env python3
"""Join P1/D1 dispatch metadata with measured latency and summarize windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": stdev,
        "cv": stdev / mean if mean else math.nan,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    dispatch_path: Path,
    requests_path: Path,
    output_dir: Path,
    discard_first: int,
    minimum_samples: int,
    max_cv: float,
) -> dict[str, Any]:
    dispatches = load_jsonl(dispatch_path)
    measured = load_csv(requests_path)
    by_id: dict[str, dict[str, str]] = {}
    for row in measured:
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in by_id:
            raise ValueError(f"invalid or duplicate measured request ID: {request_id!r}")
        by_id[request_id] = row
    if len(dispatches) != len(measured):
        raise ValueError(
            f"dispatch/measurement count mismatch: {len(dispatches)} != {len(measured)}"
        )

    joined: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    window_id = -1
    index_in_window = 0
    for dispatch in sorted(dispatches, key=lambda row: int(row["service_sequence"])):
        request_id = str(dispatch["request_id"])
        request = by_id.pop(request_id, None)
        if request is None:
            raise ValueError(f"dispatch has no measured request: {request_id}")
        if (
            request.get("prefill_endpoint_id") != "P1"
            or request.get("decode_endpoint_id") != "D1"
        ):
            raise ValueError(f"formal request did not use P1/D1: {request_id}")
        key = (
            dispatch["workload_id"],
            int(dispatch["prefill_frequency_mhz"]),
            int(dispatch["decode_frequency_mhz"]),
            bool(dispatch["table_hit"]),
            int(dispatch["table_revision"]),
        )
        if key != current_key:
            current_key = key
            window_id += 1
            index_in_window = 0
        index_in_window += 1
        included = index_in_window > discard_first
        joined.append({
            "request_id": request_id,
            "service_sequence": int(dispatch["service_sequence"]),
            "workload_id": dispatch["workload_id"],
            "configuration_window_id": window_id,
            "sample_index_in_window": index_in_window,
            "steady_state_included": included,
            "steady_exclusion_reason": "" if included else "post_frequency_change_transient",
            "table_hit": bool(dispatch["table_hit"]),
            "table_revision": int(dispatch["table_revision"]),
            "frequency_source": dispatch["frequency_source"],
            "prefill_frequency_mhz": int(dispatch["prefill_frequency_mhz"]),
            "decode_frequency_mhz": int(dispatch["decode_frequency_mhz"]),
            "frequency_changed": bool(dispatch["frequency_changed"]),
            "settle_wait_s": float(dispatch["settle_wait_s"]),
            "input_len": int(request["input_len"]),
            "output_len": int(request["requested_output_len"]),
            "ttft_ms": float(request["ttft_ms"]),
            "tpot_ms": float(request["tpot_ms"]),
            "client_observed_ttft_ms": float(request["client_observed_ttft_ms"]),
            "client_observed_tpot_ms": float(request["client_observed_tpot_ms"]),
        })
    if by_id:
        raise ValueError(f"measured requests have no dispatch records: {sorted(by_id)[:5]}")

    windows: dict[int, list[dict[str, Any]]] = {}
    for row in joined:
        windows.setdefault(int(row["configuration_window_id"]), []).append(row)
    summaries = []
    for identifier, rows in windows.items():
        selected = [row for row in rows if row["steady_state_included"]]
        base = rows[0]
        enough = len(selected) >= minimum_samples
        ttft = metric_summary([row["ttft_ms"] for row in selected]) if selected else None
        tpot = metric_summary([row["tpot_ms"] for row in selected]) if selected else None
        stable = bool(
            enough and ttft and tpot
            and ttft["cv"] <= max_cv and tpot["cv"] <= max_cv
        )
        summaries.append({
            "configuration_window_id": identifier,
            "workload_id": base["workload_id"],
            "table_hit": base["table_hit"],
            "table_revision": base["table_revision"],
            "frequency_source": base["frequency_source"],
            "prefill_frequency_mhz": base["prefill_frequency_mhz"],
            "decode_frequency_mhz": base["decode_frequency_mhz"],
            "total_samples": len(rows),
            "discarded_samples": len(rows) - len(selected),
            "steady_samples": len(selected),
            "minimum_samples": minimum_samples,
            "max_cv": max_cv,
            "stable": stable,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "service_requests_joined.csv", joined)
    write_csv(
        output_dir / "service_requests_steady.csv",
        [row for row in joined if row["steady_state_included"]],
    )
    flat = []
    for row in summaries:
        flat.append({
            key: value for key, value in row.items()
            if key not in {"ttft_ms", "tpot_ms"}
        } | {
            f"ttft_ms_{key}": value for key, value in (row["ttft_ms"] or {}).items()
        } | {
            f"tpot_ms_{key}": value for key, value in (row["tpot_ms"] or {}).items()
        })
    write_csv(output_dir / "steady_state_summary.csv", flat)
    selected_by_workload: dict[str, dict[str, Any]] = {}
    for row in summaries:
        if not row["stable"]:
            continue
        workload_id = str(row["workload_id"])
        current = selected_by_workload.get(workload_id)
        if current is None or (
            bool(row["table_hit"]), int(row["configuration_window_id"])
        ) > (
            bool(current["table_hit"]),
            int(current["configuration_window_id"]),
        ):
            selected_by_workload[workload_id] = row
    selected = [selected_by_workload[key] for key in sorted(selected_by_workload)]
    selected_flat = [
        next(item for item in flat if item["configuration_window_id"] == row["configuration_window_id"])
        for row in selected
    ]
    write_csv(output_dir / "selected_steady_state_summary.csv", selected_flat)
    result = {
        "schema_version": 1,
        "dispatch_count": len(dispatches),
        "measured_request_count": len(measured),
        "discard_first_per_configuration_window": discard_first,
        "minimum_steady_samples": minimum_samples,
        "maximum_cv": max_cv,
        "stable_window_count": sum(bool(row["stable"]) for row in summaries),
        "stable_workload_count": len(selected),
        "window_count": len(summaries),
        "windows": summaries,
        "selected_steady_state_by_workload": selected,
    }
    (output_dir / "steady_state_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-jsonl", type=Path, required=True)
    parser.add_argument("--requests-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--discard-first", type=int, default=3)
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--max-cv", type=float, default=0.10)
    args = parser.parse_args()
    if args.discard_first < 0 or args.minimum_samples < 1 or args.max_cv <= 0:
        parser.error("invalid steady-state threshold")
    result = summarize(
        args.dispatch_jsonl, args.requests_csv, args.output_dir,
        args.discard_first, args.minimum_samples, args.max_cv,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
