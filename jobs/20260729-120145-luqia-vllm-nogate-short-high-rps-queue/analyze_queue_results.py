#!/usr/bin/env python3
"""Build request-order latency and server-queue time series from job artifacts."""

import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def milliseconds(values):
    return [float(value) * 1000.0 for value in values]


def write_latency_series(out_dir, workload_id, rate, bin_seconds):
    matches = sorted(out_dir.glob(f"detailed_*_{workload_id}.json"))
    if not matches:
        return {"available": False, "reason": "detailed result missing"}
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    ttfts = milliseconds(payload.get("ttfts", []))
    tpots = milliseconds(payload.get("tpots", []))
    e2els = milliseconds(payload.get("e2els", []))
    count = max(len(ttfts), len(tpots), len(e2els))
    if count == 0:
        return {
            "available": False,
            "reason": "detailed result has no ttfts/tpots/e2els",
            "keys": sorted(payload),
        }

    requests_per_bin = max(1, int(round(rate * bin_seconds)))
    output = out_dir / "latency_by_request_order.csv"
    fields = [
        "bin",
        "request_index_start",
        "request_index_end",
        "nominal_send_start_s",
        "nominal_send_end_s",
        "count",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
    ]
    rows = []
    for start in range(0, count, requests_per_bin):
        end = min(count, start + requests_per_bin)
        ttft_chunk = ttfts[start:min(end, len(ttfts))]
        tpot_chunk = tpots[start:min(end, len(tpots))]
        e2e_chunk = e2els[start:min(end, len(e2els))]
        rows.append({
            "bin": len(rows) + 1,
            "request_index_start": start + 1,
            "request_index_end": end,
            "nominal_send_start_s": round(start / rate, 3),
            "nominal_send_end_s": round(end / rate, 3),
            "count": end - start,
            "ttft_p50_ms": percentile(ttft_chunk, 0.50),
            "ttft_p95_ms": percentile(ttft_chunk, 0.95),
            "ttft_p99_ms": percentile(ttft_chunk, 0.99),
            "tpot_p50_ms": percentile(tpot_chunk, 0.50),
            "tpot_p95_ms": percentile(tpot_chunk, 0.95),
            "tpot_p99_ms": percentile(tpot_chunk, 0.99),
            "e2e_p50_ms": percentile(e2e_chunk, 0.50),
            "e2e_p95_ms": percentile(e2e_chunk, 0.95),
            "e2e_p99_ms": percentile(e2e_chunk, 0.99),
        })
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "available": True,
        "source": str(matches[0]),
        "output": str(output),
        "request_count": count,
        "requests_per_bin": requests_per_bin,
    }


def workload_interval(out_dir, workload_id):
    events = out_dir / "events.csv"
    if not events.exists():
        return None
    start = None
    end = None
    with events.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("workload_id") != workload_id:
                continue
            if row.get("event") == "workload_start":
                start = float(row["unix_ts"])
            elif row.get("event") == "workload_end":
                end = float(row["unix_ts"])
    if start is None or end is None:
        return None
    return start, end


def write_queue_series(out_dir, workload_id, bin_seconds):
    source = out_dir / "vllm_queue_metrics.csv"
    interval = workload_interval(out_dir, workload_id)
    if not source.exists() or interval is None:
        return {"available": False, "reason": "metrics or workload interval missing"}
    start, end = interval
    buckets = {}
    with source.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scrape_ok") != "True":
                continue
            timestamp = float(row["unix_ts"])
            if timestamp < start or timestamp > end:
                continue
            bucket = int((timestamp - start) // bin_seconds)
            key = (bucket, row["component"])
            item = buckets.setdefault(key, {
                "running": [],
                "waiting": [],
                "gpu_cache": [],
            })
            for column, target in (
                ("num_running", "running"),
                ("num_waiting", "waiting"),
                ("gpu_cache_usage", "gpu_cache"),
            ):
                if row.get(column) not in ("", None):
                    item[target].append(float(row[column]))

    output = out_dir / "queue_by_time.csv"
    fields = [
        "time_start_s",
        "time_end_s",
        "component",
        "samples",
        "running_avg",
        "running_max",
        "waiting_avg",
        "waiting_max",
        "gpu_cache_usage_avg",
        "gpu_cache_usage_max",
    ]
    rows = []
    for (bucket, component), values in sorted(buckets.items()):
        def average(name):
            data = values[name]
            return sum(data) / len(data) if data else None

        def maximum(name):
            data = values[name]
            return max(data) if data else None

        rows.append({
            "time_start_s": bucket * bin_seconds,
            "time_end_s": (bucket + 1) * bin_seconds,
            "component": component,
            "samples": max(len(values["running"]), len(values["waiting"])),
            "running_avg": average("running"),
            "running_max": maximum("running"),
            "waiting_avg": average("waiting"),
            "waiting_max": maximum("waiting"),
            "gpu_cache_usage_avg": average("gpu_cache"),
            "gpu_cache_usage_max": maximum("gpu_cache"),
        })
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "available": bool(rows),
        "source": str(source),
        "output": str(output),
        "workload_start_unix": start,
        "workload_end_unix": end,
        "row_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workload-id", default="short_high_rps_queue")
    parser.add_argument("--rate", type=float, default=32.0)
    parser.add_argument("--bin-seconds", type=float, default=5.0)
    args = parser.parse_args()
    summary = {
        "workload_id": args.workload_id,
        "configured_rate_rps": args.rate,
        "latency": write_latency_series(
            args.out_dir, args.workload_id, args.rate, args.bin_seconds
        ),
        "queue": write_queue_series(
            args.out_dir, args.workload_id, args.bin_seconds
        ),
    }
    destination = args.out_dir / "queue_backlog_analysis.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
