#!/usr/bin/env python3
"""Join each proxy arrival to the most recent Prefill and Decode queue sample."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from pathlib import Path


QUEUE_FIELDS = [
    "proxy_arrival_unix_ns",
    "client_to_proxy_ms",
    "prefill_metric_unix_ns",
    "prefill_metric_age_ms",
    "prefill_running",
    "prefill_running_source",
    "prefill_waiting",
    "prefill_waiting_source",
    "prefill_waiting_growth_per_s",
    "prefill_kv_cache_usage",
    "prefill_proxy_inflight",
    "decode_proxy_inflight",
    "decode_metric_unix_ns",
    "decode_metric_age_ms",
    "decode_running",
    "decode_waiting",
    "decode_waiting_growth_per_s",
    "decode_kv_cache_usage",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_before(
    rows: list[dict[str, str]], timestamps: list[int], target_ns: int
) -> dict[str, str] | None:
    index = bisect_right(timestamps, target_ns) - 1
    return rows[index] if index >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--proxy-events", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = read_csv(args.requests)
    proxy_events = read_csv(args.proxy_events)
    metrics = read_csv(args.metrics)

    arrivals: dict[str, dict[str, str]] = {}
    for row in proxy_events:
        if row["event"] != "proxy_arrival":
            continue
        request_id = row["client_request_id"]
        arrivals.setdefault(request_id, row)

    by_role: dict[str, list[dict[str, str]]] = {"prefill": [], "decode": []}
    for row in metrics:
        if row["role"] in by_role and row.get("scrape_ok") == "True":
            by_role[row["role"]].append(row)
    for role in by_role:
        by_role[role].sort(key=lambda row: int(row["unix_ns"]))
    timestamps = {
        role: [int(row["unix_ns"]) for row in rows]
        for role, rows in by_role.items()
    }

    output_rows = []
    for request in requests:
        request_id = request["request_id"]
        client_send_ns = int(request["actual_send_unix_ns"])
        arrival = arrivals.get(request_id)
        arrival_ns = (
            int(arrival["unix_ns"]) if arrival is not None else client_send_ns
        )
        output = dict(request)
        output["proxy_arrival_unix_ns"] = arrival_ns
        output["client_to_proxy_ms"] = (arrival_ns - client_send_ns) / 1e6
        output["prefill_proxy_inflight"] = (
            arrival.get("prefill_inflight", "") if arrival else ""
        )
        output["decode_proxy_inflight"] = (
            arrival.get("decode_inflight", "") if arrival else ""
        )
        for role in ("prefill", "decode"):
            sample = latest_before(
                by_role[role], timestamps[role], arrival_ns
            )
            if sample is None:
                for suffix in (
                    "metric_unix_ns",
                    "metric_age_ms",
                    "running",
                    "waiting",
                    "waiting_growth_per_s",
                    "kv_cache_usage",
                ):
                    output[f"{role}_{suffix}"] = ""
                continue
            sample_ns = int(sample["unix_ns"])
            output[f"{role}_metric_unix_ns"] = sample_ns
            output[f"{role}_metric_age_ms"] = (arrival_ns - sample_ns) / 1e6
            output[f"{role}_running"] = sample["running"]
            output[f"{role}_waiting"] = sample["waiting"]
            output[f"{role}_waiting_growth_per_s"] = sample[
                "waiting_growth_per_s"
            ]
            output[f"{role}_kv_cache_usage"] = sample["kv_cache_usage"]
        output["prefill_running_source"] = "vllm_metrics"
        output["prefill_waiting_source"] = "vllm_metrics"
        proxy_prefill = output.get("prefill_proxy_inflight", "")
        if proxy_prefill != "":
            metric_prefill = float(output.get("prefill_running") or 0)
            proxy_prefill_value = float(proxy_prefill)
            if proxy_prefill_value > metric_prefill:
                output["prefill_running"] = proxy_prefill
                output["prefill_running_source"] = "proxy_inflight_event"
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(requests[0]) + QUEUE_FIELDS if requests else QUEUE_FIELDS
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"queue_state_rows={len(output_rows)} output={args.output}")


if __name__ == "__main__":
    main()
