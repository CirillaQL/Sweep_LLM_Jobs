#!/usr/bin/env python3
"""Scrape both vLLM /metrics endpoints into an analysis-ready wide CSV."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import aiohttp


METRIC_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')

LONG_FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "role",
    "endpoint",
    "metric",
    "labels_json",
    "value",
]

SNAPSHOT_FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "role",
    "endpoint",
    "scrape_ok",
    "scrape_duration_ms",
    "running",
    "waiting",
    "waiting_growth_per_s",
    "kv_cache_usage",
    "prefix_cache_hits",
    "prefix_cache_queries",
    "external_prefix_cache_hits",
    "external_prefix_cache_queries",
    "preemptions",
    "prompt_tokens",
    "generation_tokens",
    "successful_requests",
    "e2e_count",
    "e2e_sum_s",
    "ttft_count",
    "ttft_sum_s",
    "inter_token_count",
    "inter_token_sum_s",
    "iteration_tokens_count",
    "iteration_tokens_sum",
    "error",
]


def decode_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        key: value.replace(r"\"", '"').replace(r"\\", "\\")
        for key, value in LABEL_RE.findall(raw)
    }


def parse_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    result = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        metric, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        result.append((metric, decode_labels(raw_labels), value))
    return result


def append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def total(values: dict[str, list[float]], name: str) -> float:
    finite = [value for value in values.get(name, []) if math.isfinite(value)]
    return sum(finite) if finite else 0.0


def maximum(values: dict[str, list[float]], name: str) -> float:
    finite = [value for value in values.get(name, []) if math.isfinite(value)]
    return max(finite) if finite else 0.0


async def collect(
    endpoints: list[tuple[str, str]],
    output_dir: Path,
    interval: float,
    write_long: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "vllm_metrics_long.csv"
    snapshot_path = output_dir / "vllm_metrics_snapshots.csv"
    previous_waiting: dict[str, tuple[int, float]] = {}
    timeout = aiohttp.ClientTimeout(total=max(5.0, interval * 4))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            cycle_start = time.monotonic()
            for role, endpoint in endpoints:
                unix_ns = time.time_ns()
                monotonic_ns = time.monotonic_ns()
                started = time.monotonic()
                parsed: list[tuple[str, dict[str, str], float]] = []
                error = ""
                try:
                    async with session.get(endpoint) as response:
                        body = await response.text()
                        if response.status != 200:
                            raise RuntimeError(
                                f"HTTP {response.status}: {body[:240]}"
                            )
                        parsed = parse_metrics(body)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

                if write_long:
                    long_rows = [
                        {
                            "unix_ns": unix_ns,
                            "monotonic_ns": monotonic_ns,
                            "role": role,
                            "endpoint": endpoint,
                            "metric": metric,
                            "labels_json": json.dumps(
                                labels, sort_keys=True, ensure_ascii=False
                            ),
                            "value": value,
                        }
                        for metric, labels, value in parsed
                    ]
                    append_csv(long_path, LONG_FIELDS, long_rows)

                values: dict[str, list[float]] = {}
                for metric, _, value in parsed:
                    values.setdefault(metric, []).append(value)
                waiting = total(values, "vllm:num_requests_waiting")
                growth = 0.0
                previous = previous_waiting.get(role)
                if previous:
                    elapsed = (monotonic_ns - previous[0]) / 1_000_000_000
                    if elapsed > 0:
                        growth = (waiting - previous[1]) / elapsed
                previous_waiting[role] = (monotonic_ns, waiting)

                snapshot = {
                    "unix_ns": unix_ns,
                    "monotonic_ns": monotonic_ns,
                    "role": role,
                    "endpoint": endpoint,
                    "scrape_ok": not bool(error),
                    "scrape_duration_ms": (time.monotonic() - started) * 1000,
                    "running": total(values, "vllm:num_requests_running"),
                    "waiting": waiting,
                    "waiting_growth_per_s": growth,
                    "kv_cache_usage": maximum(
                        values, "vllm:kv_cache_usage_perc"
                    ),
                    "prefix_cache_hits": total(
                        values, "vllm:prefix_cache_hits_total"
                    ),
                    "prefix_cache_queries": total(
                        values, "vllm:prefix_cache_queries_total"
                    ),
                    "external_prefix_cache_hits": total(
                        values, "vllm:external_prefix_cache_hits_total"
                    ),
                    "external_prefix_cache_queries": total(
                        values, "vllm:external_prefix_cache_queries_total"
                    ),
                    "preemptions": total(
                        values, "vllm:num_preemptions_total"
                    ),
                    "prompt_tokens": total(
                        values, "vllm:prompt_tokens_total"
                    ),
                    "generation_tokens": total(
                        values, "vllm:generation_tokens_total"
                    ),
                    "successful_requests": total(
                        values, "vllm:request_success_total"
                    ),
                    "e2e_count": total(
                        values, "vllm:e2e_request_latency_seconds_count"
                    ),
                    "e2e_sum_s": total(
                        values, "vllm:e2e_request_latency_seconds_sum"
                    ),
                    "ttft_count": total(
                        values, "vllm:time_to_first_token_seconds_count"
                    ),
                    "ttft_sum_s": total(
                        values, "vllm:time_to_first_token_seconds_sum"
                    ),
                    "inter_token_count": total(
                        values, "vllm:inter_token_latency_seconds_count"
                    ),
                    "inter_token_sum_s": total(
                        values, "vllm:inter_token_latency_seconds_sum"
                    ),
                    "iteration_tokens_count": total(
                        values, "vllm:iteration_tokens_total_count"
                    ),
                    "iteration_tokens_sum": total(
                        values, "vllm:iteration_tokens_total_sum"
                    ),
                    "error": error,
                }
                append_csv(snapshot_path, SNAPSHOT_FIELDS, [snapshot])

            delay = interval - (time.monotonic() - cycle_start)
            await asyncio.sleep(max(0.0, delay))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        action="append",
        required=True,
        help="ROLE=URL; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument(
        "--write-long",
        action="store_true",
        help="also persist every raw Prometheus series; disabled by default",
    )
    args = parser.parse_args()
    endpoints = []
    for value in args.endpoint:
        role, endpoint = value.split("=", 1)
        endpoints.append((role, endpoint))
    asyncio.run(
        collect(
            endpoints,
            args.output_dir,
            args.interval,
            args.write_long,
        )
    )


if __name__ == "__main__":
    main()
