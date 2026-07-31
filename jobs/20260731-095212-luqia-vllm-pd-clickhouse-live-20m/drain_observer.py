#!/usr/bin/env python3
"""Observe both PD queues after load stops and record time-to-drain."""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
import time

import aiohttp

from metrics_collector import maximum, parse_metrics, total


FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "elapsed_s",
    "role",
    "running",
    "waiting",
    "kv_cache_usage",
    "scrape_ok",
    "error",
]

SUMMARY_FIELDS = [
    "start_unix_ns",
    "end_unix_ns",
    "drain_time_s",
    "drained",
    "consecutive_zero_samples",
]


def append(path: Path, fields: list[str], rows: list[dict]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


async def main_async(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "drain_samples.csv"
    summary_path = args.output_dir / "drain_summary.csv"
    endpoints = []
    for value in args.endpoint:
        role, endpoint = value.split("=", 1)
        endpoints.append((role, endpoint))

    start_unix_ns = time.time_ns()
    start = time.monotonic()
    consecutive_zero = 0
    drained = False
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=5)
    ) as session:
        while time.monotonic() - start < args.max_seconds:
            all_zero = True
            rows = []
            for role, endpoint in endpoints:
                now_ns = time.time_ns()
                values: dict[str, list[float]] = {}
                error = ""
                try:
                    async with session.get(endpoint) as response:
                        text = await response.text()
                        if response.status != 200:
                            raise RuntimeError(f"HTTP {response.status}")
                    for metric, _, value in parse_metrics(text):
                        values.setdefault(metric, []).append(value)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                running = total(values, "vllm:num_requests_running")
                waiting = total(values, "vllm:num_requests_waiting")
                if error or running != 0 or waiting != 0:
                    all_zero = False
                rows.append(
                    {
                        "unix_ns": now_ns,
                        "monotonic_ns": time.monotonic_ns(),
                        "elapsed_s": time.monotonic() - start,
                        "role": role,
                        "running": running,
                        "waiting": waiting,
                        "kv_cache_usage": maximum(
                            values, "vllm:kv_cache_usage_perc"
                        ),
                        "scrape_ok": not bool(error),
                        "error": error,
                    }
                )
            append(samples_path, FIELDS, rows)
            consecutive_zero = consecutive_zero + 1 if all_zero else 0
            if consecutive_zero >= args.zero_samples:
                drained = True
                break
            await asyncio.sleep(args.interval)

    end_unix_ns = time.time_ns()
    append(
        summary_path,
        SUMMARY_FIELDS,
        [
            {
                "start_unix_ns": start_unix_ns,
                "end_unix_ns": end_unix_ns,
                "drain_time_s": (end_unix_ns - start_unix_ns) / 1e9,
                "drained": drained,
                "consecutive_zero_samples": consecutive_zero,
            }
        ],
    )
    raise SystemExit(0 if drained else 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--zero-samples", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
