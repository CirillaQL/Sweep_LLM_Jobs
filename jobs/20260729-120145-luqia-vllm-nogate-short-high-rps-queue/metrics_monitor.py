#!/usr/bin/env python3
"""Sample vLLM queue metrics while the parent job starts and runs workloads."""

import argparse
import csv
import signal
import time
import urllib.request
from pathlib import Path


STOP = False
TARGETS = {
    "vllm:num_requests_running": "num_running",
    "vllm:num_requests_waiting": "num_waiting",
    "vllm:num_requests_swapped": "num_swapped",
    "vllm:gpu_cache_usage_perc": "gpu_cache_usage",
    "vllm:request_queue_time_seconds_sum": "queue_time_sum_s",
    "vllm:request_queue_time_seconds_count": "queue_time_count",
}


def stop(_signum, _frame):
    global STOP
    STOP = True


def fetch(url):
    with urllib.request.urlopen(url, timeout=1.5) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metrics(text):
    values = {column: "" for column in TARGETS.values()}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        metric = fields[0].split("{", 1)[0]
        column = TARGETS.get(metric)
        if column is None:
            continue
        try:
            value = float(fields[-1])
        except ValueError:
            continue
        if values[column] == "":
            values[column] = value
        else:
            values[column] = max(float(values[column]), value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    fields = [
        "unix_ts",
        "component",
        *TARGETS.values(),
        "scrape_ok",
        "error",
    ]
    endpoints = {
        "prefill": args.prefill_url,
        "decode": args.decode_url,
    }

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        while not STOP:
            started = time.time()
            for component, url in endpoints.items():
                row = {
                    "unix_ts": f"{started:.6f}",
                    "component": component,
                    "scrape_ok": False,
                    "error": "",
                }
                try:
                    row.update(parse_metrics(fetch(url)))
                    row["scrape_ok"] = True
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                writer.writerow(row)
            handle.flush()
            remaining = args.interval - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
