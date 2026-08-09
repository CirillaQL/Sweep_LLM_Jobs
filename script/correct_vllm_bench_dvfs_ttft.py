#!/usr/bin/env python3
"""Create vLLM benchmark metrics with a fixed DVFS settle wait removed."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be non-negative")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    raw_ttfts = [float(value) for value in data.get("ttfts", [])]
    wait_seconds = args.wait_seconds
    corrected_ttfts = [
        max(0.0, value - wait_seconds) if value > 0 else 0.0
        for value in raw_ttfts
    ]
    successful_ms = [value * 1000.0 for value in corrected_ttfts if value > 0]
    raw_metrics = {
        key: data.get(key)
        for key in ("mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms")
    }
    data["ttfts_raw_including_dvfs_wait"] = raw_ttfts
    data["ttfts"] = corrected_ttfts
    data["dvfs_settle_wait_excluded_seconds"] = wait_seconds
    data["raw_ttft_metrics_including_dvfs_wait"] = raw_metrics
    data["mean_ttft_ms"] = statistics.mean(successful_ms) if successful_ms else 0.0
    data["median_ttft_ms"] = statistics.median(successful_ms) if successful_ms else 0.0
    data["std_ttft_ms"] = statistics.pstdev(successful_ms) if successful_ms else 0.0
    data["p99_ttft_ms"] = percentile(successful_ms, 0.99)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": str(args.input),
        "output": str(args.output),
        "successful_ttft_count": len(successful_ms),
        "dvfs_settle_wait_excluded_seconds": wait_seconds,
        "raw_ttft_metrics_including_dvfs_wait": raw_metrics,
        "corrected_ttft_metrics": {
            "mean_ttft_ms": data["mean_ttft_ms"],
            "median_ttft_ms": data["median_ttft_ms"],
            "std_ttft_ms": data["std_ttft_ms"],
            "p99_ttft_ms": data["p99_ttft_ms"],
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
