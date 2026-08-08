#!/usr/bin/env python3
"""Validate the request-level route cycle emitted by the custom PD proxy."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROUTE_ORDER = ("P0-D0", "P0-D1", "P1-D0", "P1-D1")
ROUTE_LINE = re.compile(r"^route count=(\d+) policy_route=(P[01]-D[01])\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations: dict[int, str] = {}
    duplicate_counts: list[int] = []

    for line in args.proxy_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = ROUTE_LINE.match(line)
        if match is None:
            continue
        count = int(match.group(1))
        if count in observations:
            duplicate_counts.append(count)
        observations[count] = match.group(2)

    ordered_counts = sorted(observations)
    expected_counts = list(range(args.expected_requests))
    missing_counts = sorted(set(expected_counts) - set(ordered_counts))
    unexpected_counts = sorted(set(ordered_counts) - set(expected_counts))
    route_mismatches = [
        {
            "count": count,
            "expected": ROUTE_ORDER[count % len(ROUTE_ORDER)],
            "actual": observations[count],
        }
        for count in ordered_counts
        if observations[count] != ROUTE_ORDER[count % len(ROUTE_ORDER)]
    ]
    passed = not (
        duplicate_counts or missing_counts or unexpected_counts or route_mismatches
    )
    report = {
        "passed": passed,
        "route_order": ROUTE_ORDER,
        "expected_requests": args.expected_requests,
        "observed_requests": len(observations),
        "route_counts": dict(sorted(Counter(observations.values()).items())),
        "duplicate_counts": duplicate_counts,
        "missing_counts": missing_counts,
        "unexpected_counts": unexpected_counts,
        "route_mismatches": route_mismatches,
        "proxy_log": str(args.proxy_log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "round_robin_check "
        f"passed={str(passed).lower()} "
        f"expected={args.expected_requests} observed={len(observations)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
