#!/usr/bin/env python3
"""Validate request-level random routes emitted by the custom PD proxy."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROUTE_LINE = re.compile(r"^route count=(\d+) policy_route=(\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-count-start", type=int, default=0)
    parser.add_argument("--expected-prefill-count", type=int, default=2)
    parser.add_argument("--expected-decode-count", type=int, default=2)
    parser.add_argument("--expected-prefill-tp-sizes", default="1,1")
    parser.add_argument("--expected-decode-tp-sizes", default="1,1")
    parser.add_argument("--allow-asymmetric-tp", action="store_true")
    parser.add_argument("--require-all-routes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefill_tp_sizes = tuple(
        int(value) for value in args.expected_prefill_tp_sizes.split(",")
    )
    decode_tp_sizes = tuple(
        int(value) for value in args.expected_decode_tp_sizes.split(",")
    )
    if len(prefill_tp_sizes) != args.expected_prefill_count:
        raise ValueError(
            "expected Prefill TP list length to match expected Prefill count"
        )
    if len(decode_tp_sizes) != args.expected_decode_count:
        raise ValueError(
            "expected Decode TP list length to match expected Decode count"
        )
    supported_routes = tuple(
        f"P{prefill_index}-D{decode_index}"
        for prefill_index, prefill_tp_size in enumerate(prefill_tp_sizes)
        for decode_index, decode_tp_size in enumerate(decode_tp_sizes)
        if args.allow_asymmetric_tp or prefill_tp_size == decode_tp_size
    )
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
    expected_counts = list(
        range(
            args.expected_count_start,
            args.expected_count_start + args.expected_requests,
        )
    )
    missing_counts = sorted(set(expected_counts) - set(ordered_counts))
    unexpected_counts = sorted(set(ordered_counts) - set(expected_counts))
    invalid_routes = [
        {"count": count, "route": observations[count]}
        for count in ordered_counts
        if observations[count] not in supported_routes
    ]
    route_counts = Counter(observations.values())
    prefill_counts = Counter(
        route.split("-", 1)[0] for route in observations.values()
    )
    decode_counts = Counter(
        route.split("-", 1)[1] for route in observations.values()
    )
    unobserved_routes = sorted(set(supported_routes) - set(observations.values()))
    passed = not (
        duplicate_counts
        or missing_counts
        or unexpected_counts
        or invalid_routes
        or (args.require_all_routes and unobserved_routes)
    )
    report = {
        "passed": passed,
        "policy": (
            "independent_prefill_decode_random"
            if args.allow_asymmetric_tp
            else "prefill_first_symmetric_tp_random"
        ),
        "allow_asymmetric_tp": args.allow_asymmetric_tp,
        "prefill_tp_sizes": prefill_tp_sizes,
        "decode_tp_sizes": decode_tp_sizes,
        "supported_routes": supported_routes,
        "require_all_routes": args.require_all_routes,
        "expected_requests": args.expected_requests,
        "expected_count_start": args.expected_count_start,
        "observed_requests": len(observations),
        "route_counts": dict(sorted(route_counts.items())),
        "prefill_counts": dict(sorted(prefill_counts.items())),
        "decode_counts": dict(sorted(decode_counts.items())),
        "duplicate_counts": duplicate_counts,
        "missing_counts": missing_counts,
        "unexpected_counts": unexpected_counts,
        "invalid_routes": invalid_routes,
        "unobserved_routes": unobserved_routes,
        "proxy_log": str(args.proxy_log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "random_route_check "
        f"passed={str(passed).lower()} "
        f"expected={args.expected_requests} observed={len(observations)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
