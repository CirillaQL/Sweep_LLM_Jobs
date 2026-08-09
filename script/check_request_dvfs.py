#!/usr/bin/env python3
"""Validate request-level predictive DVFS decisions and clock telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--clock-tolerance-mhz", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors = []
    records = []
    if not args.decisions.is_file():
        errors.append(f"missing decisions file: {args.decisions}")
    else:
        for line_number, line in enumerate(
            args.decisions.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid decision line {line_number}: {exc}")

    route_counts = Counter()
    frequency_pair_counts = Counter()
    safe_count = fallback_count = success_count = 0
    actual_ttft_slo_met = actual_forwarding_ttft_slo_met = 0
    actual_tpot_slo_met = 0
    powers = []
    for item in records:
        route_counts[item.get("route", "missing")] += 1
        prediction = item.get("prediction") or {}
        status = prediction.get("status")
        if status == "OK":
            safe_count += 1
        elif status == "OVERLOAD_FALLBACK":
            fallback_count += 1
        else:
            errors.append(
                f"request {item.get('request_index')} has prediction status {status}"
            )
        actual = item.get("actual") or {}
        if actual.get("success") is True:
            success_count += 1
        else:
            errors.append(
                f"request {item.get('request_index')} failed: {actual.get('error')}"
            )
        actual_ttft_slo_met += int(actual.get("ttft_slo_met") is True)
        actual_forwarding_ttft_slo_met += int(
            actual.get("forwarding_ttft_slo_met") is True
        )
        actual_tpot_slo_met += int(actual.get("tpot_slo_met") is True)
        raw_ttft = actual.get("proxy_ttft_raw_ms")
        corrected_ttft = actual.get("proxy_ttft_ms")
        if actual.get("success") is True:
            forwarding_ttft = actual.get("proxy_forwarding_ttft_ms")
            if forwarding_ttft is None or float(forwarding_ttft) < 0:
                errors.append(
                    f"request {item.get('request_index')} has invalid forwarding TTFT"
                )
            try:
                excluded = float(raw_ttft) - float(corrected_ttft)
                expected_excluded = args.settle_seconds * 1000.0
                if abs(excluded - expected_excluded) > 10.0:
                    errors.append(
                        f"request {item.get('request_index')} excluded TTFT "
                        f"{excluded:.3f} ms != expected {expected_excluded:.3f} ms"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"request {item.get('request_index')} has invalid raw/corrected TTFT"
                )
        recommended = prediction.get("recommended") or {}
        if recommended.get("is_safe") is not True:
            errors.append(
                f"request {item.get('request_index')} selected an unsafe candidate"
            )
        if "predicted_cluster_power_w" in recommended:
            powers.append(float(recommended["predicted_cluster_power_w"]))
        clock = item.get("clock") or {}
        p_freq = clock.get("prefill_target_mhz")
        d_freq = clock.get("decode_target_mhz")
        if p_freq is None or d_freq is None:
            errors.append(f"request {item.get('request_index')} has no clock targets")
            continue
        frequency_pair_counts[f"P{p_freq}-D{d_freq}"] += 1
        for role in ("prefill", "decode"):
            target = float(clock[f"{role}_target_mhz"])
            ack = clock.get(f"{role}_ack") or {}
            if ack.get("cached_verified_target"):
                continue
            if int(ack.get("rc", -1)) != 0:
                errors.append(
                    f"request {item.get('request_index')} {role} clock rc={ack.get('rc')}"
                )
            try:
                ack_settle = float(ack.get("settle_seconds", -1))
            except (TypeError, ValueError):
                ack_settle = -1
            if abs(ack_settle - args.settle_seconds) > 0.01:
                errors.append(
                    f"request {item.get('request_index')} {role} settle_seconds="
                    f"{ack.get('settle_seconds')} expected={args.settle_seconds}"
                )
            observed = ack.get("observed_mhz", [])
            if not isinstance(observed, list):
                observed = [observed]
            try:
                if not observed or any(
                    abs(float(value) - target) > args.clock_tolerance_mhz
                    for value in observed
                ):
                    errors.append(
                        f"request {item.get('request_index')} {role} target={target} "
                        f"observed={observed}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"request {item.get('request_index')} invalid {role} observed={observed}"
                )

    telemetry_files = sorted(args.telemetry_dir.glob("gpu_telemetry_*.csv"))
    telemetry_rows = 0
    active_telemetry_rows = 0
    for path in telemetry_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                telemetry_rows += 1
                try:
                    if int(row["clock_seq"]) > 0 and int(row["target_freq_mhz"]) > 0:
                        active_telemetry_rows += 1
                except (KeyError, TypeError, ValueError):
                    pass
    if len(records) != args.expected_requests:
        errors.append(
            f"decision count {len(records)} != expected {args.expected_requests}"
        )
    if not telemetry_files or active_telemetry_rows == 0:
        errors.append("no active-clock GPU telemetry was recorded")

    result = {
        "ok": not errors,
        "expected_requests": args.expected_requests,
        "expected_settle_seconds": args.settle_seconds,
        "decision_count": len(records),
        "successful_decisions": success_count,
        "safe_decisions": safe_count,
        "overload_fallback_decisions": fallback_count,
        "actual_ttft_slo_met": actual_ttft_slo_met,
        "actual_forwarding_ttft_slo_met": actual_forwarding_ttft_slo_met,
        "actual_tpot_slo_met": actual_tpot_slo_met,
        "route_counts": dict(sorted(route_counts.items())),
        "frequency_pair_counts": dict(sorted(frequency_pair_counts.items())),
        "mean_predicted_cluster_power_w": (
            round(sum(powers) / len(powers), 3) if powers else None
        ),
        "telemetry_files": [path.name for path in telemetry_files],
        "telemetry_rows": telemetry_rows,
        "active_clock_telemetry_rows": active_telemetry_rows,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
