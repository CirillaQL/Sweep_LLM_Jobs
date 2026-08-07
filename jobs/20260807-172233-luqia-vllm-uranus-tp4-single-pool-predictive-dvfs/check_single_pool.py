#!/usr/bin/env python3
"""Compare predicted and measured single-pool TTFT/TPOT, clocks, and power."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


PATTERNS = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "actual_request_throughput_rps": (r"Request throughput \(req/s\):\s+([0-9.]+)", float),
    "actual_mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([0-9.]+)", float),
    "actual_p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
    "actual_mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([0-9.]+)", float),
    "actual_p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([0-9.]+)", float),
}


def parse_bench(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for key, (pattern, converter) in PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing {key} in {path}")
        result[key] = converter(match.group(1))
    return result


def maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--clock-tolerance-mhz", type=float, default=90)
    parser.add_argument("--slo-ttft-ms", type=float, default=500)
    parser.add_argument("--slo-tpot-ms", type=float, default=200)
    parser.add_argument("--gpu-type", choices=("l4", "l40s"), default="l40s")
    parser.add_argument("--node", default="uranus")
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()

    decisions = json.loads((args.out_dir / "decisions.json").read_text(encoding="utf-8"))
    event_bounds = {}
    with (args.out_dir / "events.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_bounds.setdefault(int(row["seq"]), {})[row["event"]] = float(row["unix_ts"])
    telemetry = []
    with (args.out_dir / "telemetry.csv").open(newline="", encoding="utf-8") as handle:
        telemetry.extend(csv.DictReader(handle))

    rows = []
    integrity_errors = []
    for decision in decisions:
        seq = int(decision["seq"])
        workload_id = decision["workload_id"]
        bench = parse_bench(args.out_dir / f"bench_{seq}_{workload_id}.txt")
        bounds = event_bounds.get(seq, {})
        if "workload_start" not in bounds or "workload_end" not in bounds:
            integrity_errors.append(f"missing event bounds for {workload_id}")
            continue
        start, end = bounds["workload_start"], bounds["workload_end"]
        samples = [
            item for item in telemetry
            if start <= float(item["unix_ts"]) <= end
        ]
        by_gpu = {}
        by_ts = {}
        for sample in samples:
            gpu = sample["gpu_uuid"]
            clock = maybe_float(sample["sm_clock_mhz"])
            if clock is not None:
                by_gpu.setdefault(gpu, []).append(clock)
            power = maybe_float(sample["power_w"])
            if power is not None:
                by_ts.setdefault(sample["unix_ts"], []).append(power)
        medians = {gpu: statistics.median(values) for gpu, values in by_gpu.items()}
        target = float(decision["selected_frequency_mhz"])
        clock_ok = (
            len(medians) == args.expected_gpus
            and all(abs(value - target) <= args.clock_tolerance_mhz for value in medians.values())
        )
        if not clock_ok:
            integrity_errors.append(
                f"clock mismatch {workload_id}: target={target} medians={medians}"
            )
        powers = [sum(values) for values in by_ts.values() if len(values) == args.expected_gpus]
        avg_power = statistics.mean(powers) if powers else None
        if avg_power is None:
            integrity_errors.append(f"missing four-GPU power samples for {workload_id}")
        actual_slo_ok = (
            bench["actual_p99_ttft_ms"] <= args.slo_ttft_ms
            and bench["actual_p99_tpot_ms"] <= args.slo_tpot_ms
        )
        row = {
            "seq": seq,
            "workload_id": workload_id,
            "input_len": decision["input_len"],
            "output_len": decision["output_len"],
            "configured_request_rate_rps": decision["request_rate_rps"],
            "target_frequency_mhz": target,
            "observed_clock_median_mhz": statistics.median(medians.values()) if medians else None,
            "observed_gpu_count": len(medians),
            "frequency_lock_ok": clock_ok,
            "predicted_p99_ttft_ms": decision["predicted_p99_ttft_ms"],
            "actual_p99_ttft_ms": bench["actual_p99_ttft_ms"],
            "ttft_signed_error_ms": bench["actual_p99_ttft_ms"] - decision["predicted_p99_ttft_ms"],
            "predicted_p99_tpot_ms": decision["predicted_p99_tpot_ms"],
            "actual_p99_tpot_ms": bench["actual_p99_tpot_ms"],
            "tpot_signed_error_ms": bench["actual_p99_tpot_ms"] - decision["predicted_p99_tpot_ms"],
            "predicted_slo_ok": decision["predicted_slo_ok"],
            "actual_slo_ok": actual_slo_ok,
            "slo_prediction_match": decision["predicted_slo_ok"] == actual_slo_ok,
            "successful_requests": bench["successful_requests"],
            "failed_requests": bench["failed_requests"],
            "actual_request_throughput_rps": bench["actual_request_throughput_rps"],
            "actual_mean_ttft_ms": bench["actual_mean_ttft_ms"],
            "actual_mean_tpot_ms": bench["actual_mean_tpot_ms"],
            "predicted_power_proxy_w": decision["predicted_power_proxy_w"],
            "actual_cluster_avg_power_w": avg_power,
            "per_gpu_clock_medians_mhz": medians,
        }
        if bench["failed_requests"] != 0:
            integrity_errors.append(f"failed requests in {workload_id}")
        rows.append(row)

    if len(rows) != len(decisions):
        integrity_errors.append(f"expected {len(decisions)} result rows, got {len(rows)}")
    ttft_mae = statistics.mean(abs(row["ttft_signed_error_ms"]) for row in rows)
    tpot_mae = statistics.mean(abs(row["tpot_signed_error_ms"]) for row in rows)
    report = {
        "topology": "single_pool",
        "node": args.node,
        "gpu_type": args.gpu_type,
        "tensor_parallel_size": args.tp,
        "integrity_ok": not integrity_errors,
        "integrity_errors": integrity_errors,
        "slo_prediction_mismatches": sum(not row["slo_prediction_match"] for row in rows),
        "aggregate_prediction_error": {"ttft_mae_ms": ttft_mae, "tpot_mae_ms": tpot_mae},
        "workloads": rows,
    }
    (args.out_dir / "prediction_error.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_fields = [key for key in rows[0] if key != "per_gpu_clock_medians_mhz"]
    with (args.out_dir / "prediction_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)
    lines = [
        f"# TP{args.tp} {args.node} {args.gpu_type.upper()} single-pool predictive DVFS validation",
        "",
        f"Integrity: **{'PASS' if not integrity_errors else 'FAIL'}**; "
        f"SLO prediction mismatches: **{report['slo_prediction_mismatches']}**.",
        "",
        "| workload | target/actual MHz | predicted/actual P99 TTFT | predicted/actual P99 TPOT | actual W | SLO match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload_id']} | {row['target_frequency_mhz']:.0f}/{row['observed_clock_median_mhz']:.1f} | "
            f"{row['predicted_p99_ttft_ms']:.1f}/{row['actual_p99_ttft_ms']:.1f} ms | "
            f"{row['predicted_p99_tpot_ms']:.1f}/{row['actual_p99_tpot_ms']:.1f} ms | "
            f"{row['actual_cluster_avg_power_w']:.1f} | {'YES' if row['slo_prediction_match'] else 'NO'} |"
        )
    lines.extend(["", f"TTFT MAE: {ttft_mae:.2f} ms; TPOT MAE: {tpot_mae:.2f} ms.", ""])
    (args.out_dir / "prediction_error.md").write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(0 if not integrity_errors else 1)


if __name__ == "__main__":
    main()
