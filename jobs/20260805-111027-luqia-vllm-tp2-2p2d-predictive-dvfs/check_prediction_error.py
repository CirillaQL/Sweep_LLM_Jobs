#!/usr/bin/env python3
"""Compare scheduler predictions with measured latency, power, and GPU clocks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


BENCH_PATTERNS = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "request_throughput_rps": (
        r"Request throughput \(req/s\):\s+([0-9.]+)", float
    ),
    "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([0-9.]+)", float),
    "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
    "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_itl_ms": (r"P99 ITL \(ms\):\s+([0-9.]+)", float),
}


def parse_bench(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, (pattern, cast) in BENCH_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            result[name] = cast(match.group(1))
    return result


def signed_error(actual: float, predicted: float) -> float:
    return actual - predicted


def percent_error(actual: float, predicted: float) -> float:
    if predicted == 0:
        return float("nan")
    return 100.0 * signed_error(actual, predicted) / predicted


def rounded(value):
    return round(value, 6) if isinstance(value, float) else value


def display(value, digits=1):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def active_clock_summary(out_dir: Path, role: str, seq: int) -> dict:
    samples = []
    per_gpu = defaultdict(list)
    observed_samples = []
    observed_per_gpu = defaultdict(list)
    for path in sorted(out_dir.glob(f"{role}_*_telemetry.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("workload_seq") != str(seq):
                    continue
                try:
                    utilization = float(row["gpu_util_pct"])
                    clock = float(row["gpu_sm_mhz"])
                except (KeyError, TypeError, ValueError):
                    continue
                gpu_uuid = row.get("gpu_uuid") or f"{path.name}:{row.get('gpu_id')}"
                observed_per_gpu[gpu_uuid].append(clock)
                observed_samples.append(clock)
                if utilization < 10:
                    continue
                per_gpu[gpu_uuid].append(clock)
                samples.append(clock)
    gpu_means = {gpu: statistics.fmean(values) for gpu, values in per_gpu.items()}
    gpu_medians = {
        gpu: statistics.median(values) for gpu, values in observed_per_gpu.items()
    }
    return {
        "active_samples": len(samples),
        "gpu_count": len(gpu_means),
        "active_clock_mean_mhz": statistics.fmean(samples) if samples else None,
        "active_clock_min_mhz": min(samples) if samples else None,
        "active_clock_max_mhz": max(samples) if samples else None,
        "per_gpu_active_clock_mean_mhz": gpu_means,
        "observed_samples": len(observed_samples),
        "observed_gpu_count": len(gpu_medians),
        "observed_clock_median_mhz": (
            statistics.median(observed_samples) if observed_samples else None
        ),
        "per_gpu_observed_clock_median_mhz": gpu_medians,
    }


def aggregate_error(rows: list[dict], predicted_key: str, actual_key: str) -> dict:
    pairs = [
        (float(row[predicted_key]), float(row[actual_key]))
        for row in rows
        if row.get(predicted_key) is not None and row.get(actual_key) is not None
    ]
    if not pairs:
        return {"samples": 0}
    errors = [actual - predicted for predicted, actual in pairs]
    absolute = [abs(value) for value in errors]
    percentages = [
        abs(100.0 * (actual - predicted) / predicted)
        for predicted, actual in pairs
        if predicted != 0
    ]
    return {
        "samples": len(pairs),
        "mean_signed_error": statistics.fmean(errors),
        "mean_absolute_error": statistics.fmean(absolute),
        "mean_absolute_percentage_error_pct": (
            statistics.fmean(percentages) if percentages else None
        ),
        "max_absolute_error": max(absolute),
    }


def p2p_errors(out_dir: Path) -> dict:
    error_lines = 0
    request_ids = set()
    pattern = re.compile(r"tensor_id['\"]?: ['\"]([^'\"]+?)-0#model")
    for path in sorted(out_dir.glob("prefill*_server.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Peer Out Of Memory/Threshold" not in line:
                continue
            error_lines += 1
            match = pattern.search(line)
            if match:
                request_ids.add(match.group(1))
    return {
        "tensor_rejection_lines": error_lines,
        "affected_request_count": len(request_ids),
        "affected_request_ids": sorted(request_ids),
        "clean": error_lines == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--expected-workloads", type=int, required=True)
    parser.add_argument("--expected-prefill-instances", type=int, required=True)
    parser.add_argument("--expected-decode-instances", type=int, required=True)
    parser.add_argument("--expected-tp", type=int, required=True)
    parser.add_argument("--expected-gpus-per-role", type=int, required=True)
    parser.add_argument("--expected-prefill-node", default="neptune")
    parser.add_argument("--expected-decode-node", default="ganymede")
    parser.add_argument("--slo-ttft-ms", type=float, required=True)
    parser.add_argument("--slo-tpot-ms", type=float, required=True)
    parser.add_argument("--clock-tolerance-mhz", type=float, required=True)
    args = parser.parse_args()

    with args.workloads.open(newline="", encoding="utf-8") as handle:
        workloads = list(csv.DictReader(handle))
    registry = json.loads((args.out_dir / "registry.json").read_text(encoding="utf-8"))
    live = json.loads((args.out_dir / "live_summary.json").read_text(encoding="utf-8"))
    live_by_id = {
        item["workload"]["workload_id"]: item for item in live.get("workloads", [])
    }

    rows = []
    structural_ok = len(workloads) == args.expected_workloads
    requests_ok = True
    metrics_ok = True
    power_metrics_ok = True
    decisions_ok = True
    clocks_ok = True
    for seq, workload in enumerate(workloads, start=1):
        workload_id = workload["workload_id"]
        decision_path = args.out_dir / f"decision_active_{seq}_{workload_id}.json"
        bench_path = args.out_dir / f"bench_{seq}_{workload_id}.txt"
        if not decision_path.exists() or not bench_path.exists():
            structural_ok = False
            continue
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        recommended = decision.get("recommended") or {}
        prefill = recommended.get("prefill") or {}
        decode = recommended.get("decode") or {}
        metrics = parse_bench(bench_path)
        required = {
            "successful_requests", "failed_requests", "request_throughput_rps",
            "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
        }
        if not required.issubset(metrics):
            metrics_ok = False
            continue
        if metrics["successful_requests"] <= 0 or metrics["failed_requests"] != 0:
            requests_ok = False

        predicted_ttft = float(prefill["p99_ttft_ms"])
        predicted_tpot = float(decode["p99_tpot_ms"])
        actual_ttft = float(metrics["p99_ttft_ms"])
        actual_tpot = float(metrics["p99_tpot_ms"])
        predicted_slo_ok = (
            predicted_ttft <= args.slo_ttft_ms
            and predicted_tpot <= args.slo_tpot_ms
            and recommended.get("is_safe") is True
        )
        actual_slo_ok = (
            actual_ttft <= args.slo_ttft_ms
            and actual_tpot <= args.slo_tpot_ms
        )
        expected_instances = decision.get("instances", {})
        status_mode_ok = (
            decision.get("status") == "OK"
            and decision.get("decision_mode") == "safe_min_power"
        ) or (
            decision.get("status") == "OVERLOAD_FALLBACK"
            and decision.get("decision_mode") == "overload_min_slo_violation"
        )
        decision_ok = (
            status_mode_ok
            and decision.get("tp_per_role") == args.expected_tp
            and expected_instances.get("prefill") == args.expected_prefill_instances
            and expected_instances.get("decode") == args.expected_decode_instances
            and prefill.get("node_group") == args.expected_prefill_node
            and decode.get("node_group") == args.expected_decode_node
        )
        decisions_ok = decisions_ok and decision_ok

        prefill_clock = active_clock_summary(args.out_dir, "prefill", seq)
        decode_clock = active_clock_summary(args.out_dir, "decode", seq)
        prefill_target = float(prefill["freq_mhz"])
        decode_target = float(decode["freq_mhz"])
        role_clock_ok = True
        for target, summary in (
            (prefill_target, prefill_clock),
            (decode_target, decode_clock),
        ):
            gpu_medians = summary["per_gpu_observed_clock_median_mhz"]
            if len(gpu_medians) != args.expected_gpus_per_role:
                role_clock_ok = False
                continue
            if any(
                abs(median - target) > args.clock_tolerance_mhz
                for median in gpu_medians.values()
            ):
                role_clock_ok = False
        clocks_ok = clocks_ok and role_clock_ok

        live_item = live_by_id.get(workload_id, {})
        actual_power = (live_item.get("energy") or {}).get("combined_avg_power_w")
        predicted_power = recommended.get("predicted_cluster_power_w")
        if actual_power is None or predicted_power is None:
            power_metrics_ok = False
        row = {
            "seq": seq,
            "workload_id": workload_id,
            "input_len": int(workload["input_len"]),
            "output_len": int(workload["output_len"]),
            "configured_request_rate_rps": float(workload["request_rate"]),
            "decision_status": decision.get("status"),
            "decision_mode": decision.get("decision_mode"),
            "predicted_slo_ok": predicted_slo_ok,
            "actual_slo_ok": actual_slo_ok,
            "slo_prediction_match": predicted_slo_ok == actual_slo_ok,
            "successful_requests": metrics["successful_requests"],
            "failed_requests": metrics["failed_requests"],
            "actual_request_throughput_rps": metrics["request_throughput_rps"],
            "predicted_p99_ttft_ms": predicted_ttft,
            "actual_p99_ttft_ms": actual_ttft,
            "ttft_signed_error_ms": signed_error(actual_ttft, predicted_ttft),
            "ttft_absolute_error_ms": abs(signed_error(actual_ttft, predicted_ttft)),
            "ttft_percent_error": percent_error(actual_ttft, predicted_ttft),
            "predicted_p99_tpot_ms": predicted_tpot,
            "actual_p99_tpot_ms": actual_tpot,
            "tpot_signed_error_ms": signed_error(actual_tpot, predicted_tpot),
            "tpot_absolute_error_ms": abs(signed_error(actual_tpot, predicted_tpot)),
            "tpot_percent_error": percent_error(actual_tpot, predicted_tpot),
            "actual_mean_ttft_ms": metrics["mean_ttft_ms"],
            "actual_mean_tpot_ms": metrics["mean_tpot_ms"],
            "prefill_target_freq_mhz": prefill_target,
            "prefill_actual_active_clock_mean_mhz": prefill_clock[
                "active_clock_mean_mhz"
            ],
            "prefill_active_gpu_count": prefill_clock["gpu_count"],
            "prefill_actual_observed_clock_median_mhz": prefill_clock[
                "observed_clock_median_mhz"
            ],
            "prefill_observed_gpu_count": prefill_clock["observed_gpu_count"],
            "decode_target_freq_mhz": decode_target,
            "decode_actual_active_clock_mean_mhz": decode_clock[
                "active_clock_mean_mhz"
            ],
            "decode_active_gpu_count": decode_clock["gpu_count"],
            "decode_actual_observed_clock_median_mhz": decode_clock[
                "observed_clock_median_mhz"
            ],
            "decode_observed_gpu_count": decode_clock["observed_gpu_count"],
            "frequency_lock_ok": role_clock_ok,
            "predicted_cluster_power_w": predicted_power,
            "actual_cluster_avg_power_w": actual_power,
            "power_signed_error_w": (
                signed_error(float(actual_power), float(predicted_power))
                if actual_power is not None and predicted_power is not None else None
            ),
            "power_percent_error": (
                percent_error(float(actual_power), float(predicted_power))
                if actual_power is not None and predicted_power is not None else None
            ),
        }
        rows.append({key: rounded(value) for key, value in row.items()})

    topology_ok = (
        registry.get("prefill_count") == args.expected_prefill_instances
        and registry.get("decode_count") == args.expected_decode_instances
        and registry.get("prefill_tp_sizes")
        == [args.expected_tp] * args.expected_prefill_instances
        and registry.get("decode_tp_sizes")
        == [args.expected_tp] * args.expected_decode_instances
    )
    power_coverage_ok = (
        live.get("gpu_stream_count")
        == 2 * args.expected_gpus_per_role
    )
    transport = p2p_errors(args.out_dir)
    checks = {
        "workload_count_ok": len(workloads) == args.expected_workloads,
        "all_outputs_present": structural_ok and len(rows) == args.expected_workloads,
        "requests_ok": requests_ok,
        "metrics_complete": metrics_ok,
        "power_metrics_complete": power_metrics_ok,
        "scheduler_decisions_and_topology_ok": decisions_ok,
        "registry_topology_ok": topology_ok,
        "all_gpu_power_covered": power_coverage_ok,
        "frequency_locks_verified": clocks_ok,
    }
    integrity_ok = all(checks.values())
    aggregates = {
        "ttft": aggregate_error(
            rows, "predicted_p99_ttft_ms", "actual_p99_ttft_ms"
        ),
        "tpot": aggregate_error(
            rows, "predicted_p99_tpot_ms", "actual_p99_tpot_ms"
        ),
        "cluster_power": aggregate_error(
            rows, "predicted_cluster_power_w", "actual_cluster_avg_power_w"
        ),
        "slo_prediction_matches": sum(
            row["slo_prediction_match"] for row in rows
        ),
        "slo_prediction_mismatches": sum(
            not row["slo_prediction_match"] for row in rows
        ),
    }
    payload = {
        "slo": {
            "ttft_ms": args.slo_ttft_ms,
            "tpot_ms": args.slo_tpot_ms,
        },
        "expected_topology": {
            "prefill_instances": args.expected_prefill_instances,
            "decode_instances": args.expected_decode_instances,
            "tp_per_instance": args.expected_tp,
            "gpus_per_role": args.expected_gpus_per_role,
            "prefill_node": args.expected_prefill_node,
            "decode_node": args.expected_decode_node,
        },
        "checks": checks,
        "integrity_ok": integrity_ok,
        "transport": transport,
        "aggregate_prediction_error": aggregates,
        "workloads": rows,
    }
    (args.out_dir / "prediction_error.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    fields = list(rows[0]) if rows else []
    with (args.out_dir / "prediction_error.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# TP{args.expected_tp} {args.expected_prefill_node}/{args.expected_decode_node} predictive DVFS validation",
        "",
        f"Integrity: **{'PASS' if integrity_ok else 'FAIL'}**; "
        f"SLO prediction mismatches: **{aggregates['slo_prediction_mismatches']}**; "
        f"P2P transport clean: **{'YES' if transport['clean'] else 'NO'}**.",
        "",
        "| workload | target P/D MHz | actual P/D MHz | predicted/actual P99 TTFT | "
        "predicted/actual P99 TPOT | predicted/actual W | SLO match |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload_id']} | "
            f"{row['prefill_target_freq_mhz']:.0f}/{row['decode_target_freq_mhz']:.0f} | "
            f"{display(row['prefill_actual_observed_clock_median_mhz'])}/"
            f"{display(row['decode_actual_observed_clock_median_mhz'])} | "
            f"{row['predicted_p99_ttft_ms']:.1f}/{row['actual_p99_ttft_ms']:.1f} ms | "
            f"{row['predicted_p99_tpot_ms']:.1f}/{row['actual_p99_tpot_ms']:.1f} ms | "
            f"{display(row['predicted_cluster_power_w'])}/"
            f"{display(row['actual_cluster_avg_power_w'])} | "
            f"{'YES' if row['slo_prediction_match'] else 'NO'} |"
        )
    lines.extend([
        "",
        f"TTFT MAE: {aggregates['ttft'].get('mean_absolute_error', float('nan')):.2f} ms; "
        f"TPOT MAE: {aggregates['tpot'].get('mean_absolute_error', float('nan')):.2f} ms; "
        f"power MAE: {aggregates['cluster_power'].get('mean_absolute_error', float('nan')):.2f} W.",
        "",
        f"P2P tensor rejection lines: {transport['tensor_rejection_lines']}; "
        f"affected requests: {transport['affected_request_count']}.",
    ])
    (args.out_dir / "prediction_error.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if integrity_ok else 1)


if __name__ == "__main__":
    main()
