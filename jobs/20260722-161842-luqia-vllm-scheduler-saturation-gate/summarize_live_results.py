#!/usr/bin/env python3
"""Summarize predicted placement, vLLM performance, energy, and GPU telemetry."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


BENCH_PATTERNS = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "benchmark_duration_s": (r"Benchmark duration \(s\):\s+([0-9.]+)", float),
    "request_throughput_rps": (r"Request throughput \(req/s\):\s+([0-9.]+)", float),
    "output_token_throughput_tps": (
        r"Output token throughput \(tok/s\):\s+([0-9.]+)", float
    ),
    "total_token_throughput_tps": (
        r"Total token throughput \(tok/s\):\s+([0-9.]+)", float
    ),
    "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
    "p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_itl_ms": (r"P99 ITL \(ms\):\s+([0-9.]+)", float),
}

TELEMETRY_FIELDS = {
    "gpu_util_pct": ("gpu_util_avg_pct", "gpu_util_max_pct"),
    "gpu_power_w": ("gpu_power_avg_w", "gpu_power_max_w"),
    "gpu_sm_mhz": ("gpu_sm_clock_avg_mhz", "gpu_sm_clock_max_mhz"),
    "gpu_memory_used_mib": ("gpu_memory_used_avg_mib", "gpu_memory_used_max_mib"),
    "gpu_mem_util_pct": ("gpu_mem_util_avg_pct", "gpu_mem_util_max_pct"),
    "gpu_power_limit_w": ("gpu_power_limit_avg_w", "gpu_power_limit_max_w"),
    "gpu_mem_clock_mhz": ("gpu_mem_clock_avg_mhz", "gpu_mem_clock_max_mhz"),
    "gpu_temperature_c": ("gpu_temperature_avg_c", "gpu_temperature_max_c"),
}


def parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_bench(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, (pattern, cast) in BENCH_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            result[name] = cast(match.group(1))
    return result


def load_events(path):
    windows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["seq"]), row["workload_id"])
            windows.setdefault(key, {})[row["event"]] = float(row["unix_ts"])
    return windows


def load_telemetry(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def telemetry_window(rows, start, end):
    selected = []
    for row in rows:
        timestamp = parse_float(row.get("unix_ts"))
        if timestamp is not None and start <= timestamp <= end:
            selected.append(row)
    result = {"samples": len(selected)}
    if not selected:
        return result

    target_values = [
        value for value in (parse_float(row.get("target_freq_mhz")) for row in selected)
        if value is not None
    ]
    if target_values:
        result["target_freq_mhz"] = target_values[-1]

    for source, (avg_name, max_name) in TELEMETRY_FIELDS.items():
        values = [
            value for value in (parse_float(row.get(source)) for row in selected)
            if value is not None
        ]
        if values:
            result[avg_name] = statistics.fmean(values)
            result[max_name] = max(values)
            if source == "gpu_sm_mhz":
                result["gpu_sm_clock_min_mhz"] = min(values)

    for source, output in (("rx_bytes", "network_rx_bytes"), ("tx_bytes", "network_tx_bytes")):
        values = [
            value for value in (parse_float(row.get(source)) for row in selected)
            if value is not None
        ]
        if len(values) >= 2:
            result[output] = max(0.0, values[-1] - values[0])

    pstates = [row.get("gpu_pstate", "").strip() for row in selected]
    pstates = [value for value in pstates if value]
    if pstates:
        result["gpu_pstate_mode"] = Counter(pstates).most_common(1)[0][0]
    return result


def rounded(value):
    return round(value, 6) if isinstance(value, float) else value


def flatten(prefix, data):
    return {f"{prefix}_{key}": rounded(value) for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--slo-ttft", type=float, required=True)
    parser.add_argument("--slo-tpot", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--no-decisions", action="store_true")
    parser.add_argument("--decision-prefix", default="decision_gate")
    args = parser.parse_args()

    with args.workloads.open(newline="", encoding="utf-8") as handle:
        workloads = list(csv.DictReader(handle))
    windows = load_events(args.out_dir / "events.csv")

    energy_path = args.out_dir / "energy_summary.json"
    energy = json.loads(energy_path.read_text(encoding="utf-8"))
    energy_by_id = {item["workload_id"]: item for item in energy["workloads"]}

    telemetry_sources = {}
    for path in sorted(args.out_dir.glob("*_telemetry.csv")):
        name = path.stem.removesuffix("_telemetry")
        telemetry_sources[name] = load_telemetry(path)

    results = []
    csv_rows = []
    for seq, workload in enumerate(workloads, start=1):
        workload_id = workload["workload_id"]
        decision_path = args.out_dir / f"{args.decision_prefix}_{seq}_{workload_id}.json"
        if args.no_decisions:
            decision = None
            recommended = None
            row = {
                "seq": seq,
                "workload_id": workload_id,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "configured_request_rate_rps": float(workload["request_rate"]),
                "decision_status": "FIXED_AUTO_DVFS",
                "prefill_gpu": "l40s",
                "decode_gpu": "l4",
            }
        else:
            if not decision_path.exists():
                csv_rows.append({
                    "seq": seq,
                    "workload_id": workload_id,
                    "input_len": int(workload["input_len"]),
                    "output_len": int(workload["output_len"]),
                    "configured_request_rate_rps": float(workload["request_rate"]),
                    "decision_status": "NOT_EXECUTED",
                })
                results.append({
                    "workload": csv_rows[-1].copy(),
                    "decision": None,
                    "performance": {},
                    "telemetry": {},
                    "energy": {},
                    "summary": csv_rows[-1].copy(),
                })
                continue
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            recommended = decision.get("recommended")
            row = {
                "seq": seq,
                "workload_id": workload_id,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "configured_request_rate_rps": float(workload["request_rate"]),
                "decision_status": decision["status"],
                "decision_mode": decision.get("decision_mode"),
                "num_candidates": decision["num_candidates"],
                "num_safe": decision["num_safe"],
            }
        nested = {"workload": row.copy(), "decision": decision}

        if recommended:
            prefill = recommended["prefill"]
            decode = recommended["decode"]
            kv_transfer = decision.get("kv_transfer_model", {})
            row.update({
                "recommended_is_safe": recommended["is_safe"],
                "prefill_gpu": prefill["gpu_type"],
                "prefill_target_freq_mhz": prefill["freq_mhz"],
                "prefill_predicted_p99_ttft_ms": prefill["p99_ttft_ms"],
                "prefill_predicted_p99_queue_plus_prefill_ms": prefill.get(
                    "p99_queue_plus_prefill_ms", prefill["p99_ttft_ms"]
                ),
                "predicted_kv_transfer_ms": prefill.get("kv_transfer_ms", 0.0),
                "predicted_dispatch_ms": prefill.get("dispatch_ms", 0.0),
                "kv_bytes_per_token": kv_transfer.get("kv_bytes_per_token"),
                "kv_total_bytes": kv_transfer.get("kv_total_bytes"),
                "kv_effective_bandwidth_gbps": kv_transfer.get(
                    "effective_bandwidth_gbps"
                ),
                "prefill_predicted_p_saturated": prefill["p_saturated"],
                "decode_gpu": decode["gpu_type"],
                "decode_target_freq_mhz": decode["freq_mhz"],
                "decode_predicted_p99_tpot_ms": decode["p99_tpot_ms"],
                "decode_predicted_p_saturated": decode["p_saturated"],
                "predicted_cluster_power_w": recommended["predicted_cluster_power_w"],
            })

        metrics = parse_bench(args.out_dir / f"bench_{seq}_{workload_id}.txt")
        required_metrics = {
            "request_throughput_rps", "p99_ttft_ms", "p99_tpot_ms"
        }
        if required_metrics.issubset(metrics):
            throughput_ratio = (
                metrics["request_throughput_rps"] / row["configured_request_rate_rps"]
            )
            metrics.update({
                "throughput_ratio": throughput_ratio,
                "measured_saturated": None,
                "saturation_assessment": "INCONCLUSIVE_FINITE_POISSON",
                "ttft_slo_ok": metrics["p99_ttft_ms"] <= args.slo_ttft,
                "tpot_slo_ok": metrics["p99_tpot_ms"] <= args.slo_tpot,
            })
        if metrics:
            row.update({f"actual_{key}": rounded(value) for key, value in metrics.items()})
        nested["performance"] = metrics

        window = windows.get((seq, workload_id), {})
        telemetry = {}
        if "workload_start" in window and "workload_end" in window:
            for name, samples in telemetry_sources.items():
                summary = telemetry_window(
                    samples, window["workload_start"], window["workload_end"]
                )
                telemetry[name] = summary
                row.update(flatten(name, summary))
        nested["telemetry"] = telemetry

        energy_item = energy_by_id.get(workload_id, {})
        energy_summary = {
            "combined_energy_j": energy_item.get("combined_energy_j"),
            "combined_avg_power_w": energy_item.get("combined_avg_power_w"),
            "energy_per_request_j": energy_item.get("energy_per_request_j"),
        }
        row.update({key: rounded(value) for key, value in energy_summary.items()})
        nested["energy"] = energy_summary
        nested["summary"] = row
        results.append(nested)
        csv_rows.append(row)

    payload = {
        "slo": {"ttft_ms": args.slo_ttft, "tpot_ms": args.slo_tpot},
        "saturation_definition": (
            "inconclusive for fixed-count finite-rate Poisson windows; "
            "throughput ratio includes arrival spacing and tail drain"
        ),
        "power_scope": energy["scope"],
        "workloads": results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = sorted({key for row in csv_rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
