#!/usr/bin/env python3
"""Compare default-DVFS baseline energy with scheduler-controlled job 249820."""

import argparse
import csv
import json
import re
from pathlib import Path

from energy_summary import integrate, load_telemetry, probe_energy


def metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
        "request_throughput_rps": r"Request throughput \(req/s\):\s+([0-9.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing {key} in {path}")
        result[key] = int(match.group(1)) if key == "successful_requests" else float(match.group(1))
    return result


def scheduler_windows(root, workloads):
    telemetry = {
        "neptune_l40s": load_telemetry(root / "prefill_neptune_telemetry.csv"),
        "ganymede_l4": load_telemetry(root / "decode_ganymede_telemetry.csv"),
    }
    seq_rows = {}
    for name, path in {
        "neptune_l40s": root / "prefill_neptune_telemetry.csv",
        "ganymede_l4": root / "decode_ganymede_telemetry.csv",
    }.items():
        values = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                values.setdefault(int(row["workload_seq"]), []).append(float(row["unix_ts"]))
        seq_rows[name] = values

    result = {}
    for index, workload in enumerate(workloads):
        seq = int(workload["seq"])
        probe_ends = []
        for node in ("neptune", "ganymede"):
            data = json.loads((root / f"clock_{seq}_{node}.json").read_text(encoding="utf-8"))
            probe_ends.append(max(float(row["unix_ts"]) for row in data["samples"]))
        start = max(probe_ends)
        if index + 1 < len(workloads):
            next_seq = int(workloads[index + 1]["seq"])
            end = min(
                min(seq_rows[node][next_seq])
                for node in ("neptune_l40s", "ganymede_l4")
            )
        else:
            end = min(samples[-1][0] for samples in telemetry.values())
        node_energy = {}
        for node, samples in telemetry.items():
            energy_j, covered_s = integrate(samples, start, end)
            node_energy[node] = {"energy_j": energy_j, "covered_s": covered_s}
        combined_j = sum(item["energy_j"] for item in node_energy.values())
        request_count = int(workload["successful_requests"])
        result[workload["workload_id"]] = {
            "start_unix_ts": start,
            "end_unix_ts": end,
            "duration_s": end - start,
            "nodes": node_energy,
            "combined_energy_j": combined_j,
            "combined_avg_power_w": combined_j / (end - start),
            "energy_per_request_j": combined_j / request_count,
        }
    total_j = sum(integrate(samples)[0] for samples in telemetry.values())
    reset_j = sum(probe_energy(path) for path in root.glob("reset_*_probe.json"))
    return result, total_j + reset_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--scheduler-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads((args.baseline_dir / "energy_summary.json").read_text(encoding="utf-8"))
    baseline_items = {item["workload_id"]: item for item in baseline["workloads"]}
    workload_order = baseline["workloads"]
    scheduler_energy, scheduler_total_j = scheduler_windows(args.scheduler_dir, workload_order)

    workload_config = {}
    with (args.baseline_dir.parent / "workloads.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            row["seq"] = index
            workload_config[row["workload_id"]] = row

    comparisons = []
    for item in workload_order:
        workload_id = item["workload_id"]
        seq = int(item["seq"])
        baseline_metrics = metrics(args.baseline_dir / f"bench_{seq}_{workload_id}.txt")
        scheduler_metrics = metrics(args.scheduler_dir / f"bench_{seq}_{workload_id}.txt")
        baseline_j = float(item["combined_energy_j"])
        scheduled = scheduler_energy[workload_id]
        scheduler_j = scheduled["combined_energy_j"]
        config = workload_config[workload_id]
        comparisons.append({
            "seq": seq,
            "workload_id": workload_id,
            "input_len": int(config["input_len"]),
            "output_len": int(config["output_len"]),
            "request_rate": float(config["request_rate"]),
            "baseline": {
                "combined_energy_j": baseline_j,
                "combined_avg_power_w": item["combined_avg_power_w"],
                "energy_per_request_j": item["energy_per_request_j"],
                **baseline_metrics,
            },
            "scheduler": {
                "combined_energy_j": scheduler_j,
                "combined_avg_power_w": scheduled["combined_avg_power_w"],
                "energy_per_request_j": scheduled["energy_per_request_j"],
                **scheduler_metrics,
            },
            "scheduler_minus_baseline": {
                "energy_j": scheduler_j - baseline_j,
                "energy_percent": 100 * (scheduler_j - baseline_j) / baseline_j,
                "avg_power_w": scheduled["combined_avg_power_w"] - item["combined_avg_power_w"],
                "energy_per_request_j": scheduled["energy_per_request_j"] - item["energy_per_request_j"],
                "p99_ttft_ms": scheduler_metrics["p99_ttft_ms"] - baseline_metrics["p99_ttft_ms"],
                "p99_tpot_ms": scheduler_metrics["p99_tpot_ms"] - baseline_metrics["p99_tpot_ms"],
            },
        })

    baseline_total_j = baseline["total_recorded_gpu_energy_j"]
    payload = {
        "baseline_policy": "default DVFS, no scheduler, no -lgc",
        "scheduler_policy": "per-workload scheduler rec_freq_mhz applied with -lgc",
        "workload_window_note": "baseline uses explicit command start/end markers; scheduler starts after both clock probes and ends before the next clock change",
        "workloads": comparisons,
        "whole_recorded_process": {
            "baseline_energy_j": baseline_total_j,
            "scheduler_energy_j": scheduler_total_j,
            "scheduler_minus_baseline_j": scheduler_total_j - baseline_total_j,
            "scheduler_minus_baseline_percent": 100 * (scheduler_total_j - baseline_total_j) / baseline_total_j,
            "warning": "whole-process totals include different control/probe overhead; use workload windows for inference comparison",
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = [
        "seq", "workload_id", "input_len", "output_len", "request_rate",
        "baseline_energy_j", "scheduler_energy_j", "scheduler_energy_delta_percent",
        "baseline_avg_power_w", "scheduler_avg_power_w",
        "baseline_energy_per_request_j", "scheduler_energy_per_request_j",
        "baseline_p99_ttft_ms", "scheduler_p99_ttft_ms",
        "baseline_p99_tpot_ms", "scheduler_p99_tpot_ms",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            writer.writerow({
                "seq": item["seq"], "workload_id": item["workload_id"],
                "input_len": item["input_len"], "output_len": item["output_len"],
                "request_rate": item["request_rate"],
                "baseline_energy_j": item["baseline"]["combined_energy_j"],
                "scheduler_energy_j": item["scheduler"]["combined_energy_j"],
                "scheduler_energy_delta_percent": item["scheduler_minus_baseline"]["energy_percent"],
                "baseline_avg_power_w": item["baseline"]["combined_avg_power_w"],
                "scheduler_avg_power_w": item["scheduler"]["combined_avg_power_w"],
                "baseline_energy_per_request_j": item["baseline"]["energy_per_request_j"],
                "scheduler_energy_per_request_j": item["scheduler"]["energy_per_request_j"],
                "baseline_p99_ttft_ms": item["baseline"]["p99_ttft_ms"],
                "scheduler_p99_ttft_ms": item["scheduler"]["p99_ttft_ms"],
                "baseline_p99_tpot_ms": item["baseline"]["p99_tpot_ms"],
                "scheduler_p99_tpot_ms": item["scheduler"]["p99_tpot_ms"],
            })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
