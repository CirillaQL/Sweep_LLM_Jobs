#!/usr/bin/env python3
"""Integrate two-node GPU board power for marked vLLM benchmark windows."""

import argparse
import csv
import json
import re
from pathlib import Path


def load_telemetry(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row["unix_ts"]), float(row["gpu_power_w"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        raise ValueError(f"insufficient telemetry: {path}")
    return sorted(rows)


def integrate(samples, start=None, end=None):
    if start is None:
        start = samples[0][0]
    if end is None:
        end = samples[-1][0]
    energy_j = 0.0
    covered_s = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if t1 <= t0 or t1 - t0 > 5:
            continue
        left = max(t0, start)
        right = min(t1, end)
        if right <= left:
            continue
        slope = (p1 - p0) / (t1 - t0)
        p_left = p0 + slope * (left - t0)
        p_right = p0 + slope * (right - t0)
        energy_j += 0.5 * (p_left + p_right) * (right - left)
        covered_s += right - left
    return energy_j, covered_s


def probe_energy(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for row in data.get("samples", []):
        try:
            samples.append((float(row["unix_ts"]), float(row["power_w"])))
        except (KeyError, TypeError, ValueError):
            pass
    return integrate(sorted(samples))[0] if len(samples) >= 2 else 0.0


def benchmark_requests(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Successful requests:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    telemetry_paths = {
        "neptune_l40s": args.out_dir / "prefill_neptune_telemetry.csv",
        "ganymede_l4": args.out_dir / "decode_ganymede_telemetry.csv",
    }
    telemetry = {name: load_telemetry(path) for name, path in telemetry_paths.items()}

    events = []
    with (args.out_dir / "events.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["unix_ts"] = float(row["unix_ts"])
            row["seq"] = int(row["seq"])
            events.append(row)

    windows = {}
    for event in events:
        key = (event["seq"], event["workload_id"])
        if event["event"] == "workload_start":
            windows.setdefault(key, {})["start"] = event["unix_ts"]
        elif event["event"] == "workload_end":
            windows.setdefault(key, {})["end"] = event["unix_ts"]

    workloads = []
    for (seq, workload_id), window in sorted(windows.items()):
        if "start" not in window or "end" not in window:
            raise ValueError(f"incomplete event window: {(seq, workload_id)} {window}")
        duration_s = window["end"] - window["start"]
        node_energy = {}
        for node, samples in telemetry.items():
            energy_j, covered_s = integrate(samples, window["start"], window["end"])
            node_energy[node] = {
                "energy_j": energy_j,
                "energy_wh": energy_j / 3600,
                "covered_s": covered_s,
            }
        combined_j = sum(item["energy_j"] for item in node_energy.values())
        bench_path = args.out_dir / f"bench_{seq}_{workload_id}.txt"
        requests = benchmark_requests(bench_path)
        workloads.append({
            "seq": seq,
            "workload_id": workload_id,
            "start_unix_ts": window["start"],
            "end_unix_ts": window["end"],
            "duration_s": duration_s,
            "nodes": node_energy,
            "combined_energy_j": combined_j,
            "combined_energy_wh": combined_j / 3600,
            "combined_avg_power_w": combined_j / duration_s,
            "successful_requests": requests,
            "energy_per_request_j": combined_j / requests if requests else None,
        })

    telemetry_total = {}
    for node, samples in telemetry.items():
        energy_j, covered_s = integrate(samples)
        telemetry_total[node] = {
            "energy_j": energy_j,
            "energy_wh": energy_j / 3600,
            "covered_s": covered_s,
            "average_power_w": energy_j / covered_s,
        }
    telemetry_total_j = sum(item["energy_j"] for item in telemetry_total.values())
    reset_energy_j = sum(
        probe_energy(path)
        for path in sorted(args.out_dir.glob("reset_*_probe.json"))
    )
    payload = {
        "scope": "two allocated GPU board-power telemetry; excludes CPU, RAM, NIC, and fans",
        "integration": "piecewise-linear trapezoidal integration",
        "sample_period_s": 0.5,
        "workloads": workloads,
        "telemetry_total": telemetry_total,
        "telemetry_total_energy_j": telemetry_total_j,
        "telemetry_total_energy_wh": telemetry_total_j / 3600,
        "reset_probe_energy_j": reset_energy_j,
        "reset_probe_energy_wh": reset_energy_j / 3600,
        "total_recorded_gpu_energy_j": telemetry_total_j + reset_energy_j,
        "total_recorded_gpu_energy_wh": (telemetry_total_j + reset_energy_j) / 3600,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = [
        "seq", "workload_id", "duration_s", "successful_requests",
        "neptune_energy_j", "ganymede_energy_j", "combined_energy_j",
        "combined_energy_wh", "combined_avg_power_w", "energy_per_request_j",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in workloads:
            writer.writerow({
                "seq": item["seq"],
                "workload_id": item["workload_id"],
                "duration_s": round(item["duration_s"], 6),
                "successful_requests": item["successful_requests"],
                "neptune_energy_j": round(item["nodes"]["neptune_l40s"]["energy_j"], 6),
                "ganymede_energy_j": round(item["nodes"]["ganymede_l4"]["energy_j"], 6),
                "combined_energy_j": round(item["combined_energy_j"], 6),
                "combined_energy_wh": round(item["combined_energy_wh"], 9),
                "combined_avg_power_w": round(item["combined_avg_power_w"], 6),
                "energy_per_request_j": round(item["energy_per_request_j"], 6),
            })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
