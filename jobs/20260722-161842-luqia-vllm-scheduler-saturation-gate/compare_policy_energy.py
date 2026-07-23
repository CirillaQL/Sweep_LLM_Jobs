#!/usr/bin/env python3
"""Compare per-window energy and SLO outcomes for two identical request traces."""

import argparse
import csv
import json
from pathlib import Path


def load(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["workload_id"]: row for row in csv.DictReader(handle)}


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduled", type=Path, required=True)
    parser.add_argument("--auto", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    scheduled = load(args.scheduled)
    auto = load(args.auto)
    rows = []
    for workload_id, auto_row in auto.items():
        scheduled_row = scheduled[workload_id]
        scheduled_energy = number(scheduled_row, "combined_energy_j")
        auto_energy = number(auto_row, "combined_energy_j")
        delta = auto_energy - scheduled_energy
        rows.append({
            "seq": int(auto_row["seq"]),
            "workload_id": workload_id,
            "scheduled_energy_j": scheduled_energy,
            "auto_dvfs_energy_j": auto_energy,
            "scheduled_saving_j": delta,
            "scheduled_saving_pct": 100 * delta / auto_energy,
            "scheduled_avg_power_w": number(scheduled_row, "combined_avg_power_w"),
            "auto_dvfs_avg_power_w": number(auto_row, "combined_avg_power_w"),
            "scheduled_p99_ttft_ms": number(scheduled_row, "actual_p99_ttft_ms"),
            "auto_dvfs_p99_ttft_ms": number(auto_row, "actual_p99_ttft_ms"),
            "scheduled_p99_tpot_ms": number(scheduled_row, "actual_p99_tpot_ms"),
            "auto_dvfs_p99_tpot_ms": number(auto_row, "actual_p99_tpot_ms"),
            "scheduled_slo_ok": (
                scheduled_row.get("actual_ttft_slo_ok") == "True"
                and scheduled_row.get("actual_tpot_slo_ok") == "True"
            ),
            "auto_dvfs_slo_ok": (
                auto_row.get("actual_ttft_slo_ok") == "True"
                and auto_row.get("actual_tpot_slo_ok") == "True"
            ),
        })
    rows.sort(key=lambda row: row["seq"])
    scheduled_total = sum(row["scheduled_energy_j"] for row in rows)
    auto_total = sum(row["auto_dvfs_energy_j"] for row in rows)
    payload = {
        "comparison": "identical finite-rate Poisson request trace",
        "scheduled_total_energy_j": scheduled_total,
        "auto_dvfs_total_energy_j": auto_total,
        "scheduled_saving_j": auto_total - scheduled_total,
        "scheduled_saving_pct": 100 * (auto_total - scheduled_total) / auto_total,
        "scheduled_slo_pass_windows": sum(row["scheduled_slo_ok"] for row in rows),
        "auto_dvfs_slo_pass_windows": sum(row["auto_dvfs_slo_ok"] for row in rows),
        "workloads": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
