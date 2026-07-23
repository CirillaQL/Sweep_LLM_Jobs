#!/usr/bin/env python3
"""Join live gate measurements with the prior no-gate and DVFS controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def bench_metrics(path: Path) -> dict:
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


def percent_delta(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--reference-comparison", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    gate_energy = json.loads((args.out_dir / "energy_summary.json").read_text(encoding="utf-8"))
    energy_by_id = {item["workload_id"]: item for item in gate_energy["workloads"]}
    reference = json.loads(args.reference_comparison.read_text(encoding="utf-8"))
    reference_by_id = {item["workload_id"]: item for item in reference["workloads"]}
    with args.workloads.open(newline="", encoding="utf-8") as handle:
        workloads = list(csv.DictReader(handle))

    rows = []
    admitted_gate_energy = 0.0
    admitted_baseline_energy = 0.0
    admitted_nogate_energy = 0.0
    for seq, workload in enumerate(workloads, start=1):
        workload_id = workload["workload_id"]
        request_rate = float(workload["request_rate"])
        gated = json.loads(
            (args.out_dir / f"decision_gate_{seq}_{workload_id}.json").read_text(encoding="utf-8")
        )
        no_gate_decision = json.loads(
            (args.out_dir / f"decision_latency_{seq}_{workload_id}.json").read_text(encoding="utf-8")
        )
        ref = reference_by_id[workload_id]
        no_gate_actual = ref["scheduler"]
        achieved_ratio = no_gate_actual["request_throughput_rps"] / request_rate
        row = {
            "seq": seq,
            "workload_id": workload_id,
            "input_len": int(workload["input_len"]),
            "output_len": int(workload["output_len"]),
            "request_rate": request_rate,
            "gate_status": gated["status"],
            "gate_num_latency_safe": gated["num_latency_safe"],
            "gate_num_saturation_safe": gated["num_saturation_safe"],
            "gate_num_safe": gated["num_safe"],
            "historical_no_gate_throughput_ratio": achieved_ratio,
            "historical_no_gate_measured_saturated": achieved_ratio < 0.95,
            "historical_no_gate_energy_j": no_gate_actual["combined_energy_j"],
            "default_dvfs_energy_j": ref["baseline"]["combined_energy_j"],
        }
        no_gate_rec = no_gate_decision["recommended"]
        row.update({
            "no_gate_prefill_freq_mhz": no_gate_rec["prefill"]["freq_mhz"],
            "no_gate_decode_freq_mhz": no_gate_rec["decode"]["freq_mhz"],
            "no_gate_prefill_p_saturated": no_gate_rec["prefill"]["p_saturated"],
            "no_gate_decode_p_saturated": no_gate_rec["decode"]["p_saturated"],
        })

        if gated["status"] in {"OK", "OVERLOAD_FALLBACK"}:
            measured = energy_by_id[workload_id]
            metrics = bench_metrics(args.out_dir / f"bench_{seq}_{workload_id}.txt")
            rec = gated["recommended"]
            energy_j = measured["combined_energy_j"]
            row.update({
                "gate_admitted": True,
                "gate_overload_fallback": gated["status"] == "OVERLOAD_FALLBACK",
                "gate_predicted_overload_violation_probability": rec[
                    "predicted_overload_violation_probability"
                ],
                "gate_prefill_freq_mhz": rec["prefill"]["freq_mhz"],
                "gate_decode_freq_mhz": rec["decode"]["freq_mhz"],
                "gate_prefill_p_saturated": rec["prefill"]["p_saturated"],
                "gate_decode_p_saturated": rec["decode"]["p_saturated"],
                "gate_energy_j": energy_j,
                "gate_avg_power_w": measured["combined_avg_power_w"],
                "gate_energy_per_request_j": measured["energy_per_request_j"],
                "gate_energy_vs_default_pct": percent_delta(
                    energy_j, ref["baseline"]["combined_energy_j"]
                ),
                "gate_energy_vs_no_gate_pct": percent_delta(
                    energy_j, no_gate_actual["combined_energy_j"]
                ),
                **{f"gate_{key}": value for key, value in metrics.items()},
            })
            admitted_gate_energy += energy_j
            admitted_baseline_energy += ref["baseline"]["combined_energy_j"]
            admitted_nogate_energy += no_gate_actual["combined_energy_j"]
        else:
            row.update({
                "gate_admitted": False,
                "gate_rejection_reason": "no candidate passed latency and p_saturated < 0.30",
            })
        rows.append(row)

    payload = {
        "policy": "latency_plus_saturation",
        "saturation_threshold": 0.30,
        "saturation_model": "five-seed mean of per-GPU local_retrained_75 classifiers",
        "measured_power_scope": "two GPU board-power readings only",
        "workloads": rows,
        "admitted_workload_aggregate": {
            "gate_energy_j": admitted_gate_energy,
            "default_dvfs_energy_j": admitted_baseline_energy,
            "historical_no_gate_energy_j": admitted_nogate_energy,
            "gate_vs_default_percent": percent_delta(admitted_gate_energy, admitted_baseline_energy),
            "gate_vs_historical_no_gate_percent": percent_delta(
                admitted_gate_energy, admitted_nogate_energy
            ),
            "warning": "historical controls are separate runs; use per-workload results and repeat for confidence intervals",
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
