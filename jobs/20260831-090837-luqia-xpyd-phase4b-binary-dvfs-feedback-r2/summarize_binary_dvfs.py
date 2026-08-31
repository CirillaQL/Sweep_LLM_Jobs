#!/usr/bin/env python3
"""Reduce cache-resident binary-DVFS evidence to compact Git artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ENDPOINTS = ("P0", "P1", "D0", "D1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    native_audit_path = args.run_dir / "binary_dvfs_audit.json"
    decisions_path = args.run_dir / "binary_search_decisions.csv"
    windows_path = args.run_dir / "window_results.csv"
    actions_path = args.run_dir / "raw" / "actuator_actions.csv"
    grids_path = args.run_dir / "frequency_grids.json"
    selected_path = args.run_dir / "selected_frequencies.json"
    native = json.loads(native_audit_path.read_text(encoding="utf-8"))
    decisions = read_csv(decisions_path)
    windows = read_csv(windows_path)
    actions = read_csv(actions_path)
    grids = json.loads(grids_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    monitor_samples: list[dict[str, Any]] = []
    for path in args.monitor:
        with path.open(encoding="utf-8") as stream:
            monitor_samples.extend(
                json.loads(line) for line in stream if line.strip()
            )
    successful_samples = [
        row for row in monitor_samples if row.get("status") == "success"
    ]

    action_checks = []
    for row in actions:
        timestamp = float(row["timestamp_unix_s"])
        endpoint_id = row["endpoint_id"]
        target = int(row["requested_freq_mhz"])
        matches = [
            sample for sample in successful_samples
            if sample["endpoint_id"] == endpoint_id
            and timestamp <= float(sample["timestamp_wall_s"]) <= timestamp + 10.0
            and int(sample["graphics_clock_mhz"]) == target
        ]
        action_checks.append({
            "decision_id": row["decision_id"], "phase": row["phase"],
            "endpoint_id": endpoint_id, "target_mhz": target,
            "command_readback_valid": (
                row["command_status"] == "success" and bool_value(row["readback_valid"])
            ),
            "independent_clock_match": bool(matches),
            "first_independent_match_unix_s": (
                matches[0]["timestamp_wall_s"] if matches else None
            ),
        })

    compact_windows = []
    independent_window_valid = []
    for row in windows:
        start = float(row["window_start_unix_s"])
        end = float(row["window_end_unix_s"])
        fractions: dict[str, float] = {}
        for endpoint_id in ENDPOINTS:
            target = int(row["%s_requested_freq_mhz" % endpoint_id])
            samples = [
                sample for sample in successful_samples
                if sample["endpoint_id"] == endpoint_id
                and start <= float(sample["timestamp_wall_s"]) <= end
            ]
            fractions[endpoint_id] = (
                sum(int(sample["graphics_clock_mhz"]) == target for sample in samples)
                / len(samples) if samples else 0.0
            )
        independent_valid = all(value >= 0.95 for value in fractions.values())
        independent_window_valid.append(independent_valid)
        compact_windows.append({
            "window_id": row["window_id"], "workload": row["workload"],
            "phase": row["phase"], "axis": row["axis"],
            "repeat": int(row["repeat"]),
            "confirmation_attempt": int(row["confirmation_attempt"]),
            "P0_frequency_mhz": int(row["P0_requested_freq_mhz"]),
            "P1_frequency_mhz": int(row["P1_requested_freq_mhz"]),
            "D0_frequency_mhz": int(row["D0_requested_freq_mhz"]),
            "D1_frequency_mhz": int(row["D1_requested_freq_mhz"]),
            "configured_max_concurrency": int(row["configured_max_concurrency"]),
            "observed_peak_concurrency": int(row["observed_peak_concurrency"]),
            "concurrency_valid": bool_value(row["concurrency_valid"]),
            "offered_rate_rps": float(row["offered_rate_rps"]),
            "achieved_throughput_rps": float(row["achieved_throughput_rps"]),
            "slo_goodput_rps": float(row["slo_goodput_rps"]),
            "joules_per_request": float(row["joules_per_request"]),
            "p99_ttft_ms": float(row["p99_ttft_ms"]),
            "p99_tpot_ms": float(row["p99_tpot_ms"]),
            "slo_violation_count": int(row["slo_violation_count"]),
            "measurement_valid": bool_value(row["measurement_valid"]),
            "route_valid": bool_value(row["route_valid"]),
            "route_distribution_json": row["route_distribution_json"],
            "internal_clock_valid": bool_value(row["clocks_valid"]),
            "independent_clock_match_min_fraction": min(fractions.values()),
            "independent_clock_valid": independent_valid,
        })

    summary_rows = []
    for workload, choice in sorted(selected.items()):
        baseline = [
            row for row in compact_windows
            if row["workload"] == workload and row["phase"] == "baseline_high"
        ]
        confirmation = [
            row for row in compact_windows
            if row["workload"] == workload and row["phase"] == "confirmation"
            and row["confirmation_attempt"] == int(choice["confirmation_attempt"])
        ]
        baseline_energy = [row["joules_per_request"] for row in baseline]
        chosen_energy = [row["joules_per_request"] for row in confirmation]
        baseline_mean = mean(baseline_energy)
        chosen_mean = mean(chosen_energy)
        summary_rows.append({
            "workload": workload,
            "P_pool_selected_frequency_mhz": int(choice["P_pool_frequency_mhz"]),
            "D_pool_selected_frequency_mhz": int(choice["D_pool_frequency_mhz"]),
            "P0_grid_index": int(choice["P0_grid_index"]),
            "D0_grid_index": int(choice["D0_grid_index"]),
            "baseline_high_joules_per_request_mean": baseline_mean,
            "baseline_high_joules_per_request_std": (
                stdev(baseline_energy) if len(baseline_energy) > 1 else 0.0
            ),
            "selected_joules_per_request_mean": chosen_mean,
            "selected_joules_per_request_std": (
                stdev(chosen_energy) if len(chosen_energy) > 1 else 0.0
            ),
            "energy_savings_vs_high_fraction": (baseline_mean - chosen_mean) / baseline_mean,
            "confirmation_p99_ttft_ms_max": max(
                row["p99_ttft_ms"] for row in confirmation
            ),
            "confirmation_p99_tpot_ms_max": max(
                row["p99_tpot_ms"] for row in confirmation
            ),
            "confirmation_slo_violation_count": sum(
                row["slo_violation_count"] for row in confirmation
            ),
            "confirmation_observed_peak_concurrency_min": min(
                row["observed_peak_concurrency"] for row in confirmation
            ),
            "confirmation_achieved_throughput_rps_mean": mean(
                row["achieved_throughput_rps"] for row in confirmation
            ),
            "confirmation_slo_goodput_rps_mean": mean(
                row["slo_goodput_rps"] for row in confirmation
            ),
            "all_confirmation_routes_balanced": bool(confirmation) and all(
                row["route_valid"] for row in confirmation
            ),
            "confirmation_window_count": len(confirmation),
            "confirmation_attempt": int(choice["confirmation_attempt"]),
            "all_confirmation_evidence_valid": bool(confirmation) and all(
                row["measurement_valid"] and row["route_valid"]
                and row["concurrency_valid"] and row["internal_clock_valid"]
                and row["independent_clock_valid"]
                for row in confirmation
            ),
        })

    gates = {
        "native_binary_dvfs_audit_valid": native.get("valid") is True,
        "four_workloads_selected": len(summary_rows) == 4,
        "prefill_grid_has_at_least_ten_levels": len(grids["P0"]) >= 10,
        "decode_grid_has_at_least_ten_levels": len(grids["D0"]) >= 10,
        "every_actuation_has_command_readback_and_independent_match": (
            bool(action_checks) and all(
                row["command_readback_valid"] and row["independent_clock_match"]
                for row in action_checks
            )
        ),
        "every_physical_window_has_independent_target_clock_match": (
            bool(independent_window_valid) and all(independent_window_valid)
        ),
        "every_physical_window_has_real_concurrency_and_balanced_routes": (
            bool(compact_windows) and all(
                row["concurrency_valid"] and row["route_valid"]
                for row in compact_windows
            )
        ),
        "every_final_confirmation_is_slo_safe": bool(summary_rows) and all(
            row["confirmation_slo_violation_count"] == 0
            and row["all_confirmation_evidence_valid"] for row in summary_rows
        ),
        "binary_decision_trace_present": bool(decisions),
        "raw_logs_outside_git_job_tree": (
            str(args.run_dir).startswith("/data/users/chjing/vllm_job_work/")
        ),
    }
    artifacts = [native_audit_path, decisions_path, windows_path, actions_path,
                 grids_path, selected_path, *args.monitor]
    manifest = [{
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "storage": "cache_only" if path not in (grids_path, selected_path) else "cache_source",
    } for path in artifacts]
    result = {
        "phase": "4B_fine_grained_binary_feedback_DVFS",
        "valid": all(gates.values()), "hard_gates": gates,
        "raw_artifact_root": str(args.run_dir),
        "raw_artifact_manifest": manifest,
        "topology": "2P2D: Uranus L40S P0/P1 + Ganymede L4 D0/D1",
        "routing": "frozen deterministic round-robin",
        "frozen_routes": ["P0->D0", "P1->D1"],
        "max_concurrency": 4,
        "slo": {"ttft_ms": 500.0, "tpot_ms": 200.0},
        "controller": "online coordinate lower-bound binary search; no model/oracle",
        "frequency_grids_mhz": grids,
        "workload_results": summary_rows,
        "decision_count": len(decisions),
        "physical_window_count": len(compact_windows),
        "independent_clock_sample_count": len(successful_samples),
        "actuation_count": len(actions),
    }
    write_json(args.output_dir / "binary_dvfs_final_summary.json", result)
    write_json(args.output_dir / "raw_artifact_manifest.json", manifest)
    write_json(args.output_dir / "frequency_grids.json", grids)
    write_json(args.output_dir / "selected_frequencies.json", selected)
    write_csv(args.output_dir / "workload_summary.csv", summary_rows)
    write_csv(args.output_dir / "decision_trace.csv", decisions)
    write_csv(args.output_dir / "window_statistics.csv", compact_windows)
    write_csv(args.output_dir / "actuation_clock_audit.csv", action_checks)

    lines = [
        "# Fine-grained binary-feedback DVFS result", "",
        "Verdict: **%s**" % ("PASS" if result["valid"] else "FAIL"), "",
        "Raw logs remain at `%s`; this Git result contains only compact statistics and audits."
        % args.run_dir, "",
        "| Workload | P-pool MHz | D-pool MHz | High J/req | Selected J/req | Savings | Min peak concurrency | Goodput req/s | Max p99 TTFT | Max p99 TPOT | Violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {workload} | {P_pool_selected_frequency_mhz} | {D_pool_selected_frequency_mhz} | "
            "{baseline_high_joules_per_request_mean:.3f} | {selected_joules_per_request_mean:.3f} | "
            "{energy_savings_vs_high_fraction:.2%} | {confirmation_observed_peak_concurrency_min} | "
            "{confirmation_slo_goodput_rps_mean:.3f} | {confirmation_p99_ttft_ms_max:.2f} | "
            "{confirmation_p99_tpot_ms_max:.2f} | {confirmation_slo_violation_count} |".format(**row)
        )
    lines.extend(["", "Hard gates: `%s`" % json.dumps(gates, sort_keys=True), ""])
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["valid"]:
        raise SystemExit("fine-grained binary-DVFS post-run audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
