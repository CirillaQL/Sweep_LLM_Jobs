#!/usr/bin/env python3
"""Audit Phase 4B energy, route, and independent live-clock evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.run_dir / "phase4b_audit.json").read_text())
    summary = json.loads((args.run_dir / "phase4b_summary.json").read_text())
    traces = read_csv(args.run_dir / "phase4b_feedback_decision_trace.csv")
    actions = read_csv(args.run_dir / "raw" / "actuator_actions.csv")
    samples = []
    for path in args.monitor:
        samples.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    successful_samples = [row for row in samples if row.get("status") == "success"]

    route_checks = []
    frequency_window_checks = []
    for row in traces:
        selected = {"%s->%s" % tuple(pair) for pair in json.loads(row["selected_pairs_json"])}
        actual = json.loads(row["actual_route_distribution_json"] or "{}")
        route_checks.append(bool(actual) and set(actual).issubset(selected))
        frequency_window_checks.append(all(
            float(row[f"{endpoint_id}_frequency_match_fraction"]) >= 0.95
            for endpoint_id in ("P0", "P1", "D0", "D1")
        ))

    controller_actions = [row for row in actions if row.get("controller_action") == "True"]
    live_clock_checks = []
    live_clock_details = []
    for row in controller_actions:
        timestamp = float(row["timestamp_unix_s"])
        endpoint_id = row["endpoint_id"]
        target = int(row["requested_freq_mhz"])
        matching = [
            sample for sample in successful_samples
            if sample["endpoint_id"] == endpoint_id
            and timestamp <= float(sample["timestamp_wall_s"]) <= timestamp + 10.0
            and int(sample["graphics_clock_mhz"]) == target
        ]
        passed = bool(matching)
        live_clock_checks.append(passed)
        live_clock_details.append({
            "control_iteration_id": row["control_iteration_id"],
            "endpoint_id": endpoint_id, "target_mhz": target,
            "matched_independent_sample": passed,
            "first_matching_timestamp": matching[0]["timestamp_wall_s"] if matching else None,
        })

    policy_rows = summary["policy_summaries"]
    full = [row for row in policy_rows if row["policy"] == "FULL_FEEDBACK"]
    savings = [row["energy_savings_vs_static"] for row in full]
    energy_result = {
        "full_feedback_savings_vs_static_by_workload": {
            row["workload"]: row["energy_savings_vs_static"] for row in full
        },
        "full_feedback_mean_savings_vs_static": mean(savings) if savings else None,
        "full_feedback_all_requests_slo_safe": bool(full) and all(row["slo_pass"] for row in full),
        "interpretation": "Measured gross four-GPU energy; positive means lower energy than same-job STATIC.",
    }
    gates = {
        "native_phase4b_audit_valid": audit.get("valid") is True,
        "all_decisions_joined_to_physical_windows": len(traces) == 80,
        "actual_routes_match_feedback_selection": bool(route_checks) and all(route_checks),
        "all_window_clock_target_match_at_least_95_percent": (
            bool(frequency_window_checks) and all(frequency_window_checks)
        ),
        "every_controller_dvfs_action_has_independent_live_clock_match": (
            bool(controller_actions) and all(live_clock_checks)
        ),
        "static_baseline_present_for_every_workload": len(full) == 4 and all(
            row.get("static_joules_per_request") is not None for row in full
        ),
    }
    result = {
        "phase": "4B-r3_stationary_actual_energy",
        "valid": all(gates.values()), "hard_gates": gates,
        "energy_result": energy_result,
        "decision_trace": str(args.run_dir / "phase4b_feedback_decision_trace.csv"),
        "controller_dvfs_action_count": len(controller_actions),
        "independent_clock_sample_count": len(successful_samples),
        "live_clock_action_checks": live_clock_details,
        "oracle_claim": "none; incompatible prior Neptune/IO oracle was not used",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["valid"]:
        raise SystemExit("Phase 4B-r3 post-run audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
