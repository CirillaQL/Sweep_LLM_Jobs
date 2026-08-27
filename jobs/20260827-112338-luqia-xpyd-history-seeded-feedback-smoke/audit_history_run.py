#!/usr/bin/env python3
"""Cross-check decisions, physical outcomes, readbacks, and live clock traces."""

import argparse
import csv
import json
from pathlib import Path


ENDPOINTS = ("P0", "P1", "D0", "D1")


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def monitor_rows(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.run_dir / "history_control_audit.json").read_text())
    decisions = csv_rows(args.run_dir / "history_decisions.csv")
    outcomes = csv_rows(args.run_dir / "history_actual_outcomes.csv")
    actions = csv_rows(args.run_dir / "raw/actuator_actions.csv")
    samples = monitor_rows(args.monitor)
    decision_ids = {row["control_iteration_id"] for row in decisions}
    clock_checks = []
    for outcome in outcomes:
        start = float(outcome["timing_start_unix_s"])
        end = float(outcome["timing_end_unix_s"])
        requested = json.loads(outcome["planned_frequencies_json"])
        for endpoint in ENDPOINTS:
            matching = [row for row in samples if row.get("status") == "success"
                        and row.get("endpoint_id") == endpoint
                        and start <= float(row["timestamp_wall_s"]) <= end]
            clock_checks.append({
                "window_id": outcome["window_id"], "endpoint_id": endpoint,
                "requested_mhz": int(requested[endpoint]),
                "sample_count": len(matching),
                "matching_sample_count": sum(
                    int(row["graphics_clock_mhz"]) == int(requested[endpoint]) for row in matching
                ),
            })
    gates = {
        "native_audit_valid": bool(audit.get("valid")),
        "decision_and_outcome_counts_match": bool(decisions) and len(decisions) == len(outcomes),
        "every_outcome_has_decision": bool(outcomes) and all(
            row["control_iteration_id"] in decision_ids for row in outcomes
        ),
        "every_window_reports_slo_and_energy": bool(outcomes) and all(
            row.get("slo_pass") not in (None, "")
            and float(row["joules_per_request"]) > 0 for row in outcomes
        ),
        "phase3c_clock_match_gate_passed": bool(outcomes) and all(
            float(row[f"{endpoint}_frequency_match_fraction"]) >= 0.95
            for row in outcomes for endpoint in ENDPOINTS
        ),
        "all_dvfs_commands_have_readback": all(
            row["command_status"] == "success" and row["readback_valid"].lower() == "true"
            for row in actions
        ),
        "live_monitor_has_no_errors": bool(samples) and all(
            row.get("status") == "success" for row in samples
        ),
        "live_monitor_observed_every_requested_window_clock": bool(clock_checks) and all(
            row["sample_count"] > 0 and row["matching_sample_count"] > 0 for row in clock_checks
        ),
    }
    report = {"valid": all(gates.values()), "hard_gates": gates,
              "decision_count": len(decisions), "outcome_count": len(outcomes),
              "actuator_action_count": len(actions), "live_sample_count": len(samples),
              "window_clock_checks": clock_checks}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(20)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
