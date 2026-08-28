#!/usr/bin/env python3
"""Fail-closed post-audit for Phase 3D-A command and live clock evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


EXPECTED_MID_WINDOWS = {
    "p0_mid": "P0",
    "p1_mid": "P1",
    "d0_mid": "D0",
    "d1_mid": "D1",
}
EXPECTED_WINDOWS = ("high_before", *EXPECTED_MID_WINDOWS, "high_after")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_monitor(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    row["source_file"] = str(path)
                    row["source_line"] = line_number
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--fixed-clock-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    actuator_audit = read_json(args.run_dir / "actuator_audit.json")
    capabilities = read_json(args.run_dir / "capabilities.json")
    monitor_rows = load_monitor(args.monitor)
    evidence = args.fixed_clock_evidence.read_text(encoding="utf-8", errors="replace")
    with (args.run_dir / "actuator_actions.csv").open(encoding="utf-8") as stream:
        action_rows = list(csv.DictReader(stream))

    endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
    samples_by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in monitor_rows:
        samples_by_endpoint[str(row.get("endpoint_id"))].append(row)

    endpoint_evidence: dict[str, Any] = {}
    identity_valid = True
    no_monitor_errors = True
    sampling_valid = True
    states_valid = True
    loaded_mid_valid = True
    memory_valid = True
    for endpoint_id, endpoint in endpoints.items():
        rows = samples_by_endpoint.get(endpoint_id, [])
        success = [row for row in rows if row.get("status") == "success"]
        errors = [row for row in rows if row.get("status") != "success"]
        capability = capabilities[endpoint_id]
        expected_uuid = str(capability["gpu_uuid"])
        expected_pci = str(capability["pci_bus_id"])
        endpoint_identity_valid = bool(success) and all(
            str(row.get("gpu_uuid")) == expected_uuid
            and str(row.get("pci_bus_id")) == expected_pci
            and int(row.get("gpu_index", -1)) == int(endpoint["gpu_ids"][0])
            for row in success
        )
        timestamps = sorted(float(row["timestamp_wall_s"]) for row in success)
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        maximum_gap_s = max(gaps, default=0.0)
        endpoint_sampling_valid = len(success) >= 10 and maximum_gap_s <= 0.75

        selected = {
            name: int(capability[f"selected_{name.lower()}_mhz"])
            for name in ("LOW", "MID", "HIGH")
        }
        graphics_counts = Counter(int(row["graphics_clock_mhz"]) for row in success)
        selected_counts = {name: graphics_counts[value] for name, value in selected.items()}
        endpoint_states_valid = all(count >= 2 for count in selected_counts.values())
        loaded_mid_count = sum(
            int(row["graphics_clock_mhz"]) == selected["MID"]
            and int(row.get("gpu_utilization_pct", 0)) > 0
            for row in success
        )
        endpoint_loaded_mid_valid = loaded_mid_count >= 1

        memory_target = int(config["fixed_clocks"][endpoint_id]["memory_mhz"])
        locked_rows = [
            row for row in success
            if int(row["graphics_clock_mhz"]) in set(selected.values())
        ]
        memory_matches = sum(
            int(row["memory_clock_mhz"]) == memory_target for row in locked_rows
        )
        memory_match_fraction = memory_matches / len(locked_rows) if locked_rows else 0.0
        endpoint_memory_valid = len(locked_rows) >= 10 and memory_match_fraction >= 0.95

        identity_valid = identity_valid and endpoint_identity_valid
        no_monitor_errors = no_monitor_errors and not errors
        sampling_valid = sampling_valid and endpoint_sampling_valid
        states_valid = states_valid and endpoint_states_valid
        loaded_mid_valid = loaded_mid_valid and endpoint_loaded_mid_valid
        memory_valid = memory_valid and endpoint_memory_valid
        endpoint_evidence[endpoint_id] = {
            "sample_count": len(rows),
            "successful_sample_count": len(success),
            "error_sample_count": len(errors),
            "identity_valid": endpoint_identity_valid,
            "maximum_sampling_gap_s": maximum_gap_s,
            "sampling_valid": endpoint_sampling_valid,
            "selected_frequencies_mhz": selected,
            "selected_frequency_sample_counts": selected_counts,
            "loaded_mid_sample_count": loaded_mid_count,
            "loaded_mid_valid": endpoint_loaded_mid_valid,
            "memory_target_mhz": memory_target,
            "memory_match_fraction_while_graphics_locked": memory_match_fraction,
            "memory_valid": endpoint_memory_valid,
        }

    window_evidence: dict[str, Any] = {}
    serving_windows_valid = True
    per_endpoint_mid_window_valid = True
    for window_id in EXPECTED_WINDOWS:
        window_dir = args.run_dir / "workload_windows" / window_id
        audit = read_json(window_dir / "audit.json")
        summary = read_json(window_dir / "summary.json")
        endpoint_clocks = summary.get("endpoint_clocks", {})
        window_valid = bool(audit.get("valid")) and all(
            bool(value.get("valid")) for value in endpoint_clocks.values()
        )
        changed_endpoint = EXPECTED_MID_WINDOWS.get(window_id)
        changed_valid = True
        if changed_endpoint is not None:
            expected_mid = int(capabilities[changed_endpoint]["selected_mid_mhz"])
            observed = endpoint_clocks.get(changed_endpoint, {})
            changed_valid = (
                bool(observed.get("valid"))
                and int(observed.get("graphics", {}).get("target_mhz", -1)) == expected_mid
                and float(observed.get("graphics", {}).get("target_match_fraction", 0.0)) >= 0.95
                and float(observed.get("memory", {}).get("target_match_fraction", 0.0)) >= 0.95
            )
        serving_windows_valid = serving_windows_valid and window_valid
        per_endpoint_mid_window_valid = per_endpoint_mid_window_valid and changed_valid
        window_evidence[window_id] = {
            "valid": window_valid,
            "changed_endpoint": changed_endpoint,
            "changed_endpoint_mid_valid": changed_valid,
            "endpoint_clocks": endpoint_clocks,
        }

    command_evidence_valid = True
    for endpoint_id, endpoint in endpoints.items():
        graphics = int(config["fixed_clocks"][endpoint_id]["graphics_mhz"])
        memory = int(config["fixed_clocks"][endpoint_id]["memory_mhz"])
        marker = (
            f"endpoint={endpoint_id} node={endpoint['node']} event=after_lock "
            f"target_graphics_mhz={graphics} target_memory_mhz={memory}"
        )
        command_evidence_valid = command_evidence_valid and marker in evidence
    command_evidence_valid = command_evidence_valid and bool(action_rows) and all(
        row.get("command_status") == "success"
        and row.get("readback_valid") == "True"
        for row in action_rows
    )

    hard_gates = {
        "phase3d_actuator_audit": bool(actuator_audit.get("valid")),
        "sudo_lgc_lmc_command_evidence": command_evidence_valid,
        "continuous_monitor_identity": identity_valid,
        "continuous_monitor_no_errors": no_monitor_errors,
        "continuous_sampling_gap": sampling_valid,
        "all_low_mid_high_states_observed_live": states_valid,
        "each_endpoint_mid_observed_under_load_live": loaded_mid_valid,
        "memory_clock_observed_while_graphics_locked": memory_valid,
        "all_serving_windows_valid": serving_windows_valid,
        "each_endpoint_mid_serving_window_valid": per_endpoint_mid_window_valid,
    }
    report = {
        "phase": "3D-A_sudo_actuator_plus_independent_live_clock_validation",
        "valid": all(hard_gates.values()),
        "hard_gates": hard_gates,
        "endpoint_evidence": endpoint_evidence,
        "window_evidence": window_evidence,
        "actuator_action_count": len(action_rows),
        "claim_boundary": "actuator validation only; no dynamic policy or energy-optimality claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
