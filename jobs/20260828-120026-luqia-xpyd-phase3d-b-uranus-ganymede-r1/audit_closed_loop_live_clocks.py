#!/usr/bin/env python3
"""Cross-check native Phase 3D-B evidence against independent NVML clock traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_ENDPOINTS = {
    "P0": ("uranus", 0, "NVIDIA L40S", 9001),
    "P1": ("uranus", 1, "NVIDIA L40S", 9001),
    "D0": ("ganymede", 0, "NVIDIA L4", 6251),
    "D1": ("ganymede", 1, "NVIDIA L4", 6251),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_samples(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-actuator-audit", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_audit = read_json(args.run_dir / "closed_loop_audit.json")
    prerequisite = read_json(args.accepted_actuator_audit)
    config = read_json(args.config)
    iterations = read_csv(args.run_dir / "control_iterations.csv")
    actions = read_csv(args.run_dir / "dvfs_actions.csv")
    samples = read_samples(args.monitor)

    configured = {
        row["endpoint_id"]: (row["node"], int(row["gpu_ids"][0]), row["gpu_type"])
        for row in config["endpoints"]
    }
    expected_configured = {
        endpoint: values[:3] for endpoint, values in EXPECTED_ENDPOINTS.items()
    }

    endpoint_evidence: dict[str, dict[str, Any]] = {}
    identity_valid = True
    sampling_valid = True
    no_sample_errors = True
    for endpoint, (node, gpu_index, gpu_name, memory_target) in EXPECTED_ENDPOINTS.items():
        endpoint_rows = [row for row in samples if row.get("endpoint_id") == endpoint]
        success = [row for row in endpoint_rows if row.get("status") == "success"]
        errors = [row for row in endpoint_rows if row.get("status") != "success"]
        timestamps = sorted(float(row["timestamp_wall_s"]) for row in success)
        maximum_gap = max(
            (right - left for left, right in zip(timestamps, timestamps[1:])),
            default=None,
        )
        endpoint_identity = bool(success) and all(
            row.get("node") == node
            and int(row.get("gpu_index", -1)) == gpu_index
            and row.get("gpu_name") == gpu_name
            for row in success
        )
        endpoint_sampling = len(success) >= 10 and maximum_gap is not None and maximum_gap <= 1.0
        identity_valid = identity_valid and endpoint_identity
        sampling_valid = sampling_valid and endpoint_sampling
        no_sample_errors = no_sample_errors and not errors
        endpoint_evidence[endpoint] = {
            "sample_count": len(endpoint_rows),
            "successful_sample_count": len(success),
            "error_sample_count": len(errors),
            "maximum_sampling_gap_s": maximum_gap,
            "identity_valid": endpoint_identity,
            "sampling_valid": endpoint_sampling,
            "memory_target_mhz": memory_target,
        }

    action_evidence: list[dict[str, Any]] = []
    action_readbacks_valid = True
    every_action_observed_live = True
    for index, action in enumerate(actions):
        endpoint = action["endpoint_id"]
        started = float(action["timestamp_unix_s"])
        target = int(action["requested_freq_mhz"])
        next_started = min(
            (
                float(other["timestamp_unix_s"])
                for other in actions[index + 1 :]
                if other["endpoint_id"] == endpoint
            ),
            default=started + 10.0,
        )
        deadline = min(next_started, started + 10.0)
        memory_target = EXPECTED_ENDPOINTS[endpoint][3]
        matching = [
            row for row in samples
            if row.get("status") == "success"
            and row.get("endpoint_id") == endpoint
            and started <= float(row["timestamp_wall_s"]) <= deadline
            and int(row["graphics_clock_mhz"]) == target
            and int(row["memory_clock_mhz"]) == memory_target
        ]
        readback_valid = (
            action.get("command_status") == "success"
            and action.get("readback_valid", "").lower() == "true"
            and int(action["observed_freq_after_mhz"]) == target
        )
        observed_live = bool(matching)
        action_readbacks_valid = action_readbacks_valid and readback_valid
        every_action_observed_live = every_action_observed_live and observed_live
        action_evidence.append({
            "action_index": index,
            "endpoint_id": endpoint,
            "reason": action["reason"],
            "target_graphics_mhz": target,
            "target_memory_mhz": memory_target,
            "readback_valid": readback_valid,
            "independent_matching_sample_count": len(matching),
            "observed_live": observed_live,
        })

    controller_actions = [
        row for row in actions if row["reason"] != "final_safe_high_restoration"
    ]
    feedback_iterations = [
        row for row in iterations[1:]
        if not row.get("fallback_reason")
        and row.get("telemetry_fresh", "").lower() == "true"
    ]
    feedback_single_route = any(
        len(json.loads(row["selected_pairs_json"])) == 1 for row in feedback_iterations
    )

    gates = {
        "accepted_phase3d_a_valid": bool(prerequisite.get("valid")),
        "native_closed_loop_audit_valid": bool(native_audit.get("valid")),
        "configured_topology_is_uranus_ganymede": configured == expected_configured,
        "continuous_monitor_identity": identity_valid,
        "continuous_monitor_no_errors": no_sample_errors,
        "continuous_sampling_gap": sampling_valid,
        "all_native_action_readbacks_valid": bool(actions) and action_readbacks_valid,
        "every_action_target_observed_by_independent_monitor": (
            bool(actions) and every_action_observed_live
        ),
        "controller_performed_physical_dvfs_action": bool(controller_actions),
        "fresh_feedback_selected_single_route": bool(feedback_iterations) and feedback_single_route,
        "light_moderate_light_and_post_control_recorded": len(iterations) == 4,
    }
    report = {
        "phase": "3D-B_feedback_only_closed_loop_plus_independent_live_clock_validation",
        "valid": all(gates.values()),
        "hard_gates": gates,
        "control_iteration_count": len(iterations),
        "feedback_iteration_count": len(feedback_iterations),
        "actuator_action_count": len(actions),
        "controller_action_count": len(controller_actions),
        "endpoint_evidence": endpoint_evidence,
        "action_evidence": action_evidence,
        "claim_boundary": "functional feedback-only validation; no energy-optimality claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
