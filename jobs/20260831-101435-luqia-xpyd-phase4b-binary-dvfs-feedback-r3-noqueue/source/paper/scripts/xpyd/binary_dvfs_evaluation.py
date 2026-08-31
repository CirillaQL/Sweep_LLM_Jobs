"""Feedback-only fine-grained DVFS search on the real 2P2D substrate.

The experiment intentionally freezes routing to the disjoint balanced routes
P0->D0 and P1->D1.
For every workload it first measures an all-HIGH baseline, then performs a
lower-bound binary search over a hardware-supported P-pool grid while the
D pool stays at HIGH, followed by the same search over the D pool while the
P pool stays at its selected frequency. P0/P1 always share a target, as do
D0/D1. Every probe is a real concurrent request window and every choice is based
only on already completed windows.  No model, oracle, or offline energy table
is used by the controller.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import (
    Phase3CSubstrateHarness,
    _read_json,
    _read_jsonl,
    _write_json,
)
from xpyd.phase3d_control import (
    EndpointClockCapability,
    NvidiaSmiClockBackend,
    PerEndpointClockActuator,
    Phase3DError,
    _phase3c_window_config,
    _csv,
)
from xpyd.phase4b_evaluation import Phase4BEvaluationHarness


class BinaryDVFSError(RuntimeError):
    """Fail-closed configuration, physical actuation, or evidence error."""


ENDPOINTS = ("P0", "P1", "D0", "D1")
ROUTES = (("P0", "D0"), ("P1", "D1"))


def peak_observed_concurrency(requests: Sequence[Mapping[str, Any]]) -> int:
    """Compute actual overlapping client requests from send/complete timestamps."""
    events: list[tuple[float, int]] = []
    for request in requests:
        if not request.get("ok"):
            continue
        if request.get("send_unix_s") is None or request.get("complete_unix_s") is None:
            continue
        events.append((float(request["send_unix_s"]), 1))
        events.append((float(request["complete_unix_s"]), -1))
    active = 0
    maximum = 0
    # Complete before send at an identical timestamp to avoid false overlap.
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def load_client_request_results(window_dir: Path) -> list[dict[str, Any]]:
    """Load the replay client's per-request timing evidence.

    replay_synthetic_trace deliberately removes request_results from the
    aggregate summary and persists them in client/requests.jsonl instead.
    """
    path = window_dir / "client" / "requests.jsonl"
    if not path.is_file():
        raise BinaryDVFSError("client request timing evidence is missing: %s" % path)
    rows = _read_jsonl(path)
    if not rows:
        raise BinaryDVFSError("client request timing evidence is empty: %s" % path)
    return rows


def balanced_frozen_routes_valid(
    route_matrix: Mapping[str, Any], expected_requests: int,
) -> bool:
    """Require only the two frozen routes and deterministic count balance."""
    positive = {
        str(key): int(value) for key, value in route_matrix.items()
        if int(value) > 0
    }
    allowed = {"P0->D0", "P1->D1"}
    counts = [positive.get(route, 0) for route in sorted(allowed)]
    return (
        set(positive).issubset(allowed)
        and sum(counts) == int(expected_requests)
        and min(counts) > 0
        and abs(counts[0] - counts[1]) <= 1
    )


def select_frequency_grid(
    supported: Sequence[int], minimum_mhz: int, maximum_mhz: int, levels: int,
) -> tuple[int, ...]:
    """Select evenly spread, real hardware frequencies including both bounds."""
    if levels < 10:
        raise BinaryDVFSError("fine-grained DVFS requires at least ten levels")
    values = sorted({
        int(value) for value in supported
        if minimum_mhz <= int(value) <= maximum_mhz
    })
    if maximum_mhz not in values:
        raise BinaryDVFSError("configured safe-HIGH is not hardware-supported")
    if len(values) < levels:
        raise BinaryDVFSError(
            "hardware exposes only %d supported points in [%d, %d], need %d"
            % (len(values), minimum_mhz, maximum_mhz, levels)
        )
    indices = [round(index * (len(values) - 1) / (levels - 1)) for index in range(levels)]
    grid = tuple(values[index] for index in indices)
    if len(set(grid)) != levels or tuple(sorted(grid)) != grid:
        raise BinaryDVFSError("could not construct a strictly ordered frequency grid")
    return grid


def binary_update(low: int, high: int, candidate: int, feasible: bool) -> tuple[int, int]:
    """Return lower-bound binary-search bounds after one observed probe."""
    if not (0 <= low <= candidate <= high):
        raise ValueError("candidate must be within binary-search bounds")
    return (low, candidate) if feasible else (candidate + 1, high)


def _percentile_99(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


class BinaryDVFSEvaluationHarness(Phase4BEvaluationHarness):
    def __init__(
        self, config: Mapping[str, Any], run_id: Optional[str] = None,
        backend: Optional[NvidiaSmiClockBackend] = None,
    ) -> None:
        super().__init__(config, run_id=run_id, backend=backend, stage="stationary")
        self.settings = dict(self.config["binary_dvfs"])
        self.run_dir = Path(os.path.expandvars(str(self.settings["output_root"]))) / self.run_id
        self.window_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.action_rows: list[dict[str, Any]] = []
        self.selected: dict[str, dict[str, Any]] = {}
        self._window_sequence = 0

    def _discover_fine_grids(
        self,
    ) -> tuple[dict[str, EndpointClockCapability], dict[str, tuple[int, ...]]]:
        capabilities: dict[str, EndpointClockCapability] = {}
        grids: dict[str, tuple[int, ...]] = {}
        role_grid = self.settings["frequency_grids"]
        for endpoint in self.config["endpoints"]:
            endpoint_id = str(endpoint["endpoint_id"])
            role = "prefill" if endpoint_id.startswith("P") else "decode"
            spec = role_grid[role]
            capability = self.backend.discover(
                endpoint_id, str(endpoint["node"]), int(endpoint["gpu_ids"][0]),
                int(spec["maximum_mhz"]),
            )
            expected_name = str(endpoint.get("expected_gpu_name", endpoint["gpu_type"]))
            if expected_name.lower() not in capability.gpu_name.lower():
                raise BinaryDVFSError("GPU model mismatch for %s" % endpoint_id)
            grid = select_frequency_grid(
                capability.supported_graphics_mhz,
                int(spec["minimum_mhz"]), int(spec["maximum_mhz"]),
                int(spec["levels"]),
            )
            capabilities[endpoint_id] = capability
            grids[endpoint_id] = grid
        if grids["P0"] != grids["P1"] or grids["D0"] != grids["D1"]:
            raise BinaryDVFSError("same-type endpoints exposed different selected frequency grids")
        return capabilities, grids

    def _actuate_targets(
        self, actuator: PerEndpointClockActuator, targets: Mapping[str, int],
        *, workload: str, phase: str, decision_id: str, reason: str,
    ) -> None:
        for endpoint_id in ENDPOINTS:
            target = int(targets[endpoint_id])
            if actuator.requested.get(endpoint_id) == target:
                continue
            row = actuator.actuate(endpoint_id, target, reason)
            row.update({
                "workload": workload, "phase": phase,
                "decision_id": decision_id, "controller_action": phase != "restore",
            })
            self.action_rows.append(row)
            if row["command_status"] != "success" or not row["readback_valid"]:
                raise BinaryDVFSError("physical DVFS actuation failed for %s" % endpoint_id)

    def _run_window(
        self, window_id: str, workload: Mapping[str, Any], count: int,
        requested: Mapping[str, int], pairs: Sequence[Sequence[str]],
        windows_root: Path, *, require_all_pairs: bool = True,
    ) -> Path:
        """Run a concurrent window with both frozen disjoint routes required."""
        spec = dict(workload)
        spec["count"] = int(count)
        window_config = _phase3c_window_config(
            self.config, windows_root, [spec], requested,
            required_pairs=pairs if require_all_pairs else None,
        )
        window_config["server_logs"] = {}
        window_config["client"]["max_concurrency"] = int(
            self.settings["max_concurrency"]
        )
        window_config["client"]["dispatch_mode"] = "closed_loop_batches"
        window_config["client"]["closed_loop_batch_size"] = int(
            self.settings["closed_loop_batch_size"]
        )
        window_config["request_id_namespace"] = "binary-dvfs-%s" % window_id
        Phase3CSubstrateHarness(window_config, run_id=window_id).run()
        return windows_root / window_id

    def _measure_window(
        self, window_id: str, workload: Mapping[str, Any], phase: str,
        repeat: int, targets: Mapping[str, int], window_dir: Path,
        *, axis: str = "", candidate_index: Optional[int] = None,
        confirmation_attempt: int = 0,
    ) -> dict[str, Any]:
        summary = _read_json(window_dir / "summary.json")
        audit = _read_json(window_dir / "audit.json")
        client = _read_json(window_dir / "client" / "summary.json")
        with (window_dir / "requests.csv").open(newline="", encoding="utf-8") as stream:
            requests = list(csv.DictReader(stream))
        ttft = [float(row["ttft_ms"]) for row in requests]
        client_ttft = [float(row["client_ttft_ms"]) for row in requests]
        tpot = [float(row["tpot_ms"]) for row in requests]
        queue_delays = [float(row["client_queue_delay_ms"]) for row in requests]
        ttft_slo = float(self.settings["slo"]["ttft_ms"])
        tpot_slo = float(self.settings["slo"]["tpot_ms"])
        violations = sum(
            value_ttft > ttft_slo or value_tpot > tpot_slo
            for value_ttft, value_tpot in zip(ttft, tpot)
        )
        positive_routes = {
            key: int(value) for key, value in summary["route_matrix"].items()
            if int(value) > 0
        }
        frequency_matches = {
            endpoint_id: float(
                summary["endpoint_clocks"][endpoint_id]["graphics"]["target_match_fraction"]
            ) for endpoint_id in ENDPOINTS
        }
        p99_ttft = _percentile_99(ttft)
        p99_tpot = _percentile_99(tpot)
        completed = int(client["successful_requests"])
        expected = int(self.settings["requests_per_window"])
        measurement_valid = bool(audit.get("valid"))
        route_valid = balanced_frozen_routes_valid(positive_routes, expected)
        clocks_valid = all(value >= 0.95 for value in frequency_matches.values())
        request_results = load_client_request_results(window_dir)
        observed_peak_concurrency = peak_observed_concurrency(request_results)
        concurrency_valid = (
            int(self.settings["minimum_observed_concurrency"])
            <= observed_peak_concurrency
            <= int(self.settings["maximum_observed_concurrency"])
        )
        max_client_queue_delay_ms = max(queue_delays) if queue_delays else math.inf
        client_queue_valid = max_client_queue_delay_ms <= float(
            self.settings["maximum_client_queue_delay_ms"]
        )
        duration_s = (
            float(client["timing_end_unix_s"])
            - float(client["timing_start_unix_s"])
        )
        slo_pass = (
            completed == expected and violations == 0
            and p99_ttft <= ttft_slo and p99_tpot <= tpot_slo
        )
        row = {
            "window_id": window_id, "workload": str(workload["id"]),
            "phase": phase, "axis": axis, "repeat": repeat,
            "confirmation_attempt": confirmation_attempt,
            "candidate_index": candidate_index,
            "P0_requested_freq_mhz": int(targets["P0"]),
            "P1_requested_freq_mhz": int(targets["P1"]),
            "D0_requested_freq_mhz": int(targets["D0"]),
            "D1_requested_freq_mhz": int(targets["D1"]),
            "completed_requests": completed,
            "configured_max_concurrency": int(client["max_concurrency"]),
            "observed_peak_concurrency": observed_peak_concurrency,
            "concurrency_valid": concurrency_valid,
            "dispatch_mode": str(client.get("dispatch_mode")),
            "closed_loop_batch_size": int(client.get("closed_loop_batch_size", 0)),
            "client_queue_valid": client_queue_valid,
            "max_client_queue_delay_ms": max_client_queue_delay_ms,
            "p99_client_queue_delay_ms": _percentile_99(queue_delays),
            "nominal_trace_rate_rps": float(workload["rate_rps"]),
            "offered_rate_rps": None,
            "achieved_throughput_rps": completed / duration_s,
            "slo_goodput_rps": (completed - violations) / duration_s,
            "measurement_valid": measurement_valid,
            "route_valid": route_valid, "clocks_valid": clocks_valid,
            "slo_pass": slo_pass, "slo_violation_count": violations,
            "ttft_measurement_scope": "proxy_request_received_to_first_real_decode_chunk",
            "mean_ttft_ms": mean(ttft),
            "p99_ttft_ms": p99_ttft,
            "max_ttft_ms": max(ttft) if ttft else None,
            "mean_client_ttft_ms": mean(client_ttft),
            "p99_client_ttft_ms": _percentile_99(client_ttft),
            "mean_tpot_ms": mean(tpot),
            "p99_tpot_ms": p99_tpot,
            "max_tpot_ms": max(tpot) if tpot else None,
            "joules_per_request": float(summary["joules_per_request"]),
            "total_gpu_gross_energy_j": float(summary["energy_j"]["total"]),
            "P_pool_energy_j": float(summary["energy_j"]["P_pool"]),
            "D_pool_energy_j": float(summary["energy_j"]["D_pool"]),
            "route_distribution_json": json.dumps(summary["route_matrix"], sort_keys=True),
            "frequency_match_fraction_json": json.dumps(frequency_matches, sort_keys=True),
            "window_start_unix_s": float(client["timing_start_unix_s"]),
            "window_end_unix_s": float(client["timing_end_unix_s"]),
            "hard_gates_json": json.dumps(audit.get("hard_gates", {}), sort_keys=True),
        }
        self.window_rows.append(row)
        return row

    def _run_repeats(
        self, actuator: PerEndpointClockActuator, windows_root: Path,
        workload: Mapping[str, Any], phase: str, targets: Mapping[str, int],
        repeats: int, *, axis: str = "", candidate_index: Optional[int] = None,
        confirmation_attempt: int = 0, decision_id: str,
    ) -> list[dict[str, Any]]:
        self._actuate_targets(
            actuator, targets, workload=str(workload["id"]), phase=phase,
            decision_id=decision_id, reason="binary_feedback:%s" % decision_id,
        )
        self._write_routes(ROUTES, "binary_feedback_frozen_balanced_disjoint_routes")
        rows = []
        for repeat in range(1, repeats + 1):
            self._window_sequence += 1
            window_id = "%03d-%s-%s-r%02d" % (
                self._window_sequence, workload["id"], phase, repeat,
            )
            window_dir = self._run_window(
                window_id, workload, int(self.settings["requests_per_window"]),
                targets, ROUTES, windows_root, require_all_pairs=True,
            )
            rows.append(self._measure_window(
                window_id, workload, phase, repeat, targets, window_dir,
                axis=axis, candidate_index=candidate_index,
                confirmation_attempt=confirmation_attempt,
            ))
        return rows

    @staticmethod
    def _probe_feasible(rows: Sequence[Mapping[str, Any]]) -> bool:
        return bool(rows) and all(
            row["measurement_valid"] and row["route_valid"]
            and row["clocks_valid"] and row["concurrency_valid"]
            and row["client_queue_valid"]
            and row["slo_pass"] for row in rows
        )

    @staticmethod
    def _evidence_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
        """Validate a physical measurement independently of SLO feasibility."""
        return bool(rows) and all(
            row["measurement_valid"] and row["route_valid"]
            and row["clocks_valid"] and row["concurrency_valid"]
            and row["client_queue_valid"] for row in rows
        )

    def _binary_search_axis(
        self, axis: str, grid: Sequence[int], fixed_targets: Mapping[str, int],
        actuator: PerEndpointClockActuator, windows_root: Path,
        workload: Mapping[str, Any],
    ) -> int:
        endpoint_ids = ("P0", "P1") if axis == "P" else ("D0", "D1")
        low, high = 0, len(grid) - 1
        iteration = 0
        while low < high:
            iteration += 1
            before_low, before_high = low, high
            candidate = (low + high) // 2
            targets = dict(fixed_targets)
            for endpoint_id in endpoint_ids:
                targets[endpoint_id] = int(grid[candidate])
            decision_id = "%s-%s-binary-%02d" % (workload["id"], axis, iteration)
            rows = self._run_repeats(
                actuator, windows_root, workload, "%s_search" % axis,
                targets, int(self.settings["probe_repeats"]), axis=axis,
                candidate_index=candidate, decision_id=decision_id,
            )
            feasible = self._probe_feasible(rows)
            low, high = binary_update(low, high, candidate, feasible)
            self.decision_rows.append({
                "decision_id": decision_id, "workload": str(workload["id"]),
                "axis": axis, "iteration": iteration,
                "low_index_before": before_low, "high_index_before": before_high,
                "candidate_index": candidate,
                "candidate_frequency_mhz": int(grid[candidate]),
                "P0_target_mhz": int(targets["P0"]),
                "P1_target_mhz": int(targets["P1"]),
                "D0_target_mhz": int(targets["D0"]),
                "D1_target_mhz": int(targets["D1"]),
                "probe_window_ids_json": json.dumps([row["window_id"] for row in rows]),
                "feedback_p99_ttft_ms_json": json.dumps([row["p99_ttft_ms"] for row in rows]),
                "feedback_p99_tpot_ms_json": json.dumps([row["p99_tpot_ms"] for row in rows]),
                "feedback_slo_violations_json": json.dumps([
                    row["slo_violation_count"] for row in rows
                ]),
                "feedback_joules_per_request_json": json.dumps([
                    row["joules_per_request"] for row in rows
                ]),
                "feasible": feasible,
                "decision": "SEARCH_LOWER_HALF" if feasible else "SEARCH_UPPER_HALF",
                "low_index_after": low, "high_index_after": high,
                "next_interval_frequency_mhz_json": json.dumps([
                    int(grid[low]), int(grid[high])
                ]),
            })
        return low

    def run_binary(self) -> Path:
        if self.run_dir.exists():
            raise BinaryDVFSError("run directory already exists: %s" % self.run_dir)
        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        capabilities, grids = self._discover_fine_grids()
        _write_json(self.run_dir / "raw" / "capabilities.json", {
            key: asdict(value) for key, value in capabilities.items()
        })
        _write_json(self.run_dir / "frequency_grids.json", {
            key: list(value) for key, value in grids.items()
        })
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        high = {endpoint_id: grids[endpoint_id][-1] for endpoint_id in ENDPOINTS}
        workloads = list(self.config["workloads"])
        restoration_valid = False
        error: Optional[str] = None
        try:
            self._actuate_targets(
                actuator, high, workload="global", phase="warmup",
                decision_id="global-high-warmup", reason="binary_feedback_global_safe_high",
            )
            self._write_routes(ROUTES, "binary_feedback_global_warmup")
            warmup = {"id": "global_warmup", "input_len": 128, "output_len": 16,
                      "rate_rps": float(self.settings["warmup_rate_rps"])}
            self._run_repeats(
                actuator, windows_root, warmup, "warmup", high, 1,
                decision_id="global-high-warmup",
            )

            for workload in workloads:
                workload_id = str(workload["id"])
                baseline = self._run_repeats(
                    actuator, windows_root, workload, "baseline_high", high,
                    int(self.settings["baseline_repeats"]),
                    decision_id="%s-baseline-high" % workload_id,
                )
                if not self._evidence_valid(baseline):
                    raise BinaryDVFSError(
                        "all-HIGH baseline does not satisfy physical evidence gates for %s"
                        % workload_id
                    )
                if not self._probe_feasible(baseline):
                    self.selected[workload_id] = {
                        "status": "MAX_FREQ_INFEASIBLE",
                        "P_pool_frequency_mhz": int(high["P0"]),
                        "D_pool_frequency_mhz": int(high["D0"]),
                        "P0_frequency_mhz": int(high["P0"]),
                        "P1_frequency_mhz": int(high["P1"]),
                        "D0_frequency_mhz": int(high["D0"]),
                        "D1_frequency_mhz": int(high["D1"]),
                        "P0_grid_index": len(grids["P0"]) - 1,
                        "D0_grid_index": len(grids["D0"]) - 1,
                        "confirmation_attempt": -1,
                        "confirmation_window_ids": [],
                        "baseline_window_ids": [row["window_id"] for row in baseline],
                        "evaluation_window_ids": [row["window_id"] for row in baseline],
                        "baseline_joules_per_request_mean": mean(
                            row["joules_per_request"] for row in baseline
                        ),
                        "max_frequency_joules_per_request_mean": mean(
                            row["joules_per_request"] for row in baseline
                        ),
                        "max_frequency_p99_ttft_ms": max(
                            row["p99_ttft_ms"] for row in baseline
                        ),
                        "max_frequency_p99_tpot_ms": max(
                            row["p99_tpot_ms"] for row in baseline
                        ),
                        "max_frequency_slo_violation_count": sum(
                            row["slo_violation_count"] for row in baseline
                        ),
                    }
                    self.decision_rows.append({
                        "decision_id": "%s-max-frequency-infeasible" % workload_id,
                        "workload": workload_id, "axis": "MAX_FREQUENCY_BASELINE",
                        "iteration": 0, "low_index_before": "", "high_index_before": "",
                        "candidate_index": "", "candidate_frequency_mhz": "",
                        "P0_target_mhz": int(high["P0"]),
                        "P1_target_mhz": int(high["P1"]),
                        "D0_target_mhz": int(high["D0"]),
                        "D1_target_mhz": int(high["D1"]),
                        "probe_window_ids_json": json.dumps([
                            row["window_id"] for row in baseline
                        ]),
                        "feedback_p99_ttft_ms_json": json.dumps([
                            row["p99_ttft_ms"] for row in baseline
                        ]),
                        "feedback_p99_tpot_ms_json": json.dumps([
                            row["p99_tpot_ms"] for row in baseline
                        ]),
                        "feedback_slo_violations_json": json.dumps([
                            row["slo_violation_count"] for row in baseline
                        ]),
                        "feedback_joules_per_request_json": json.dumps([
                            row["joules_per_request"] for row in baseline
                        ]),
                        "feasible": False,
                        "decision": "RECORD_MAX_FREQ_INFEASIBLE_CONTINUE",
                        "low_index_after": "", "high_index_after": "",
                        "next_interval_frequency_mhz_json": "[]",
                    })
                    continue

                p_index = self._binary_search_axis(
                    "P", grids["P0"], high, actuator, windows_root, workload,
                )
                after_p = dict(high)
                after_p["P0"] = grids["P0"][p_index]
                after_p["P1"] = grids["P1"][p_index]
                d_index = self._binary_search_axis(
                    "D", grids["D0"], after_p, actuator, windows_root, workload,
                )

                max_corrections = int(self.settings["maximum_confirmation_corrections"])
                confirmation_rows: list[dict[str, Any]] = []
                confirmation_valid = False
                final_attempt = 0
                for attempt in range(max_corrections + 1):
                    final_attempt = attempt
                    targets = dict(high)
                    targets["P0"] = grids["P0"][p_index]
                    targets["P1"] = grids["P1"][p_index]
                    targets["D0"] = grids["D0"][d_index]
                    targets["D1"] = grids["D1"][d_index]
                    confirmation_rows = self._run_repeats(
                        actuator, windows_root, workload, "confirmation", targets,
                        int(self.settings["confirmation_repeats"]),
                        confirmation_attempt=attempt,
                        decision_id="%s-confirm-%02d" % (workload_id, attempt),
                    )
                    confirmation_valid = self._probe_feasible(confirmation_rows)
                    self.decision_rows.append({
                        "decision_id": "%s-confirm-%02d" % (workload_id, attempt),
                        "workload": workload_id, "axis": "JOINT_CONFIRMATION",
                        "iteration": attempt + 1,
                        "low_index_before": "", "high_index_before": "",
                        "candidate_index": "", "candidate_frequency_mhz": "",
                        "P0_target_mhz": int(targets["P0"]),
                        "P1_target_mhz": int(targets["P1"]),
                        "D0_target_mhz": int(targets["D0"]),
                        "D1_target_mhz": int(targets["D1"]),
                        "probe_window_ids_json": json.dumps([
                            row["window_id"] for row in confirmation_rows
                        ]),
                        "feedback_p99_ttft_ms_json": json.dumps([
                            row["p99_ttft_ms"] for row in confirmation_rows
                        ]),
                        "feedback_p99_tpot_ms_json": json.dumps([
                            row["p99_tpot_ms"] for row in confirmation_rows
                        ]),
                        "feedback_slo_violations_json": json.dumps([
                            row["slo_violation_count"] for row in confirmation_rows
                        ]),
                        "feedback_joules_per_request_json": json.dumps([
                            row["joules_per_request"] for row in confirmation_rows
                        ]),
                        "feasible": confirmation_valid,
                        "decision": (
                            "ACCEPT_FINAL" if confirmation_valid
                            else "ESCALATE_VIOLATING_AXIS"
                        ),
                        "low_index_after": "", "high_index_after": "",
                        "next_interval_frequency_mhz_json": "[]",
                    })
                    if confirmation_valid:
                        break
                    ttft_failed = any(not row["slo_pass"] and (
                        row["p99_ttft_ms"] > float(self.settings["slo"]["ttft_ms"])
                        or row["max_ttft_ms"] > float(self.settings["slo"]["ttft_ms"])
                    ) for row in confirmation_rows)
                    tpot_failed = any(not row["slo_pass"] and (
                        row["p99_tpot_ms"] > float(self.settings["slo"]["tpot_ms"])
                        or row["max_tpot_ms"] > float(self.settings["slo"]["tpot_ms"])
                    ) for row in confirmation_rows)
                    if not ttft_failed and not tpot_failed:
                        ttft_failed = tpot_failed = True
                    if ttft_failed:
                        p_index = min(p_index + 1, len(grids["P0"]) - 1)
                    if tpot_failed:
                        d_index = min(d_index + 1, len(grids["D0"]) - 1)
                if not confirmation_valid:
                    final_attempt = max_corrections + 1
                    confirmation_rows = self._run_repeats(
                        actuator, windows_root, workload, "confirmation", high,
                        int(self.settings["confirmation_repeats"]),
                        confirmation_attempt=final_attempt,
                        decision_id="%s-fallback-high" % workload_id,
                    )
                    if not self._evidence_valid(confirmation_rows):
                        raise BinaryDVFSError(
                            "maximum-frequency fallback lacks valid evidence for %s"
                            % workload_id
                        )
                    confirmation_valid = self._probe_feasible(confirmation_rows)
                    self.decision_rows.append({
                        "decision_id": "%s-fallback-high" % workload_id,
                        "workload": workload_id, "axis": "JOINT_FALLBACK_HIGH",
                        "iteration": final_attempt + 1,
                        "low_index_before": "", "high_index_before": "",
                        "candidate_index": "", "candidate_frequency_mhz": "",
                        "P0_target_mhz": int(high["P0"]),
                        "P1_target_mhz": int(high["P1"]),
                        "D0_target_mhz": int(high["D0"]),
                        "D1_target_mhz": int(high["D1"]),
                        "probe_window_ids_json": json.dumps([
                            row["window_id"] for row in confirmation_rows
                        ]),
                        "feedback_p99_ttft_ms_json": json.dumps([
                            row["p99_ttft_ms"] for row in confirmation_rows
                        ]),
                        "feedback_p99_tpot_ms_json": json.dumps([
                            row["p99_tpot_ms"] for row in confirmation_rows
                        ]),
                        "feedback_slo_violations_json": json.dumps([
                            row["slo_violation_count"] for row in confirmation_rows
                        ]),
                        "feedback_joules_per_request_json": json.dumps([
                            row["joules_per_request"] for row in confirmation_rows
                        ]),
                        "feasible": confirmation_valid,
                        "decision": (
                            "ACCEPT_MAX_FREQUENCY"
                            if confirmation_valid
                            else "RECORD_MAX_FREQ_INFEASIBLE_CONTINUE"
                        ),
                        "low_index_after": "", "high_index_after": "",
                        "next_interval_frequency_mhz_json": "[]",
                    })
                    p_index = len(grids["P0"]) - 1
                    d_index = len(grids["D0"]) - 1
                    if not confirmation_valid:
                        self.selected[workload_id] = {
                            "status": "MAX_FREQ_INFEASIBLE",
                            "P_pool_frequency_mhz": int(high["P0"]),
                            "D_pool_frequency_mhz": int(high["D0"]),
                            "P0_frequency_mhz": int(high["P0"]),
                            "P1_frequency_mhz": int(high["P1"]),
                            "D0_frequency_mhz": int(high["D0"]),
                            "D1_frequency_mhz": int(high["D1"]),
                            "P0_grid_index": p_index,
                            "D0_grid_index": d_index,
                            "confirmation_attempt": -1,
                            "confirmation_window_ids": [
                                row["window_id"] for row in confirmation_rows
                            ],
                            "baseline_window_ids": [
                                row["window_id"] for row in baseline
                            ],
                            "evaluation_window_ids": [
                                row["window_id"] for row in confirmation_rows
                            ],
                            "baseline_joules_per_request_mean": mean(
                                row["joules_per_request"] for row in baseline
                            ),
                            "max_frequency_joules_per_request_mean": mean(
                                row["joules_per_request"] for row in confirmation_rows
                            ),
                            "max_frequency_p99_ttft_ms": max(
                                row["p99_ttft_ms"] for row in confirmation_rows
                            ),
                            "max_frequency_p99_tpot_ms": max(
                                row["p99_tpot_ms"] for row in confirmation_rows
                            ),
                            "max_frequency_slo_violation_count": sum(
                                row["slo_violation_count"] for row in confirmation_rows
                            ),
                        }
                        continue
                self.selected[workload_id] = {
                    "status": "CONFIRMED_SLO_SAFE",
                    "P_pool_frequency_mhz": int(grids["P0"][p_index]),
                    "D_pool_frequency_mhz": int(grids["D0"][d_index]),
                    "P0_frequency_mhz": int(grids["P0"][p_index]),
                    "P1_frequency_mhz": int(grids["P1"][p_index]),
                    "D0_frequency_mhz": int(grids["D0"][d_index]),
                    "D1_frequency_mhz": int(grids["D1"][d_index]),
                    "P0_grid_index": p_index, "D0_grid_index": d_index,
                    "confirmation_attempt": final_attempt,
                    "confirmation_window_ids": [
                        row["window_id"] for row in confirmation_rows
                    ],
                    "confirmation_joules_per_request_mean": mean(
                        row["joules_per_request"] for row in confirmation_rows
                    ),
                    "baseline_joules_per_request_mean": mean(
                        row["joules_per_request"] for row in baseline
                    ),
                }
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            try:
                self._actuate_targets(
                    actuator, high, workload="global", phase="restore",
                    decision_id="final-safe-high-restore",
                    reason="binary_feedback_final_safe_high_restoration",
                )
                restoration_valid = all(
                    actuator.read(endpoint_id).graphics_clock_mhz == high[endpoint_id]
                    for endpoint_id in ENDPOINTS
                )
            except Exception:
                restoration_valid = False

        window_fields = tuple(self.window_rows[0]) if self.window_rows else ()
        decision_fields = (
            "decision_id", "workload", "axis", "iteration",
            "low_index_before", "high_index_before", "candidate_index",
            "candidate_frequency_mhz", "P0_target_mhz", "P1_target_mhz",
            "D0_target_mhz", "D1_target_mhz",
            "probe_window_ids_json", "feedback_p99_ttft_ms_json",
            "feedback_p99_tpot_ms_json", "feedback_slo_violations_json",
            "feedback_joules_per_request_json", "feasible", "decision",
            "low_index_after", "high_index_after",
            "next_interval_frequency_mhz_json",
        )
        action_fields = tuple(self.action_rows[0]) if self.action_rows else ()
        _csv(self.run_dir / "window_results.csv", self.window_rows, window_fields)
        _csv(self.run_dir / "binary_search_decisions.csv", self.decision_rows, decision_fields)
        _csv(self.run_dir / "raw" / "actuator_actions.csv", self.action_rows, action_fields)
        _write_json(self.run_dir / "selected_frequencies.json", self.selected)
        hard_gates = {
            "at_least_ten_frequency_levels_per_endpoint": all(
                len(grid) >= 10 for grid in grids.values()
            ),
            "frozen_balanced_disjoint_routes": True,
            "all_recorded_windows_valid": bool(self.window_rows) and all(
                row["measurement_valid"] and row["route_valid"]
                and row["clocks_valid"] and row["concurrency_valid"]
                and row["client_queue_valid"]
                for row in self.window_rows
            ),
            "every_workload_has_recorded_outcome": (
                len(self.selected) == len(workloads)
            ),
            "recorded_outcomes_are_explicit_and_valid": (
                len(self.selected) == len(workloads)
                and all(
                    choice.get("status") in {
                        "CONFIRMED_SLO_SAFE", "MAX_FREQ_INFEASIBLE"
                    }
                    and (
                        choice.get("status") != "MAX_FREQ_INFEASIBLE"
                        or (
                            int(choice["P_pool_frequency_mhz"]) == int(high["P0"])
                            and int(choice["D_pool_frequency_mhz"]) == int(high["D0"])
                            and int(choice["confirmation_attempt"]) == -1
                        )
                    )
                    for choice in self.selected.values()
                )
            ),
            "every_actuation_readback_valid": bool(self.action_rows) and all(
                row["command_status"] == "success" and row["readback_valid"]
                for row in self.action_rows
            ),
            "safe_high_restoration": restoration_valid,
            "feedback_only_no_model_or_oracle": True,
            "no_unresolved_error": error is None,
        }
        audit = {
            "phase": "4B_fine_grained_binary_feedback_DVFS",
            "valid": all(hard_gates.values()), "hard_gates": hard_gates,
            "error": error, "routes": [list(pair) for pair in ROUTES],
            "max_concurrency": int(self.settings["max_concurrency"]),
            "slo": self.settings["slo"],
            "controller_inputs": "completed physical probe windows only",
            "models_or_oracles_used": [],
            "raw_artifact_root": str(self.run_dir),
        }
        _write_json(self.run_dir / "binary_dvfs_audit.json", audit)
        if not audit["valid"]:
            raise BinaryDVFSError(error or "binary DVFS evidence audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    settings = config.get("binary_dvfs")
    if not isinstance(settings, dict):
        raise BinaryDVFSError("config requires binary_dvfs")
    endpoints = {str(row["endpoint_id"]): row for row in config.get("endpoints", [])}
    if set(endpoints) != set(ENDPOINTS):
        raise BinaryDVFSError("binary DVFS requires exactly P0/P1/D0/D1")
    if any(endpoints[key]["node"] != "uranus" for key in ("P0", "P1")):
        raise BinaryDVFSError("P0/P1 must run on Uranus")
    if any(endpoints[key]["node"] != "ganymede" for key in ("D0", "D1")):
        raise BinaryDVFSError("D0/D1 must run on Ganymede")
    if settings.get("routes") != [["P0", "D0"], ["P1", "D1"]]:
        raise BinaryDVFSError(
            "this experiment freezes balanced disjoint P0->D0 and P1->D1 routes"
        )
    if settings.get("slo") != {"ttft_ms": 500.0, "tpot_ms": 200.0}:
        raise BinaryDVFSError("SLO must be TTFT=500ms and TPOT=200ms")
    if int(settings.get("probe_repeats", 0)) < 2:
        raise BinaryDVFSError("each binary decision requires at least two probe windows")
    if int(settings.get("baseline_repeats", 0)) < 3:
        raise BinaryDVFSError("all-HIGH baseline requires at least three repeats")
    if int(settings.get("confirmation_repeats", 0)) < 5:
        raise BinaryDVFSError("final selection requires at least five confirmation windows")
    if int(settings.get("max_concurrency", 0)) != 2:
        raise BinaryDVFSError("no-queue frozen routing requires max_concurrency=2")
    if int(settings.get("closed_loop_batch_size", 0)) != 2:
        raise BinaryDVFSError("no-queue frozen routing requires closed_loop_batch_size=2")
    if (
        int(settings.get("minimum_observed_concurrency", 0)) != 2
        or int(settings.get("maximum_observed_concurrency", 0)) != 2
    ):
        raise BinaryDVFSError("observed concurrency must be exactly two")
    if float(settings.get("maximum_client_queue_delay_ms", -1)) <= 0:
        raise BinaryDVFSError("maximum client queue delay gate must be positive")
    if int(settings.get("requests_per_window", 0)) < 4 * int(
        settings["max_concurrency"]
    ):
        raise BinaryDVFSError("each window requires at least four requests per concurrency slot")
    for role in ("prefill", "decode"):
        grid = settings.get("frequency_grids", {}).get(role, {})
        if int(grid.get("levels", 0)) < 10:
            raise BinaryDVFSError("%s grid requires at least ten levels" % role)
        if int(grid.get("minimum_mhz", 0)) >= int(grid.get("maximum_mhz", 0)):
            raise BinaryDVFSError("%s frequency bounds are invalid" % role)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    BinaryDVFSEvaluationHarness(load_config(args.config), args.run_id).run_binary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
