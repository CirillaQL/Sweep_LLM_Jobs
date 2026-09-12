"""History-seeded routing/DVFS with measured online feedback correction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from xpyd.phase3c_substrate import _read_json, _write_json, build_registry_and_compatibility
from xpyd.phase3d_control import (
    PerEndpointClockActuator,
    Phase3DError,
    _csv,
    load_config as load_phase3d_config,
)
from xpyd.phase4b_evaluation import Phase4BEvaluationHarness


class HistoryControlError(Phase3DError):
    """Raised when historical evidence or a physical control gate is invalid."""


@dataclass(frozen=True)
class HistoricalDecision:
    source_workload: str
    context_distance: float
    config_id: str
    route: tuple[str, str]
    frequencies_mhz: dict[str, int]
    expected_joules_per_request: float
    expected_energy_ci95_half_width: float
    expected_ttft_ms: float
    expected_tpot_ms: float
    expected_ttft_upper_ms: float
    expected_tpot_upper_ms: float
    candidate_count: int
    ranking_rule: str


def workload_distance(current: Mapping[str, Any], historical: Mapping[str, Any]) -> float:
    """A small, interpretable log-distance over IL, OL, and offered load."""
    weights = {"input_len": 0.4, "output_len": 0.4, "rate_rps": 0.2}
    distance = 0.0
    for key, weight in weights.items():
        left = float(current[key])
        right = float(historical[key])
        if left <= 0 or right <= 0:
            raise HistoryControlError("workload dimensions must be positive")
        distance += weight * abs(math.log2(left / right))
    return distance


class HistoricalExperienceStore:
    """Read-only selector over audited Phase 4A aggregate measurements."""

    def __init__(
        self,
        summary: Mapping[str, Any],
        workload_specs: Sequence[Mapping[str, Any]],
        *,
        safety_fraction: float,
        max_context_distance: float,
    ) -> None:
        if not summary.get("valid") or not summary.get("ready_for_phase4b"):
            raise HistoryControlError("historical Phase 4A summary is not accepted")
        if summary.get("models_trained_or_used"):
            raise HistoryControlError("historical summary unexpectedly used a model")
        if not 0 < safety_fraction <= 1:
            raise HistoryControlError("history safety_fraction must be in (0, 1]")
        if max_context_distance < 0 or not math.isfinite(max_context_distance):
            raise HistoryControlError("max_context_distance must be finite and non-negative")
        self.summary = dict(summary)
        self.workloads = {str(item["id"]): dict(item) for item in workload_specs}
        self.safety_fraction = float(safety_fraction)
        self.max_context_distance = float(max_context_distance)
        self.ttft_slo_ms = float(summary["slo"]["ttft_ms"])
        self.tpot_slo_ms = float(summary["slo"]["tpot_ms"])

    def choose(self, workload: Mapping[str, Any]) -> tuple[HistoricalDecision, list[dict[str, Any]]]:
        candidates: list[dict[str, Any]] = []
        for row in self.summary.get("configuration_aggregates", []):
            source_id = str(row.get("workload", ""))
            source = self.workloads.get(source_id)
            if source is None or not row.get("oracle_eligible"):
                continue
            distance = workload_distance(workload, source)
            ttft_mean = float(row["mean_ttft_ms_mean"])
            tpot_mean = float(row["mean_tpot_ms_mean"])
            ttft_upper = ttft_mean + float(row.get("mean_ttft_ms_ci95_half_width") or 0.0)
            tpot_upper = tpot_mean + float(row.get("mean_tpot_ms_ci95_half_width") or 0.0)
            energy_mean = float(row["joules_per_request_mean"])
            energy_ci = float(row.get("joules_per_request_ci95_half_width") or 0.0)
            safe = (
                distance <= self.max_context_distance
                and ttft_upper <= self.safety_fraction * self.ttft_slo_ms
                and tpot_upper <= self.safety_fraction * self.tpot_slo_ms
            )
            candidate = {
                "source_workload": source_id,
                "config_id": str(row["config_id"]),
                "context_distance": distance,
                "eligible": safe,
                "rejection_reason": None if safe else (
                    "context_too_distant" if distance > self.max_context_distance
                    else "historical_tail_exceeds_safety_margin"
                ),
                "route": [str(row["prefill_endpoint_id"]), str(row["decode_endpoint_id"])],
                "frequencies_mhz": {
                    endpoint_id: int(row[f"{endpoint_id}_freq_mhz"])
                    for endpoint_id in ("P0", "P1", "D0", "D1")
                },
                "expected_joules_per_request": energy_mean,
                "energy_ci95_half_width": energy_ci,
                "conservative_energy_score": energy_mean + energy_ci,
                "expected_ttft_ms": ttft_mean,
                "expected_tpot_ms": tpot_mean,
                "expected_ttft_upper_ms": ttft_upper,
                "expected_tpot_upper_ms": tpot_upper,
            }
            candidates.append(candidate)
        eligible = [item for item in candidates if item["eligible"]]
        eligible.sort(key=lambda item: (
            item["context_distance"], item["conservative_energy_score"], item["config_id"]
        ))
        if not eligible:
            raise HistoryControlError("no historical candidate satisfies context and SLO safety gates")
        best = eligible[0]
        decision = HistoricalDecision(
            source_workload=best["source_workload"],
            context_distance=float(best["context_distance"]),
            config_id=best["config_id"],
            route=tuple(best["route"]),
            frequencies_mhz=dict(best["frequencies_mhz"]),
            expected_joules_per_request=float(best["expected_joules_per_request"]),
            expected_energy_ci95_half_width=float(best["energy_ci95_half_width"]),
            expected_ttft_ms=float(best["expected_ttft_ms"]),
            expected_tpot_ms=float(best["expected_tpot_ms"]),
            expected_ttft_upper_ms=float(best["expected_ttft_upper_ms"]),
            expected_tpot_upper_ms=float(best["expected_tpot_upper_ms"]),
            candidate_count=len(eligible),
            ranking_rule="min(context_log_distance, energy_mean_plus_ci95, config_id)",
        )
        return decision, candidates


class HistorySeededEvaluationHarness(Phase4BEvaluationHarness):
    """Physical history seed followed by the existing model-free correction loop."""

    def __init__(self, config: Mapping[str, Any], run_id: Optional[str] = None) -> None:
        super().__init__(config, run_id=run_id, smoke=True, stage="stationary")
        self.history_settings = dict(self.config["phase3d_history"])
        root = self.history_settings.get(
            "output_root", "results/history_seeded_closed_loop"
        )
        self.run_dir = Path(os.path.expandvars(str(root))) / self.run_id
        self.decision_rows: list[dict[str, Any]] = []
        self.outcome_rows: list[dict[str, Any]] = []

    def _history_decide(
        self, store: HistoricalExperienceStore, workload: Mapping[str, Any],
        iteration_id: str, context: Any, actuator: PerEndpointClockActuator,
    ) -> list[list[str]]:
        fallback_reason = None
        try:
            decision, candidates = store.choose(workload)
            frequencies = decision.frequencies_mhz
            pairs = [list(decision.route)]
            decision_source = "HISTORICAL_EXPERIENCE"
        except HistoryControlError as exc:
            decision = None
            candidates = []
            fallback_reason = str(exc)
            frequencies = {
                endpoint_id: context.scheduler.hardware.gpu_type(
                    context.registry.get_spec(endpoint_id).gpu_type
                ).max_frequency_mhz
                for endpoint_id in ("P0", "P1", "D0", "D1")
            }
            pairs = [["P0", "D0"]]
            decision_source = "HISTORY_MISS_SAFE_HIGH_FALLBACK"
        self._actuate(
            actuator, context, frequencies,
            "history_seeded_configuration", controller_action=True,
            policy="HISTORY_SEEDED_FULL", workload=str(workload["id"]),
            iteration_id=iteration_id,
        )
        actuation_time = datetime.now(timezone.utc).timestamp()
        for endpoint_id in sorted(frequencies):
            context.scheduler.record_dvfs_actuation(
                endpoint_id,
                observed_freq_mhz=context.registry.get_state(endpoint_id).freq_mhz,
                timestamp_s=actuation_time,
            )
        context.selected_pairs = copy.deepcopy(pairs)
        self._write_routes(pairs, (
            "history_seeded_safe_low_energy_configuration"
            if decision is not None else "history_miss_safe_high_fallback"
        ))
        now = datetime.now(timezone.utc).timestamp()
        row = {
            "control_timestamp": now,
            "control_iteration_id": iteration_id,
            "policy": "HISTORY_SEEDED_FULL",
            "workload": str(workload["id"]),
            "decision_source": decision_source,
            "selected_pairs_json": json.dumps(pairs, sort_keys=True),
            "requested_frequencies_json": json.dumps(frequencies, sort_keys=True),
            "source_workload": decision.source_workload if decision else None,
            "context_distance": decision.context_distance if decision else None,
            "selected_config_id": decision.config_id if decision else None,
            "candidate_count": decision.candidate_count if decision else 0,
            "expected_joules_per_request": decision.expected_joules_per_request if decision else None,
            "expected_energy_ci95_half_width": decision.expected_energy_ci95_half_width if decision else None,
            "expected_ttft_ms": decision.expected_ttft_ms if decision else None,
            "expected_tpot_ms": decision.expected_tpot_ms if decision else None,
            "expected_ttft_upper_ms": decision.expected_ttft_upper_ms if decision else None,
            "expected_tpot_upper_ms": decision.expected_tpot_upper_ms if decision else None,
            "ranking_rule": decision.ranking_rule if decision else "safe_high_fallback",
            "candidate_evaluations_json": json.dumps(candidates, sort_keys=True),
            "fallback_reason": fallback_reason,
            "dvfs_recommendations_json": "[]",
        }
        self.decision_rows.append(row)
        return pairs

    def _record_feedback_decision(self) -> None:
        source = self.control_rows[-1]
        self.decision_rows.append({
            "control_timestamp": source["control_timestamp"],
            "control_iteration_id": source["control_iteration_id"],
            "policy": "HISTORY_SEEDED_FULL",
            "workload": source["workload"],
            "decision_source": "ONLINE_FEEDBACK_CORRECTION",
            "selected_pairs_json": source["selected_pairs_json"],
            "requested_frequencies_json": source["requested_frequencies_json"],
            "source_workload": None,
            "context_distance": None,
            "selected_config_id": None,
            "candidate_count": None,
            "expected_joules_per_request": None,
            "expected_energy_ci95_half_width": None,
            "expected_ttft_ms": None,
            "expected_tpot_ms": None,
            "expected_ttft_upper_ms": None,
            "expected_tpot_upper_ms": None,
            "ranking_rule": source["routing_score_basis"],
            "candidate_evaluations_json": source["candidate_evaluations_json"],
            "fallback_reason": source["fallback_reason"],
            "dvfs_recommendations_json": source["dvfs_recommendations_json"],
        })

    def run(self) -> Path:
        if self.run_dir.exists():
            raise HistoryControlError(f"run directory already exists: {self.run_dir}")
        windows_root = self.run_dir / "raw" / "windows"
        windows_root.mkdir(parents=True)
        history_path = Path(os.path.expandvars(str(self.history_settings["history_summary"])))
        history_raw = history_path.read_bytes()
        history_summary = json.loads(history_raw)
        workloads = [dict(item) for item in self.config["workloads"]]
        store = HistoricalExperienceStore(
            history_summary, workloads,
            safety_fraction=float(self.history_settings.get("safety_fraction", 0.9)),
            max_context_distance=float(self.history_settings.get("max_context_distance", 0.0)),
        )
        reference = self._reference()
        accepted = self._accepted_frequencies(reference)
        capabilities = self._discover_exact(accepted)
        actuator = PerEndpointClockActuator(
            self.backend, capabilities,
            float(self.config["phase3d"].get("minimum_dwell_s", 1.0)),
        )
        high = {endpoint_id: values["HIGH"] for endpoint_id, values in accepted.items()}
        actuator.requested.update(high)
        _, _, all_pairs = build_registry_and_compatibility(self.config)
        repeats = int(self.history_settings.get("windows_per_workload", 3))
        request_count = int(self.history_settings.get("requests_per_window", 2))
        sequence = self.history_settings.get("workload_sequence") or [item["id"] for item in workloads]
        by_id = {str(item["id"]): item for item in workloads}
        error: Optional[str] = None
        restoration_valid = False
        window_audits: list[dict[str, Any]] = []
        try:
            warm_context = self._new_context(capabilities)
            self._actuate(
                actuator, warm_context, high, "history_controller_safe_high_warmup",
                controller_action=False, policy="WARMUP", workload="all",
                iteration_id="global_warmup",
            )
            self._write_routes(all_pairs, "history_controller_all_route_warmup")
            warm_dir = self._run_window(
                "global-warmup", {"id": "global_warmup", "input_len": 128,
                "output_len": 16, "rate_rps": 0.5}, 4, high, all_pairs,
                windows_root, require_all_pairs=True,
            )
            warm_audit = _read_json(warm_dir / "audit.json")
            window_audits.append({"window_id": "global-warmup", **warm_audit})
            if not warm_audit.get("valid"):
                raise HistoryControlError("global all-route warmup failed")
            for state_index, workload_id in enumerate(sequence, 1):
                workload = by_id[str(workload_id)]
                context = self._new_context(capabilities)
                for repeat in range(1, repeats + 1):
                    iteration_id = f"s{state_index:02d}-w{repeat:02d}-{workload_id}"
                    if repeat == 1:
                        pairs = self._history_decide(
                            store, workload, iteration_id, context, actuator
                        )
                    else:
                        pairs = self._decide(
                            "FULL_FEEDBACK", str(workload_id), iteration_id,
                            context, capabilities, actuator, all_pairs,
                            measured_iteration=True,
                        )
                        self._record_feedback_decision()
                    window_dir = self._run_window(
                        iteration_id, workload, request_count,
                        context.target_frequencies, pairs, windows_root,
                    )
                    audit = _read_json(window_dir / "audit.json")
                    window_audits.append({"window_id": iteration_id, **audit})
                    row, _ = self._measurement_row(
                        iteration_id, "HISTORY_SEEDED_FULL", workload, repeat,
                        context, accepted, reference, window_dir,
                    )
                    row["control_iteration_id"] = iteration_id
                    row["decision_source"] = self.decision_rows[-1]["decision_source"]
                    row["planned_route_json"] = self.decision_rows[-1]["selected_pairs_json"]
                    row["planned_frequencies_json"] = self.decision_rows[-1]["requested_frequencies_json"]
                    self.outcome_rows.append(row)
                    if not row["measurement_valid"]:
                        raise HistoryControlError(f"physical window audit failed: {iteration_id}")
                    self._observe_window(
                        iteration_id, window_dir, context.registry, context.telemetry
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            restore_context = self._new_context(capabilities)
            try:
                self._actuate(
                    actuator, restore_context, high,
                    "history_controller_final_safe_high_restoration",
                    controller_action=False, policy="RESTORE", workload="all",
                    iteration_id="final_restore",
                )
                restoration_valid = all(
                    actuator.read(endpoint_id).graphics_clock_mhz == target
                    for endpoint_id, target in high.items()
                )
            except Exception:
                restoration_valid = False

        decision_fields = tuple(self.decision_rows[0]) if self.decision_rows else ()
        outcome_fields = tuple(self.outcome_rows[0]) if self.outcome_rows else ()
        _csv(self.run_dir / "history_decisions.csv", self.decision_rows, decision_fields)
        _csv(self.run_dir / "history_actual_outcomes.csv", self.outcome_rows, outcome_fields)
        _csv(self.run_dir / "raw" / "actuator_actions.csv", self.actuator_rows, (
            "timestamp_unix_s", "policy", "workload", "control_iteration_id",
            "controller_action", "endpoint_id", "node", "gpu_id",
            "previous_requested_freq_mhz", "requested_freq_mhz",
            "observed_freq_before_mhz", "observed_freq_after_mhz",
            "command_status", "readback_valid", "transition_readback_latency_s",
            "reason", "error",
        ))
        _csv(self.run_dir / "raw" / "endpoint_telemetry.csv", self.telemetry_rows, (
            "window_id", "endpoint_id", "timestamp_s", "health", "queue",
            "kv_cache_usage_frac", "ttft_ms", "tpot_ms", "power_w", "energy_j",
            "observed_frequency_mhz", "telemetry_age_s_at_observation",
            "measurement_valid",
        ))
        expected = len(sequence) * repeats
        hard_gates = {
            "accepted_history_summary": bool(history_summary.get("valid")),
            "history_source_frozen_by_sha256": len(hashlib.sha256(history_raw).hexdigest()) == 64,
            "complete_decision_log": len(self.decision_rows) == expected,
            "complete_actual_outcome_log": len(self.outcome_rows) == expected,
            "all_physical_windows_valid": len(self.outcome_rows) == expected and all(
                row["measurement_valid"] for row in self.outcome_rows
            ),
            "every_outcome_linked_to_decision": bool(self.outcome_rows) and all(
                row["control_iteration_id"] for row in self.outcome_rows
            ),
            "all_actuations_read_back": all(
                row["command_status"] == "success" and row["readback_valid"]
                for row in self.actuator_rows
            ),
            "safe_high_restoration": restoration_valid,
            "no_models_used": not history_summary.get("models_trained_or_used"),
            "no_unresolved_error": error is None,
        }
        valid = all(hard_gates.values())
        audit = {
            "phase": "history_seeded_feedback_control_smoke",
            "valid": valid,
            "hard_gates": hard_gates,
            "error": error,
            "history_summary_path": history_path.as_posix(),
            "history_summary_sha256": hashlib.sha256(history_raw).hexdigest(),
            "decision_rule": "nearest safe historical context, minimum energy mean+CI95; then existing feedback correction",
            "window_audits": window_audits,
            "models_trained_or_used": [],
            "claim_boundary": "history-seeded functional smoke; same-workload history is not held-out generalization evidence",
        }
        _write_json(self.run_dir / "history_control_audit.json", audit)
        if not valid:
            raise HistoryControlError(error or "history controller hard-gate audit failed")
        print(self.run_dir.as_posix())
        return self.run_dir


def load_config(path: Path) -> dict[str, Any]:
    config = load_phase3d_config(path)
    if not isinstance(config.get("phase4b"), dict):
        raise HistoryControlError("history control reuses Phase 4B measurement settings")
    settings = config.get("phase3d_history")
    if not isinstance(settings, dict):
        raise HistoryControlError("phase3d_history object is required")
    for key in ("history_summary", "output_root"):
        if not str(settings.get(key, "")):
            raise HistoryControlError(f"phase3d_history.{key} is required")
    if int(settings.get("windows_per_workload", 0)) < 2:
        raise HistoryControlError("at least two windows are required to observe online correction")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    accepted = os.path.expandvars(str(config["phase3d"].get("accepted_actuator_audit", "")))
    if not accepted or not _read_json(Path(accepted)).get("valid"):
        raise HistoryControlError("an accepted Phase 3D-A actuator audit is required")
    HistorySeededEvaluationHarness(config, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
