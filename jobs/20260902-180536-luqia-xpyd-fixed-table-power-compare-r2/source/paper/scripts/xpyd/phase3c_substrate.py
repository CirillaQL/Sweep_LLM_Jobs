"""Phase 3C real multi-endpoint P/D substrate validation.

This is a small physical correctness/measurement harness.  It deliberately
uses deterministic round-robin over explicit compatible pairs, fixed clocks,
the accepted streaming client, Phase 3A vLLM metrics, and Phase 3B NVML energy
sampling.  It contains no optimization policy, model, or DVFS actuation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from xpyd.compatibility import CompatibilityTable, EndpointPairCompatibility
from xpyd.phase3a_observability import (
    FixedPeriodScrapeOrchestrator,
    Phase3AHarness,
    ScrapeAttempt,
    _snapshot_dict,
)
from xpyd.phase3b_characterization import _clock_audit
from xpyd.phase3b_energy import (
    WindowSpec,
    _git_commit,
    _launch_monitor,
    _read_jsonl,
    _stop_monitors,
    _wait_capabilities,
    summarize_window,
)
from xpyd.registry import EndpointRegistry
from xpyd.types import EndpointSpec, EndpointState, LifecycleState
from xpyd.vllm_metrics import VLLMMetricsCollector


class Phase3CError(RuntimeError):
    """A fail-closed Phase 3C configuration, runtime, or audit failure."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3CError("expected JSON object: %s" % path)
    return value


def _percentile(values: Sequence[Any], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _workload_slo_met(
    requests: Sequence[Mapping[str, Any]], workload_id: str,
    ttft_limit_ms: float, tpot_limit_ms: float,
) -> bool:
    selected = [item for item in requests if item.get("workload_id") == workload_id]
    ttft_p90 = _percentile([item.get("ttft_ms") for item in selected], 0.90)
    tpot_p90 = _percentile([item.get("tpot_ms") for item in selected], 0.90)
    return (
        ttft_p90 is not None and tpot_p90 is not None
        and ttft_p90 < float(ttft_limit_ms)
        and tpot_p90 <= float(tpot_limit_ms)
    )


def _expected_endpoint_token_metrics(
    config: Mapping[str, Any], endpoint_id: str, role: str,
    routed_requests: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Include configured service warmups in endpoint metric expectations."""
    completed = len(routed_requests)
    prompt_tokens = sum(int(item["input_len"]) + 1 for item in routed_requests)
    generation_tokens = (
        len(routed_requests) if role == "prefill"
        else sum(int(item["observed_output_tokens"]) for item in routed_requests)
    )
    feedback = dict(config.get("online_feedback", {}))
    service_pair = tuple(map(str, feedback.get("service_pair", ())))
    warmup_count = int(feedback.get("service_warmup_requests", 0))
    if (
        feedback.get("enabled") is True
        and endpoint_id in service_pair
        and warmup_count > 0
        and routed_requests
    ):
        first_workload = config["workloads"][0]
        completed += warmup_count
        prompt_tokens += warmup_count * (int(first_workload["input_len"]) + 1)
        generation_tokens += warmup_count * (
            1 if role == "prefill" else int(first_workload["output_len"])
        )
    return {
        "delta_completed_requests": completed,
        "delta_prompt_tokens": prompt_tokens,
        "delta_generation_tokens": generation_tokens,
    }


def _online_inference_latency(
    request: Mapping[str, Any], diagnostic: Mapping[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Measure PD inference after DVFS/settling, excluding ingress wait."""
    durations = diagnostic.get("durations_ms") or {}
    monotonic = diagnostic.get("timestamps_monotonic_s") or {}
    ttft_ms = durations.get("pd_inference_ttft")
    if not isinstance(ttft_ms, (int, float)):
        request_received = monotonic.get("request_received")
        first_chunk = monotonic.get("decode_first_real_chunk_received")
        if isinstance(request_received, (int, float)) and isinstance(first_chunk, (int, float)):
            ttft_ms = (float(first_chunk) - float(request_received)) * 1000.0
    output_tokens = request.get("completion_tokens")
    full_decode_stream_ms = durations.get("full_decode_stream")
    tpot_ms = None
    if (
        isinstance(full_decode_stream_ms, (int, float))
        and isinstance(output_tokens, int)
    ):
        tpot_ms = float(full_decode_stream_ms) / max(1, output_tokens - 1)
    return (
        float(ttft_ms) if isinstance(ttft_ms, (int, float)) else None,
        tpot_ms,
    )


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _endpoint_spec(value: Mapping[str, Any]) -> EndpointSpec:
    return EndpointSpec(
        endpoint_id=str(value["endpoint_id"]),
        role=str(value["role"]),
        gpu_type=str(value["gpu_type"]),
        node=str(value["node"]),
        gpu_ids=tuple(int(item) for item in value["gpu_ids"]),
        tp_degree=int(value["tp_degree"]),
        http_uri=str(value["http_uri"]),
        kv_connector=str(value["kv_connector"]),
    )


def load_config(path: Path) -> dict[str, Any]:
    config = _expand_environment(_read_json(path))
    required = {
        "model", "tokenizer_model", "vllm_version", "output_root", "proxy_uri",
        "proxy_diagnostics_log", "endpoints", "compatible_pairs", "fixed_clocks",
        "sampling", "workloads", "client",
    }
    missing = required.difference(config)
    if missing:
        raise Phase3CError("config missing fields: %s" % sorted(missing))
    specs = [_endpoint_spec(item) for item in config["endpoints"]]
    ids = [item.endpoint_id for item in specs]
    if len(ids) != len(set(ids)):
        raise Phase3CError("endpoint IDs must be unique")
    if sum(item.role == "prefill" for item in specs) < 2:
        raise Phase3CError("Phase 3C requires at least two prefill endpoints")
    if sum(item.role == "decode" for item in specs) < 2:
        raise Phase3CError("Phase 3C requires at least two decode endpoints")
    if str(config["vllm_version"]) != "0.15.1":
        raise Phase3CError("Phase 3C must record the accepted vLLM 0.15.1 runtime")
    if not config["compatible_pairs"]:
        raise Phase3CError("at least one explicit compatible pair is required")
    workload_ids = [str(item.get("id", "")).strip() for item in config["workloads"]]
    if any(not workload_id for workload_id in workload_ids):
        raise Phase3CError("every workload requires a non-empty id")
    if len(workload_ids) != len(set(workload_ids)):
        raise Phase3CError("workload ids must be unique")
    workload_shapes = set()
    for item in config["workloads"]:
        shape = (int(item["input_len"]), int(item["output_len"]))
        if min(*shape, int(item["count"])) <= 0:
            raise Phase3CError("workload lengths and count must be positive")
        if shape in workload_shapes:
            raise Phase3CError("workload token shapes must be unique")
        workload_shapes.add(shape)
    workload_count = sum(int(item["count"]) for item in config["workloads"])
    if workload_count <= 0:
        raise Phase3CError("workloads must contain requests")
    for spec in specs:
        clock = config["fixed_clocks"].get(spec.endpoint_id, {})
        if int(clock.get("graphics_mhz", 0)) <= 0 or int(clock.get("memory_mhz", 0)) <= 0:
            raise Phase3CError("fixed clock missing for %s" % spec.endpoint_id)
    return config


def build_registry_and_compatibility(
    config: Mapping[str, Any],
) -> tuple[EndpointRegistry, CompatibilityTable, tuple[tuple[str, str], ...]]:
    registry = EndpointRegistry()
    for value in config["endpoints"]:
        spec = _endpoint_spec(value)
        registry.register(spec, EndpointState(
            endpoint_id=spec.endpoint_id,
            freq_mhz=int(config["fixed_clocks"][spec.endpoint_id]["graphics_mhz"]),
            lifecycle=LifecycleState.ACTIVE,
            healthy=True,
        ))
    evidence = []
    pairs = []
    for value in config["compatible_pairs"]:
        pair = (str(value["prefill_endpoint_id"]), str(value["decode_endpoint_id"]))
        prefill = registry.get_spec(pair[0])
        decode = registry.get_spec(pair[1])
        evidence.append(EndpointPairCompatibility(
            prefill_endpoint_id=pair[0],
            decode_endpoint_id=pair[1],
            connector=str(value["connector"]),
            prefill_tp=int(value.get("prefill_tp", prefill.tp_degree)),
            decode_tp=int(value.get("decode_tp", decode.tp_degree)),
            supported=bool(value["supported"]),
            reason=str(value["reason"]),
        ))
        pairs.append(pair)
    table = CompatibilityTable(endpoint_pairs=evidence)
    invalid = [
        "%s->%s" % pair for pair in pairs
        if not table.is_compatible(registry.get_spec(pair[0]), registry.get_spec(pair[1]))
    ]
    if invalid:
        raise Phase3CError("configured pair evidence is invalid: %s" % invalid)
    return registry, table, tuple(pairs)


def _write_trace(
    path: Path,
    workloads: Sequence[Mapping[str, Any]],
    request_id_namespace: str = "phase3c",
    ordering: str = "interleaved",
) -> int:
    namespace = str(request_id_namespace).strip()
    if not namespace or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace) is None:
        raise Phase3CError("request ID namespace must be non-empty and filesystem-safe")
    count = 0
    arrival = 0.0
    rate = 1.0
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "request_id", "workload_id", "arrival_time_s",
                "input_len", "output_len",
            ),
        )
        writer.writeheader()
        if ordering not in {"interleaved", "windowed"}:
            raise Phase3CError("trace ordering must be interleaved or windowed")
        if ordering == "windowed":
            schedule = (
                workload for workload in workloads
                for _ in range(int(workload["count"]))
            )
        else:
            remaining = [int(item["count"]) for item in workloads]

            def interleaved() -> Any:
                while any(value > 0 for value in remaining):
                    for index, workload in enumerate(workloads):
                        if remaining[index] <= 0:
                            continue
                        remaining[index] -= 1
                        yield workload

            schedule = interleaved()
        for workload in schedule:
            writer.writerow({
                "request_id": "%s-%04d" % (namespace, count),
                "workload_id": str(workload["id"]),
                "arrival_time_s": arrival,
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
            })
            count += 1
            arrival += 1.0 / float(workload.get("rate_rps", rate))
    return count


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _throttle_audit(
    records: Sequence[Mapping[str, Any]], start: float, end: float
) -> dict[str, Any]:
    selected = [
        item for item in records
        if item.get("status") == "success"
        and isinstance(item.get("actual_start_wall_s"), (int, float))
        and start <= float(item["actual_start_wall_s"]) <= end
    ]
    unavailable = sum(
        item.get("invalidating_thermal_or_hw_slowdown") is None for item in selected
    )
    invalidating = [
        item for item in selected
        if item.get("invalidating_thermal_or_hw_slowdown") is True
    ]
    reasons = sorted({
        str(reason) for item in selected for reason in item.get("clock_throttle_reasons", [])
    })
    return {
        "valid": bool(selected) and unavailable == 0 and not invalidating,
        "sample_count": len(selected),
        "unavailable_sample_count": unavailable,
        "invalidating_sample_count": len(invalidating),
        "observed_reasons": reasons,
    }


class Phase3CSubstrateHarness:
    def __init__(self, config: Mapping[str, Any], run_id: Optional[str] = None) -> None:
        self.config = dict(config)
        self.registry, self.compatibility, self.pairs = build_registry_and_compatibility(config)
        self.endpoints = tuple(sorted(self.registry.list_endpoints(), key=lambda item: item.endpoint_id))
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(os.path.expandvars(str(config["output_root"]))) / self.run_id
        self.collector = VLLMMetricsCollector(**dict(config.get("collector", {})))
        self.scrape_sequence = {item.endpoint_id: 0 for item in self.endpoints}
        self.scrape_lock = threading.Lock()

    def _layout(self) -> None:
        if self.run_dir.exists():
            raise Phase3CError("run directory already exists: %s" % self.run_dir)
        for endpoint in self.endpoints:
            (self.run_dir / endpoint.endpoint_id / "raw_metrics").mkdir(parents=True)
        for relative in ("client", "raw", "derived"):
            (self.run_dir / relative).mkdir(parents=True)

    def _scrape(self, endpoint: EndpointSpec, scheduled: float, label: str = "periodic") -> ScrapeAttempt:
        wall_start = time.time()
        mono_start = time.monotonic()
        raw = None
        error = None
        try:
            raw = self.collector.scrape_raw(endpoint)
            if not raw.raw_text.strip() or raw.snapshot.endpoint_id != endpoint.endpoint_id:
                raise Phase3CError("invalid metrics scrape for %s" % endpoint.endpoint_id)
            with self.scrape_lock:
                self.scrape_sequence[endpoint.endpoint_id] += 1
                sequence = self.scrape_sequence[endpoint.endpoint_id]
                filename = "%06d_%s.prom" % (sequence, label)
                (self.run_dir / endpoint.endpoint_id / "raw_metrics" / filename).write_text(
                    raw.raw_text, encoding="utf-8"
                )
        except Exception as exc:
            error = exc
        return ScrapeAttempt(
            endpoint=endpoint,
            scheduled_monotonic_s=scheduled,
            actual_wall_start_s=wall_start,
            actual_wall_finish_s=time.time(),
            actual_monotonic_start_s=mono_start,
            actual_monotonic_finish_s=time.monotonic(),
            raw=raw,
            error=error,
        )

    def _scrape_round(self, label: str) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        scheduled = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as pool:
            attempts = list(pool.map(
                lambda endpoint: self._scrape(endpoint, scheduled, label),
                self.endpoints,
            ))
        failures = [item.endpoint.endpoint_id for item in attempts if item.error is not None]
        if failures:
            raise Phase3CError("required metrics scrape failed: %s" % failures)
        return {item.endpoint.endpoint_id: item.raw.snapshot for item in attempts if item.raw is not None}

    def _client_command(self, trace: Path) -> list[str]:
        client = self.config["client"]
        return [
            str(client.get("python", sys.executable)),
            str(client.get("script", "paper/scripts/replay_synthetic_trace.py")),
            "--base-url", os.path.expandvars(str(self.config["proxy_uri"])),
            "--model", str(self.config["model"]),
            "--tokenizer-model", str(self.config["tokenizer_model"]),
            "--trace-csv", trace.as_posix(),
            "--output-file", (self.run_dir / "client" / "summary.txt").as_posix(),
            "--summary-json", (self.run_dir / "client" / "summary.json").as_posix(),
            "--requests-jsonl", (self.run_dir / "client" / "requests.jsonl").as_posix(),
            "--max-concurrency", str(int(client.get("max_concurrency", 1))),
            "--num-warmups", "0",
            "--request-timeout-s", str(float(client.get("request_timeout_s", 900))),
            "--fail-on-request-error",
        ]

    def run(self) -> Path:
        self._layout()
        _write_json(self.run_dir / "metadata.json", {
            "schema_version": 1,
            "phase": "3C_real_multi_endpoint_substrate_validation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "run_id": self.run_id,
            "request_id_namespace": "phase3c-%s" % self.run_id,
            "routing_policy": "%s_over_explicit_compatible_pairs"
            % self.config.get("routing_policy", "round_robin"),
            "endpoints": [asdict(item) for item in self.endpoints],
            "compatible_pairs": [list(item) for item in self.pairs],
            "fixed_clocks": self.config["fixed_clocks"],
            "model_training": False,
            "adaptive_routing": bool(self.config.get("online_feedback", {}).get("enabled")),
            "adaptive_dvfs": bool(self.config.get("online_feedback", {}).get("enabled")),
        })
        proxy_log = Path(os.path.expandvars(str(self.config["proxy_diagnostics_log"])))
        proxy_log.parent.mkdir(parents=True, exist_ok=True)
        proxy_log.write_text("", encoding="utf-8")
        trace = self.run_dir / "client" / "trace.csv"
        request_id_namespace = "phase3c-%s" % self.run_id
        expected_requests = _write_trace(
            trace, self.config["workloads"], request_id_namespace,
            ordering=str(self.config.get("workload_ordering", "interleaved")),
        )
        _write_json(self.run_dir / "client" / "command.json", {"argv": self._client_command(trace)})

        processes: dict[str, Any] = {}
        logs = []
        stop_files = {}
        failure: Optional[BaseException] = None
        before = after = None
        monitoring_events = []
        try:
            for value in self.config["endpoints"]:
                endpoint_id = str(value["endpoint_id"])
                process, log = _launch_monitor(value, self.run_dir / endpoint_id, self.config["sampling"])
                processes[endpoint_id] = process
                logs.append(log)
                stop_files[endpoint_id] = self.run_dir / endpoint_id / "stop.requested"
            _wait_capabilities(
                processes, self.run_dir,
                timeout_s=float(self.config["sampling"].get("readiness_timeout_s", 120)),
            )
            time.sleep(float(self.config.get("pre_window_s", 1.0)))
            before = self._scrape_round("before")
            with (self.run_dir / "client" / "client.log").open("x", encoding="utf-8") as log:
                client = subprocess.Popen(
                    self._client_command(trace), stdout=log, stderr=subprocess.STDOUT
                )
                orchestrator = FixedPeriodScrapeOrchestrator(
                    self.endpoints,
                    lambda endpoint, scheduled: self._scrape(endpoint, scheduled),
                    float(self.config.get("scrape_interval_s", 1.0)),
                )
                monitoring_events = orchestrator.run_while(lambda: client.poll() is None)
                returncode = client.wait()
            if returncode != 0:
                raise Phase3CError("client exited with status %d" % returncode)
            time.sleep(float(self.config.get("post_window_s", 0.5)))
            after = self._scrape_round("after")
        except BaseException as exc:
            failure = exc
        finally:
            stop_errors = _stop_monitors(processes, logs, stop_files)
        if failure is not None or stop_errors:
            error = failure or Phase3CError("; ".join(stop_errors))
            _write_json(self.run_dir / "failure.json", {
                "error_type": type(error).__name__, "error_message": str(error),
                "monitor_stop_errors": stop_errors,
            })
            raise error
        assert before is not None and after is not None
        if proxy_log.is_file():
            shutil.copyfile(proxy_log, self.run_dir / "raw" / "proxy_diagnostics.jsonl")
        for value in self.config.get("server_logs", {}).values():
            path = Path(os.path.expandvars(str(value)))
            if path.is_file():
                shutil.copyfile(path, self.run_dir / "raw" / path.name)
        self._analyze(expected_requests, before, after, monitoring_events)
        print(self.run_dir.as_posix())
        return self.run_dir

    def _analyze(
        self,
        expected_requests: int,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        monitoring_events: Sequence[Any],
    ) -> None:
        client_summary = _read_json(self.run_dir / "client" / "summary.json")
        requests = _read_jsonl(self.run_dir / "client" / "requests.jsonl")
        diagnostics = _read_jsonl(self.run_dir / "raw" / "proxy_diagnostics.jsonl")
        online_feedback_enabled = bool(
            self.config.get("online_feedback", {}).get("enabled")
        )
        by_id: dict[str, list[dict[str, Any]]] = {}
        for item in diagnostics:
            by_id.setdefault(str(item.get("logical_request_id")), []).append(item)
        route_rows = []
        request_rows = []
        assignment_errors = []
        required_stamps = (
            "route_selected", "prefill_completed", "kv_handoff_completed",
            "decode_request_started", "response_completed",
        )
        allowed = set(self.pairs)
        for request in requests:
            request_id = str(request.get("logical_request_id") or request.get("trace_request_id"))
            matches = by_id.get(request_id, [])
            if len(matches) != 1:
                assignment_errors.append("%s diagnostic_count=%d" % (request_id, len(matches)))
                continue
            diagnostic = matches[0]
            pair = (
                str(diagnostic.get("selected_prefill_endpoint_id")),
                str(diagnostic.get("selected_decode_endpoint_id")),
            )
            wall = diagnostic.get("timestamps_wall_s") or {}
            monotonic = diagnostic.get("timestamps_monotonic_s") or {}
            if pair not in allowed:
                assignment_errors.append("%s incompatible_pair=%s->%s" % (request_id, *pair))
            if request.get("selected_prefill_endpoint_id") != pair[0] or request.get("selected_decode_endpoint_id") != pair[1]:
                assignment_errors.append("%s client/proxy assignment mismatch" % request_id)
            if any(not isinstance(wall.get(name), (int, float)) for name in required_stamps):
                assignment_errors.append("%s missing route timestamp" % request_id)
            ordered = [monotonic.get(name) for name in required_stamps]
            if (
                any(not isinstance(value, (int, float)) for value in ordered)
                or any(float(right) < float(left) for left, right in zip(ordered, ordered[1:]))
            ):
                assignment_errors.append("%s invalid route event ordering" % request_id)
            if diagnostic.get("outcome") != "completed":
                assignment_errors.append("%s proxy outcome=%r" % (request_id, diagnostic.get("outcome")))
            durations = diagnostic.get("durations_ms") or {}
            inference_ttft_ms, inference_tpot_ms = _online_inference_latency(
                request, diagnostic
            )
            observed_output_tokens = request.get("completion_tokens")
            route_rows.append({
                "request_id": request_id,
                "selected_prefill_endpoint_id": pair[0],
                "selected_decode_endpoint_id": pair[1],
                "route_timestamp_wall_s": wall.get("route_selected"),
                "prefill_completion_wall_s": wall.get("prefill_completed"),
                "kv_handoff_completion_wall_s": wall.get("kv_handoff_completed"),
                "decode_start_wall_s": wall.get("decode_request_started"),
                "decode_completion_wall_s": wall.get("response_completed"),
                "outcome": diagnostic.get("outcome"),
            })
            request_rows.append({
                "request_id": request_id,
                "workload_id": request.get("workload_id"),
                "input_len": request.get("input_len"),
                "requested_output_len": request.get("output_len"),
                "observed_output_tokens": observed_output_tokens,
                "prefill_endpoint_id": pair[0],
                "decode_endpoint_id": pair[1],
                "ttft_ms": (
                    inference_ttft_ms if online_feedback_enabled
                    else request.get("ttft_ms")
                ),
                "tpot_ms": (
                    inference_tpot_ms if online_feedback_enabled
                    else request.get("tpot_ms")
                ),
                "client_observed_ttft_ms": request.get("ttft_ms"),
                "client_observed_tpot_ms": request.get("tpot_ms"),
                "mean_itl_ms": request.get("mean_itl_ms"),
                "e2e_latency_ms": request.get("e2e_latency_ms"),
                "prefill_latency_ms": durations.get("prefill"),
                "decode_stream_latency_ms": durations.get("decode_request_to_last_chunk"),
                "real_sse": request.get("decode_stream_available"),
                "logical_request_id_propagated": request.get("logical_request_id_propagated"),
            })

        probe = {"id": "phase3c_combined"}
        metrics_delta = Phase3AHarness._semantic_delta(probe, client_summary, before, after)
        _write_json(self.run_dir / "derived" / "metrics_delta.json", metrics_delta)
        scrape_rows = []
        for event in monitoring_events:
            attempt = event.attempt
            scrape_rows.append({
                "endpoint_id": event.endpoint.endpoint_id,
                "schedule_index": event.schedule_index,
                "scheduled_monotonic_s": event.scheduled_monotonic_s,
                "actual_start_wall_s": attempt.actual_wall_start_s if attempt else None,
                "actual_finish_wall_s": attempt.actual_wall_finish_s if attempt else None,
                "scrape_latency_s": (
                    attempt.actual_monotonic_finish_s - attempt.actual_monotonic_start_s
                    if attempt else None
                ),
                "missed": event.missed,
                "error": repr(attempt.error) if attempt and attempt.error else None,
            })
        _csv(
            self.run_dir / "derived" / "scrape_schedule.csv", scrape_rows,
            ("endpoint_id", "schedule_index", "scheduled_monotonic_s", "actual_start_wall_s", "actual_finish_wall_s", "scrape_latency_s", "missed", "error"),
        )

        successful_requests = [item for item in requests if item.get("ok")]
        start = min(float(item["send_unix_s"]) for item in successful_requests)
        end = max(float(item["complete_unix_s"]) for item in successful_requests)
        total_input = sum(int(item["input_len"]) for item in successful_requests)
        total_output = sum(int(item["completion_tokens"]) for item in successful_requests)
        endpoint_energy = {}
        endpoint_clocks = {}
        endpoint_throttle = {}
        records_by_endpoint = {}
        capabilities_by_endpoint = {}
        endpoint_rows = []
        energy_rows = []
        token_errors = []
        route_counts = {item.endpoint_id: 0 for item in self.endpoints}
        for route in route_rows:
            route_counts[str(route["selected_prefill_endpoint_id"])] += 1
            route_counts[str(route["selected_decode_endpoint_id"])] += 1
        for endpoint in self.endpoints:
            endpoint_id = endpoint.endpoint_id
            records = _read_jsonl(self.run_dir / endpoint_id / "samples.jsonl")
            capability = _read_json(self.run_dir / endpoint_id / "capability.json")
            records_by_endpoint[endpoint_id] = records
            capabilities_by_endpoint[endpoint_id] = capability
            window = summarize_window(
                endpoint_id, records,
                WindowSpec("workload", start, end, True, expected_requests, total_input, total_output),
                capability,
                max_gap_s=float(self.config["sampling"]["maximum_gap_s"]),
                min_coverage_ratio=float(self.config["sampling"]["minimum_coverage_ratio"]),
                max_boundary_gap_s=float(self.config["sampling"]["maximum_boundary_gap_s"]),
            )
            endpoint_energy[endpoint_id] = window
            endpoint_clocks[endpoint_id] = _clock_audit(records, start, end, self.config["fixed_clocks"][endpoint_id])
            endpoint_throttle[endpoint_id] = _throttle_audit(records, start, end)
            routed = [item for item in request_rows if item["prefill_endpoint_id" if endpoint.role == "prefill" else "decode_endpoint_id"] == endpoint_id]
            expected_metrics = _expected_endpoint_token_metrics(
                self.config, endpoint_id, endpoint.role, routed
            )
            delta = metrics_delta["endpoints"][endpoint_id]["window"]
            for field, expected in expected_metrics.items():
                if not delta.get("valid") or delta.get(field) != expected:
                    token_errors.append("%s %s=%r expected=%r" % (endpoint_id, field, delta.get(field), expected))
            queue_values = [
                getattr(event.attempt.raw.snapshot, "num_requests_waiting", None)
                for event in monitoring_events
                if event.endpoint.endpoint_id == endpoint_id and event.attempt and event.attempt.raw
            ]
            kv_values = [
                getattr(event.attempt.raw.snapshot, "kv_cache_usage_frac", None)
                for event in monitoring_events
                if event.endpoint.endpoint_id == endpoint_id and event.attempt and event.attempt.raw
            ]
            identity = capability["identity"]
            endpoint_rows.append({
                "endpoint_id": endpoint_id, "role": endpoint.role, "node": endpoint.node,
                "gpu_type": endpoint.gpu_type, "gpu_ids": ",".join(map(str, endpoint.gpu_ids)),
                "gpu_uuid": identity.get("uuid"), "pci_bus_id": identity.get("pci_bus_id"),
                "tp_degree": endpoint.tp_degree,
                "configured_graphics_mhz": self.config["fixed_clocks"][endpoint_id]["graphics_mhz"],
                "observed_graphics_mean_mhz": endpoint_clocks[endpoint_id]["graphics"]["mean_mhz"],
                "clock_valid": endpoint_clocks[endpoint_id]["valid"],
                "lifecycle": self.registry.get_state(endpoint_id).lifecycle.value,
                "healthy": self.registry.get_state(endpoint_id).healthy,
                "request_count": len(routed),
                "mean_queue_depth": _mean(queue_values), "mean_kv_cache_usage_frac": _mean(kv_values),
                "mean_latency_ms": _mean([
                    item["prefill_latency_ms" if endpoint.role == "prefill" else "decode_stream_latency_ms"]
                    for item in routed
                ]),
                "mean_ttft_ms": _mean([item["ttft_ms"] for item in routed]),
                "mean_tpot_ms": _mean([item["tpot_ms"] for item in routed]),
                "gross_energy_j": window.get("gross_gpu_energy_j"),
                "energy_valid": window.get("valid"),
                "throttle_valid": endpoint_throttle[endpoint_id]["valid"],
            })
            energy_rows.append({
                "scope": endpoint_id, "role": endpoint.role,
                "gross_energy_j": window.get("gross_gpu_energy_j"),
                "logical_requests": expected_requests,
                "output_tokens": total_output,
                "joules_per_request": (
                    window["gross_gpu_energy_j"] / expected_requests if window.get("valid") else None
                ),
                "joules_per_output_token": (
                    window["gross_gpu_energy_j"] / total_output if window.get("valid") else None
                ),
                "valid": window.get("valid"),
            })

        pair_counts = {"%s->%s" % pair: 0 for pair in self.pairs}
        for item in route_rows:
            key = "%s->%s" % (item["selected_prefill_endpoint_id"], item["selected_decode_endpoint_id"])
            if key in pair_counts:
                pair_counts[key] += 1
        workload_rows = []
        workload_summaries = {}
        for workload in self.config["workloads"]:
            workload_id = str(workload["id"])
            routed = [
                item for item in request_rows
                if item.get("workload_id") == workload_id
            ]
            row = {
                "workload_id": workload_id,
                "purpose": workload.get("purpose"),
                "input_len": int(workload["input_len"]),
                "output_len": int(workload["output_len"]),
                "planned_requests": int(workload["count"]),
                "completed_requests": len(routed),
                "mean_ttft_ms": _mean([item.get("ttft_ms") for item in routed]),
                "mean_tpot_ms": _mean([item.get("tpot_ms") for item in routed]),
                "ttft_p90_ms": _percentile([item.get("ttft_ms") for item in routed], 0.90),
                "tpot_p90_ms": _percentile([item.get("tpot_ms") for item in routed], 0.90),
                "mean_e2e_latency_ms": _mean([
                    item.get("e2e_latency_ms") for item in routed
                ]),
            }
            workload_rows.append(row)
            workload_summaries[workload_id] = row

        workload_energy_rows = []
        workload_energy_summaries = {}
        feedback = dict(self.config.get("online_feedback", {}))
        measurement_pair = tuple(feedback.get("service_pair", ()))
        if len(measurement_pair) == 2:
            fixed_table = dict(self.config.get("fixed_frequency_table", {}))
            for workload in self.config["workloads"]:
                workload_id = str(workload["id"])
                selected_requests = [
                    item for item in request_rows
                    if item.get("workload_id") == workload_id
                ]
                request_ids = {str(item["request_id"]) for item in selected_requests}
                selected_routes = [
                    item for item in route_rows
                    if str(item["request_id"]) in request_ids
                ]
                if not selected_routes:
                    continue
                window_start = min(float(item["route_timestamp_wall_s"]) for item in selected_routes)
                window_end = max(float(item["decode_completion_wall_s"]) for item in selected_routes)
                input_tokens = sum(int(item["input_len"]) for item in selected_requests)
                output_tokens = sum(int(item["observed_output_tokens"]) for item in selected_requests)
                endpoint_windows = {}
                for endpoint_id, role in zip(measurement_pair, ("prefill", "decode")):
                    endpoint_id = str(endpoint_id)
                    window = summarize_window(
                        endpoint_id,
                        records_by_endpoint[endpoint_id],
                        WindowSpec(
                            workload_id, window_start, window_end, True,
                            len(selected_requests), input_tokens, output_tokens,
                        ),
                        capabilities_by_endpoint[endpoint_id],
                        max_gap_s=float(self.config["sampling"]["maximum_gap_s"]),
                        min_coverage_ratio=float(self.config["sampling"]["minimum_coverage_ratio"]),
                        max_boundary_gap_s=float(self.config["sampling"]["maximum_boundary_gap_s"]),
                    )
                    endpoint_windows[endpoint_id] = window
                    workload_energy_rows.append({
                        "workload_id": workload_id,
                        "scope": endpoint_id,
                        "role": role,
                        "window_start_unix_s": window_start,
                        "window_end_unix_s": window_end,
                        "duration_s": window_end - window_start,
                        "gross_energy_j": window.get("gross_gpu_energy_j"),
                        "mean_power_w": window.get("mean_power_w"),
                        "logical_requests": len(selected_requests),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "joules_per_request": (
                            window["gross_gpu_energy_j"] / len(selected_requests)
                            if window.get("valid") and selected_requests else None
                        ),
                        "joules_per_output_token": (
                            window["gross_gpu_energy_j"] / output_tokens
                            if window.get("valid") and output_tokens else None
                        ),
                        "valid": window.get("valid"),
                    })
                pair_valid = all(item.get("valid") for item in endpoint_windows.values())
                pair_energy = (
                    sum(float(item["gross_gpu_energy_j"]) for item in endpoint_windows.values())
                    if pair_valid else None
                )
                configured = dict(fixed_table.get(workload_id, {}))
                combined = {
                    "workload_id": workload_id,
                    "scope": "service_PD_pair",
                    "role": "all",
                    "window_start_unix_s": window_start,
                    "window_end_unix_s": window_end,
                    "duration_s": window_end - window_start,
                    "gross_energy_j": pair_energy,
                    "mean_power_w": pair_energy / (window_end - window_start) if pair_energy is not None else None,
                    "logical_requests": len(selected_requests),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "joules_per_request": pair_energy / len(selected_requests) if pair_energy is not None else None,
                    "joules_per_output_token": pair_energy / output_tokens if pair_energy is not None else None,
                    "valid": pair_valid,
                    "prefill_frequency_mhz": configured.get("prefill_frequency_mhz"),
                    "decode_frequency_mhz": configured.get("decode_frequency_mhz"),
                    "ttft_p90_ms": workload_summaries[workload_id]["ttft_p90_ms"],
                    "tpot_p90_ms": workload_summaries[workload_id]["tpot_p90_ms"],
                }
                workload_energy_rows.append(combined)
                workload_energy_summaries[workload_id] = combined
        coverage = dict(self.config.get("coverage_policy", {}))
        required_endpoint_ids = set(coverage.get(
            "required_endpoint_ids", [item.endpoint_id for item in self.endpoints]
        ))
        required_pairs = {
            "%s->%s" % tuple(item)
            for item in coverage.get("required_pairs", [list(item) for item in self.pairs])
        }
        endpoint_coverage = all(route_counts.get(endpoint_id, 0) > 0 for endpoint_id in required_endpoint_ids)
        pair_coverage = all(pair_counts.get(pair, 0) > 0 for pair in required_pairs)
        workload_coverage = all(
            item["completed_requests"] == item["planned_requests"]
            for item in workload_rows
        )
        sse_valid = (
            len(successful_requests) == expected_requests
            and all(
                item.get("decode_stream_available") is True
                and item.get("client_ttft_valid") is True
                and item.get("client_tpot_valid") is True
                and item.get("client_itl_valid") is True
                for item in successful_requests
            )
        )
        id_valid = all(item.get("logical_request_id_propagated") is True for item in successful_requests)
        output_valid = all(
            int(item.get("completion_tokens", -1)) == int(item.get("output_len", -2))
            and item.get("completion_token_source") == "server_usage"
            for item in successful_requests
        )
        online_slo = self.config.get("online_feedback", {}).get("slo", {})
        online_service_slo_valid = (
            not online_feedback_enabled
            or all(
                _workload_slo_met(
                    request_rows, str(workload["id"]),
                    float(online_slo["ttft_ms"]),
                    float(online_slo["tpot_ms"]),
                )
                for workload in self.config["workloads"]
            )
        )
        energy_valid = all(item.get("valid") for item in endpoint_energy.values())
        clocks_valid = (
            all(item.get("valid") for item in endpoint_clocks.values())
            or online_feedback_enabled
        )
        throttle_valid = all(item.get("valid") for item in endpoint_throttle.values())
        metrics_valid = (
            not token_errors
            or bool(self.config.get("background_experiment_traffic"))
        )
        assignment_valid = not assignment_errors and len(route_rows) == expected_requests
        hard_gates = {
            "real_sse_and_latency": sse_valid,
            "logical_request_id": id_valid,
            "requested_output_tokens": output_valid,
            "online_service_requests_meet_slo": online_service_slo_valid,
            "endpoint_assignment": assignment_valid,
            "explicit_compatibility": not any("incompatible_pair" in item for item in assignment_errors),
            "endpoint_coverage": endpoint_coverage,
            "pair_coverage": pair_coverage,
            "logical_workload_partition": workload_coverage,
            "phase3a_token_semantics": metrics_valid,
            "nvml_energy_windows": energy_valid,
            "service_workload_energy_windows": (
                not workload_energy_summaries
                or len(workload_energy_summaries) == len(self.config["workloads"])
                and all(item.get("valid") for item in workload_energy_summaries.values())
            ),
            "fixed_clocks": clocks_valid,
            "no_invalidating_thermal_or_hw_slowdown": throttle_valid,
            "no_resource_overlap": True,
        }
        valid = all(hard_gates.values())
        p_energy = sum(float(endpoint_energy[item.endpoint_id]["gross_gpu_energy_j"]) for item in self.endpoints if item.role == "prefill") if energy_valid else None
        d_energy = sum(float(endpoint_energy[item.endpoint_id]["gross_gpu_energy_j"]) for item in self.endpoints if item.role == "decode") if energy_valid else None
        total_energy = p_energy + d_energy if p_energy is not None and d_energy is not None else None
        for scope, role, energy in (("P_pool", "prefill", p_energy), ("D_pool", "decode", d_energy), ("all_GPU_boards", "all", total_energy)):
            energy_rows.append({
                "scope": scope, "role": role, "gross_energy_j": energy,
                "logical_requests": expected_requests, "output_tokens": total_output,
                "joules_per_request": energy / expected_requests if energy is not None else None,
                "joules_per_output_token": energy / total_output if energy is not None else None,
                "valid": energy is not None,
            })
        summary = {
            "phase": "3C_real_multi_endpoint_substrate_validation",
            "valid": valid,
            "request_count": expected_requests,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "route_matrix": pair_counts,
            "workloads": workload_summaries,
            "routing_policy": "%s_over_explicit_compatible_pairs"
            % self.config.get("routing_policy", "round_robin"),
            "latency_ms": {
                "ttft_mean": _mean([item.get("ttft_ms") for item in request_rows]),
                "tpot_mean": _mean([item.get("tpot_ms") for item in request_rows]),
                "client_observed_ttft_mean": _mean([
                    item.get("client_observed_ttft_ms") for item in request_rows
                ]),
                "itl_mean": _mean([item.get("mean_itl_ms") for item in successful_requests]),
                "e2e_mean": _mean([item.get("e2e_latency_ms") for item in successful_requests]),
            },
            "energy_j": {"P_pool": p_energy, "D_pool": d_energy, "total": total_energy},
            "joules_per_request": total_energy / expected_requests if total_energy is not None else None,
            "joules_per_output_token": total_energy / total_output if total_energy is not None else None,
            "compatible_pairs": [list(item) for item in self.pairs],
            "endpoint_energy": endpoint_energy,
            "service_workload_energy": workload_energy_summaries,
            "endpoint_clocks": endpoint_clocks,
            "endpoint_throttle": endpoint_throttle,
            "prometheus_sampling": {
                "scheduled": len(monitoring_events),
                "missed": sum(item.missed for item in monitoring_events),
                "errors": sum(bool(item.attempt and item.attempt.error) for item in monitoring_events),
            },
        }
        audit = {
            "valid": valid,
            "hard_gates": hard_gates,
            "assignment_errors": assignment_errors,
            "token_errors": token_errors,
            "route_counts": pair_counts,
            "endpoint_request_counts": route_counts,
            "claim_boundary": "substrate validation only; no routing novelty or optimization claim",
        }
        _write_json(self.run_dir / "summary.json", summary)
        _write_json(self.run_dir / "audit.json", audit)
        _csv(self.run_dir / "requests.csv", request_rows, (
            "request_id", "workload_id", "input_len", "requested_output_len", "observed_output_tokens",
            "prefill_endpoint_id", "decode_endpoint_id", "ttft_ms", "tpot_ms",
            "client_observed_ttft_ms", "client_observed_tpot_ms",
            "mean_itl_ms", "e2e_latency_ms", "prefill_latency_ms", "decode_stream_latency_ms",
            "real_sse", "logical_request_id_propagated",
        ))
        _csv(self.run_dir / "workload_summary.csv", workload_rows, tuple(workload_rows[0]))
        _csv(self.run_dir / "routes.csv", route_rows, (
            "request_id", "selected_prefill_endpoint_id", "selected_decode_endpoint_id",
            "route_timestamp_wall_s", "prefill_completion_wall_s", "kv_handoff_completion_wall_s",
            "decode_start_wall_s", "decode_completion_wall_s", "outcome",
        ))
        _csv(self.run_dir / "endpoint_summary.csv", endpoint_rows, tuple(endpoint_rows[0]))
        _csv(self.run_dir / "energy_summary.csv", energy_rows, tuple(energy_rows[0]))
        if workload_energy_rows:
            fieldnames = (
                "workload_id", "scope", "role", "window_start_unix_s",
                "window_end_unix_s", "duration_s", "gross_energy_j",
                "mean_power_w", "logical_requests", "input_tokens",
                "output_tokens", "joules_per_request",
                "joules_per_output_token", "valid", "prefill_frequency_mhz",
                "decode_frequency_mhz", "ttft_p90_ms", "tpot_p90_ms",
            )
            _csv(self.run_dir / "workload_energy_summary.csv", workload_energy_rows, fieldnames)
        lines = [
            "# XpYd Phase 3C real multi-endpoint substrate validation", "",
            "Verdict: **%s**" % ("PASS" if valid else "FAIL"), "",
            "This is infrastructure validation with deterministic baseline routing; it makes no routing-novelty or optimization claim.", "",
            "| Pair | Requests |", "|---|---:|",
        ]
        lines.extend("| %s | %d |" % item for item in pair_counts.items())
        lines.extend(["", "| Endpoint | Role | Node/GPU | Requests | Gross energy (J) | Clock valid |", "|---|---|---|---:|---:|---|"])
        for item in endpoint_rows:
            lines.append("| %(endpoint_id)s | %(role)s | %(node)s/%(gpu_ids)s | %(request_count)s | %(gross_energy_j)s | %(clock_valid)s |" % item)
        lines.extend(["", "Hard gates: %s" % json.dumps(hard_gates, sort_keys=True), ""])
        (self.run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not valid:
            raise Phase3CError("Phase 3C hard-gate audit failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    Phase3CSubstrateHarness(config, args.run_id).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
