"""Online 2P2D service/experiment feedback controller.

P1/D1 always serves the original request.  A missing workload-table entry
causes one deduplicated clone to be explored serially on P0/D0. Exploration
searches P first and D second. Each axis uses binary search to locate the
lowest SLO-feasible grid index, then follows measured neighboring frequencies
to a local minimum-energy valley. Binary measurements are reused, and minimum
frequency is not treated as minimum energy.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from xpyd.phase3d_control import NvidiaSmiClockBackend, PerEndpointClockActuator
from xpyd.workload_frequency_table import WorkloadFrequencyTable


WORKLOAD_SHAPES = {
    "small_light": (128, 64),
    "prefill_medium": (1024, 64),
    "prefill_heavy": (2048, 64),
    "decode_medium": (128, 128),
    "decode_heavy": (128, 256),
    "balanced_medium": (512, 128),
    "both_heavy": (2048, 256),
}
DEFAULT_AXIS_SEARCH_LEVELS = {"prefill": 17, "decode": 15}
_JSONL_APPEND_LOCK = threading.Lock()


class OnlineFeedbackError(RuntimeError):
    pass


def classify_workload(body: Mapping[str, Any]) -> str:
    """Classify one request from explicit token-shape request parameters."""
    try:
        input_len = int(body["xpyd_input_len"])
        output_len = int(body.get("xpyd_output_len", body["max_tokens"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlineFeedbackError(
            "feedback requests require xpyd_input_len and max_tokens"
        ) from exc
    declared = body.get("xpyd_workload_id")
    matches = [key for key, shape in WORKLOAD_SHAPES.items()
               if shape == (input_len, output_len)]
    if len(matches) != 1:
        raise OnlineFeedbackError(
            "unsupported workload token shape: %d/%d" % (input_len, output_len)
        )
    if declared is not None and str(declared) != matches[0]:
        raise OnlineFeedbackError("declared workload does not match token shape")
    return matches[0]


def strip_feedback_metadata(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    for key in ("xpyd_input_len", "xpyd_output_len", "xpyd_workload_id"):
        value.pop(key, None)
    return value


@dataclass(frozen=True)
class ProbeResult:
    """One raw probe or an aggregate whose latency fields are P95 values."""
    prefill_frequency_mhz: int
    decode_frequency_mhz: int
    measured_power_w: float
    measured_energy_j: float
    ttft_ms: float
    tpot_ms: float
    sample_count: int = 1

    def feasible(self, ttft_slo_ms: float, tpot_slo_ms: float) -> bool:
        return (
            math.isfinite(self.measured_power_w) and self.measured_power_w > 0
            and math.isfinite(self.measured_energy_j)
            and self.measured_energy_j > 0
            and self.ttft_ms < ttft_slo_ms
            and self.tpot_ms <= tpot_slo_ms
        )


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise OnlineFeedbackError("cannot compute percentile without samples")
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_probe_results(results: Sequence[ProbeResult]) -> ProbeResult:
    if not results:
        raise OnlineFeedbackError("candidate requires at least one probe result")
    first = results[0]
    if any(
        item.prefill_frequency_mhz != first.prefill_frequency_mhz
        or item.decode_frequency_mhz != first.decode_frequency_mhz
        for item in results
    ):
        raise OnlineFeedbackError("cannot aggregate different frequency candidates")
    return ProbeResult(
        prefill_frequency_mhz=first.prefill_frequency_mhz,
        decode_frequency_mhz=first.decode_frequency_mhz,
        measured_power_w=statistics.fmean(item.measured_power_w for item in results),
        measured_energy_j=statistics.fmean(item.measured_energy_j for item in results),
        ttft_ms=percentile([item.ttft_ms for item in results], 0.95),
        tpot_ms=percentile([item.tpot_ms for item in results], 0.95),
        sample_count=sum(item.sample_count for item in results),
    )


def coefficient_of_variation(values: Sequence[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return math.inf
    mean = statistics.fmean(numbers)
    if not math.isfinite(mean) or mean <= 0:
        return math.inf
    return statistics.pstdev(numbers) / mean


Actuate = Callable[[str, int, int], Awaitable[bool]]
Probe = Callable[[Mapping[str, Any], str, int, int, str], Awaitable[ProbeResult]]
Sleep = Callable[[float], Awaitable[None]]


def pd_inference_metrics(
    stamps: Mapping[str, Any], output_len: int,
) -> tuple[float, float]:
    """Return PD inference TTFT/TPOT, starting after DVFS settling."""
    request_received = stamps.get("request_received")
    first = stamps.get("decode_first_real_chunk_received")
    last = stamps.get("decode_last_chunk_received")
    if any(value is None for value in (request_received, first, last)):
        raise OnlineFeedbackError("request lacks complete PD latency timestamps")
    ttft_ms = (float(first) - float(request_received)) * 1000.0
    tpot_ms = ((float(last) - float(first)) * 1000.0 / max(1, output_len - 1))
    if ttft_ms < 0 or tpot_ms < 0:
        raise OnlineFeedbackError("request has invalid PD latency timestamp order")
    return ttft_ms, tpot_ms


class OnlineFeedbackController:
    """Concurrent-safe service routing plus one serialized exploration worker."""

    def __init__(
        self, table: WorkloadFrequencyTable, service_core: Any,
        actuate: Actuate, probe: Probe, prefill_grid: Sequence[int],
        decode_grid: Sequence[int], *, probe_interval_s: float = 0.0,
        service_settle_s: float = 10.0,
        ttft_slo_ms: float = 500.0, tpot_slo_ms: float = 200.0,
        service_warmup_requests: int = 0,
        experiment_warmup_requests: int = 0,
        probe_samples_per_candidate: int = 3,
        minimum_probe_samples: int = 3,
        candidate_stability_cv: float = 0.05,
        candidate_slo_headroom_ratio: float = 0.90,
        exploration_shutdown_timeout_s: float = 36000.0,
        energy_refinement_candidate_budget: int = 9,
        exploration_max_attempts: int = 3,
        exploration_retry_backoff_s: float = 5.0,
        event_log: Optional[str] = None,
        service_request_log: Optional[str] = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.table = table
        self.service_core = service_core
        self.actuate = actuate
        self.probe = probe
        self.prefill_grid = tuple(sorted(set(map(int, prefill_grid))))
        self.decode_grid = tuple(sorted(set(map(int, decode_grid))))
        if len(self.prefill_grid) < 2 or len(self.decode_grid) < 2:
            raise OnlineFeedbackError("both frequency grids need at least two levels")
        if not 0.0 <= float(probe_interval_s) <= 5.0:
            raise OnlineFeedbackError("probe interval must be 0..5 seconds")
        self.probe_interval_s = float(probe_interval_s)
        if not 0.0 <= float(service_settle_s) <= 60.0:
            raise OnlineFeedbackError("service frequency settle must be 0..60 seconds")
        self.service_settle_s = float(service_settle_s)
        self.ttft_slo_ms = float(ttft_slo_ms)
        self.tpot_slo_ms = float(tpot_slo_ms)
        if not 0 <= int(service_warmup_requests) <= 5:
            raise OnlineFeedbackError("service warmup requests must be 0..5")
        if not 0 <= int(experiment_warmup_requests) <= 5:
            raise OnlineFeedbackError("experiment warmup requests must be 0..5")
        self.service_warmup_requests = int(service_warmup_requests)
        self.experiment_warmup_requests = int(experiment_warmup_requests)
        if not 2 <= int(probe_samples_per_candidate) <= 20:
            raise OnlineFeedbackError("probe samples per candidate must be 2..20")
        self.probe_samples_per_candidate = int(probe_samples_per_candidate)
        if not 2 <= int(minimum_probe_samples) <= self.probe_samples_per_candidate:
            raise OnlineFeedbackError("minimum probe samples must be 2..candidate maximum")
        self.minimum_probe_samples = int(minimum_probe_samples)
        if not 0.0 < float(candidate_stability_cv) <= 0.25:
            raise OnlineFeedbackError("candidate stability CV must be in (0, 0.25]")
        self.candidate_stability_cv = float(candidate_stability_cv)
        if not 0.5 <= float(candidate_slo_headroom_ratio) <= 1.0:
            raise OnlineFeedbackError("candidate SLO headroom ratio must be 0.5..1.0")
        self.candidate_slo_headroom_ratio = float(candidate_slo_headroom_ratio)
        if not 60.0 <= float(exploration_shutdown_timeout_s) <= 43200.0:
            raise OnlineFeedbackError(
                "exploration shutdown timeout must be 60..43200 seconds"
            )
        self.exploration_shutdown_timeout_s = float(exploration_shutdown_timeout_s)
        if not 5 <= int(energy_refinement_candidate_budget) <= 17:
            raise OnlineFeedbackError(
                "energy refinement candidate budget must be 5..17"
            )
        self.energy_refinement_candidate_budget = int(
            energy_refinement_candidate_budget
        )
        if not 1 <= int(exploration_max_attempts) <= 5:
            raise OnlineFeedbackError("exploration max attempts must be 1..5")
        if not 0.0 <= float(exploration_retry_backoff_s) <= 60.0:
            raise OnlineFeedbackError("exploration retry backoff must be 0..60 seconds")
        self.exploration_max_attempts = int(exploration_max_attempts)
        self.exploration_retry_backoff_s = float(exploration_retry_backoff_s)
        self.event_log = Path(os.path.expandvars(event_log)) if event_log else None
        self.service_request_log = (
            Path(os.path.expandvars(service_request_log))
            if service_request_log else None
        )
        self.sleep = sleep
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], str]] = asyncio.Queue()
        self._pending: set[str] = set()
        self._exploration_attempts: dict[str, int] = {}
        self._pending_lock = asyncio.Lock()
        self._service_lock = asyncio.Lock()
        self._worker: Optional[asyncio.Task[Any]] = None
        self._active_workload: Optional[str] = None
        self._service_warmed = False
        self._experiment_warmed = False
        self._service_sequence = 0

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._log({
                "event": "exploration_shutdown_wait_started",
                "timeout_s": self.exploration_shutdown_timeout_s,
            })
            try:
                await asyncio.wait_for(
                    self._queue.join(), timeout=self.exploration_shutdown_timeout_s
                )
            except asyncio.TimeoutError:
                self._log({
                    "event": "exploration_shutdown_wait_timed_out",
                    "timeout_s": self.exploration_shutdown_timeout_s,
                })
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def status(self) -> dict[str, Any]:
        return {
            "active_workload": self._active_workload,
            "pending_workloads": sorted(self._pending),
            "queue_depth": self._queue.qsize(),
            "table": self.table.snapshot(),
        }

    def _log(self, value: Mapping[str, Any]) -> None:
        if self.event_log is None:
            return
        row = dict(value) | {"timestamp_unix_s": time.time()}
        self._append_jsonl(self.event_log, row)

    def _append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        with _JSONL_APPEND_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(value), sort_keys=True) + "\n")

    def _record_service_request(self, value: Mapping[str, Any]) -> None:
        if self.service_request_log is None:
            return
        self._append_jsonl(
            self.service_request_log,
            dict(value) | {"dispatch_unix_s": time.time()},
        )

    async def _enqueue(self, workload_id: str, body: Mapping[str, Any], request_id: str) -> None:
        async with self._pending_lock:
            if self.table.read(workload_id).value is not None or workload_id in self._pending:
                return
            self._pending.add(workload_id)
            await self._queue.put((workload_id, copy.deepcopy(dict(body)), request_id))
            self._log({"event": "exploration_queued", "workload_id": workload_id})

    async def handle(self, body: Mapping[str, Any], logical_request_id: str) -> Any:
        workload_id = classify_workload(body)
        async with self._service_lock:
            entry = self.table.read(workload_id)
            if entry.value is None:
                p_mhz, d_mhz = self.prefill_grid[-1], self.decode_grid[-1]
                frequency_changed = await self.actuate(
                    "service", p_mhz, d_mhz
                )
            else:
                p_mhz = entry.value.prefill_frequency_mhz
                d_mhz = entry.value.decode_frequency_mhz
                frequency_changed = await self.actuate(
                    "service", p_mhz, d_mhz,
                )
            settle_wait_s = self.service_settle_s if frequency_changed else 0.0
            if frequency_changed and self.service_settle_s:
                self._log({
                    "event": "service_frequency_settle",
                    "workload_id": workload_id,
                    "settle_s": self.service_settle_s,
                })
                await self.sleep(self.service_settle_s)
            await self._ensure_service_warmup(body, workload_id, logical_request_id)
            if entry.value is None:
                await self._enqueue(workload_id, body, logical_request_id)
            self._service_sequence += 1
            self._record_service_request({
                "schema_version": 1,
                "request_id": logical_request_id,
                "service_sequence": self._service_sequence,
                "workload_id": workload_id,
                "input_len": int(body["xpyd_input_len"]),
                "output_len": int(body.get("xpyd_output_len", body["max_tokens"])),
                "table_hit": entry.value is not None,
                "table_revision": entry.revision,
                "frequency_source": "table" if entry.value is not None else "safe_high",
                "prefill_frequency_mhz": int(p_mhz),
                "decode_frequency_mhz": int(d_mhz),
                "frequency_changed": frequency_changed,
                "settle_wait_s": settle_wait_s,
                "prefill_endpoint_id": "P1",
                "decode_endpoint_id": "D1",
            })
            return await self.service_core.prepare(
                strip_feedback_metadata(body), logical_request_id
            )

    @staticmethod
    async def _consume_prepared(prepared: Any) -> None:
        stream = getattr(prepared, "stream", None)
        if stream is not None:
            async for _ in stream:
                pass

    async def _ensure_service_warmup(
        self, body: Mapping[str, Any], workload_id: str, request_id: str,
    ) -> None:
        if self._service_warmed or self.service_warmup_requests == 0:
            return
        self._log({
            "event": "service_warmup_started",
            "workload_id": workload_id,
            "request_count": self.service_warmup_requests,
        })
        warmup_body = strip_feedback_metadata(dict(body) | {"stream": True})
        for index in range(self.service_warmup_requests):
            warmup_id = "%s-service-warmup-%d" % (request_id, index + 1)
            prepared = await self.service_core.prepare(warmup_body, warmup_id)
            await self._consume_prepared(prepared)
            self._log({
                "event": "service_warmup_completed",
                "workload_id": workload_id,
                "request_id": warmup_id,
                "warmup_index": index + 1,
            })
        self._service_warmed = True

    async def _worker_loop(self) -> None:
        while True:
            workload_id, body, request_id = await self._queue.get()
            self._active_workload = workload_id
            retry_queued = False
            attempt = self._exploration_attempts.get(workload_id, 0) + 1
            self._exploration_attempts[workload_id] = attempt
            try:
                await self._explore(workload_id, body, request_id)
            except Exception as exc:
                self._log({"event": "exploration_failed", "workload_id": workload_id,
                           "attempt": attempt,
                           "error": "%s: %s" % (type(exc).__name__, exc)})
                if attempt < self.exploration_max_attempts:
                    if self.exploration_retry_backoff_s:
                        await self.sleep(self.exploration_retry_backoff_s)
                    await self._queue.put((workload_id, body, request_id))
                    retry_queued = True
                    self._log({
                        "event": "exploration_retry_queued",
                        "workload_id": workload_id,
                        "next_attempt": attempt + 1,
                    })
            finally:
                if not retry_queued:
                    async with self._pending_lock:
                        self._pending.discard(workload_id)
                    self._exploration_attempts.pop(workload_id, None)
                self._active_workload = None
                self._queue.task_done()

    async def _one_probe(
        self, body: Mapping[str, Any], workload_id: str,
        p_mhz: int, d_mhz: int, probe_id: str,
    ) -> ProbeResult:
        result = await self.probe(body, workload_id, p_mhz, d_mhz, probe_id)
        self._log({"event": "probe_completed", "workload_id": workload_id,
                   "probe_id": probe_id, **result.__dict__,
                   "slo_met": result.feasible(self.ttft_slo_ms, self.tpot_slo_ms)})
        return result

    async def _probe_candidate(
        self, body: Mapping[str, Any], workload_id: str,
        p_mhz: int, d_mhz: int, probe_id: str, *, adaptive: bool = True,
    ) -> ProbeResult:
        if self.probe_interval_s:
            await self.sleep(self.probe_interval_s)
        samples = []
        early_stop_reason = None
        for index in range(self.probe_samples_per_candidate):
            samples.append(await self._one_probe(
                body, workload_id, p_mhz, d_mhz,
                "%s-sample-%d" % (probe_id, index + 1),
            ))
            if adaptive and len(samples) >= self.minimum_probe_samples:
                early_stop_reason = self._candidate_early_stop_reason(samples)
                if early_stop_reason is not None:
                    self._log({
                        "event": "probe_candidate_early_stop",
                        "workload_id": workload_id,
                        "probe_id": probe_id,
                        "sample_count": len(samples),
                        "reason": early_stop_reason,
                    })
                    break
        result = aggregate_probe_results(samples)
        self._log({
            "event": "probe_candidate_aggregated",
            "workload_id": workload_id,
            "probe_id": probe_id,
            **result.__dict__,
            "latency_statistic": "p95",
            "energy_statistic": "mean_request_energy_j",
            "slo_met": result.feasible(self.ttft_slo_ms, self.tpot_slo_ms),
            "adaptive": adaptive,
            "early_stop_reason": early_stop_reason,
        })
        return result

    def _candidate_early_stop_reason(
        self, samples: Sequence[ProbeResult],
    ) -> Optional[str]:
        if all(not item.feasible(self.ttft_slo_ms, self.tpot_slo_ms) for item in samples):
            return "all_minimum_samples_outside_slo"
        aggregate = aggregate_probe_results(samples)
        stable = all(
            coefficient_of_variation(values) <= self.candidate_stability_cv
            for values in (
                [item.measured_energy_j for item in samples],
                [item.ttft_ms for item in samples],
                [item.tpot_ms for item in samples],
            )
        )
        has_headroom = (
            aggregate.ttft_ms < self.ttft_slo_ms * self.candidate_slo_headroom_ratio
            and aggregate.tpot_ms <= self.tpot_slo_ms * self.candidate_slo_headroom_ratio
        )
        if stable and has_headroom:
            return "stable_with_slo_headroom"
        return None

    async def _ensure_experiment_warmup(
        self, body: Mapping[str, Any], workload_id: str, request_id: str,
    ) -> None:
        if self._experiment_warmed or self.experiment_warmup_requests == 0:
            return
        self._log({
            "event": "experiment_warmup_started",
            "workload_id": workload_id,
            "request_count": self.experiment_warmup_requests,
        })
        for index in range(self.experiment_warmup_requests):
            warmup_id = "%s-warmup-%d" % (request_id, index + 1)
            result = await self.probe(
                body, workload_id, self.prefill_grid[-1],
                self.decode_grid[-1], warmup_id,
            )
            self._log({
                "event": "experiment_warmup_completed",
                "workload_id": workload_id,
                "probe_id": warmup_id,
                "warmup_index": index + 1,
                **result.__dict__,
            })
        self._experiment_warmed = True

    async def _search_axis(
        self, axis: str, grid: Sequence[int], fixed_mhz: int,
        body: Mapping[str, Any], workload_id: str, request_id: str,
    ) -> tuple[int, ProbeResult]:
        measured: dict[int, ProbeResult] = {}

        async def measure(candidate: int, phase: str) -> ProbeResult:
            if candidate not in measured:
                frequency = int(grid[candidate])
                p_mhz = frequency if axis == "P" else int(fixed_mhz)
                d_mhz = int(fixed_mhz) if axis == "P" else frequency
                measured[candidate] = await self._probe_candidate(
                    body, workload_id, p_mhz, d_mhz,
                    "%s-%s-%s-%d" % (request_id, phase, axis, candidate),
                )
            return measured[candidate]

        low, high = 0, len(grid) - 1
        boundary = None
        while low <= high:
            candidate = (low + high) // 2
            result = await measure(candidate, "binary-slo")
            if result.feasible(self.ttft_slo_ms, self.tpot_slo_ms):
                boundary = candidate
                high = candidate - 1
            else:
                low = candidate + 1
        if boundary is None:
            raise OnlineFeedbackError("no SLO-safe %s frequency" % axis)

        # Feasibility is assumed monotone, while energy is locally valley-like.
        # Reuse the binary probes, then walk from their best feasible energy
        # point until both adjacent levels have been measured without finding
        # an improvement. A hard candidate budget bounds Canary time.
        while len(measured) < min(len(grid), self.energy_refinement_candidate_budget):
            safe = [
                (candidate, result) for candidate, result in measured.items()
                if candidate >= boundary
                and result.feasible(self.ttft_slo_ms, self.tpot_slo_ms)
            ]
            current, _ = min(
                safe, key=lambda item: (item[1].measured_energy_j, item[0])
            )
            neighbors = [
                candidate for candidate in (current - 1, current + 1)
                if boundary <= candidate < len(grid) and candidate not in measured
            ]
            if not neighbors:
                break
            for candidate in neighbors:
                if len(measured) >= self.energy_refinement_candidate_budget:
                    break
                await measure(candidate, "energy-refine")
        feasible = [
            (int(grid[candidate]), result)
            for candidate, result in measured.items()
            if candidate >= boundary
            and result.feasible(self.ttft_slo_ms, self.tpot_slo_ms)
        ]
        self._log({
            "event": "axis_search_completed",
            "workload_id": workload_id,
            "axis": axis,
            "slo_boundary_index": boundary,
            "slo_boundary_mhz": int(grid[boundary]),
            "unique_candidate_count": len(measured),
            "feasible_candidate_count": len(feasible),
            "objective": "minimum_mean_request_energy_j",
            "search_strategy": "binary_slo_then_bounded_neighbor_energy_refinement",
            "candidate_budget": self.energy_refinement_candidate_budget,
        })
        return min(
            feasible,
            key=lambda item: (item[1].measured_energy_j, item[0]),
        )

    async def _explore(
        self, workload_id: str, body: Mapping[str, Any], request_id: str,
    ) -> None:
        await self._ensure_experiment_warmup(body, workload_id, request_id)
        p_mhz, _ = await self._search_axis(
            "P", self.prefill_grid, self.decode_grid[-1],
            body, workload_id, request_id,
        )
        d_mhz, _ = await self._search_axis(
            "D", self.decode_grid, p_mhz, body, workload_id, request_id,
        )
        final = await self._probe_candidate(
            body, workload_id, p_mhz, d_mhz, "%s-confirm" % request_id,
            adaptive=False,
        )
        if not final.feasible(self.ttft_slo_ms, self.tpot_slo_ms):
            raise OnlineFeedbackError("joint P/D confirmation failed SLO")
        self.table.write(workload_id, {
            "prefill_frequency_mhz": p_mhz,
            "decode_frequency_mhz": d_mhz,
            "measured_power_w": final.measured_power_w,
            "measured_energy_j": final.measured_energy_j,
            "ttft_ms": final.ttft_ms,
            "tpot_ms": final.tpot_ms,
            "prefill_endpoint_id": "P0",
            "decode_endpoint_id": "D0",
            "sample_count": final.sample_count,
            "source": "binary_slo_bounded_neighbor_energy_refinement_p95",
            "slo_met": True,
        })
        self._log({"event": "table_updated", "workload_id": workload_id,
                   "prefill_frequency_mhz": p_mhz,
                   "decode_frequency_mhz": d_mhz})


def select_grid(
    supported: Sequence[int], minimum: int, maximum: int, levels: int,
    preferred: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    values = sorted({int(item) for item in supported if minimum <= int(item) <= maximum})
    if maximum not in values or len(values) < levels or levels < 2:
        raise OnlineFeedbackError("hardware frequency grid cannot satisfy configuration")
    if preferred is not None:
        grid = tuple(sorted({int(item) for item in preferred}))
        if len(grid) != levels or grid[-1] != maximum or any(item not in values for item in grid):
            raise OnlineFeedbackError("preferred frequency grid is not hardware-supported")
        return grid
    indices = [round(index * (len(values) - 1) / (levels - 1)) for index in range(levels)]
    grid = tuple(values[index] for index in indices)
    if len(set(grid)) != levels:
        raise OnlineFeedbackError("frequency grid is not strictly ordered")
    return grid


class PhysicalFeedbackRuntime:
    """Real Slurm/NVML actuator and repeated-request energy probe backend."""

    def __init__(self, config: Mapping[str, Any], experiment_core: Any) -> None:
        self.config = config
        self.experiment_core = experiment_core
        self.backend = NvidiaSmiClockBackend()
        endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
        settings = config["online_feedback"]
        configured_levels = settings.get("axis_search_levels", {})
        capabilities = {}
        grids = {}
        for endpoint_id in ("P0", "P1", "D0", "D1"):
            endpoint = endpoints[endpoint_id]
            role = "prefill" if endpoint_id.startswith("P") else "decode"
            grid_spec = settings["frequency_grids"][role]
            capability = self.backend.discover(
                endpoint_id, str(endpoint["node"]), int(endpoint["gpu_ids"][0]),
                int(grid_spec["maximum_mhz"]),
            )
            capabilities[endpoint_id] = capability
            grids[endpoint_id] = select_grid(
                capability.supported_graphics_mhz, int(grid_spec["minimum_mhz"]),
                int(grid_spec["maximum_mhz"]),
                int(configured_levels.get(role, DEFAULT_AXIS_SEARCH_LEVELS[role])),
                grid_spec.get("candidate_mhz"),
            )
        self.prefill_grid = grids["P0"]
        self.decode_grid = grids["D0"]
        self.actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(settings.get("minimum_dwell_s", 1.0))
        )
        self.endpoints = endpoints
        self.event_log = Path(os.path.expandvars(str(settings["event_log"])))

    def _log_actuation(self, group: str, result: Mapping[str, Any]) -> None:
        row = dict(result) | {"event": "frequency_actuation", "group": group}
        with _JSONL_APPEND_LOCK:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _actuate_sync(self, group: str, p_mhz: int, d_mhz: int) -> bool:
        ids = ("P1", "D1") if group == "service" else ("P0", "D0")
        changed = False
        for endpoint_id, target in zip(ids, (p_mhz, d_mhz)):
            if self.actuator.requested.get(endpoint_id) == int(target):
                continue
            result = self.actuator.actuate(endpoint_id, int(target), "online_feedback_%s" % group)
            self._log_actuation(group, result)
            if result["command_status"] != "success" or not result["readback_valid"]:
                raise OnlineFeedbackError("frequency actuation failed for %s" % endpoint_id)
            changed = True
        return changed

    async def actuate(self, group: str, p_mhz: int, d_mhz: int) -> bool:
        return await asyncio.to_thread(self._actuate_sync, group, p_mhz, d_mhz)

    def _energy_mj(self, endpoint_id: str) -> float:
        endpoint = self.endpoints[endpoint_id]
        code = (
            "import pynvml; pynvml.nvmlInit(); "
            "h=pynvml.nvmlDeviceGetHandleByIndex(%d); "
            "print(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))"
            % int(endpoint["gpu_ids"][0])
        )
        command = [
            "srun", "--overlap", "--nodes=1", "--ntasks=1", "--mem=128M",
            "--nodelist=%s" % endpoint["node"],
            "--gpus-per-node=2", "--gpu-bind=none",
            "env", "CUDA_VISIBLE_DEVICES=0,1",
            os.environ.get("PYTHON_BIN", "python3"), "-c", code,
        ]
        result = None
        for attempt in range(6):
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[-1])
            output = result.stdout + "\n" + result.stderr
            transient = any(marker in output for marker in (
                "Unable to create step",
                "Memory required by task is not available",
                "Failed to retrive the job ID from slurmctld",
            ))
            if not transient or attempt == 5:
                break
            time.sleep(min(2 ** attempt, 8))
        assert result is not None
        detail = (result.stdout + "\n" + result.stderr).strip()[-1000:]
        raise OnlineFeedbackError(
            "energy read failed for %s after retries: %s"
            % (endpoint_id, detail)
        )

    async def probe(
        self, body: Mapping[str, Any], workload_id: str, p_mhz: int,
        d_mhz: int, probe_id: str,
    ) -> ProbeResult:
        frequency_changed = await self.actuate("experiment", p_mhz, d_mhz)
        settle_s = float(
            self.config["online_feedback"].get("experiment_frequency_settle_s", 10.0)
        )
        if frequency_changed and settle_s:
            await asyncio.sleep(settle_s)
        before = await asyncio.to_thread(
            lambda: self._energy_mj("P0") + self._energy_mj("D0")
        )
        started = time.monotonic()
        prepared = await self.experiment_core.prepare(
            strip_feedback_metadata(dict(body) | {"stream": True}), probe_id
        )
        if prepared.stream is None:
            raise OnlineFeedbackError("experiment clone did not return real SSE")
        async for _ in prepared.stream:
            pass
        duration = time.monotonic() - started
        after = await asyncio.to_thread(
            lambda: self._energy_mj("P0") + self._energy_mj("D0")
        )
        stamps = prepared.diagnostics.timestamps_monotonic_s
        output_len = int(body.get("xpyd_output_len", body["max_tokens"]))
        ttft_ms, tpot_ms = pd_inference_metrics(stamps, output_len)
        measured_energy_j = (after - before) / 1000.0
        return ProbeResult(
            p_mhz, d_mhz,
            measured_energy_j / max(duration, 1e-9), measured_energy_j,
            ttft_ms, tpot_ms,
        )
