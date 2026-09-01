"""Online 2P2D service/experiment feedback controller.

P1/D1 always serves the original request.  A missing workload-table entry
causes one deduplicated clone to be explored serially on P0/D0.  Exploration
searches P first and D second with lower-bound binary search.
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
    prefill_frequency_mhz: int
    decode_frequency_mhz: int
    measured_power_w: float
    ttft_ms: float
    tpot_ms: float
    sample_count: int = 1

    def feasible(self, ttft_slo_ms: float, tpot_slo_ms: float) -> bool:
        return (
            math.isfinite(self.measured_power_w) and self.measured_power_w > 0
            and self.ttft_ms < ttft_slo_ms
            and self.tpot_ms <= tpot_slo_ms
        )


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
        decode_grid: Sequence[int], *, probe_interval_s: float = 25.0,
        service_settle_s: float = 10.0,
        ttft_slo_ms: float = 500.0, tpot_slo_ms: float = 200.0,
        service_warmup_requests: int = 0,
        experiment_warmup_requests: int = 0,
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
        if not 20.0 <= float(probe_interval_s) <= 30.0:
            raise OnlineFeedbackError("test-stage probe interval must be 20..30 seconds")
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
        self.event_log = Path(os.path.expandvars(event_log)) if event_log else None
        self.service_request_log = (
            Path(os.path.expandvars(service_request_log))
            if service_request_log else None
        )
        self.sleep = sleep
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], str]] = asyncio.Queue()
        self._pending: set[str] = set()
        self._pending_lock = asyncio.Lock()
        self._service_lock = asyncio.Lock()
        self._worker: Optional[asyncio.Task[Any]] = None
        self._active_workload: Optional[str] = None
        self._service_warmed = False
        self._experiment_warmed = False
        self._service_sequence = 0
        self._log_lock = threading.Lock()

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker is not None:
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
        with self._log_lock:
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
            try:
                await self._explore(workload_id, body, request_id)
            except Exception as exc:
                self._log({"event": "exploration_failed", "workload_id": workload_id,
                           "error": "%s: %s" % (type(exc).__name__, exc)})
            finally:
                async with self._pending_lock:
                    self._pending.discard(workload_id)
                self._active_workload = None
                self._queue.task_done()

    async def _one_probe(
        self, body: Mapping[str, Any], workload_id: str,
        p_mhz: int, d_mhz: int, probe_id: str,
    ) -> ProbeResult:
        await self.sleep(self.probe_interval_s)
        result = await self.probe(body, workload_id, p_mhz, d_mhz, probe_id)
        self._log({"event": "probe_completed", "workload_id": workload_id,
                   "probe_id": probe_id, **result.__dict__,
                   "slo_met": result.feasible(self.ttft_slo_ms, self.tpot_slo_ms)})
        return result

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
        low, high = 0, len(grid) - 1
        observed: dict[int, ProbeResult] = {}
        while low < high:
            candidate = (low + high) // 2
            p_mhz = int(grid[candidate]) if axis == "P" else int(fixed_mhz)
            d_mhz = int(fixed_mhz) if axis == "P" else int(grid[candidate])
            result = await self._one_probe(
                body, workload_id, p_mhz, d_mhz,
                "%s-explore-%s-%d" % (request_id, axis, candidate),
            )
            observed[candidate] = result
            if result.feasible(self.ttft_slo_ms, self.tpot_slo_ms):
                high = candidate
            else:
                low = candidate + 1
        if low not in observed:
            p_mhz = int(grid[low]) if axis == "P" else int(fixed_mhz)
            d_mhz = int(fixed_mhz) if axis == "P" else int(grid[low])
            observed[low] = await self._one_probe(
                body, workload_id, p_mhz, d_mhz,
                "%s-explore-%s-final" % (request_id, axis),
            )
        result = observed[low]
        if not result.feasible(self.ttft_slo_ms, self.tpot_slo_ms):
            raise OnlineFeedbackError("no SLO-safe %s frequency" % axis)
        return int(grid[low]), result

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
        final = await self._one_probe(
            body, workload_id, p_mhz, d_mhz, "%s-confirm" % request_id
        )
        if not final.feasible(self.ttft_slo_ms, self.tpot_slo_ms):
            raise OnlineFeedbackError("joint P/D confirmation failed SLO")
        self.table.write(workload_id, {
            "prefill_frequency_mhz": p_mhz,
            "decode_frequency_mhz": d_mhz,
            "measured_power_w": final.measured_power_w,
            "ttft_ms": final.ttft_ms,
            "tpot_ms": final.tpot_ms,
            "prefill_endpoint_id": "P0",
            "decode_endpoint_id": "D0",
            "sample_count": final.sample_count,
            "source": "online_binary_feedback",
            "slo_met": True,
        })
        self._log({"event": "table_updated", "workload_id": workload_id,
                   "prefill_frequency_mhz": p_mhz,
                   "decode_frequency_mhz": d_mhz})


def select_grid(supported: Sequence[int], minimum: int, maximum: int, levels: int) -> tuple[int, ...]:
    values = sorted({int(item) for item in supported if minimum <= int(item) <= maximum})
    if maximum not in values or len(values) < levels or levels < 2:
        raise OnlineFeedbackError("hardware frequency grid cannot satisfy configuration")
    indices = [round(index * (len(values) - 1) / (levels - 1)) for index in range(levels)]
    grid = tuple(values[index] for index in indices)
    if len(set(grid)) != levels:
        raise OnlineFeedbackError("frequency grid is not strictly ordered")
    return grid


class PhysicalFeedbackRuntime:
    """Real Slurm/NVML actuator and single-request energy probe backend."""

    def __init__(self, config: Mapping[str, Any], experiment_core: Any) -> None:
        self.config = config
        self.experiment_core = experiment_core
        self.backend = NvidiaSmiClockBackend()
        endpoints = {str(item["endpoint_id"]): item for item in config["endpoints"]}
        settings = config["online_feedback"]
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
                int(grid_spec["maximum_mhz"]), int(grid_spec["levels"]),
            )
        self.prefill_grid = grids["P0"]
        self.decode_grid = grids["D0"]
        self.actuator = PerEndpointClockActuator(
            self.backend, capabilities, float(settings.get("minimum_dwell_s", 1.0))
        )
        self.endpoints = endpoints

    def _actuate_sync(self, group: str, p_mhz: int, d_mhz: int) -> bool:
        ids = ("P1", "D1") if group == "service" else ("P0", "D0")
        changed = False
        for endpoint_id, target in zip(ids, (p_mhz, d_mhz)):
            if self.actuator.requested.get(endpoint_id) == int(target):
                continue
            result = self.actuator.actuate(endpoint_id, int(target), "online_feedback_%s" % group)
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
        result = subprocess.run([
            "srun", "--overlap", "--nodes=1", "--ntasks=1",
            "--nodelist=%s" % endpoint["node"],
            "--gpus-per-node=2", "--gpu-bind=none",
            "env", "CUDA_VISIBLE_DEVICES=0,1",
            os.environ.get("PYTHON_BIN", "python3"), "-c", code,
        ], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise OnlineFeedbackError("energy read failed for %s" % endpoint_id)
        return float(result.stdout.strip().splitlines()[-1])

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
        return ProbeResult(
            p_mhz, d_mhz, (after - before) / 1000.0 / max(duration, 1e-9),
            ttft_ms, tpot_ms,
        )
