"""One auditable request record for Canary and fixed-tier Production."""
from __future__ import annotations

from collections import deque
import math
import time
from typing import Any, Mapping

from xpyd.types import EndpointSpec
from xpyd.vllm_metrics import VLLMMetricsCollector, VLLMWindowTracker


class NodeGroupTelemetry:
    """Collect read-only vLLM state without turning missing metrics into data."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        contract = dict(config["telemetry_contract"])
        self.window_s = float(contract["arrival_rate_window_s"])
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("arrival_rate_window_s must be positive")
        self.p_capacity = contract.get("prefill_capacity_input_tokens_per_s")
        self.d_capacity = contract.get("decode_capacity_output_tokens_per_s")
        self.endpoints = {
            str(item["endpoint_id"]): dict(item) for item in config["endpoints"]
        }
        self.specs = {
            endpoint_id: EndpointSpec(
                endpoint_id=endpoint_id,
                role=str(item["role"]),
                gpu_type=str(item["gpu_type"]),
                node=str(item["node"]),
                gpu_ids=tuple(int(value) for value in item["gpu_ids"]),
                tp_degree=int(item["tp_degree"]),
                http_uri=str(item["http_uri"]),
                kv_connector=str(item["kv_connector"]),
            )
            for endpoint_id, item in self.endpoints.items()
        }
        self.collector = VLLMMetricsCollector()
        self.windows = VLLMWindowTracker()
        self.arrivals: deque[float] = deque()

    @staticmethod
    def _count(value: Any) -> int | None:
        if value is None or not float(value).is_integer() or value < 0:
            return None
        return int(value)

    @staticmethod
    def _rho(rate: Any, capacity: Any) -> float | None:
        if rate is None or capacity is None:
            return None
        rate, capacity = float(rate), float(capacity)
        if not math.isfinite(rate) or not math.isfinite(capacity) or capacity <= 0:
            return None
        return rate / capacity

    def _scrape(self, endpoint_id: str) -> tuple[dict[str, Any], list[str]]:
        try:
            snapshot = self.collector.scrape(self.specs[endpoint_id])
            window = self.windows.observe(snapshot)
            return {
                "running": self._count(snapshot.num_requests_running),
                "waiting": self._count(snapshot.num_requests_waiting),
                "kv_usage": snapshot.kv_cache_usage_frac,
                "prompt_rate": window.prompt_tokens_per_s if window.valid else None,
                "generation_rate": window.generation_tokens_per_s if window.valid else None,
                "missing_metrics": list(snapshot.missing_metrics),
                "window_reason": window.reason,
            }, []
        except Exception as exc:  # Telemetry must expose, not invent, availability.
            return {}, [f"{endpoint_id}:{type(exc).__name__}:{exc}"]

    def build(
        self,
        *,
        source: str,
        sent_timestamp_s: float,
        pair: tuple[str, str],
        input_tokens: int,
        max_output_tokens: int,
        actual_output_tokens: int,
        targets_mhz: tuple[int, int],
        endpoint_energy: Mapping[str, Mapping[str, Any]],
        ttft_ms: float,
        tpot_ms: float,
        slo_met: bool,
    ) -> dict[str, Any]:
        if source not in {"canary", "production"}:
            raise ValueError("source must be canary or production")
        now = time.time()
        self.arrivals.append(float(sent_timestamp_s))
        while self.arrivals and self.arrivals[0] < now - self.window_s:
            self.arrivals.popleft()
        span = max(now - self.arrivals[0], 1e-9)
        arrival_rate = len(self.arrivals) / span
        p_state, p_errors = self._scrape(pair[0])
        d_state, d_errors = self._scrape(pair[1])
        p_energy = endpoint_energy.get(pair[0], {})
        d_energy = endpoint_energy.get(pair[1], {})
        running = [value for value in (p_state.get("running"), d_state.get("running")) if value is not None]
        errors = p_errors + d_errors
        return {
            "schema_version": 1,
            "timestamp": now,
            "source": source,
            "input_tokens": int(input_tokens),
            "max_output_tokens": int(max_output_tokens),
            "actual_output_tokens": int(actual_output_tokens),
            "output_tokens": int(actual_output_tokens),
            "batch_size": max(running) if running else None,
            "arrival_rate": arrival_rate,
            "P_gpu": self.endpoints[pair[0]]["gpu_type"],
            "D_gpu": self.endpoints[pair[1]]["gpu_type"],
            "P_rho": self._rho(p_state.get("prompt_rate"), self.p_capacity),
            "D_rho": self._rho(d_state.get("generation_rate"), self.d_capacity),
            "P_queue": p_state.get("waiting"),
            "D_queue": d_state.get("waiting"),
            "D_KV_usage": d_state.get("kv_usage"),
            "P_freq": {"target_mhz": int(targets_mhz[0]), "observed_min_mhz": p_energy.get("observed_clock_min_mhz"), "observed_max_mhz": p_energy.get("observed_clock_max_mhz")},
            "D_freq": {"target_mhz": int(targets_mhz[1]), "observed_min_mhz": d_energy.get("observed_clock_min_mhz"), "observed_max_mhz": d_energy.get("observed_clock_max_mhz")},
            "TTFT": float(ttft_ms),
            "TPOT": float(tpot_ms),
            "P_energy": p_energy.get("energy_j"),
            "D_energy": d_energy.get("energy_j"),
            "total_energy": (p_energy.get("energy_j") or 0.0) + (d_energy.get("energy_j") or 0.0),
            "SLO_met": bool(slo_met),
            "prefill_endpoint_id": pair[0],
            "decode_endpoint_id": pair[1],
            "telemetry_errors": errors,
            "P_metrics_missing": p_state.get("missing_metrics", []),
            "D_metrics_missing": d_state.get("missing_metrics", []),
            "P_metrics_window_reason": p_state.get("window_reason"),
            "D_metrics_window_reason": d_state.get("window_reason"),
        }
