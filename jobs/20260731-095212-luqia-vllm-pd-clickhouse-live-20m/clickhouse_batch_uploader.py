#!/usr/bin/env python3
"""Batch-upload vLLM observability CSVs to ClickHouse over HTTPS.

Live mode periodically uploads wide engine and GPU samples while the benchmark
runs. Final mode flushes live samples and uploads completed request-level,
window, environment, scheduler, OTEL and drain artifacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SCHEMA_VERSION = 1
STOP = False


def stop_requested(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value: Any, default: int = 0) -> int:
    return int(number(value, float(default)))


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ns_time(value: Any) -> str:
    raw = integer(value)
    seconds, nanos = divmod(raw, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{nanos:09d}"


def ns_time_nullable(value: Any) -> str | None:
    return ns_time(value) if integer(value) > 0 else None


def unix_float_time(value: Any, precision: int = 6) -> str:
    raw = number(value)
    seconds = math.floor(raw)
    fraction = int(round((raw - seconds) * (10**precision)))
    if fraction >= 10**precision:
        seconds += 1
        fraction = 0
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{fraction:0{precision}d}"


def now_time(precision: int = 3) -> str:
    raw = time.time()
    seconds = math.floor(raw)
    fraction = int((raw - seconds) * (10**precision))
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{fraction:0{precision}d}"


def compact_json(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8") + b"\n"


def string_map(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            result[str(key)] = json.dumps(
                item, ensure_ascii=False, separators=(",", ":")
            )
        elif item is None:
            result[str(key)] = ""
        else:
            result[str(key)] = str(item)
    return result


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


class ClickHouse:
    def __init__(
        self,
        url: str,
        user: str,
        password: str,
        database: str,
    ) -> None:
        self.url = url.rstrip("/") + "/"
        self.user = user
        self.password = password
        self.database = database

    def request(
        self,
        query: str,
        body: bytes = b"",
        *,
        content_encoding: str = "",
        settings: dict[str, str] | None = None,
        query_id: str = "",
        timeout: int = 120,
    ) -> tuple[bytes, dict[str, str]]:
        params = {"database": self.database, "query": query}
        if settings:
            params.update(settings)
        request = Request(
            self.url + "?" + urlencode(params),
            data=body,
            method="POST",
        )
        request.add_header("X-ClickHouse-User", self.user)
        request.add_header("X-ClickHouse-Key", self.password)
        # The endpoint is behind Cloudflare; avoid Python urllib's bot-like
        # default User-Agent while retaining ClickHouse header authentication.
        request.add_header("User-Agent", "vllm-clickhouse-uploader/1.0")
        request.add_header("Accept", "text/plain")
        if query_id:
            request.add_header("X-ClickHouse-Query-Id", query_id)
        if content_encoding:
            request.add_header("Content-Encoding", content_encoding)
            request.add_header("Content-Type", "application/x-ndjson")
        try:
            with urlopen(request, timeout=timeout) as response:
                headers = {key.lower(): val for key, val in response.headers.items()}
                return response.read(), headers
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"ClickHouse HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"ClickHouse connection failed: {exc}") from exc

    def preflight(self) -> str:
        body, _ = self.request("SELECT version() FORMAT TabSeparated", timeout=20)
        return body.decode("utf-8", "replace").strip()

    def insert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        batch_label: str,
    ) -> tuple[str, int]:
        if not rows:
            return "", 0
        payload = b"".join(compact_json(row) for row in rows)
        digest = hashlib.sha256(
            (
                f"{SCHEMA_VERSION}\0{table}\0{batch_label}\0".encode("utf-8")
                + payload
            )
        ).hexdigest()
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        query_id = f"vllm-{table}-{digest[:20]}"
        last_error: Exception | None = None
        for attempt, delay in enumerate((0, 2, 5, 10, 20, 40), start=1):
            if delay:
                time.sleep(delay)
            try:
                self.request(
                    f"INSERT INTO {self.database}.{table} FORMAT JSONEachRow",
                    compressed,
                    content_encoding="gzip",
                    query_id=query_id,
                    settings={
                        "wait_end_of_query": "1",
                        "async_insert": "0",
                        "insert_deduplication_token": digest,
                    },
                )
                return digest, len(rows)
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                retryable = any(
                    marker in message
                    for marker in (
                        "connection failed",
                        "HTTP 408",
                        "HTTP 425",
                        "HTTP 429",
                        "HTTP 500",
                        "HTTP 502",
                        "HTTP 503",
                        "HTTP 504",
                    )
                )
                if not retryable or attempt == 6:
                    break
                print(
                    f"clickhouse_retry table={table} attempt={attempt} "
                    f"reason={type(exc).__name__}",
                    flush=True,
                )
        raise RuntimeError(
            f"batch insert failed table={table}: {last_error}"
        )


class Uploader:
    def __init__(
        self,
        client: ClickHouse,
        output_dir: Path,
        workload_file: Path,
        job_id: str,
        job_name: str,
        started_unix_ns: int,
        state_file: Path,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.workload_file = workload_file
        self.job_id = job_id
        self.job_name = job_name
        self.started_unix_ns = started_unix_ns
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {"sources": {}, "static_started": False, "finalized": False}
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def save_state(self) -> None:
        temp = self.state_file.with_suffix(".tmp")
        temp.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.state_file)

    def insert_chunks(
        self,
        table: str,
        rows: list[dict[str, Any]],
        label: str,
        chunk_size: int = 20_000,
    ) -> int:
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            _, count = self.client.insert(
                table,
                chunk,
                f"{label}-{start // chunk_size:06d}",
            )
            total += count
        if rows:
            print(f"clickhouse_insert table={table} rows={total}", flush=True)
        return total

    def static_rows(self, status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stamp = now_time()
        experiment = {
            "experiment_id": "pd-clickhouse-live-20m-20260731",
            "pair_group_id": "",
            "dataset_name": "mixed-20m-live-ingest",
            "description": "L40S prefill + L4 decode, automatic DVFS, live ClickHouse batches",
            "created_at": ns_time(self.started_unix_ns)[:23],
            "tags": {
                "attention_backend": "FLASH_ATTN",
                "ingestion": "live_batch",
                "duration_target": "20m",
            },
            "updated_at": stamp,
        }
        job = {
            "job_id": self.job_id,
            "experiment_id": experiment["experiment_id"],
            "pair_group_id": "",
            "variant_id": "auto-dvfs-live-clickhouse",
            "repeat_no": 1,
            "slurm_job_id": self.job_id,
            "job_name": self.job_name,
            "status": status,
            "started_at": ns_time(self.started_unix_ns)[:23],
            "ended_at": None if status == "running" else stamp,
            "model": "mistralai/Mistral-7B-v0.1",
            "topology": "pd",
            "prefill_node": "neptune",
            "decode_node": "ganymede",
            "attention_backend": "FLASH_ATTN",
            "kv_connector": "P2pNcclConnector",
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.82,
            "tensor_parallel_size": 1,
            "dvfs_mode": "automatic_hardware",
            "manual_frequency_control": False,
            "scheduler_prediction": False,
            "policy_variant": "vllm_default",
            "kv_cache_metrics": True,
            "kv_cache_metrics_sample": 1.0,
            "logging_iteration_details": True,
            "collect_detailed_traces": "all",
            "prefix_caching": True,
            "request_id_headers": True,
            "config": {
                "clickhouse_live_interval_s": "60",
                "metrics_long": "disabled",
            },
            "updated_at": stamp,
        }
        return [experiment], [job]

    def ensure_started(self) -> None:
        if self.state.get("static_started"):
            return
        experiments, jobs = self.static_rows("running")
        self.insert_chunks("experiments", experiments, "experiment-start")
        self.insert_chunks("jobs", jobs, "job-start")
        self.state["static_started"] = True
        self.save_state()

    def new_source_rows(
        self,
        source_name: str,
    ) -> tuple[list[dict[str, str]], int]:
        rows = read_csv(self.output_dir / source_name)
        start = integer(self.state["sources"].get(source_name, 0))
        if start > len(rows):
            start = 0
        return rows[start:], start

    def mark_source(self, source_name: str, count: int) -> None:
        self.state["sources"][source_name] = count
        self.save_state()

    def engine_rows(self, source: list[dict[str, str]]) -> list[dict[str, Any]]:
        result = []
        for row in source:
            result.append(
                {
                    "job_id": self.job_id,
                    "event_time": ns_time(row["unix_ns"]),
                    "monotonic_ns": integer(row["monotonic_ns"]),
                    "role": row["role"],
                    "endpoint": row["endpoint"],
                    "scrape_ok": boolean(row["scrape_ok"]),
                    "scrape_duration_ms": number(row["scrape_duration_ms"]),
                    "running": integer(row["running"]),
                    "waiting": integer(row["waiting"]),
                    "waiting_growth_per_s": number(row["waiting_growth_per_s"]),
                    "kv_cache_usage": number(row["kv_cache_usage"]),
                    "prefix_cache_hits": integer(row["prefix_cache_hits"]),
                    "prefix_cache_queries": integer(row["prefix_cache_queries"]),
                    "external_prefix_cache_hits": integer(
                        row["external_prefix_cache_hits"]
                    ),
                    "external_prefix_cache_queries": integer(
                        row["external_prefix_cache_queries"]
                    ),
                    "preemptions": integer(row["preemptions"]),
                    "prompt_tokens": integer(row["prompt_tokens"]),
                    "generation_tokens": integer(row["generation_tokens"]),
                    "successful_requests": integer(row["successful_requests"]),
                    "failed_requests": 0,
                    "e2e_count": integer(row["e2e_count"]),
                    "e2e_sum_s": number(row["e2e_sum_s"]),
                    "ttft_count": integer(row["ttft_count"]),
                    "ttft_sum_s": number(row["ttft_sum_s"]),
                    "inter_token_count": integer(row["inter_token_count"]),
                    "inter_token_sum_s": number(row["inter_token_sum_s"]),
                    "iteration_tokens_count": integer(row["iteration_tokens_count"]),
                    "iteration_tokens_sum": integer(row["iteration_tokens_sum"]),
                    "engine_sleep_state": -1,
                    "prompt_tps": 0.0,
                    "generation_tps": 0.0,
                    "completion_rps": 0.0,
                    "error": row.get("error", ""),
                }
            )
        return result

    def gpu_rows(
        self,
        source: list[dict[str, str]],
        role: str,
        hostname: str,
    ) -> list[dict[str, Any]]:
        result = []
        for row in source:
            result.append(
                {
                    "job_id": self.job_id,
                    "event_time": unix_float_time(row["unix_ts"], 6),
                    "role": role,
                    "hostname": hostname,
                    "gpu_index": 0,
                    "gpu_uuid": "",
                    "network_interface": "",
                    "rx_bytes": integer(row["rx_bytes"]),
                    "tx_bytes": integer(row["tx_bytes"]),
                    "rx_bytes_per_s": 0.0,
                    "tx_bytes_per_s": 0.0,
                    "gpu_util_pct": number(row["gpu_util_pct"]),
                    "gpu_power_w": number(row["gpu_power_w"]),
                    "gpu_sm_mhz": integer(row["gpu_sm_mhz"]),
                    "gpu_memory_used_mib": integer(row["gpu_memory_used_mib"]),
                    "gpu_mem_util_pct": number(row["gpu_mem_util_pct"]),
                    "gpu_power_limit_w": number(row["gpu_power_limit_w"]),
                    "gpu_mem_clock_mhz": integer(row["gpu_mem_clock_mhz"]),
                    "gpu_temperature_c": number(row["gpu_temperature_c"]),
                    "gpu_memory_total_mib": integer(row["gpu_memory_total_mib"]),
                    "gpu_pstate": row.get("gpu_pstate", ""),
                    "dvfs_mode": row.get("dvfs_mode", "automatic_hardware"),
                    "throttle_reasons": [],
                    "pcie_rx_bytes_per_s": 0.0,
                    "pcie_tx_bytes_per_s": 0.0,
                }
            )
        return result

    def flush_live_source(
        self,
        source_name: str,
        table: str,
        transform: Callable[[list[dict[str, str]]], list[dict[str, Any]]],
    ) -> int:
        source, start = self.new_source_rows(source_name)
        if not source:
            return 0
        rows = transform(source)
        count = self.insert_chunks(
            table,
            rows,
            f"{source_name}-{start}",
        )
        self.mark_source(source_name, start + len(source))
        return count

    def flush_live(self) -> int:
        total = 0
        total += self.flush_live_source(
            "vllm_metrics_snapshots.csv",
            "engine_samples",
            self.engine_rows,
        )
        total += self.flush_live_source(
            "prefill_neptune_telemetry.csv",
            "gpu_samples",
            lambda rows: self.gpu_rows(rows, "prefill", "neptune"),
        )
        total += self.flush_live_source(
            "decode_ganymede_telemetry.csv",
            "gpu_samples",
            lambda rows: self.gpu_rows(rows, "decode", "ganymede"),
        )
        return total

    def request_context(
        self,
    ) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        rows = read_csv(self.output_dir / "queue_state_at_arrival.csv")
        by_request = {row["request_id"]: row for row in rows}
        by_trace = {
            row["trace_id"]: row
            for row in rows
            if row.get("trace_id")
        }
        return rows, by_request, by_trace

    def token_rows(
        self,
        by_request: dict[str, dict[str, str]],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in read_csv(self.output_dir / "token_timestamps.csv"):
            grouped.setdefault(row["request_id"], []).append(row)
        result = []
        stored: set[str] = set()
        for request_id, items in grouped.items():
            request = by_request.get(request_id)
            if not request or integer(request.get("actual_send_unix_ns")) <= 0:
                continue
            items.sort(key=lambda row: integer(row["token_event_index"]))
            offsets = [
                max(0, int(round(number(row["since_send_ms"]) * 1000)))
                for row in items
            ]
            inter = [
                max(0, int(round(number(row["since_previous_token_ms"]) * 1000)))
                for row in items
            ]
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": request["window_id"],
                    "request_id": request_id,
                    "actual_send_time": ns_time(request["actual_send_unix_ns"]),
                    "sample_reason": "all_requests_smoke_test",
                    "token_count": len(items),
                    "arrival_offset_us": offsets,
                    "inter_token_us": inter,
                    "fragment_bytes": [
                        integer(row["fragment_bytes"]) for row in items
                    ],
                    "mean_itl_us": statistics.fmean(inter) if inter else 0.0,
                    "p50_itl_us": percentile(inter, 0.50),
                    "p95_itl_us": percentile(inter, 0.95),
                    "max_itl_us": max(inter, default=0),
                    "stall_count": sum(value >= 100_000 for value in inter),
                }
            )
            stored.add(request_id)
        return result, stored

    def final_request_rows(
        self,
        rows: list[dict[str, str]],
        token_stored: set[str],
    ) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": row["window_id"],
                    "request_index": integer(row["request_index"]),
                    "request_id": row["request_id"],
                    "response_request_id": row.get("response_request_id", ""),
                    "trace_id": row.get("trace_id", ""),
                    "client_span_id": row.get("client_span_id", ""),
                    "planned_send_time": ns_time(row["planned_send_unix_ns"]),
                    "client_ready_time": ns_time_nullable(row["client_ready_unix_ns"]),
                    "actual_send_time": ns_time_nullable(row["actual_send_unix_ns"]),
                    "proxy_arrival_time": ns_time_nullable(row.get("proxy_arrival_unix_ns")),
                    "response_headers_time": ns_time_nullable(
                        row["response_headers_unix_ns"]
                    ),
                    "first_token_time": ns_time_nullable(row["first_token_unix_ns"]),
                    "completed_time": ns_time_nullable(row["completed_unix_ns"]),
                    "planned_to_ready_ms": number(row["planned_to_ready_ms"]),
                    "client_queue_delay_ms": number(row["client_queue_delay_ms"]),
                    "send_lag_ms": number(row["send_lag_ms"]),
                    "client_to_proxy_ms": number(row.get("client_to_proxy_ms")),
                    "ttft_ms": number(row["ttft_ms"]),
                    "tpot_ms": number(row["tpot_ms"]),
                    "e2e_ms": number(row["e2e_ms"]),
                    "planned_input_tokens": integer(row["planned_input_tokens"]),
                    "actual_input_tokens": integer(row["actual_input_tokens"]),
                    "max_tokens": integer(row["max_tokens"]),
                    "actual_output_tokens": integer(row["actual_output_tokens"]),
                    "token_event_count": integer(row["token_event_count"]),
                    "http_status": integer(row["http_status"]),
                    "outcome": row.get("outcome", ""),
                    "timeout": boolean(row["timeout"]),
                    "cancelled": boolean(row["cancelled"]),
                    "retry_count": integer(row["retry_count"]),
                    "attempt_count": integer(row["attempt_count"]),
                    "response_bytes": integer(row["response_bytes"]),
                    "error": row.get("error", ""),
                    "prefill_metric_time": ns_time_nullable(
                        row.get("prefill_metric_unix_ns")
                    ),
                    "prefill_metric_age_ms": number(
                        row.get("prefill_metric_age_ms")
                    ),
                    "prefill_running": integer(row.get("prefill_running")),
                    "prefill_waiting": integer(row.get("prefill_waiting")),
                    "prefill_waiting_growth_per_s": number(
                        row.get("prefill_waiting_growth_per_s")
                    ),
                    "prefill_kv_cache_usage": number(
                        row.get("prefill_kv_cache_usage")
                    ),
                    "decode_metric_time": ns_time_nullable(
                        row.get("decode_metric_unix_ns")
                    ),
                    "decode_metric_age_ms": number(row.get("decode_metric_age_ms")),
                    "decode_running": integer(row.get("decode_running")),
                    "decode_waiting": integer(row.get("decode_waiting")),
                    "decode_waiting_growth_per_s": number(
                        row.get("decode_waiting_growth_per_s")
                    ),
                    "decode_kv_cache_usage": number(
                        row.get("decode_kv_cache_usage")
                    ),
                    "token_series_stored": row["request_id"] in token_stored,
                    "token_series_reason": (
                        "all_requests_smoke_test"
                        if row["request_id"] in token_stored
                        else ""
                    ),
                }
            )
        return result

    def event_rows(
        self,
        by_request: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        result = []
        for row in read_csv(self.output_dir / "client_events.csv"):
            request = by_request.get(row["request_id"], {})
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": row.get("window_id", request.get("window_id", "")),
                    "request_id": row["request_id"],
                    "internal_request_id": "",
                    "trace_id": request.get("trace_id", ""),
                    "event_time": ns_time(row["unix_ns"]),
                    "monotonic_ns": integer(row["monotonic_ns"]),
                    "source": "client",
                    "role": "",
                    "stage": "",
                    "event": row["event"],
                    "attempt": integer(row["attempt"]),
                    "route_index": -1,
                    "upstream_url": "",
                    "http_status": integer(row["http_status"]),
                    "payload_bytes": 0,
                    "detail": {"detail": row.get("detail", "")},
                    "message": "",
                }
            )
        for row in read_csv(self.output_dir / "proxy_events.csv"):
            request_id = row["client_request_id"]
            request = by_request.get(request_id, {})
            traceparent = row.get("traceparent", "")
            parts = traceparent.split("-")
            trace_id = parts[1] if len(parts) >= 4 else request.get("trace_id", "")
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": request.get("window_id", ""),
                    "request_id": request_id,
                    "internal_request_id": row.get("internal_request_id", ""),
                    "trace_id": trace_id,
                    "event_time": ns_time(row["unix_ns"]),
                    "monotonic_ns": integer(row["monotonic_ns"]),
                    "source": "proxy",
                    "role": row.get("stage", ""),
                    "stage": row.get("stage", ""),
                    "event": row["event"],
                    "attempt": 0,
                    "route_index": integer(row.get("route_index"), -1),
                    "upstream_url": row.get("upstream_url", ""),
                    "http_status": integer(row["http_status"]),
                    "payload_bytes": integer(row["bytes"]),
                    "detail": {
                        "prefill_http": row.get("prefill_http", ""),
                        "decode_http": row.get("decode_http", ""),
                        "error": row.get("error", ""),
                    },
                    "message": "",
                }
            )
        return result

    def attempt_rows(
        self,
        by_request: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        attempts = [
            row
            for row in read_csv(self.output_dir / "client_events.csv")
            if row.get("event") == "attempt_send"
        ]
        result = []
        for row in attempts:
            request = by_request.get(row["request_id"], {})
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": row.get("window_id", request.get("window_id", "")),
                    "request_id": row["request_id"],
                    "attempt": integer(row["attempt"]),
                    "started_at": ns_time(row["unix_ns"]),
                    "response_headers_at": ns_time_nullable(
                        request.get("response_headers_unix_ns")
                    ),
                    "completed_at": ns_time_nullable(request.get("completed_unix_ns")),
                    "http_status": integer(request.get("http_status")),
                    "outcome": request.get("outcome", ""),
                    "timeout": boolean(request.get("timeout")),
                    "cancelled": boolean(request.get("cancelled")),
                    "retry_scheduled": integer(request.get("retry_count")) > 0,
                    "retry_delay_ms": 0.0,
                    "response_bytes": integer(request.get("response_bytes")),
                    "error": request.get("error", ""),
                }
            )
        return result

    def window_rows(self) -> list[dict[str, Any]]:
        configs = {
            row["window_id"]: row for row in read_csv(self.workload_file)
        }
        result = []
        for row in read_csv(self.output_dir / "window_summary.csv"):
            config = configs.get(row["window_id"], {})
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": row["window_id"],
                    "workload_name": row["window_id"],
                    "action_id": "vllm_auto_dvfs",
                    "action_config": {
                        "manual_frequency_control": "false",
                        "scheduler_prediction": "false",
                    },
                    "window_start": ns_time(row["window_start_unix_ns"]),
                    "sending_stopped": ns_time_nullable(
                        row["sending_stopped_unix_ns"]
                    ),
                    "window_end": ns_time_nullable(row["window_end_unix_ns"]),
                    "input_len": integer(config.get("input_tokens")),
                    "max_tokens": integer(config.get("max_tokens")),
                    "request_rate": number(config.get("request_rate")),
                    "concurrency": integer(config.get("max_concurrency")),
                    "num_prompts": integer(row["planned_requests"]),
                    "timeout_ms": int(
                        round(number(config.get("timeout_s")) * 1000)
                    ),
                    "random_seed": 0,
                    "planned_requests": integer(row["planned_requests"]),
                    "completed_requests": integer(row["completed_requests"]),
                    "failed_requests": integer(row["failed_requests"]),
                    "timeout_requests": integer(row["timeout_requests"]),
                    "cancelled_requests": integer(row["cancelled_requests"]),
                    "retry_attempts": integer(row["retry_attempts"]),
                    "client_observed_drain_s": number(
                        row["client_observed_drain_s"]
                    ),
                    "actual_send_rps": number(row["actual_send_rps"]),
                    "completion_output_tps": number(
                        row["completion_output_tps"]
                    ),
                    "mean_ttft_ms": number(row["mean_ttft_ms"]),
                    "p50_ttft_ms": number(row["p50_ttft_ms"]),
                    "p95_ttft_ms": number(row["p95_ttft_ms"]),
                    "p99_ttft_ms": number(row["p99_ttft_ms"]),
                    "mean_tpot_ms": number(row["mean_tpot_ms"]),
                    "p99_tpot_ms": number(row["p99_tpot_ms"]),
                    "mean_e2e_ms": number(row["mean_e2e_ms"]),
                    "p99_e2e_ms": number(row["p99_e2e_ms"]),
                    "updated_at": now_time(),
                }
            )
        return result

    def node_rows(self) -> list[dict[str, Any]]:
        result = []
        for name in ("environment_neptune.csv", "environment_ganymede.csv"):
            for row in read_csv(self.output_dir / name):
                gpu_blob = row.get("gpu", "")
                uuid_match = re.search(r"GPU-[0-9a-fA-F-]+", gpu_blob)
                memory_match = re.search(r"(\d+)\s+MiB", gpu_blob)
                result.append(
                    {
                        "job_id": self.job_id,
                        "captured_at": ns_time(row["unix_ns"]),
                        "role": row.get("role", ""),
                        "node_group": row.get("node_group", ""),
                        "hostname": row.get("hostname", ""),
                        "node_ip": row.get("node_ip", ""),
                        "peer_ip": row.get("peer_ip", ""),
                        "network_interface": row.get("interface", ""),
                        "link_speed_mbps": integer(row.get("link_speed_mbps")),
                        "link_mtu": integer(row.get("link_mtu")),
                        "kernel": row.get("kernel", ""),
                        "cpu_count": integer(row.get("cpu_count")),
                        "gpu_name": row.get("expected_gpu", ""),
                        "gpu_uuid": uuid_match.group(0) if uuid_match else "",
                        "gpu_memory_total_mib": (
                            integer(memory_match.group(1))
                            if memory_match
                            else 0
                        ),
                        "driver_version": "",
                        "cuda_version": "",
                        "vllm_version": "",
                        "container_image": "",
                        "node_work_dir": row.get("node_work_dir", ""),
                        "runtime_cwd": row.get("runtime_cwd", ""),
                        "environment": {
                            "attention_backend": row.get(
                                "attention_backend", ""
                            ),
                            "dvfs_mode": row.get("dvfs_mode", ""),
                            "slurm_node_list": row.get("slurm_node_list", ""),
                        },
                    }
                )
        return result

    def otel_rows(
        self,
        by_trace: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        result = []
        for row in read_csv(self.output_dir / "otel_spans.csv"):
            request = by_trace.get(row["trace_id"], {})
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": request.get("window_id", ""),
                    "request_id": request.get("request_id", ""),
                    "received_at": ns_time(row["received_unix_ns"]),
                    "batch_id": row["batch_id"],
                    "service_name": row["service_name"],
                    "scope_name": row["scope_name"],
                    "scope_version": row["scope_version"],
                    "trace_id": row["trace_id"],
                    "span_id": row["span_id"],
                    "parent_span_id": row["parent_span_id"],
                    "span_name": row["name"],
                    "span_kind": integer(row["kind"]),
                    "start_time": ns_time(row["start_time_unix_ns"]),
                    "end_time": ns_time(row["end_time_unix_ns"]),
                    "duration_us": int(round(number(row["duration_ms"]) * 1000)),
                    "status_code": integer(row["status_code"]),
                    "status_message": row["status_message"],
                    "resource_attributes": string_map(
                        row["resource_attributes_json"]
                    ),
                    "span_attributes": string_map(row["span_attributes_json"]),
                    "events_json": row["events_json"],
                    "links_count": integer(row["links_count"]),
                }
            )
        return result

    def scheduler_rows(self) -> list[dict[str, Any]]:
        result = []
        year = datetime.fromtimestamp(
            self.started_unix_ns / 1_000_000_000,
            tz=timezone.utc,
        ).year
        for index, row in enumerate(
            read_csv(self.output_dir / "vllm_observability_log_events.csv")
        ):
            if row.get("category") not in {"iteration", "scheduler", "engine"}:
                continue
            try:
                parsed = datetime.strptime(
                    f"{year}-{row['log_timestamp']}",
                    "%Y-%m-%d %H:%M:%S",
                ).replace(tzinfo=ZoneInfo("Europe/Stockholm"))
                parsed = parsed.astimezone(timezone.utc)
                event_time = parsed.strftime("%Y-%m-%d %H:%M:%S.000000000")
            except ValueError:
                event_time = ns_time(self.started_unix_ns)
            result.append(
                {
                    "job_id": self.job_id,
                    "event_time": event_time,
                    "role": row["role"],
                    "iteration_id": index,
                    "iteration_duration_us": 0,
                    "running": 0,
                    "waiting": 0,
                    "prefill_running": 0,
                    "decode_running": 0,
                    "scheduled_sequences": 0,
                    "scheduled_tokens": 0,
                    "batch_tokens": 0,
                    "preempted_sequences": 0,
                    "finished_sequences": 0,
                    "kv_cache_usage": 0.0,
                    "scheduled_request_ids": [],
                    "preempted_request_ids": [],
                    "finished_request_ids": [],
                    "detail": {
                        "source_file": row.get("source_file", ""),
                        "line_number": row.get("line_number", ""),
                        "category": row.get("category", ""),
                    },
                    "raw_message": row.get("message", ""),
                }
            )
        return result

    def drain_rows(self) -> list[dict[str, Any]]:
        result = []
        summaries = read_csv(self.output_dir / "drain_summary.csv")
        drained = boolean(summaries[0].get("drained")) if summaries else False
        for row in read_csv(self.output_dir / "drain_samples.csv"):
            result.append(
                {
                    "job_id": self.job_id,
                    "window_id": "job_final_drain",
                    "event_time": ns_time(row["unix_ns"]),
                    "monotonic_ns": integer(row["monotonic_ns"]),
                    "elapsed_s": number(row["elapsed_s"]),
                    "role": row["role"],
                    "running": integer(row["running"]),
                    "waiting": integer(row["waiting"]),
                    "kv_cache_usage": number(row["kv_cache_usage"]),
                    "scrape_ok": boolean(row["scrape_ok"]),
                    "drained": False,
                    "error": row.get("error", ""),
                }
            )
        if result and drained:
            final_elapsed = max(row["elapsed_s"] for row in result)
            for row in result:
                row["drained"] = row["elapsed_s"] == final_elapsed
        return result

    def finalize(self) -> None:
        self.ensure_started()
        self.flush_live()
        request_rows, by_request, by_trace = self.request_context()
        if not request_rows:
            raise RuntimeError("queue_state_at_arrival.csv has no request rows")

        tokens, token_stored = self.token_rows(by_request)
        self.insert_chunks(
            "request_token_series",
            tokens,
            "final-token-series",
            chunk_size=5_000,
        )
        self.insert_chunks(
            "requests",
            self.final_request_rows(request_rows, token_stored),
            "final-requests",
        )
        self.insert_chunks(
            "request_events",
            self.event_rows(by_request),
            "final-events",
        )
        self.insert_chunks(
            "request_attempts",
            self.attempt_rows(by_request),
            "final-attempts",
        )
        self.insert_chunks(
            "workload_windows",
            self.window_rows(),
            "final-windows",
        )
        self.insert_chunks("job_nodes", self.node_rows(), "final-nodes")
        self.insert_chunks(
            "otel_spans",
            self.otel_rows(by_trace),
            "final-otel",
        )
        self.insert_chunks(
            "scheduler_iterations",
            self.scheduler_rows(),
            "final-scheduler",
        )
        self.insert_chunks(
            "drain_samples",
            self.drain_rows(),
            "final-drain",
        )
        _, jobs = self.static_rows("complete")
        self.insert_chunks("jobs", jobs, "job-complete")
        self.state["finalized"] = True
        self.save_state()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "final"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--started-unix-ns", type=int, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument(
        "--url", default=os.environ.get("CLICKHOUSE_URL", "")
    )
    parser.add_argument(
        "--user", default=os.environ.get("CLICKHOUSE_USER", "")
    )
    parser.add_argument(
        "--password", default=os.environ.get("CLICKHOUSE_PASSWORD", "")
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("CLICKHOUSE_DATABASE", "vllm_observability"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.url or not args.user or not args.password:
        print("clickhouse_credentials_missing=true", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = ClickHouse(args.url, args.user, args.password, args.database)
    version = client.preflight()
    print(
        f"clickhouse_preflight_ok=true version={version} "
        f"database={args.database}",
        flush=True,
    )
    uploader = Uploader(
        client,
        args.output_dir,
        args.workloads,
        args.job_id,
        args.job_name,
        args.started_unix_ns,
        args.state_file,
    )
    if args.mode == "final":
        uploader.finalize()
        print("clickhouse_finalize_success=true", flush=True)
        return 0

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    uploader.ensure_started()
    while not STOP:
        try:
            uploader.flush_live()
        except Exception as exc:
            print(
                f"clickhouse_live_flush_error={type(exc).__name__}: {exc}",
                flush=True,
            )
        deadline = time.monotonic() + args.interval
        while not STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
    uploader.flush_live()
    print("clickhouse_live_uploader_stopped=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
