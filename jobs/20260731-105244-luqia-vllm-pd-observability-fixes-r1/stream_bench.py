#!/usr/bin/env python3
"""Scheduled streaming benchmark with request IDs and request/token CSV traces."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import random
import statistics
import time
import uuid
from typing import Any

import aiohttp


REQUEST_FIELDS = [
    "window_id",
    "request_index",
    "request_id",
    "trace_id",
    "client_span_id",
    "planned_send_unix_ns",
    "client_ready_unix_ns",
    "actual_send_unix_ns",
    "response_headers_unix_ns",
    "first_token_unix_ns",
    "completed_unix_ns",
    "planned_to_ready_ms",
    "client_queue_delay_ms",
    "send_lag_ms",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "planned_input_tokens",
    "actual_input_tokens",
    "max_tokens",
    "actual_output_tokens",
    "token_event_count",
    "stream_chunk_count",
    "token_id_event_count",
    "token_timestamp_complete",
    "http_status",
    "response_request_id",
    "outcome",
    "timeout",
    "cancelled",
    "retry_count",
    "attempt_count",
    "response_bytes",
    "error",
]

TOKEN_FIELDS = [
    "window_id",
    "request_id",
    "token_event_index",
    "stream_chunk_index",
    "token_index_in_chunk",
    "tokens_in_chunk",
    "token_id",
    "timestamp_source",
    "arrival_unix_ns",
    "since_send_ms",
    "since_previous_token_ms",
    "fragment_bytes",
    "fragment_json",
]

CHUNK_FIELDS = [
    "window_id",
    "request_id",
    "stream_chunk_index",
    "arrival_unix_ns",
    "since_send_ms",
    "since_previous_chunk_ms",
    "tokens_in_chunk",
    "token_ids_json",
    "fragment_bytes",
    "fragment_json",
]

EVENT_FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "window_id",
    "request_id",
    "attempt",
    "event",
    "http_status",
    "detail",
]

WINDOW_FIELDS = [
    "window_id",
    "window_start_unix_ns",
    "sending_stopped_unix_ns",
    "window_end_unix_ns",
    "client_observed_drain_s",
    "planned_requests",
    "completed_requests",
    "failed_requests",
    "timeout_requests",
    "cancelled_requests",
    "retry_attempts",
    "actual_send_rps",
    "completion_output_tps",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_ms",
    "p99_e2e_ms",
]

MARKER_FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "event",
    "window_id",
    "detail",
]


def ns_to_ms(value: int | None, origin: int | None) -> float | str:
    if value is None or origin is None:
        return ""
    return (value - origin) / 1_000_000


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def prompt_tokens(length: int, shared_prefix: int, request_index: int) -> list[int]:
    if length < 1:
        raise ValueError("input token length must be positive")
    shared_prefix = max(1, min(shared_prefix, length))
    common = [1]
    common.extend(1000 + (index % 127) for index in range(shared_prefix - 1))
    suffix = [
        2000 + ((request_index * 131 + index * 17) % 4096)
        for index in range(length - shared_prefix)
    ]
    return common + suffix


class CsvSink:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()

    async def append(
        self, filename: str, fields: list[str], rows: list[dict[str, Any]]
    ) -> None:
        if not rows:
            return
        path = self.output_dir / filename
        async with self.lock:
            is_new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if is_new:
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()

    async def event(
        self,
        window_id: str,
        request_id: str,
        attempt: int,
        event: str,
        http_status: int | str = "",
        detail: str = "",
    ) -> None:
        await self.append(
            "client_events.csv",
            EVENT_FIELDS,
            [
                {
                    "unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "window_id": window_id,
                    "request_id": request_id,
                    "attempt": attempt,
                    "event": event,
                    "http_status": http_status,
                    "detail": detail,
                }
            ],
        )

    async def marker(self, event: str, window_id: str, detail: str = "") -> None:
        await self.append(
            "workload_events.csv",
            MARKER_FIELDS,
            [
                {
                    "unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "event": event,
                    "window_id": window_id,
                    "detail": detail,
                }
            ],
        )


def parse_sse_event(raw_event: bytes) -> dict[str, Any] | None:
    payload_parts = []
    for line in raw_event.splitlines():
        if line.startswith(b"data:"):
            payload_parts.append(line[5:].strip())
    if not payload_parts:
        return None
    payload = b"\n".join(payload_parts)
    if payload == b"[DONE]":
        return {"done": True}
    return json.loads(payload)


async def run_attempt(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    window: dict[str, Any],
    request_index: int,
    request_id: str,
    traceparent: str,
    attempt: int,
    sink: CsvSink,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt_tokens(
            int(window["input_tokens"]),
            int(window["shared_prefix_tokens"]),
            request_index,
        ),
        "max_tokens": int(window["max_tokens"]),
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM 0.15.1 returns delta token IDs in each streaming response when
        # this extension is enabled. This lets us expand a multi-token SSE
        # chunk into one timestamp row per token without re-tokenizing text.
        "return_token_ids": True,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
        "traceparent": traceparent,
    }
    await sink.event(window["window_id"], request_id, attempt, "attempt_send")
    response_headers_ns = None
    first_token_ns = None
    completed_ns = None
    status = 0
    response_request_id = ""
    token_rows = []
    chunk_rows = []
    response_bytes = 0
    actual_input_tokens = 0
    actual_output_tokens = 0
    previous_token_ns = None
    previous_chunk_ns = None
    buffer = b""

    request_timeout = aiohttp.ClientTimeout(total=float(window["timeout_s"]))
    async with session.post(
        endpoint,
        json=body,
        headers=headers,
        timeout=request_timeout,
    ) as response:
        status = response.status
        response_headers_ns = time.time_ns()
        response_request_id = response.headers.get("X-Request-Id", "")
        await sink.event(
            window["window_id"],
            request_id,
            attempt,
            "response_headers",
            status,
            f"response_request_id={response_request_id}",
        )
        if status != 200:
            error_body = await response.text()
            raise RuntimeError(f"HTTP {status}: {error_body[:500]}")

        async for chunk in response.content.iter_any():
            response_bytes += len(chunk)
            buffer += chunk
            while b"\n\n" in buffer:
                raw_event, buffer = buffer.split(b"\n\n", 1)
                decoded = parse_sse_event(raw_event)
                if not decoded or decoded.get("done"):
                    continue
                usage = decoded.get("usage") or {}
                if usage:
                    actual_input_tokens = int(
                        usage.get("prompt_tokens", actual_input_tokens) or 0
                    )
                    actual_output_tokens = int(
                        usage.get("completion_tokens", actual_output_tokens) or 0
                    )
                choices = decoded.get("choices") or []
                for choice in choices:
                    fragment = choice.get("text", "")
                    token_ids = choice.get("token_ids") or []
                    if not isinstance(token_ids, list):
                        token_ids = []
                    if not fragment and not token_ids:
                        continue
                    arrival_ns = time.time_ns()
                    if first_token_ns is None:
                        first_token_ns = arrival_ns
                        await sink.event(
                            window["window_id"],
                            request_id,
                            attempt,
                            "first_token",
                        )
                    chunk_index = len(chunk_rows)
                    tokens_in_chunk = len(token_ids)
                    timestamp_source = (
                        "server_token_ids" if token_ids else "chunk_fallback"
                    )
                    if not token_ids:
                        # Preserve a timestamp even if an older/incompatible
                        # server omits token_ids. The per-window integrity
                        # check below will reject successful incomplete traces.
                        token_ids = [None]
                        tokens_in_chunk = 1
                    chunk_rows.append(
                        {
                            "window_id": window["window_id"],
                            "request_id": request_id,
                            "stream_chunk_index": chunk_index,
                            "arrival_unix_ns": arrival_ns,
                            "since_send_ms": "",
                            "since_previous_chunk_ms": (
                                ""
                                if previous_chunk_ns is None
                                else (arrival_ns - previous_chunk_ns) / 1_000_000
                            ),
                            "tokens_in_chunk": tokens_in_chunk,
                            "token_ids_json": json.dumps(token_ids),
                            "fragment_bytes": len(fragment.encode("utf-8")),
                            "fragment_json": json.dumps(
                                fragment, ensure_ascii=False
                            ),
                        }
                    )
                    for token_index, token_id in enumerate(token_ids):
                        token_rows.append(
                            {
                                "window_id": window["window_id"],
                                "request_id": request_id,
                                "token_event_index": len(token_rows),
                                "stream_chunk_index": chunk_index,
                                "token_index_in_chunk": token_index,
                                "tokens_in_chunk": tokens_in_chunk,
                                "token_id": "" if token_id is None else token_id,
                                "timestamp_source": timestamp_source,
                                "arrival_unix_ns": arrival_ns,
                                "since_send_ms": "",
                                "since_previous_token_ms": (
                                    ""
                                    if previous_token_ns is None
                                    else (
                                        arrival_ns - previous_token_ns
                                    )
                                    / 1_000_000
                                ),
                                # Chunk text belongs to the first token row;
                                # later token rows share its arrival timestamp.
                                "fragment_bytes": (
                                    len(fragment.encode("utf-8"))
                                    if token_index == 0
                                    else 0
                                ),
                                "fragment_json": (
                                    json.dumps(fragment, ensure_ascii=False)
                                    if token_index == 0
                                    else ""
                                ),
                            }
                        )
                        previous_token_ns = arrival_ns
                    previous_chunk_ns = arrival_ns
        completed_ns = time.time_ns()

    return {
        "response_headers_ns": response_headers_ns,
        "first_token_ns": first_token_ns,
        "completed_ns": completed_ns,
        "http_status": status,
        "response_request_id": response_request_id,
        "token_rows": token_rows,
        "chunk_rows": chunk_rows,
        "response_bytes": response_bytes,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens or len(token_rows),
    }


async def scheduled_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    window: dict[str, Any],
    request_index: int,
    window_start_monotonic: float,
    window_start_unix_ns: int,
    semaphore: asyncio.Semaphore,
    sink: CsvSink,
) -> dict[str, Any]:
    rate = float(window["request_rate"])
    offset_s = request_index / rate
    planned_monotonic = window_start_monotonic + offset_s
    planned_unix_ns = window_start_unix_ns + int(offset_s * 1_000_000_000)
    await asyncio.sleep(max(0.0, planned_monotonic - time.monotonic()))
    ready_ns = time.time_ns()
    request_id = f"{window['window_id']}-{request_index:05d}-{uuid.uuid4().hex[:12]}"
    trace_id = uuid.uuid4().hex
    client_span_id = uuid.uuid4().hex[:16]
    traceparent = f"00-{trace_id}-{client_span_id}-01"
    await sink.event(window["window_id"], request_id, 0, "client_ready")

    async with semaphore:
        actual_send_ns = time.time_ns()
        await sink.event(window["window_id"], request_id, 0, "semaphore_acquired")
        max_attempts = int(window["retries"]) + 1
        error = ""
        result: dict[str, Any] = {}
        timeout_hit = False
        cancelled = False
        attempts = 0
        last_http_status = 0
        cancel_fraction = float(window.get("cancel_fraction", 0) or 0)
        cancel_after_s = float(window.get("cancel_after_s", 0) or 0)
        cancel_stride = (
            max(1, round(1 / cancel_fraction)) if cancel_fraction > 0 else 0
        )
        should_cancel = (
            cancel_stride > 0
            and cancel_after_s > 0
            and request_index % cancel_stride == 0
        )
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                attempt_task = asyncio.create_task(
                    run_attempt(
                        session,
                        endpoint,
                        model,
                        window,
                        request_index,
                        request_id,
                        traceparent,
                        attempt,
                        sink,
                    )
                )
                if should_cancel and attempt == 1:
                    done, _ = await asyncio.wait(
                        {attempt_task}, timeout=cancel_after_s
                    )
                    if not done:
                        attempt_task.cancel()
                        try:
                            await attempt_task
                        except asyncio.CancelledError:
                            pass
                        cancelled = True
                        error = "CancelledByClient"
                        await sink.event(
                            window["window_id"],
                            request_id,
                            attempt,
                            "cancelled_by_client",
                        )
                        break
                result = await attempt_task
                error = ""
                break
            except asyncio.TimeoutError:
                timeout_hit = True
                error = "TimeoutError"
                await sink.event(
                    window["window_id"], request_id, attempt, "timeout"
                )
            except asyncio.CancelledError:
                cancelled = True
                error = "CancelledError"
                await sink.event(
                    window["window_id"], request_id, attempt, "cancelled"
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if "HTTP " in error:
                    try:
                        last_http_status = int(
                            error.split("HTTP ", 1)[1].split(":", 1)[0]
                        )
                    except ValueError:
                        pass
                await sink.event(
                    window["window_id"],
                    request_id,
                    attempt,
                    "attempt_error",
                    detail=error,
                )
            if attempt < max_attempts:
                await sink.event(
                    window["window_id"], request_id, attempt, "retry_scheduled"
                )
                await asyncio.sleep(min(1.0, 0.1 * (2 ** (attempt - 1))))

        completed_ns = result.get("completed_ns", time.time_ns())
        first_token_ns = result.get("first_token_ns")
        actual_output_tokens = int(result.get("actual_output_tokens", 0))
        tpot_ms: float | str = ""
        if first_token_ns and actual_output_tokens > 1:
            tpot_ms = (
                (completed_ns - first_token_ns)
                / 1_000_000
                / (actual_output_tokens - 1)
            )
        for token_row in result.get("token_rows", []):
            token_row["since_send_ms"] = (
                token_row["arrival_unix_ns"] - actual_send_ns
            ) / 1_000_000
        for chunk_row in result.get("chunk_rows", []):
            chunk_row["since_send_ms"] = (
                chunk_row["arrival_unix_ns"] - actual_send_ns
            ) / 1_000_000
        await sink.append(
            "token_timestamps.csv",
            TOKEN_FIELDS,
            result.get("token_rows", []),
        )
        await sink.append(
            "stream_chunk_events.csv",
            CHUNK_FIELDS,
            result.get("chunk_rows", []),
        )
        token_rows = result.get("token_rows", [])
        token_timestamp_complete = bool(
            not error
            and actual_output_tokens == len(token_rows)
            and all(
                item.get("timestamp_source") == "server_token_ids"
                for item in token_rows
            )
        )
        row = {
            "window_id": window["window_id"],
            "request_index": request_index,
            "request_id": request_id,
            "trace_id": trace_id,
            "client_span_id": client_span_id,
            "planned_send_unix_ns": planned_unix_ns,
            "client_ready_unix_ns": ready_ns,
            "actual_send_unix_ns": actual_send_ns,
            "response_headers_unix_ns": result.get("response_headers_ns", ""),
            "first_token_unix_ns": first_token_ns or "",
            "completed_unix_ns": completed_ns,
            "planned_to_ready_ms": ns_to_ms(ready_ns, planned_unix_ns),
            "client_queue_delay_ms": ns_to_ms(actual_send_ns, ready_ns),
            "send_lag_ms": ns_to_ms(actual_send_ns, planned_unix_ns),
            "ttft_ms": ns_to_ms(first_token_ns, actual_send_ns),
            "tpot_ms": tpot_ms,
            "e2e_ms": ns_to_ms(completed_ns, actual_send_ns),
            "planned_input_tokens": int(window["input_tokens"]),
            "actual_input_tokens": result.get("actual_input_tokens", 0),
            "max_tokens": int(window["max_tokens"]),
            "actual_output_tokens": actual_output_tokens,
            "token_event_count": len(token_rows),
            "stream_chunk_count": len(result.get("chunk_rows", [])),
            "token_id_event_count": sum(
                item.get("timestamp_source") == "server_token_ids"
                for item in token_rows
            ),
            "token_timestamp_complete": token_timestamp_complete,
            "http_status": result.get("http_status", last_http_status),
            "response_request_id": result.get("response_request_id", ""),
            "outcome": "success" if not error else "failed",
            "timeout": timeout_hit,
            "cancelled": cancelled,
            "retry_count": max(0, attempts - 1),
            "attempt_count": attempts,
            "response_bytes": result.get("response_bytes", 0),
            "error": error,
        }
        await sink.append("requests.csv", REQUEST_FIELDS, [row])
        await sink.event(
            window["window_id"],
            request_id,
            attempts,
            "request_complete" if not error else "request_failed",
            row["http_status"],
            error,
        )
        return row


def load_windows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("workload file contains no windows")
    return rows


async def run(args: argparse.Namespace) -> None:
    windows = load_windows(args.workloads)
    sink = CsvSink(args.output_dir)
    random.seed(args.seed)
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for window in windows:
            window_id = window["window_id"]
            rate = float(window["request_rate"])
            duration_s = float(window["duration_s"])
            planned_requests = max(1, round(rate * duration_s))
            semaphore = asyncio.Semaphore(int(window["max_concurrency"]))
            start_unix_ns = time.time_ns()
            start_monotonic = time.monotonic()
            await sink.marker(
                "window_start",
                window_id,
                json.dumps(window, sort_keys=True),
            )
            tasks = [
                asyncio.create_task(
                    scheduled_request(
                        session,
                        args.endpoint,
                        args.model,
                        window,
                        index,
                        start_monotonic,
                        start_unix_ns,
                        semaphore,
                        sink,
                    )
                )
                for index in range(planned_requests)
            ]
            await asyncio.sleep(
                max(0.0, start_monotonic + duration_s - time.monotonic())
            )
            sending_stopped_unix_ns = time.time_ns()
            await sink.marker("sending_stopped", window_id)
            results = await asyncio.gather(*tasks)
            end_unix_ns = time.time_ns()
            await sink.marker("window_end", window_id)

            successful = [row for row in results if row["outcome"] == "success"]
            incomplete_token_traces = [
                row for row in successful if not row["token_timestamp_complete"]
            ]
            ttfts = [float(row["ttft_ms"]) for row in successful if row["ttft_ms"] != ""]
            tpots = [float(row["tpot_ms"]) for row in successful if row["tpot_ms"] != ""]
            e2es = [float(row["e2e_ms"]) for row in successful if row["e2e_ms"] != ""]
            send_times = [int(row["actual_send_unix_ns"]) for row in results]
            completion_times = [int(row["completed_unix_ns"]) for row in results]
            send_span_s = max(1e-9, (max(send_times) - min(send_times)) / 1e9)
            completion_span_s = max(
                1e-9, (max(completion_times) - min(send_times)) / 1e9
            )
            output_tokens = sum(int(row["actual_output_tokens"]) for row in successful)
            summary = {
                "window_id": window_id,
                "window_start_unix_ns": start_unix_ns,
                "sending_stopped_unix_ns": sending_stopped_unix_ns,
                "window_end_unix_ns": end_unix_ns,
                "client_observed_drain_s": max(
                    0.0, (end_unix_ns - sending_stopped_unix_ns) / 1e9
                ),
                "planned_requests": planned_requests,
                "completed_requests": len(successful),
                "failed_requests": len(results) - len(successful),
                "timeout_requests": sum(bool(row["timeout"]) for row in results),
                "cancelled_requests": sum(bool(row["cancelled"]) for row in results),
                "retry_attempts": sum(int(row["retry_count"]) for row in results),
                "actual_send_rps": len(results) / send_span_s,
                "completion_output_tps": output_tokens / completion_span_s,
                "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else math.nan,
                "p50_ttft_ms": percentile(ttfts, 0.50),
                "p95_ttft_ms": percentile(ttfts, 0.95),
                "p99_ttft_ms": percentile(ttfts, 0.99),
                "mean_tpot_ms": statistics.fmean(tpots) if tpots else math.nan,
                "p99_tpot_ms": percentile(tpots, 0.99),
                "mean_e2e_ms": statistics.fmean(e2es) if e2es else math.nan,
                "p99_e2e_ms": percentile(e2es, 0.99),
            }
            await sink.append("window_summary.csv", WINDOW_FIELDS, [summary])
            if incomplete_token_traces:
                raise RuntimeError(
                    f"window {window_id} has "
                    f"{len(incomplete_token_traces)} successful requests "
                    "without complete server token-ID timestamps"
                )
            if window_id != "timeout_retry_probe" and not successful:
                raise RuntimeError(
                    f"window {window_id} produced no successful requests"
                )
            pause_s = float(window["inter_window_pause_s"])
            if pause_s > 0:
                await sink.marker("inter_window_pause_start", window_id)
                await asyncio.sleep(pause_s)
                await sink.marker("inter_window_pause_end", window_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
