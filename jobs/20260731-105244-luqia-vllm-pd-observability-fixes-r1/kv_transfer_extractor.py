#!/usr/bin/env python3
"""Build request-correlated proxy-observed KV handoff upper bounds.

P2pNcclConnector in vLLM 0.15.1 does not expose request-level transfer
timestamps or byte counts. The proxy does know the full Prefill request
interval and the exact transition to Decode. We store that interval as an
explicit upper bound and label estimated fields so it cannot be mistaken for
pure NCCL transfer time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = [
    "window_id",
    "request_id",
    "trace_id",
    "transfer_id",
    "started_unix_ns",
    "completed_unix_ns",
    "source_role",
    "destination_role",
    "transport",
    "status",
    "blocks",
    "transfer_bytes",
    "queue_delay_us",
    "transfer_duration_us",
    "retry_count",
    "error",
    "attributes_json",
]

BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2
BLOCK_SIZE_TOKENS = 16


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first(
    rows: list[dict[str, str]],
    event: str,
    stage: str = "",
) -> dict[str, str] | None:
    for row in rows:
        if row.get("event") != event:
            continue
        if stage and row.get("stage") != stage:
            continue
        return row
    return None


def ns(row: dict[str, str] | None) -> int:
    if not row:
        return 0
    try:
        return int(row.get("unix_ns", "0"))
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--proxy-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = read_csv(args.requests)
    grouped: dict[str, list[dict[str, str]]] = {}
    for event in read_csv(args.proxy_events):
        request_id = event.get("client_request_id", "")
        if request_id:
            grouped.setdefault(request_id, []).append(event)

    output_rows = []
    for request in requests:
        request_id = request["request_id"]
        events = sorted(
            grouped.get(request_id, []),
            key=lambda row: ns(row),
        )
        arrival = first(events, "proxy_arrival")
        route = first(events, "proxy_route")
        prefill_dispatch = first(events, "upstream_dispatch", "prefill")
        prefill_headers = first(events, "upstream_headers", "prefill")
        prefill_first_chunk = first(
            events, "upstream_first_chunk", "prefill"
        )
        prefill_complete = first(events, "upstream_complete", "prefill")
        decode_dispatch = first(events, "upstream_dispatch", "decode")

        started_ns = ns(prefill_dispatch)
        completed_ns = ns(prefill_complete)
        if not started_ns:
            continue
        input_tokens = int(float(request.get("actual_input_tokens") or 0))
        blocks = math.ceil(input_tokens / BLOCK_SIZE_TOKENS) if input_tokens else 0
        status = (
            "proxy_observed_upper_bound_complete"
            if completed_ns
            else "proxy_observed_upper_bound_incomplete"
        )
        error = "" if completed_ns else "prefill upstream_complete missing"
        arrival_ns = ns(arrival)
        decode_dispatch_ns = ns(decode_dispatch)
        attributes = {
            "measurement_scope": "prefill_request_including_kv_handoff",
            "duration_is_pure_kv_transfer": "false",
            "duration_is_upper_bound": "true",
            "duration_includes": (
                "prefill_queue,model_prefill,one_output_token,"
                "kv_put_async,http_response"
            ),
            "bytes_are_estimated": "true",
            "bytes_estimate_kind": "mistral7b_fp16_kv_upper_bound",
            "bytes_per_input_token": str(BYTES_PER_TOKEN),
            "block_size_tokens": str(BLOCK_SIZE_TOKENS),
            "prefill_headers_unix_ns": str(ns(prefill_headers)),
            "prefill_first_chunk_unix_ns": str(ns(prefill_first_chunk)),
            "decode_dispatch_unix_ns": str(decode_dispatch_ns),
            "prefill_complete_to_decode_dispatch_us": str(
                max(0, (decode_dispatch_ns - completed_ns) // 1000)
                if decode_dispatch_ns and completed_ns
                else 0
            ),
            "prefill_http": route.get("prefill_http", "") if route else "",
            "decode_http": route.get("decode_http", "") if route else "",
            "send_type": "PUT_ASYNC",
            "connector": "P2pNcclConnector",
        }
        output_rows.append(
            {
                "window_id": request["window_id"],
                "request_id": request_id,
                "trace_id": request.get("trace_id", ""),
                "transfer_id": (
                    route.get("internal_request_id", "")
                    if route
                    else request_id
                ),
                "started_unix_ns": started_ns,
                "completed_unix_ns": completed_ns or "",
                "source_role": "prefill",
                "destination_role": "decode",
                "transport": "P2pNcclConnector/PUT_ASYNC/NCCL-Socket",
                "status": status,
                "blocks": blocks,
                "transfer_bytes": input_tokens * BYTES_PER_TOKEN,
                "queue_delay_us": (
                    max(0, (started_ns - arrival_ns) // 1000)
                    if arrival_ns
                    else 0
                ),
                "transfer_duration_us": (
                    max(0, (completed_ns - started_ns) // 1000)
                    if completed_ns
                    else 0
                ),
                "retry_count": int(float(request.get("retry_count") or 0)),
                "error": error,
                "attributes_json": json.dumps(
                    attributes,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"kv_transfer_events={len(output_rows)} "
        f"measurement=proxy_observed_upper_bound output={args.output}"
    )


if __name__ == "__main__":
    main()
