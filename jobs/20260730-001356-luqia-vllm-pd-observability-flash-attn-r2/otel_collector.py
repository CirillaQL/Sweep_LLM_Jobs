#!/usr/bin/env python3
"""Small OTLP/HTTP trace receiver that writes every vLLM span to CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import time
import uuid
from typing import Any

from aiohttp import web
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


SPAN_FIELDS = [
    "received_unix_ns",
    "batch_id",
    "service_name",
    "scope_name",
    "scope_version",
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "kind",
    "start_time_unix_ns",
    "end_time_unix_ns",
    "duration_ms",
    "status_code",
    "status_message",
    "resource_attributes_json",
    "span_attributes_json",
    "events_json",
    "links_count",
]

BATCH_FIELDS = [
    "received_unix_ns",
    "batch_id",
    "content_length",
    "span_count",
    "decode_ok",
    "error",
]


def any_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bytes_value":
        return value.bytes_value.hex()
    if kind == "array_value":
        return [any_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return attributes(value.kvlist_value.values)
    return None


def attributes(items: Any) -> dict[str, Any]:
    return {item.key: any_value(item.value) for item in items}


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


class Collector:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "otel_raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def export(self, request: web.Request) -> web.Response:
        received_ns = time.time_ns()
        batch_id = uuid.uuid4().hex
        payload = await request.read()
        if request.headers.get("Content-Encoding", "").lower() == "gzip":
            payload = gzip.decompress(payload)
        (self.raw_dir / f"{received_ns}-{batch_id}.pb").write_bytes(payload)

        span_rows: list[dict[str, Any]] = []
        error = ""
        try:
            export_request = ExportTraceServiceRequest()
            export_request.ParseFromString(payload)
            for resource_spans in export_request.resource_spans:
                resource_attrs = attributes(resource_spans.resource.attributes)
                service_name = str(
                    resource_attrs.get("service.name", "unknown-service")
                )
                resource_json = json.dumps(
                    resource_attrs, sort_keys=True, ensure_ascii=False
                )
                for scope_spans in resource_spans.scope_spans:
                    scope = scope_spans.scope
                    for span in scope_spans.spans:
                        span_attrs = attributes(span.attributes)
                        event_rows = []
                        for event in span.events:
                            event_rows.append(
                                {
                                    "name": event.name,
                                    "time_unix_ns": event.time_unix_nano,
                                    "attributes": attributes(event.attributes),
                                }
                            )
                        span_rows.append(
                            {
                                "received_unix_ns": received_ns,
                                "batch_id": batch_id,
                                "service_name": service_name,
                                "scope_name": scope.name,
                                "scope_version": scope.version,
                                "trace_id": span.trace_id.hex(),
                                "span_id": span.span_id.hex(),
                                "parent_span_id": span.parent_span_id.hex(),
                                "name": span.name,
                                "kind": int(span.kind),
                                "start_time_unix_ns": span.start_time_unix_nano,
                                "end_time_unix_ns": span.end_time_unix_nano,
                                "duration_ms": (
                                    span.end_time_unix_nano
                                    - span.start_time_unix_nano
                                )
                                / 1_000_000,
                                "status_code": int(span.status.code),
                                "status_message": span.status.message,
                                "resource_attributes_json": resource_json,
                                "span_attributes_json": json.dumps(
                                    span_attrs,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                ),
                                "events_json": json.dumps(
                                    event_rows,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                ),
                                "links_count": len(span.links),
                            }
                        )
            append_rows(
                self.output_dir / "otel_spans.csv",
                SPAN_FIELDS,
                span_rows,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        append_rows(
            self.output_dir / "otel_batches.csv",
            BATCH_FIELDS,
            [
                {
                    "received_unix_ns": received_ns,
                    "batch_id": batch_id,
                    "content_length": len(payload),
                    "span_count": len(span_rows),
                    "decode_ok": not bool(error),
                    "error": error,
                }
            ],
        )
        if error:
            return web.json_response({"error": error}, status=400)
        return web.Response(body=b"", content_type="application/x-protobuf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    collector = Collector(args.output_dir)
    app = web.Application(client_max_size=128 * 1024**2)
    app.router.add_get("/health", collector.health)
    app.router.add_post("/", collector.export)
    app.router.add_post("/v1/traces", collector.export)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
