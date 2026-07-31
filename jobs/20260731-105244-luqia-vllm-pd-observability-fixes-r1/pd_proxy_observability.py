#!/usr/bin/env python3
"""Minimal vLLM P2P-NCCL xPyD proxy, adapted from vLLM v0.15.1."""

import csv
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid
from typing import Any, AsyncIterator

import aiohttp
import msgpack
import zmq
from aiohttp import web


PREFILL_INSTANCES: dict[str, tuple[str, float]] = {}
DECODE_INSTANCES: dict[str, tuple[str, float]] = {}
PREFILL_LOCK = threading.Lock()
DECODE_LOCK = threading.Lock()
PING_TTL_SECONDS = 10
REQUEST_COUNT = 0
PREFILL_INFLIGHT = 0
DECODE_INFLIGHT = 0
EVENTS_PATH = Path(os.environ["PROXY_EVENTS_CSV"])
EVENTS_LOCK = threading.Lock()
EVENT_FIELDS = [
    "unix_ns",
    "monotonic_ns",
    "event",
    "client_request_id",
    "internal_request_id",
    "route_index",
    "stage",
    "upstream_url",
    "http_status",
    "bytes",
    "prefill_http",
    "decode_http",
    "traceparent",
    "error",
    "prefill_inflight",
    "decode_inflight",
]


def inflight_snapshot() -> tuple[int, int]:
    return PREFILL_INFLIGHT, DECODE_INFLIGHT


def change_inflight(stage: str, delta: int) -> None:
    global PREFILL_INFLIGHT, DECODE_INFLIGHT
    if stage == "prefill":
        PREFILL_INFLIGHT = max(0, PREFILL_INFLIGHT + delta)
    elif stage == "decode":
        DECODE_INFLIGHT = max(0, DECODE_INFLIGHT + delta)


def emit_event(event: str, **values: Any) -> None:
    row = {field: values.get(field, "") for field in EVENT_FIELDS}
    prefill_inflight, decode_inflight = inflight_snapshot()
    row["unix_ns"] = time.time_ns()
    row["monotonic_ns"] = time.monotonic_ns()
    row["event"] = event
    row["prefill_inflight"] = values.get(
        "prefill_inflight", prefill_inflight
    )
    row["decode_inflight"] = values.get("decode_inflight", decode_inflight)
    with EVENTS_LOCK:
        is_new = not EVENTS_PATH.exists()
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


def remove_expired(instances: dict[str, tuple[str, float]]) -> None:
    now = time.time()
    for key, value in list(instances.items()):
        if value[1] <= now:
            print(f"registry_remove http={key} zmq={value[0]}", flush=True)
            instances.pop(key, None)


def listen_for_registration(poller: zmq.Poller, router: zmq.Socket) -> None:
    while True:
        sockets = dict(poller.poll())
        if router not in sockets:
            continue
        _, message = router.recv_multipart()
        data = msgpack.loads(message, raw=False)
        instance_type = data.get("type")
        http_address = data.get("http_address")
        zmq_address = data.get("zmq_address")
        if not http_address or not zmq_address or instance_type not in {"P", "D"}:
            print(f"registry_invalid data={data!r}", flush=True)
            continue

        instances = PREFILL_INSTANCES if instance_type == "P" else DECODE_INSTANCES
        lock = PREFILL_LOCK if instance_type == "P" else DECODE_LOCK
        with lock:
            is_new = http_address not in instances
            instances[http_address] = (zmq_address, time.time() + PING_TTL_SECONDS)
            remove_expired(instances)
        if is_new:
            role = "prefill" if instance_type == "P" else "decode"
            print(
                f"registry_add role={role} http={http_address} zmq={zmq_address}",
                flush=True,
            )


def start_service_discovery(host: str, port: int) -> threading.Thread:
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(f"tcp://{host}:{port}")
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    thread = threading.Thread(
        target=listen_for_registration,
        args=(poller, router),
        daemon=True,
    )
    thread.start()
    return thread


HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)
app = web.Application()


async def forward_request(
    url: str,
    data: dict[str, Any],
    internal_request_id: str,
    client_request_id: str,
    route_index: int,
    stage: str,
    traceparent: str,
) -> AsyncIterator[bytes]:
    headers = {"X-Request-Id": internal_request_id}
    if traceparent:
        headers["traceparent"] = traceparent
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    change_inflight(stage, 1)
    try:
        emit_event(
            "upstream_dispatch",
            client_request_id=client_request_id,
            internal_request_id=internal_request_id,
            route_index=route_index,
            stage=stage,
            upstream_url=url,
            traceparent=traceparent,
        )
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data, headers=headers) as response:
                emit_event(
                    "upstream_headers",
                    client_request_id=client_request_id,
                    internal_request_id=internal_request_id,
                    route_index=route_index,
                    stage=stage,
                    upstream_url=url,
                    http_status=response.status,
                    traceparent=traceparent,
                )
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"upstream status={response.status} "
                        f"url={url} body={body}"
                    )
                total_bytes = 0
                first_chunk = True
                async for chunk in response.content.iter_chunked(1024):
                    total_bytes += len(chunk)
                    if first_chunk:
                        emit_event(
                            "upstream_first_chunk",
                            client_request_id=client_request_id,
                            internal_request_id=internal_request_id,
                            route_index=route_index,
                            stage=stage,
                            upstream_url=url,
                            http_status=response.status,
                            bytes=len(chunk),
                            traceparent=traceparent,
                        )
                        first_chunk = False
                    yield chunk
                emit_event(
                    "upstream_complete",
                    client_request_id=client_request_id,
                    internal_request_id=internal_request_id,
                    route_index=route_index,
                    stage=stage,
                    upstream_url=url,
                    http_status=response.status,
                    bytes=total_bytes,
                    traceparent=traceparent,
                )
    finally:
        change_inflight(stage, -1)
        emit_event(
            "upstream_inflight_released",
            client_request_id=client_request_id,
            internal_request_id=internal_request_id,
            route_index=route_index,
            stage=stage,
            upstream_url=url,
            traceparent=traceparent,
        )


def snapshot_registry() -> tuple[list[tuple[str, tuple[str, float]]], list[tuple[str, tuple[str, float]]]]:
    with PREFILL_LOCK:
        remove_expired(PREFILL_INSTANCES)
        prefill = list(PREFILL_INSTANCES.items())
    with DECODE_LOCK:
        remove_expired(DECODE_INSTANCES)
        decode = list(DECODE_INSTANCES.items())
    return prefill, decode


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def registry(_: web.Request) -> web.Response:
    prefill, decode = snapshot_registry()
    return web.json_response(
        {
            "prefill": [item[0] for item in prefill],
            "decode": [item[0] for item in decode],
        }
    )


async def handle_request(request: web.Request) -> web.StreamResponse:
    global REQUEST_COUNT
    client_request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    client_request_id = re.sub(r"[^A-Za-z0-9._:-]", "_", client_request_id)[:160]
    traceparent = request.headers.get("traceparent", "")
    emit_event(
        "proxy_arrival",
        client_request_id=client_request_id,
        traceparent=traceparent,
    )
    try:
        original = await request.json()
        if not isinstance(original, dict):
            return web.json_response(
                {"error": "request body must be a JSON object"}, status=400
            )

        prefill, decode = snapshot_registry()
        if not prefill or not decode:
            return web.json_response(
                {
                    "error": "PD instances are not ready",
                    "prefill_count": len(prefill),
                    "decode_count": len(decode),
                },
                status=503,
            )

        index = REQUEST_COUNT
        REQUEST_COUNT += 1
        prefill_http, (prefill_zmq, _) = prefill[index % len(prefill)]
        decode_http, (decode_zmq, _) = decode[index % len(decode)]
        request_id = (
            f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_"
            f"{client_request_id}_{uuid.uuid4().hex}"
        )
        print(
            f"route count={index} client_request_id={client_request_id} "
            f"internal_request_id={request_id} "
            f"prefill_http={prefill_http} decode_http={decode_http}",
            flush=True,
        )
        emit_event(
            "proxy_route",
            client_request_id=client_request_id,
            internal_request_id=request_id,
            route_index=index,
            prefill_http=prefill_http,
            decode_http=decode_http,
            traceparent=traceparent,
        )

        prefill_body = dict(original)
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1

        async for _ in forward_request(
            f"http://{prefill_http}{request.path}",
            prefill_body,
            request_id,
            client_request_id,
            index,
            "prefill",
            traceparent,
        ):
            pass

        content_type = (
            "text/event-stream"
            if bool(original.get("stream"))
            else "application/json"
        )
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": content_type,
                "X-Request-Id": client_request_id,
            },
        )
        await response.prepare(request)
        async for chunk in forward_request(
            f"http://{decode_http}{request.path}",
            original,
            request_id,
            client_request_id,
            index,
            "decode",
            traceparent,
        ):
            await response.write(chunk)
        await response.write_eof()
        emit_event(
            "proxy_complete",
            client_request_id=client_request_id,
            internal_request_id=request_id,
            route_index=index,
            prefill_http=prefill_http,
            decode_http=decode_http,
            traceparent=traceparent,
        )
        return response
    except Exception as exc:
        import traceback

        emit_event(
            "proxy_error",
            client_request_id=client_request_id,
            traceparent=traceparent,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"proxy_error type={type(exc).__name__} error={exc}", flush=True)
        traceback.print_exc()
        return web.json_response(
            {"error": str(exc), "type": type(exc).__name__},
            status=502,
            headers={"X-Request-Id": client_request_id},
        )


app.router.add_get("/health", health)
app.router.add_get("/registry", registry)
app.router.add_post("/v1/completions", handle_request)
app.router.add_post("/v1/chat/completions", handle_request)


if __name__ == "__main__":
    register_host = os.environ.get("PROXY_REGISTER_HOST", "0.0.0.0")
    register_port = int(os.environ.get("PROXY_REGISTER_PORT", "30001"))
    http_host = os.environ.get("PROXY_HTTP_HOST", "0.0.0.0")
    http_port = int(os.environ.get("PROXY_HTTP_PORT", "10001"))
    print(
        f"proxy_start hostname={socket.gethostname()} http={http_host}:{http_port} "
        f"registry={register_host}:{register_port}",
        flush=True,
    )
    start_service_discovery(register_host, register_port)
    web.run_app(app, host=http_host, port=http_port, print=None)
