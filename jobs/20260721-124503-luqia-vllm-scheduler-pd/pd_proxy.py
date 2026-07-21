#!/usr/bin/env python3
"""Minimal vLLM P2P-NCCL xPyD proxy, adapted from vLLM v0.15.1."""

import os
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
    url: str, data: dict[str, Any], request_id: str
) -> AsyncIterator[bytes]:
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"upstream status={response.status} url={url} body={body}")
            async for chunk in response.content.iter_chunked(1024):
                yield chunk


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
            f"{uuid.uuid4().hex}"
        )
        print(
            f"route count={index} request_id={request_id} "
            f"prefill_http={prefill_http} decode_http={decode_http}",
            flush=True,
        )

        prefill_body = dict(original)
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1

        async for _ in forward_request(
            f"http://{prefill_http}{request.path}", prefill_body, request_id
        ):
            pass

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/json"},
        )
        await response.prepare(request)
        async for chunk in forward_request(
            f"http://{decode_http}{request.path}", original, request_id
        ):
            await response.write(chunk)
        await response.write_eof()
        return response
    except Exception as exc:
        import traceback

        print(f"proxy_error type={type(exc).__name__} error={exc}", flush=True)
        traceback.print_exc()
        return web.json_response(
            {"error": str(exc), "type": type(exc).__name__}, status=502
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
