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
from quart import Quart, jsonify, make_response, request


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
app = Quart(__name__)


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


@app.get("/health")
async def health():
    return jsonify({"ok": True})


@app.get("/registry")
async def registry():
    prefill, decode = snapshot_registry()
    return jsonify(
        {
            "prefill": [item[0] for item in prefill],
            "decode": [item[0] for item in decode],
        }
    )


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def handle_request():
    global REQUEST_COUNT
    try:
        original = await request.get_json()
        if not isinstance(original, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400

        prefill, decode = snapshot_registry()
        if not prefill or not decode:
            return jsonify(
                {
                    "error": "PD instances are not ready",
                    "prefill_count": len(prefill),
                    "decode_count": len(decode),
                }
            ), 503

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

        response = await make_response(
            forward_request(
                f"http://{decode_http}{request.path}", original, request_id
            )
        )
        response.timeout = None
        return response
    except Exception as exc:
        import traceback

        print(f"proxy_error type={type(exc).__name__} error={exc}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(exc), "type": type(exc).__name__}), 502


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
    discovery_thread = start_service_discovery(register_host, register_port)
    app.run(host=http_host, port=http_port)
    discovery_thread.join()
