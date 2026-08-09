#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Custom instance-pair scheduler for a multi-prefill/multi-decode PD proxy.

This follows the two-stage forwarding pattern in vLLM's
``examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`` while
retaining this repository's P2pNcclConnector service-discovery protocol.

Callers can select any registered pair whose Prefill and Decode tensor
parallel sizes match, for example with TP layouts ``P=[1, 1, 2]`` and
``D=[1, 1, 2]``::

    P0-D0, P0-D1, P1-D0, P1-D1, P2-D2

Selection priority is:

1. ``X-PD-Route`` request header.
2. ``pd_route`` query parameter.
3. ``pd_route`` or ``_pd_route`` JSON field (removed before forwarding).
4. The configured default policy.

By default, a Prefill index is sampled first from the live registry and its
Decode index is then sampled from instances with the same TP size.
``POST /control/default-route`` can switch the default to a fixed pair,
``random``, or ``round_robin``. Fixed routes with asymmetric TP are rejected.
Aliases are assigned by increasing HTTP port, making them stable across
requests.
"""

from __future__ import annotations

import os
import random
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp
import msgpack
import zmq
from aiohttp import web


ROUTE_PATTERN = re.compile(r"^P([0-9]+)-D([0-9]+)$")
ROUND_ROBIN = "round_robin"
RANDOM_ROUTE = "random"
PING_TTL_SECONDS = int(os.environ.get("CUSTOM_PD_PING_TTL_SECONDS", "10"))
DEFAULT_TP_SIZE = int(os.environ.get("PROXY_DEFAULT_TP_SIZE", "1"))
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)


def parse_tp_sizes(name: str) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    values = tuple(int(value) for value in raw.split(","))
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers: {raw!r}")
    return values


def optional_port(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


PREFILL_TP_SIZES = parse_tp_sizes("PREFILL_TP_SIZES")
DECODE_TP_SIZES = parse_tp_sizes("DECODE_TP_SIZES")
PREFILL_HTTP_PORT_BASE = optional_port("PREFILL_HTTP_PORT_BASE")
DECODE_HTTP_PORT_BASE = optional_port("DECODE_HTTP_PORT_BASE")


@dataclass(frozen=True)
class Instance:
    http_address: str
    zmq_address: str
    expires_at: float
    tp_size: int


PREFILL_INSTANCES: dict[str, Instance] = {}
DECODE_INSTANCES: dict[str, Instance] = {}
PREFILL_LOCK = threading.Lock()
DECODE_LOCK = threading.Lock()
CLOCK_ACKS: dict[tuple[str, int], dict[str, Any]] = {}


def normalize_route(value: Any, *, allow_policy: bool = False) -> str:
    """Return a canonical route such as P0-D1 or raise ValueError."""
    if not isinstance(value, str):
        raise ValueError("route must be a string")
    normalized = value.strip().upper().replace(" ", "")
    normalized = normalized.replace("->", "-").replace(":", "-")
    if allow_policy:
        if normalized in {"RANDOM", "RAND", "AUTO"}:
            return RANDOM_ROUTE
        if normalized in {"ROUND_ROBIN", "RR"}:
            return ROUND_ROBIN
    if not ROUTE_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"unsupported route {value!r}; expected P<index>-D<index>"
        )
    return normalized


def compatible_route_choices(
    prefill: list[Instance], decode: list[Instance]
) -> tuple[str, ...]:
    """Return symmetric-TP P-D pairs in stable Prefill-major order."""
    return tuple(
        f"P{prefill_index}-D{decode_index}"
        for prefill_index, prefill_instance in enumerate(prefill)
        for decode_index, decode_instance in enumerate(decode)
        if prefill_instance.tp_size == decode_instance.tp_size
    )


def http_sort_key(address: str) -> tuple[int, str, str]:
    """Sort host:port endpoints predictably; malformed ports sort last."""
    try:
        host, port_text = address.rsplit(":", 1)
        return int(port_text), host, address
    except (ValueError, TypeError):
        return 65536, address, address


def configured_tp_size(
    instance_type: str,
    http_address: str,
    reported_tp_size: int,
) -> int:
    """Resolve mixed TP from the configured per-instance port mapping."""
    if instance_type == "P":
        tp_sizes = PREFILL_TP_SIZES
        port_base = PREFILL_HTTP_PORT_BASE
    else:
        tp_sizes = DECODE_TP_SIZES
        port_base = DECODE_HTTP_PORT_BASE
    if not tp_sizes or port_base is None:
        return reported_tp_size
    try:
        port = int(http_address.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return reported_tp_size
    ordinal = port - port_base
    if 0 <= ordinal < len(tp_sizes):
        return tp_sizes[ordinal]
    return reported_tp_size


def remove_expired(instances: dict[str, Instance]) -> None:
    now = time.time()
    for key, value in list(instances.items()):
        if value.expires_at <= now:
            print(
                f"registry_remove http={key} zmq={value.zmq_address}", flush=True
            )
            instances.pop(key, None)


def listen_for_registration(poller: zmq.Poller, router: zmq.Socket) -> None:
    while True:
        sockets = dict(poller.poll())
        if router not in sockets:
            continue
        _, message = router.recv_multipart()
        try:
            data = msgpack.loads(message, raw=False)
            instance_type = data.get("type")
            http_address = str(data.get("http_address") or "")
            zmq_address = str(data.get("zmq_address") or "")
            reported_tp_size = int(data.get("tp_size", DEFAULT_TP_SIZE))
            tp_size = configured_tp_size(
                instance_type, http_address, reported_tp_size
            )
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"registry_invalid error={exc} message={message!r}", flush=True)
            continue
        if (
            not http_address
            or not zmq_address
            or instance_type not in {"P", "D"}
            or tp_size <= 0
        ):
            print(f"registry_invalid data={data!r}", flush=True)
            continue

        instances = PREFILL_INSTANCES if instance_type == "P" else DECODE_INSTANCES
        lock = PREFILL_LOCK if instance_type == "P" else DECODE_LOCK
        with lock:
            is_new = http_address not in instances
            instances[http_address] = Instance(
                http_address=http_address,
                zmq_address=zmq_address,
                expires_at=time.time() + PING_TTL_SECONDS,
                tp_size=tp_size,
            )
            remove_expired(instances)
        if is_new:
            role = "prefill" if instance_type == "P" else "decode"
            print(
                f"registry_add role={role} http={http_address} "
                f"zmq={zmq_address} tp_size={tp_size} "
                f"reported_tp_size={reported_tp_size}",
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


def snapshot_registry() -> tuple[list[Instance], list[Instance]]:
    with PREFILL_LOCK:
        remove_expired(PREFILL_INSTANCES)
        prefill = sorted(PREFILL_INSTANCES.values(), key=lambda x: http_sort_key(x.http_address))
    with DECODE_LOCK:
        remove_expired(DECODE_INSTANCES)
        decode = sorted(DECODE_INSTANCES.values(), key=lambda x: http_sort_key(x.http_address))
    return prefill, decode


class CustomPairSchedulingPolicy:
    """Select one pair from the currently registered P and D instances."""

    def __init__(
        self,
        default_route: str = RANDOM_ROUTE,
        *,
        random_seed: Any = None,
    ) -> None:
        self._lock = threading.Lock()
        self._round_robin_index = 0
        self._rng = random.Random(random_seed)
        self._default_route = normalize_route(default_route, allow_policy=True)

    @property
    def default_route(self) -> str:
        with self._lock:
            return self._default_route

    def set_default_route(self, route: str) -> str:
        normalized = normalize_route(route, allow_policy=True)
        with self._lock:
            self._default_route = normalized
            if normalized == ROUND_ROBIN:
                self._round_robin_index = 0
        return normalized

    def choose_route(
        self,
        requested_route: str | None,
        prefill: list[Instance],
        decode: list[Instance],
    ) -> str:
        if requested_route is not None:
            return normalize_route(requested_route)
        with self._lock:
            if self._default_route == RANDOM_ROUTE:
                prefill_index = self._rng.randrange(len(prefill))
                prefill_tp_size = prefill[prefill_index].tp_size
                compatible_decode_indices = [
                    decode_index
                    for decode_index, decode_instance in enumerate(decode)
                    if decode_instance.tp_size == prefill_tp_size
                ]
                if not compatible_decode_indices:
                    raise RuntimeError(
                        f"no Decode instance matches P{prefill_index} "
                        f"TP={prefill_tp_size}"
                    )
                decode_index = self._rng.choice(compatible_decode_indices)
                return f"P{prefill_index}-D{decode_index}"
            if self._default_route != ROUND_ROBIN:
                return self._default_route
            choices = compatible_route_choices(prefill, decode)
            if not choices:
                raise RuntimeError(
                    "no symmetric-TP Prefill-Decode route is available"
                )
            route = choices[self._round_robin_index % len(choices)]
            self._round_robin_index += 1
            return route

    def schedule(
        self,
        prefill: list[Instance],
        decode: list[Instance],
        requested_route: str | None,
    ) -> tuple[str, Instance, Instance]:
        if not prefill or not decode:
            raise RuntimeError(
                "custom pair scheduling requires live Prefill and Decode "
                f"instances; got P={len(prefill)} D={len(decode)}"
            )
        route = self.choose_route(requested_route, prefill, decode)
        match = ROUTE_PATTERN.fullmatch(route)
        assert match is not None
        prefill_index, decode_index = (int(value) for value in match.groups())
        if prefill_index >= len(prefill) or decode_index >= len(decode):
            raise RuntimeError(
                f"route {route} is unavailable with P={len(prefill)} "
                f"D={len(decode)}"
            )
        prefill_instance = prefill[prefill_index]
        decode_instance = decode[decode_index]
        if prefill_instance.tp_size != decode_instance.tp_size:
            raise RuntimeError(
                f"asymmetric TP route {route} is unsupported: "
                f"prefill_tp={prefill_instance.tp_size} "
                f"decode_tp={decode_instance.tp_size}"
            )
        return route, prefill_instance, decode_instance


POLICY = CustomPairSchedulingPolicy(
    os.environ.get("CUSTOM_PD_DEFAULT_ROUTE", RANDOM_ROUTE),
    random_seed=os.environ.get("CUSTOM_PD_RANDOM_SEED") or None,
)
REQUEST_COUNT = 0
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
                raise RuntimeError(
                    f"upstream status={response.status} url={url} body={body}"
                )
            async for chunk in response.content.iter_chunked(1024):
                yield chunk


def route_from_request(request: web.Request, body: dict[str, Any]) -> str | None:
    header_route = request.headers.get("X-PD-Route")
    query_route = request.query.get("pd_route")
    body_route = body.pop("pd_route", None)
    private_body_route = body.pop("_pd_route", None)
    supplied = [
        ("header", header_route),
        ("query", query_route),
        ("body", body_route),
        ("private_body", private_body_route),
    ]
    selected = [(source, value) for source, value in supplied if value is not None]
    if not selected:
        return None
    canonical = {normalize_route(value) for _, value in selected}
    if len(canonical) != 1:
        raise ValueError(f"conflicting route selectors: {selected!r}")
    return canonical.pop()


async def health(_: web.Request) -> web.Response:
    prefill, decode = snapshot_registry()
    return web.json_response(
        {
            "ok": True,
            "ready": len(prefill) >= 2 and len(decode) >= 2,
            "prefill_count": len(prefill),
            "decode_count": len(decode),
        }
    )


async def registry(_: web.Request) -> web.Response:
    prefill, decode = snapshot_registry()
    return web.json_response(
        {
            # Keep the original proxy fields so existing job readiness checks
            # can use this scheduler as a drop-in replacement.
            "prefill": [instance.http_address for instance in prefill],
            "decode": [instance.http_address for instance in decode],
            "prefill_count": len(prefill),
            "decode_count": len(decode),
            "prefill_tp_sizes": [instance.tp_size for instance in prefill],
            "decode_tp_sizes": [instance.tp_size for instance in decode],
            "prefill_aliases": {
                f"P{index}": {
                    "http_address": instance.http_address,
                    "zmq_address": instance.zmq_address,
                    "tp_size": instance.tp_size,
                }
                for index, instance in enumerate(prefill)
            },
            "decode_aliases": {
                f"D{index}": {
                    "http_address": instance.http_address,
                    "zmq_address": instance.zmq_address,
                    "tp_size": instance.tp_size,
                }
                for index, instance in enumerate(decode)
            },
            "supported_routes": compatible_route_choices(prefill, decode),
            "default_route": POLICY.default_route,
        }
    )


def require_admin_token(request: web.Request) -> None:
    expected = os.environ.get("CUSTOM_POLICY_ADMIN_TOKEN")
    if expected and request.headers.get("X-Admin-Token") != expected:
        raise web.HTTPForbidden(text="invalid or missing X-Admin-Token")


async def set_default_route(request: web.Request) -> web.Response:
    require_admin_token(request)
    try:
        body = await request.json()
        route = POLICY.set_default_route(body["route"])
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    print(f"default_route_update route={route}", flush=True)
    return web.json_response({"ok": True, "default_route": route})


async def publish_clock_ack(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        node_group = str(payload["node_group"])
        seq = int(payload["seq"])
        target_mhz = int(payload["target_mhz"])
        rc = int(payload["rc"])
        observed_mhz = str(payload["observed_mhz"])
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "invalid clock acknowledgement"}, status=400)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", node_group) or seq < 1:
        return web.json_response({"error": "invalid clock acknowledgement"}, status=400)
    CLOCK_ACKS[(node_group, seq)] = {
        "node_group": node_group,
        "seq": seq,
        "target_mhz": target_mhz,
        "rc": rc,
        "observed_mhz": observed_mhz,
        "published_at": time.time(),
    }
    return web.json_response({"ok": True})


async def read_clock_ack(request: web.Request) -> web.Response:
    node_group = request.match_info["node_group"]
    try:
        seq = int(request.match_info["seq"])
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid sequence") from exc
    ack = CLOCK_ACKS.get((node_group, seq))
    if ack is None:
        raise web.HTTPNotFound(text="clock acknowledgement not available")
    return web.Response(
        text=(
            f"{ack['seq']} {ack['target_mhz']} {ack['rc']} "
            f"{ack['observed_mhz']}\n"
        ),
        content_type="text/plain",
    )


async def handle_request(request: web.Request) -> web.StreamResponse:
    global REQUEST_COUNT
    try:
        original = await request.json()
        if not isinstance(original, dict):
            return web.json_response(
                {"error": "request body must be a JSON object"}, status=400
            )
        requested_route = route_from_request(request, original)
        prefill, decode = snapshot_registry()
        route, prefill_instance, decode_instance = POLICY.schedule(
            prefill, decode, requested_route
        )
        match = ROUTE_PATTERN.fullmatch(route)
        assert match is not None
        prefill_alias = f"P{match.group(1)}"
        decode_alias = f"D{match.group(2)}"
        index = REQUEST_COUNT
        REQUEST_COUNT += 1
        request_id = (
            f"___prefill_addr_{prefill_instance.zmq_address}"
            f"___decode_addr_{decode_instance.zmq_address}_{uuid.uuid4().hex}"
        )
        print(
            f"route count={index} policy_route={route} request_id={request_id} "
            f"prefill_http={prefill_instance.http_address} "
            f"prefill_tp={prefill_instance.tp_size} "
            f"decode_http={decode_instance.http_address} "
            f"decode_tp={decode_instance.tp_size}",
            flush=True,
        )

        prefill_body = dict(original)
        prefill_kv_params = dict(prefill_body.get("kv_transfer_params") or {})
        prefill_kv_params["remote_tp_size"] = decode_instance.tp_size
        prefill_body["kv_transfer_params"] = prefill_kv_params
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1
        async for _ in forward_request(
            f"http://{prefill_instance.http_address}{request.path}",
            prefill_body,
            request_id,
        ):
            pass

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/json",
                "X-PD-Route": route,
                "X-Prefill-Instance": prefill_alias,
                "X-Decode-Instance": decode_alias,
            },
        )
        await response.prepare(request)
        decode_body = dict(original)
        decode_kv_params = dict(decode_body.get("kv_transfer_params") or {})
        decode_kv_params["remote_tp_size"] = prefill_instance.tp_size
        decode_body["kv_transfer_params"] = decode_kv_params
        async for chunk in forward_request(
            f"http://{decode_instance.http_address}{request.path}",
            decode_body,
            request_id,
        ):
            await response.write(chunk)
        await response.write_eof()
        return response
    except ValueError as exc:
        return web.json_response({"error": str(exc), "type": "route_error"}, status=400)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc), "type": "routing_error"}, status=503)
    except Exception as exc:
        import traceback

        print(f"proxy_error type={type(exc).__name__} error={exc}", flush=True)
        traceback.print_exc()
        return web.json_response(
            {"error": str(exc), "type": type(exc).__name__}, status=502
        )


app.router.add_get("/health", health)
app.router.add_get("/registry", registry)
app.router.add_post("/control/default-route", set_default_route)
app.router.add_post("/control/clock-ack", publish_clock_ack)
app.router.add_get("/control/clock-ack/{node_group}/{seq}", read_clock_ack)
app.router.add_post("/v1/completions", handle_request)
app.router.add_post("/v1/chat/completions", handle_request)


if __name__ == "__main__":
    register_host = os.environ.get("PROXY_REGISTER_HOST", "0.0.0.0")
    register_port = int(os.environ.get("PROXY_REGISTER_PORT", "30001"))
    http_host = os.environ.get("PROXY_HTTP_HOST", "0.0.0.0")
    http_port = int(os.environ.get("PROXY_HTTP_PORT", "10001"))
    print(
        f"custom_proxy_start hostname={socket.gethostname()} "
        f"http={http_host}:{http_port} registry={register_host}:{register_port} "
        f"default_route={POLICY.default_route} routes=dynamic_symmetric_tp_pairs",
        flush=True,
    )
    start_service_discovery(register_host, register_port)
    web.run_app(app, host=http_host, port=http_port, print=None)
