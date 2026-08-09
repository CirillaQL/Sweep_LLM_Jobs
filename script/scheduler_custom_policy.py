#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Custom instance-pair scheduler for a multi-prefill/multi-decode PD proxy.

This follows the two-stage forwarding pattern in vLLM's
``tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`` while adding
request-level instance-pair scheduling, including an opt-in asymmetric-TP mode.

With ``CUSTOM_PD_ALLOW_ASYMMETRIC_TP=true``, callers can select every
registered pair. For TP layouts ``P=[1, 1, 2]`` and ``D=[1, 1, 2]`` this is
the full 3x3 Cartesian product::

    P0-D0, P0-D1, P0-D2, P1-D0, P1-D1, P1-D2,
    P2-D0, P2-D1, P2-D2

Selection priority is:

1. ``X-PD-Route`` request header.
2. ``pd_route`` query parameter.
3. ``pd_route`` or ``_pd_route`` JSON field (removed before forwarding).
4. The configured default policy.

In asymmetric mode, Prefill and Decode indices are sampled independently.
``POST /control/default-route`` can switch the default to a fixed pair,
``random``, or ``round_robin``.
Aliases are assigned by increasing HTTP port, making them stable across
requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import socket
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
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
KV_CONNECTOR = os.environ.get("PD_KV_CONNECTOR", "NixlConnector")
ALLOW_ASYMMETRIC_TP = os.environ.get(
    "CUSTOM_PD_ALLOW_ASYMMETRIC_TP", "false"
).strip().lower() in {"1", "true", "yes", "on"}
ENABLE_PREDICTIVE_DVFS = os.environ.get(
    "PD_ENABLE_PREDICTIVE_DVFS", "false"
).strip().lower() in {"1", "true", "yes", "on"}
DVFS_SLO_TTFT_MS = int(os.environ.get("PD_DVFS_SLO_TTFT_MS", "500"))
DVFS_SLO_TPOT_MS = int(os.environ.get("PD_DVFS_SLO_TPOT_MS", "200"))
DVFS_EXPECTED_REQUEST_RATE = float(
    os.environ.get("PD_DVFS_EXPECTED_REQUEST_RATE", "0")
)
DVFS_MIN_REQUEST_RATE = float(os.environ.get("PD_DVFS_MIN_REQUEST_RATE", "0.25"))
DVFS_RATE_WINDOW_SECONDS = float(
    os.environ.get("PD_DVFS_RATE_WINDOW_SECONDS", "10")
)
DVFS_CLOCK_TIMEOUT_SECONDS = float(
    os.environ.get("PD_DVFS_CLOCK_TIMEOUT_SECONDS", "30")
)
DVFS_CLOCK_TOLERANCE_MHZ = int(
    os.environ.get("PD_DVFS_CLOCK_TOLERANCE_MHZ", "30")
)
DVFS_SETTLE_SECONDS = float(os.environ.get("PD_DVFS_SETTLE_SECONDS", "0"))
DVFS_OVERLOAD_ACTION = os.environ.get("PD_DVFS_OVERLOAD_ACTION", "reject")
DVFS_DECISIONS_FILE = Path(
    os.environ.get(
        "PD_DVFS_DECISIONS_FILE",
        str(Path(os.environ.get("PD_OUT_DIR", ".")) / "request_dvfs_decisions.jsonl"),
    )
)
REQUEST_TRACE_RAW = os.environ.get("PD_REQUEST_TRACE_FILE", "")
REQUEST_TRACE_FILE = Path(REQUEST_TRACE_RAW) if REQUEST_TRACE_RAW else None


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
    instance_name: str = "unknown"
    node_name: str = "unknown"
    gpu_type: str = "unknown"


PREFILL_INSTANCES: dict[str, Instance] = {}
DECODE_INSTANCES: dict[str, Instance] = {}
PREFILL_LOCK = threading.Lock()
DECODE_LOCK = threading.Lock()
CLOCK_ACKS: dict[tuple[str, int], dict[str, Any]] = {}
CLOCK_COMMANDS: dict[str, dict[str, Any]] = {}
CLOCK_TARGETS: dict[str, int] = {}
CLOCK_SEQUENCE = 0
CLOCK_SEQUENCE_LOCK = asyncio.Lock()
INSTANCE_DVFS_LOCKS: dict[str, asyncio.Lock] = {}
DECISION_LOG_LOCK = asyncio.Lock()
ARRIVAL_TIMES: deque[float] = deque()
IN_FLIGHT = 0
INSTANCE_IN_FLIGHT: dict[str, int] = {}

DVFS_PREDICTOR = None
if ENABLE_PREDICTIVE_DVFS:
    from request_dvfs_predictor import RequestDVFSPredictor

    bandwidth_raw = os.environ.get("PD_DVFS_KV_EFFECTIVE_BANDWIDTH_GBPS", "")
    DVFS_PREDICTOR = RequestDVFSPredictor(
        os.environ["PD_DVFS_SCHEDULER_SCRIPT"],
        os.environ["PD_DVFS_MODEL_BUNDLE"],
        os.environ["PD_DVFS_SATURATION_BUNDLE"],
        saturation_threshold=(
            float(os.environ["PD_DVFS_SATURATION_THRESHOLD"])
            if os.environ.get("PD_DVFS_SATURATION_THRESHOLD") else None
        ),
        kv_effective_bandwidth_gbps=(
            float(bandwidth_raw) if bandwidth_raw else None
        ),
        dispatch_ms=float(os.environ.get("PD_DVFS_DISPATCH_MS", "0")),
    )


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


def supported_route_choices(
    prefill: list[Instance], decode: list[Instance]
) -> tuple[str, ...]:
    """Return enabled P-D pairs in stable Prefill-major order."""
    return tuple(
        f"P{prefill_index}-D{decode_index}"
        for prefill_index, prefill_instance in enumerate(prefill)
        for decode_index, decode_instance in enumerate(decode)
        if ALLOW_ASYMMETRIC_TP
        or prefill_instance.tp_size == decode_instance.tp_size
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
            existing = instances.get(http_address)
            instances[http_address] = Instance(
                http_address=http_address,
                zmq_address=zmq_address,
                expires_at=time.time() + PING_TTL_SECONDS,
                tp_size=tp_size,
                instance_name=(
                    existing.instance_name if existing is not None else "unknown"
                ),
                node_name=existing.node_name if existing is not None else "unknown",
                gpu_type=existing.gpu_type if existing is not None else "unknown",
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
                    if ALLOW_ASYMMETRIC_TP
                    or decode_instance.tp_size == prefill_tp_size
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
            choices = supported_route_choices(prefill, decode)
            if not choices:
                raise RuntimeError("no Prefill-Decode route is available")
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
        if (
            not ALLOW_ASYMMETRIC_TP
            and prefill_instance.tp_size != decode_instance.tp_size
        ):
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


async def post_json(
    url: str, data: dict[str, Any], request_id: str
) -> dict[str, Any]:
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
            payload = await response.json(content_type=None)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"upstream returned non-object JSON url={url}"
                )
            return payload


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
                    "kv_address": instance.zmq_address,
                    "tp_size": instance.tp_size,
                    "instance_name": instance.instance_name,
                    "node_name": instance.node_name,
                    "gpu_type": instance.gpu_type,
                }
                for index, instance in enumerate(prefill)
            },
            "decode_aliases": {
                f"D{index}": {
                    "http_address": instance.http_address,
                    "zmq_address": instance.zmq_address,
                    "kv_address": instance.zmq_address,
                    "tp_size": instance.tp_size,
                    "instance_name": instance.instance_name,
                    "node_name": instance.node_name,
                    "gpu_type": instance.gpu_type,
                }
                for index, instance in enumerate(decode)
            },
            "supported_routes": supported_route_choices(prefill, decode),
            "allow_asymmetric_tp": ALLOW_ASYMMETRIC_TP,
            "default_route": POLICY.default_route,
            "kv_connector": KV_CONNECTOR,
            "predictive_dvfs": ENABLE_PREDICTIVE_DVFS,
        }
    )


def require_admin_token(request: web.Request) -> None:
    expected = os.environ.get("CUSTOM_POLICY_ADMIN_TOKEN")
    if expected and request.headers.get("X-Admin-Token") != expected:
        raise web.HTTPForbidden(text="invalid or missing X-Admin-Token")


async def register_instance(request: web.Request) -> web.Response:
    require_admin_token(request)
    try:
        payload = await request.json()
        instance_type = str(payload["type"]).upper()
        http_address = str(payload["http_address"])
        kv_address = str(payload["kv_address"])
        reported_tp_size = int(payload["tp_size"])
        instance_name = str(payload["instance_name"])
        node_name = str(payload["node_name"])
        gpu_type = str(payload["gpu_type"]).lower()
        tp_size = configured_tp_size(
            instance_type, http_address, reported_tp_size
        )
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if (
        instance_type not in {"P", "D"}
        or not http_address
        or not kv_address
        or tp_size <= 0
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", instance_name)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", node_name)
        or gpu_type not in {"l40s", "l4"}
    ):
        return web.json_response({"error": "invalid instance"}, status=400)
    instances = PREFILL_INSTANCES if instance_type == "P" else DECODE_INSTANCES
    lock = PREFILL_LOCK if instance_type == "P" else DECODE_LOCK
    with lock:
        instances[http_address] = Instance(
            http_address=http_address,
            zmq_address=kv_address,
            expires_at=float("inf"),
            tp_size=tp_size,
            instance_name=instance_name,
            node_name=node_name,
            gpu_type=gpu_type,
        )
    role = "prefill" if instance_type == "P" else "decode"
    print(
        f"registry_add source=http role={role} http={http_address} "
        f"kv={kv_address} tp_size={tp_size} "
        f"reported_tp_size={reported_tp_size} instance={instance_name} "
        f"node={node_name} gpu_type={gpu_type}",
        flush=True,
    )
    return web.json_response({"ok": True})


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
    require_admin_token(request)
    try:
        payload = await request.json()
        instance = str(payload.get("instance") or payload["node_group"])
        seq = int(payload["seq"])
        target_mhz = int(payload["target_mhz"])
        rc = int(payload["rc"])
        observed_mhz = payload["observed_mhz"]
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "invalid clock acknowledgement"}, status=400)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", instance) or seq < 1:
        return web.json_response({"error": "invalid clock acknowledgement"}, status=400)
    CLOCK_ACKS[(instance, seq)] = {
        **payload,
        "instance": instance,
        "seq": seq,
        "target_mhz": target_mhz,
        "rc": rc,
        "observed_mhz": observed_mhz,
        "published_at": time.time(),
    }
    return web.json_response({"ok": True})


async def read_clock_ack(request: web.Request) -> web.Response:
    instance = request.match_info["node_group"]
    try:
        seq = int(request.match_info["seq"])
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid sequence") from exc
    ack = CLOCK_ACKS.get((instance, seq))
    if ack is None:
        raise web.HTTPNotFound(text="clock acknowledgement not available")
    return web.Response(
        text=(
            f"{ack['seq']} {ack['target_mhz']} {ack['rc']} "
            f"{ack['observed_mhz']}\n"
        ),
        content_type="text/plain",
    )


async def read_clock_command(request: web.Request) -> web.Response:
    require_admin_token(request)
    instance = request.match_info["instance"]
    try:
        after_seq = int(request.query.get("after_seq", "0"))
    except ValueError as exc:
        raise web.HTTPBadRequest(text="invalid after_seq") from exc
    command = CLOCK_COMMANDS.get(instance)
    if command is None or int(command["seq"]) <= after_seq:
        return web.Response(status=204)
    return web.json_response(command)


def request_shape(request: web.Request, body: dict[str, Any]) -> tuple[int, int, str]:
    explicit = request.headers.get("X-Input-Tokens")
    private = body.pop("_pd_input_tokens", None)
    configured = os.environ.get("PD_DVFS_INPUT_TOKENS_OVERRIDE")
    if explicit is not None:
        il, source = int(explicit), "header"
    elif private is not None:
        il, source = int(private), "body"
    elif configured:
        il, source = int(configured), "environment"
    else:
        prompt = body.get("prompt")
        if prompt is None:
            prompt = json.dumps(body.get("messages", []), ensure_ascii=False)
        if isinstance(prompt, list):
            prompt = json.dumps(prompt, ensure_ascii=False)
        # This fallback is deliberately recorded. Production callers should
        # provide X-Input-Tokens when exact tokenizer accounting is required.
        il = max(1, (len(str(prompt).encode("utf-8")) + 3) // 4)
        source = "utf8_bytes_div4_estimate"
    ol = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1)
    if il <= 0 or ol <= 0:
        raise ValueError(f"invalid request shape il={il} ol={ol}")
    return il, ol, source


def request_slos(request: web.Request, body: dict[str, Any]) -> tuple[int, int]:
    ttft = request.headers.get("X-SLO-TTFT-MS", body.pop("_slo_ttft_ms", None))
    tpot = request.headers.get("X-SLO-TPOT-MS", body.pop("_slo_tpot_ms", None))
    return int(ttft or DVFS_SLO_TTFT_MS), int(tpot or DVFS_SLO_TPOT_MS)


def observed_rate(instance_count: int) -> tuple[float, float]:
    now = time.monotonic()
    ARRIVAL_TIMES.append(now)
    cutoff = now - DVFS_RATE_WINDOW_SECONDS
    while ARRIVAL_TIMES and ARRIVAL_TIMES[0] < cutoff:
        ARRIVAL_TIMES.popleft()
    live_total = len(ARRIVAL_TIMES) / DVFS_RATE_WINDOW_SECONDS
    total = max(DVFS_EXPECTED_REQUEST_RATE, live_total)
    per_instance = max(DVFS_MIN_REQUEST_RATE, total / max(1, instance_count))
    return round(total, 6), round(per_instance, 6)


@asynccontextmanager
async def hold_instance_clocks(*instances: Instance):
    locks = [
        INSTANCE_DVFS_LOCKS.setdefault(name, asyncio.Lock())
        for name in sorted({item.instance_name for item in instances})
    ]
    started = time.monotonic()
    for lock in locks:
        await lock.acquire()
    try:
        yield round((time.monotonic() - started) * 1000.0, 3)
    finally:
        for lock in reversed(locks):
            lock.release()


async def apply_clock(instance: Instance, target_mhz: int, request_id: str) -> dict[str, Any]:
    global CLOCK_SEQUENCE
    async with CLOCK_SEQUENCE_LOCK:
        CLOCK_SEQUENCE += 1
        seq = CLOCK_SEQUENCE
    command = {
        "instance": instance.instance_name,
        "seq": seq,
        "target_mhz": target_mhz,
        "request_id": request_id,
        "issued_at": time.time(),
    }
    CLOCK_COMMANDS[instance.instance_name] = command
    deadline = time.monotonic() + DVFS_CLOCK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ack = CLOCK_ACKS.get((instance.instance_name, seq))
        if ack is not None:
            if int(ack["rc"]) != 0 or int(ack["target_mhz"]) != target_mhz:
                raise RuntimeError(f"clock apply failed: {ack}")
            observed_raw = ack.get("observed_mhz", [])
            observed = observed_raw if isinstance(observed_raw, list) else [observed_raw]
            try:
                mismatched = [
                    value for value in observed
                    if abs(float(value) - target_mhz) > DVFS_CLOCK_TOLERANCE_MHZ
                ]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid clock acknowledgement: {ack}") from exc
            if mismatched:
                raise RuntimeError(
                    f"clock verification mismatch instance={instance.instance_name} "
                    f"target={target_mhz} observed={observed}"
                )
            CLOCK_TARGETS[instance.instance_name] = target_mhz
            return ack
        await asyncio.sleep(0.05)
    raise RuntimeError(
        f"clock acknowledgement timeout instance={instance.instance_name} "
        f"target={target_mhz}"
    )


async def append_decision(record: dict[str, Any]) -> None:
    targets: list[Path] = []
    if ENABLE_PREDICTIVE_DVFS:
        targets.append(DVFS_DECISIONS_FILE)
    if REQUEST_TRACE_FILE is not None and REQUEST_TRACE_FILE not in targets:
        targets.append(REQUEST_TRACE_FILE)
    if not targets:
        return
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    async with DECISION_LOG_LOCK:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


async def execute_scheduled_request(
    request: web.Request,
    original: dict[str, Any],
    *,
    index: int,
    route: str,
    prefill_alias: str,
    decode_alias: str,
    prefill_instance: Instance,
    decode_instance: Instance,
    prefill_count: int,
    decode_count: int,
    request_id: str,
    queue_snapshot: dict[str, Any],
) -> web.StreamResponse:
    started_wall = time.time()
    started = time.monotonic()
    first_decode_chunk_at: float | None = None
    forwarding_started: float | None = None
    forwarding_detail: dict[str, Any] = {}
    experiment_tag = request.headers.get("X-Experiment-Tag")
    traced_input_tokens = request.headers.get("X-Input-Tokens")
    record: dict[str, Any] = {
        "request_index": index,
        "request_id": request_id,
        "arrival_unix_ts": started_wall,
        "route": route,
        "prefill_alias": prefill_alias,
        "decode_alias": decode_alias,
        "prefill_instance": prefill_instance.instance_name,
        "decode_instance": decode_instance.instance_name,
        "prefill_tp": prefill_instance.tp_size,
        "decode_tp": decode_instance.tp_size,
        "prefill_gpu": prefill_instance.gpu_type,
        "decode_gpu": decode_instance.gpu_type,
        "queue_snapshot": queue_snapshot,
        "experiment_tag": experiment_tag,
        "traced_input_tokens": (
            int(traced_input_tokens) if traced_input_tokens is not None else None
        ),
        "output_tokens_requested": int(
            original.get("max_tokens") or original.get("max_completion_tokens") or 1
        ),
    }
    success = False
    error: str | None = None
    lock_wait_ms = 0.0
    try:
        if ENABLE_PREDICTIVE_DVFS:
            il, ol, token_source = request_shape(request, original)
            slo_ttft, slo_tpot = request_slos(request, original)
            total_rate, prefill_rate = observed_rate(prefill_count)
            decode_rate = max(DVFS_MIN_REQUEST_RATE, total_rate / max(1, decode_count))
            record.update({
                "input_tokens": il,
                "output_tokens_requested": ol,
                "input_token_source": token_source,
                "observed_or_configured_total_rate": total_rate,
                "prefill_rate_per_instance": prefill_rate,
                "decode_rate_per_instance": round(decode_rate, 6),
                "slo_ttft_ms": slo_ttft,
                "slo_tpot_ms": slo_tpot,
            })
            async with hold_instance_clocks(
                prefill_instance, decode_instance
            ) as lock_wait_ms:
                assert DVFS_PREDICTOR is not None
                decision = await asyncio.to_thread(
                    DVFS_PREDICTOR.recommend,
                    il=il,
                    ol=ol,
                    prefill_rate=prefill_rate,
                    decode_rate=decode_rate,
                    slo_ttft_ms=slo_ttft,
                    slo_tpot_ms=slo_tpot,
                    prefill_gpu=prefill_instance.gpu_type,
                    decode_gpu=decode_instance.gpu_type,
                    prefill_tp=prefill_instance.tp_size,
                    decode_tp=decode_instance.tp_size,
                    overload_action=DVFS_OVERLOAD_ACTION,
                )
                record["prediction"] = decision
                record["clock_lock_wait_ms"] = lock_wait_ms
                if decision["status"] == "NO_SAFE_CONFIG":
                    raise RuntimeError(
                        f"no predicted SLO-safe clock pair for route {route}"
                    )
                selected = decision["recommended"]
                p_freq = int(selected["prefill"]["freq_mhz"])
                d_freq = int(selected["decode"]["freq_mhz"])
                clock_started = time.monotonic()
                p_ack, d_ack = await asyncio.gather(
                    apply_clock(prefill_instance, p_freq, request_id),
                    apply_clock(decode_instance, d_freq, request_id),
                )
                record["clock"] = {
                    "prefill_target_mhz": p_freq,
                    "decode_target_mhz": d_freq,
                    "prefill_ack": p_ack,
                    "decode_ack": d_ack,
                    "apply_total_ms": round(
                        (time.monotonic() - clock_started) * 1000.0, 3
                    ),
                    "settle_seconds_excluded_from_ttft": DVFS_SETTLE_SECONDS,
                }
                print(
                    f"dvfs count={index} route={route} status={decision['status']} "
                    f"prefill_mhz={p_freq} decode_mhz={d_freq} "
                    f"predicted_ttft_ms={selected['prefill']['p99_ttft_ms']} "
                    f"predicted_tpot_ms={selected['decode']['p99_tpot_ms']} "
                    f"predicted_power_w={selected['predicted_cluster_power_w']} "
                    f"clock_lock_wait_ms={lock_wait_ms} "
                    f"settle_seconds={DVFS_SETTLE_SECONDS}",
                    flush=True,
                )
                forwarding_started = time.monotonic()
                response, first_decode_chunk_at, forwarding_detail = await run_pd_forwarding(
                    request, original, route, prefill_alias, decode_alias,
                    prefill_instance, decode_instance, request_id,
                    p_freq=p_freq, d_freq=d_freq,
                )
        else:
            forwarding_started = time.monotonic()
            response, first_decode_chunk_at, forwarding_detail = await run_pd_forwarding(
                request, original, route, prefill_alias, decode_alias,
                prefill_instance, decode_instance, request_id,
            )
        success = True
        return response
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ended = time.monotonic()
        raw_e2e_ms = (ended - started) * 1000.0
        raw_ttft_ms = (
            (first_decode_chunk_at - started) * 1000.0
            if first_decode_chunk_at is not None else None
        )
        excluded_settle_ms = (
            DVFS_SETTLE_SECONDS * 1000.0 if ENABLE_PREDICTIVE_DVFS else 0.0
        )
        e2e_ms = max(0.0, raw_e2e_ms - excluded_settle_ms)
        ttft_ms = (
            max(0.0, raw_ttft_ms - excluded_settle_ms)
            if raw_ttft_ms is not None else None
        )
        forwarding_ttft_ms = (
            (first_decode_chunk_at - forwarding_started) * 1000.0
            if first_decode_chunk_at is not None and forwarding_started is not None
            else None
        )
        ol = int(record.get("output_tokens_requested", 1))
        proxy_tpot_ms = (
            max(0.0, raw_e2e_ms - raw_ttft_ms) / max(1, ol - 1)
            if raw_ttft_ms is not None and ol > 1 else None
        )
        record["actual"] = {
            "success": success,
            "error": error,
            "clock_lock_wait_ms": lock_wait_ms,
            "proxy_ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
            "proxy_ttft_raw_ms": (
                round(raw_ttft_ms, 3) if raw_ttft_ms is not None else None
            ),
            "proxy_ttft_excluded_settle_ms": round(excluded_settle_ms, 3),
            "proxy_forwarding_ttft_ms": (
                round(forwarding_ttft_ms, 3)
                if forwarding_ttft_ms is not None else None
            ),
            "proxy_tpot_estimate_ms": (
                round(proxy_tpot_ms, 3) if proxy_tpot_ms is not None else None
            ),
            "proxy_e2e_ms": round(e2e_ms, 3),
            "proxy_e2e_raw_ms": round(raw_e2e_ms, 3),
            "ttft_slo_met": (
                ttft_ms <= float(record["slo_ttft_ms"])
                if ttft_ms is not None and "slo_ttft_ms" in record else None
            ),
            "tpot_slo_met": (
                proxy_tpot_ms <= float(record["slo_tpot_ms"])
                if proxy_tpot_ms is not None and "slo_tpot_ms" in record else None
            ),
            "finished_unix_ts": time.time(),
            **forwarding_detail,
        }
        await append_decision(record)


async def run_pd_forwarding(
    request: web.Request,
    original: dict[str, Any],
    route: str,
    prefill_alias: str,
    decode_alias: str,
    prefill_instance: Instance,
    decode_instance: Instance,
    request_id: str,
    *,
    p_freq: int | None = None,
    d_freq: int | None = None,
) -> tuple[web.StreamResponse, float | None, dict[str, Any]]:
    prefill_started = time.monotonic()
    prefill_body = dict(original)
    prefill_body["max_tokens"] = 1
    if "max_completion_tokens" in prefill_body:
        prefill_body["max_completion_tokens"] = 1
    if KV_CONNECTOR == "NixlConnector":
        prefill_body["stream"] = False
        prefill_body.pop("stream_options", None)
        prefill_body.pop("min_tokens", None)
        prefill_body.pop("min_completion_tokens", None)
        prefill_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_result = await post_json(
            f"http://{prefill_instance.http_address}{request.path}",
            prefill_body,
            request_id,
        )
        kv_transfer_params = prefill_result.get("kv_transfer_params")
        if not isinstance(kv_transfer_params, dict) or not kv_transfer_params:
            raise RuntimeError(
                f"Prefill {prefill_alias} returned no kv_transfer_params"
            )
    else:
        prefill_kv_params = dict(prefill_body.get("kv_transfer_params") or {})
        prefill_kv_params["remote_tp_size"] = decode_instance.tp_size
        prefill_body["kv_transfer_params"] = prefill_kv_params
        async for _ in forward_request(
            f"http://{prefill_instance.http_address}{request.path}",
            prefill_body,
            request_id,
        ):
            pass
    prefill_finished = time.monotonic()

    headers = {
        "Content-Type": "application/json",
        "X-PD-Route": route,
        "X-Prefill-Instance": prefill_alias,
        "X-Decode-Instance": decode_alias,
    }
    if p_freq is not None and d_freq is not None:
        headers["X-Prefill-Clock-MHz"] = str(p_freq)
        headers["X-Decode-Clock-MHz"] = str(d_freq)
    response = web.StreamResponse(status=200, headers=headers)
    await response.prepare(request)
    decode_body = dict(original)
    if KV_CONNECTOR == "NixlConnector":
        decode_body["kv_transfer_params"] = kv_transfer_params
    else:
        decode_kv_params = dict(decode_body.get("kv_transfer_params") or {})
        decode_kv_params["remote_tp_size"] = prefill_instance.tp_size
        decode_body["kv_transfer_params"] = decode_kv_params
    first_chunk_at = None
    decode_started = time.monotonic()
    async for chunk in forward_request(
        f"http://{decode_instance.http_address}{request.path}",
        decode_body,
        request_id,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic()
        await response.write(chunk)
    await response.write_eof()
    detail = {
        "prefill_http_ms": round((prefill_finished - prefill_started) * 1000.0, 3),
        "decode_to_first_chunk_ms": (
            round((first_chunk_at - decode_started) * 1000.0, 3)
            if first_chunk_at is not None else None
        ),
        "prefill_to_decode_dispatch_ms": round(
            (decode_started - prefill_finished) * 1000.0, 3
        ),
    }
    return response, first_chunk_at, detail


async def handle_request(request: web.Request) -> web.StreamResponse:
    global REQUEST_COUNT, IN_FLIGHT
    prefill_instance = decode_instance = None
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
        prefill_alias, decode_alias = f"P{match.group(1)}", f"D{match.group(2)}"
        index = REQUEST_COUNT
        REQUEST_COUNT += 1
        request_id = uuid.uuid4().hex if KV_CONNECTOR == "NixlConnector" else (
            f"___prefill_addr_{prefill_instance.zmq_address}"
            f"___decode_addr_{decode_instance.zmq_address}_{uuid.uuid4().hex}"
        )
        IN_FLIGHT += 1
        for instance in (prefill_instance, decode_instance):
            INSTANCE_IN_FLIGHT[instance.instance_name] = (
                INSTANCE_IN_FLIGHT.get(instance.instance_name, 0) + 1
            )
        queue_snapshot = {
            "proxy_in_flight": IN_FLIGHT,
            "prefill_instance_in_flight": INSTANCE_IN_FLIGHT[prefill_instance.instance_name],
            "decode_instance_in_flight": INSTANCE_IN_FLIGHT[decode_instance.instance_name],
        }
        print(
            f"route count={index} policy_route={route} request_id={request_id} "
            f"prefill_instance={prefill_instance.instance_name} "
            f"prefill_http={prefill_instance.http_address} "
            f"prefill_tp={prefill_instance.tp_size} "
            f"decode_instance={decode_instance.instance_name} "
            f"decode_http={decode_instance.http_address} "
            f"decode_tp={decode_instance.tp_size} predictive_dvfs={ENABLE_PREDICTIVE_DVFS}",
            flush=True,
        )
        return await execute_scheduled_request(
            request, original, index=index, route=route,
            prefill_alias=prefill_alias, decode_alias=decode_alias,
            prefill_instance=prefill_instance, decode_instance=decode_instance,
            prefill_count=len(prefill), decode_count=len(decode),
            request_id=request_id, queue_snapshot=queue_snapshot,
        )
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
    finally:
        if prefill_instance is not None and decode_instance is not None:
            IN_FLIGHT = max(0, IN_FLIGHT - 1)
            for instance in (prefill_instance, decode_instance):
                INSTANCE_IN_FLIGHT[instance.instance_name] = max(
                    0, INSTANCE_IN_FLIGHT.get(instance.instance_name, 1) - 1
                )


app.router.add_get("/health", health)
app.router.add_get("/registry", registry)
app.router.add_post("/control/register-instance", register_instance)
app.router.add_post("/control/default-route", set_default_route)
app.router.add_post("/control/clock-ack", publish_clock_ack)
app.router.add_get("/control/clock-ack/{node_group}/{seq}", read_clock_ack)
app.router.add_get("/control/clock-command/{instance}", read_clock_command)
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
        f"kv_connector={KV_CONNECTOR} "
        f"default_route={POLICY.default_route} "
        f"allow_asymmetric_tp={str(ALLOW_ASYMMETRIC_TP).lower()} "
        f"predictive_dvfs={str(ENABLE_PREDICTIVE_DVFS).lower()} "
        f"dvfs_slo_ttft_ms={DVFS_SLO_TTFT_MS} "
        f"dvfs_slo_tpot_ms={DVFS_SLO_TPOT_MS} "
        f"dvfs_settle_seconds={DVFS_SETTLE_SECONDS} "
        f"routes={'all_pairs' if ALLOW_ASYMMETRIC_TP else 'symmetric_tp_pairs'}",
        flush=True,
    )
    if KV_CONNECTOR == "P2pNcclConnector":
        start_service_discovery(register_host, register_port)
    web.run_app(app, host=http_host, port=http_port, print=None)
