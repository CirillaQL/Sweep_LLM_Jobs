#!/usr/bin/env python3
"""Warm every supported Prefill-Decode route with an exact-token request."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(","))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid TP sizes: {raw!r}")
    return values


def send_request(
    *,
    base_url: str,
    model: str,
    route: str,
    input_tokens: int,
    output_tokens: int,
    token_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            # Token IDs avoid tokenizer ambiguity and warm the requested KV size.
            "prompt": [token_id] * input_tokens,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-PD-Route": route,
            "X-Experiment-Tag": f"warmup-{route}",
            "X-Input-Tokens": str(input_tokens),
        },
    )
    started = time.monotonic()
    first_data_at: float | None = None
    event_count = 0
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                json.loads(payload)
                event_count += 1
                if first_data_at is None:
                    first_data_at = time.monotonic()
    except urllib.error.HTTPError as exc:
        error = f"HTTPError {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:  # noqa: BLE001 - preserve the remote failure details.
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    return {
        "route": route,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": (
            round((first_data_at - started) * 1000.0, 3)
            if first_data_at is not None
            else None
        ),
        "e2e_ms": round((ended - started) * 1000.0, 3),
        "stream_event_count": event_count,
        "success": error is None and first_data_at is not None,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-tp-sizes", required=True)
    parser.add_argument("--decode-tp-sizes", required=True)
    parser.add_argument("--allow-asymmetric-tp", action="store_true")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefill_tp = parse_sizes(args.prefill_tp_sizes)
    decode_tp = parse_sizes(args.decode_tp_sizes)
    if args.input_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("warmup token counts must be positive")

    routes = [
        f"P{p_index}-D{d_index}"
        for p_index, p_size in enumerate(prefill_tp)
        for d_index, d_size in enumerate(decode_tp)
        if args.allow_asymmetric_tp or p_size == d_size
    ]
    requests = []
    for route in routes:
        item = send_request(
            base_url=args.base_url,
            model=args.model,
            route=route,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            token_id=args.token_id,
            timeout_seconds=args.timeout_seconds,
        )
        requests.append(item)
        print(json.dumps({"phase": "route_warmup", **item}, sort_keys=True), flush=True)
        if not item["success"]:
            break

    successful_routes = [item["route"] for item in requests if item["success"]]
    result = {
        "ok": successful_routes == routes,
        "allow_asymmetric_tp": args.allow_asymmetric_tp,
        "prefill_tp_sizes": prefill_tp,
        "decode_tp_sizes": decode_tp,
        "expected_routes": routes,
        "request_count": len(requests),
        "successful_routes": successful_routes,
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
