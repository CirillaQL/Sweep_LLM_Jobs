#!/usr/bin/env python3
"""Run isolated, tagged NIXL PD requests at several exact token lengths."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r"KV Transfer metrics: Num successful transfers=(?P<count>\d+), "
    r"Avg xfer time \(ms\)=(?P<xfer>[0-9.]+), "
    r"P90 xfer time \(ms\)=(?P<p90_xfer>[0-9.]+), "
    r"Avg post time \(ms\)=(?P<post>[0-9.]+), "
    r"P90 post time \(ms\)=(?P<p90_post>[0-9.]+), "
    r"Avg MB per transfer=(?P<mb>[0-9.]+), "
    r"Throughput \(MB/s\)=(?P<throughput>[0-9.]+), "
    r"Avg number of descriptors=(?P<descriptors>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--proxy-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lengths", default="128,256,512,1024,2048,4096")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--warmup-length", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--metrics-wait-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--token-id", type=int, default=1000)
    return parser.parse_args()


def send_request(
    *,
    url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    token_id: int,
    tag: str,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            # vLLM's Completions API accepts a list of token IDs. This avoids
            # tokenizer ambiguity and makes the transferred KV length exact.
            "prompt": [token_id] * input_tokens,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-PD-Route": "P0-D0",
            "X-Experiment-Tag": tag,
            "X-Input-Tokens": str(input_tokens),
        },
    )
    started_wall = time.time()
    started = time.monotonic()
    first_data_at: float | None = None
    event_count = 0
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    return {
        "tag": tag,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "started_unix_ts": started_wall,
        "ttft_ms": (
            round((first_data_at - started) * 1000.0, 3)
            if first_data_at is not None else None
        ),
        "e2e_ms": round((ended - started) * 1000.0, 3),
        "stream_event_count": event_count,
        "success": error is None and first_data_at is not None,
        "error": error,
    }


def read_log_delta(path: Path, offset: int) -> tuple[str, int]:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
        return data.decode("utf-8", errors="replace"), handle.tell()


def parse_metrics(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        values = match.groupdict()
        records.append(
            {
                "num_successful_transfers": int(values["count"]),
                "avg_xfer_ms": float(values["xfer"]),
                "p90_xfer_ms": float(values["p90_xfer"]),
                "avg_post_ms": float(values["post"]),
                "p90_post_ms": float(values["p90_post"]),
                "avg_mb_per_transfer": float(values["mb"]),
                "throughput_mb_s": float(values["throughput"]),
                "avg_descriptors": float(values["descriptors"]),
                "raw_line": line,
            }
        )
    return records


def weighted(records: list[dict[str, Any]], field: str) -> float | None:
    denominator = sum(item["num_successful_transfers"] for item in records)
    if denominator == 0:
        return None
    return sum(
        item[field] * item["num_successful_transfers"] for item in records
    ) / denominator


def summarize_group(group: dict[str, Any]) -> None:
    requests = [item for item in group["requests"] if item["success"]]
    metrics = group["nixl_metrics"]
    input_tokens = group["input_tokens"]
    # Mistral-7B: 32 layers * 8 KV heads * 128 head dim * K/V * fp16.
    kv_bytes = input_tokens * 32 * 8 * 128 * 2 * 2
    group.update(
        {
            "expected_kv_bytes": kv_bytes,
            "expected_kv_mib": round(kv_bytes / (1024**2), 3),
            "successful_requests": len(requests),
            "failed_requests": len(group["requests"]) - len(requests),
            "mean_client_ttft_ms": (
                round(statistics.mean(item["ttft_ms"] for item in requests), 3)
                if requests else None
            ),
            "mean_client_e2e_ms": (
                round(statistics.mean(item["e2e_ms"] for item in requests), 3)
                if requests else None
            ),
            "nixl_reported_transfers": sum(
                item["num_successful_transfers"] for item in metrics
            ),
            "nixl_avg_xfer_ms": (
                round(value, 3)
                if (value := weighted(metrics, "avg_xfer_ms")) is not None else None
            ),
            "nixl_avg_post_ms": (
                round(value, 3)
                if (value := weighted(metrics, "avg_post_ms")) is not None else None
            ),
            "nixl_avg_mb_per_transfer": (
                round(value, 3)
                if (value := weighted(metrics, "avg_mb_per_transfer")) is not None else None
            ),
            "nixl_reported_throughput_mb_s": (
                round(value, 3)
                if (value := weighted(metrics, "throughput_mb_s")) is not None else None
            ),
        }
    )
    throughput = group["nixl_reported_throughput_mb_s"]
    group["nixl_reported_effective_gbps"] = (
        round(throughput * 8.0 / 1000.0, 4) if throughput is not None else None
    )


def merge_proxy_trace(groups: list[dict[str, Any]], path: Path) -> None:
    traces = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_tag = {item.get("experiment_tag"): item for item in traces}
    for group in groups:
        prefix = f"kvlen-{group['input_tokens']}-rep-"
        matched = [item for tag, item in by_tag.items() if tag and tag.startswith(prefix)]
        actual = [item.get("actual", {}) for item in matched]
        group["proxy_trace_count"] = len(matched)
        for output_name, trace_name in (
            ("mean_proxy_ttft_ms", "proxy_ttft_ms"),
            ("mean_prefill_http_ms", "prefill_http_ms"),
            ("mean_decode_to_first_chunk_ms", "decode_to_first_chunk_ms"),
            ("mean_prefill_to_decode_dispatch_ms", "prefill_to_decode_dispatch_ms"),
        ):
            values = [item[trace_name] for item in actual if item.get(trace_name) is not None]
            group[output_name] = round(statistics.mean(values), 3) if values else None


def main() -> int:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if any(value <= 0 for value in lengths):
        raise ValueError(f"invalid lengths: {lengths}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_log = args.output_dir / "tagged_request_events.jsonl"
    result: dict[str, Any] = {
        "model": args.model,
        "route": "P0-D0",
        "topology": {
            "prefill_gpu": "L40S",
            "prefill_tp": 1,
            "decode_gpu": "L4",
            "decode_tp": 1,
            "connector": "NixlConnector",
        },
        "lengths": lengths,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "output_tokens": args.output_tokens,
        "metrics_wait_seconds": args.metrics_wait_seconds,
        "groups": [],
    }

    for index in range(args.warmups):
        item = send_request(
            url=args.base_url,
            model=args.model,
            input_tokens=args.warmup_length,
            output_tokens=args.output_tokens,
            token_id=args.token_id,
            tag=f"warmup-len{args.warmup_length}-rep{index}",
            timeout=args.timeout_seconds,
        )
        with raw_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(json.dumps({"phase": "warmup", **item}, sort_keys=True), flush=True)
        if not item["success"]:
            return 2

    time.sleep(args.metrics_wait_seconds)
    log_offset = args.decode_log.stat().st_size

    for input_tokens in lengths:
        group: dict[str, Any] = {"input_tokens": input_tokens, "requests": []}
        group_started = time.monotonic()
        for repeat in range(args.repeats):
            item = send_request(
                url=args.base_url,
                model=args.model,
                input_tokens=input_tokens,
                output_tokens=args.output_tokens,
                token_id=args.token_id,
                tag=f"kvlen-{input_tokens}-rep-{repeat}",
                timeout=args.timeout_seconds,
            )
            group["requests"].append(item)
            with raw_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(json.dumps({"phase": "measure", **item}, sort_keys=True), flush=True)
        time.sleep(args.metrics_wait_seconds)
        log_delta, log_offset = read_log_delta(args.decode_log, log_offset)
        delta_path = args.output_dir / f"decode_metrics_len_{input_tokens}.log"
        delta_path.write_text(log_delta, encoding="utf-8")
        group["nixl_metrics"] = parse_metrics(log_delta)
        group["group_wall_seconds"] = round(time.monotonic() - group_started, 3)
        summarize_group(group)
        result["groups"].append(group)
        print(
            json.dumps(
                {
                    "phase": "group_summary",
                    "input_tokens": input_tokens,
                    "expected_kv_mib": group["expected_kv_mib"],
                    "nixl_avg_xfer_ms": group["nixl_avg_xfer_ms"],
                    "nixl_reported_throughput_mb_s": group[
                        "nixl_reported_throughput_mb_s"
                    ],
                    "mean_client_ttft_ms": group["mean_client_ttft_ms"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    merge_proxy_trace(result["groups"], args.proxy_trace)
    result["ok"] = all(
        group["successful_requests"] == args.repeats
        and group["nixl_reported_transfers"] > 0
        and group["proxy_trace_count"] == args.repeats
        for group in result["groups"]
    )
    (args.output_dir / "nixl_length_profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "nixl_length_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "input_tokens",
            "expected_kv_mib",
            "successful_requests",
            "failed_requests",
            "mean_client_ttft_ms",
            "mean_client_e2e_ms",
            "proxy_trace_count",
            "mean_proxy_ttft_ms",
            "mean_prefill_http_ms",
            "mean_decode_to_first_chunk_ms",
            "mean_prefill_to_decode_dispatch_ms",
            "nixl_reported_transfers",
            "nixl_avg_xfer_ms",
            "nixl_avg_post_ms",
            "nixl_avg_mb_per_transfer",
            "nixl_reported_throughput_mb_s",
            "nixl_reported_effective_gbps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["groups"])
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
