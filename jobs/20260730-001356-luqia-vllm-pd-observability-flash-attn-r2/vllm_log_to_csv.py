#!/usr/bin/env python3
"""Extract request, scheduler, engine, iteration, and KV events from vLLM logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


FIELDS = [
    "role",
    "source_file",
    "line_number",
    "log_timestamp",
    "category",
    "request_id",
    "message",
]

TIMESTAMP_RE = re.compile(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
REQUEST_ID_RE = re.compile(
    r"(?:request[_ -]?id|request_id=)[=: ]+([A-Za-z0-9_.:/-]+)",
    re.IGNORECASE,
)


def category(line: str) -> str | None:
    lowered = line.lower()
    if "iteration" in lowered:
        return "iteration"
    if "request_id" in lowered or "request id" in lowered:
        return "request"
    if "kv cache" in lowered or "kv_cache" in lowered:
        return "kv_cache"
    if "scheduler" in lowered or "running:" in lowered or "waiting:" in lowered:
        return "scheduler"
    if "engine" in lowered and (
        "throughput" in lowered or "cache" in lowered or "config" in lowered
    ):
        return "engine"
    if "preempt" in lowered or "recompute" in lowered:
        return "preemption"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="ROLE=PATH; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.log:
        role, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                message = raw_line.rstrip("\n")
                event_category = category(message)
                if event_category is None:
                    continue
                timestamp_match = TIMESTAMP_RE.search(message)
                request_match = REQUEST_ID_RE.search(message)
                rows.append(
                    {
                        "role": role,
                        "source_file": str(path),
                        "line_number": line_number,
                        "log_timestamp": (
                            timestamp_match.group(1) if timestamp_match else ""
                        ),
                        "category": event_category,
                        "request_id": (
                            request_match.group(1) if request_match else ""
                        ),
                        "message": message,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"vllm_log_events={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
