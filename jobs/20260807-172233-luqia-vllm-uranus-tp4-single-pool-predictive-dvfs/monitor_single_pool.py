#!/usr/bin/env python3
"""Record per-GPU power, SM clock, and utilization without failing on N/A."""

import argparse
import csv
import signal
import socket
import subprocess
import time
from pathlib import Path


running = True


def stop(_signum, _frame):
    global running
    running = False


def number(text: str):
    cleaned = text.strip().replace("[N/A]", "").replace("N/A", "")
    for suffix in (" W", " MHz", " %"):
        cleaned = cleaned.replace(suffix, "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".", 1)[0]
    fields = [
        "unix_ts", "host", "gpu_index", "gpu_uuid", "power_w",
        "sm_clock_mhz", "gpu_util_pct",
    ]
    with args.output.open("w", newline="", encoding="utf-8", buffering=1) as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        while running:
            started = time.monotonic()
            timestamp = time.time()
            command = [
                "nvidia-smi", "-i", args.gpu_ids,
                "--query-gpu=index,uuid,power.draw,clocks.sm,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            try:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    values = [item.strip() for item in line.split(",")]
                    if len(values) != 5:
                        continue
                    writer.writerow(
                        {
                            "unix_ts": f"{timestamp:.6f}",
                            "host": host,
                            "gpu_index": values[0],
                            "gpu_uuid": values[1],
                            "power_w": number(values[2]),
                            "sm_clock_mhz": number(values[3]),
                            "gpu_util_pct": number(values[4]),
                        }
                    )
            except Exception as exc:
                print(f"telemetry_query_failed host={host} error={exc}", flush=True)
            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
