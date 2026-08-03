#!/usr/bin/env python3
"""Continuously record board power for every GPU visible to this Slurm step."""

import argparse
import csv
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


RUNNING = True


def stop(_signum, _frame):
    global RUNNING
    RUNNING = False


def parse_expected_host_gpus(value):
    """Parse HOST=COUNT pairs used by asymmetric multi-node allocations."""
    expected = {}
    if not value:
        return expected
    for item in value.split(","):
        try:
            host, count_text = item.split("=", 1)
            host = host.strip().split(".", 1)[0]
            count = int(count_text)
        except (TypeError, ValueError):
            raise ValueError(f"invalid host GPU count: {item!r}") from None
        if not host or count <= 0 or host in expected:
            raise ValueError(f"invalid host GPU count: {item!r}")
        expected[host] = count
    return expected


def query_gpus(visible_devices, expected_gpus):
    command = ["nvidia-smi"]
    if visible_devices:
        command.extend(["-i", visible_devices])
    command.extend(
        [
            "--query-gpu=index,uuid,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        gpu_index, gpu_uuid, power_w = fields
        rows.append((gpu_index, gpu_uuid, float(power_w)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU power samples")
    if expected_gpus is not None and len(rows) != expected_gpus:
        raise RuntimeError(
            f"expected {expected_gpus} visible GPUs but nvidia-smi returned {len(rows)}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument(
        "--expected-host-gpus",
        help="comma-separated host-specific counts, for example ganymede=8,neptune=4",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        expected_by_host = parse_expected_host_gpus(args.expected_host_gpus)
    except ValueError as exc:
        parser.error(str(exc))

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    host = socket.gethostname().split(".", 1)[0]
    expected_gpus = expected_by_host.get(host, args.expected_gpus)
    if expected_by_host and host not in expected_by_host:
        parser.error(
            f"host {host!r} is missing from --expected-host-gpus={args.expected_host_gpus!r}"
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    print(
        f"power_monitor_start host={host} "
        f"visible_devices={visible_devices or 'all'} expected_gpus={expected_gpus}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"allocation_{host}_power.csv"
    write_header = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8", buffering=1) as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["unix_ts", "host", "gpu_index", "gpu_uuid", "gpu_power_w"])
        while RUNNING:
            started = time.monotonic()
            unix_ts = time.time()
            try:
                for gpu_index, gpu_uuid, power_w in query_gpus(
                    visible_devices, expected_gpus
                ):
                    writer.writerow([f"{unix_ts:.6f}", host, gpu_index, gpu_uuid, power_w])
            except Exception as exc:
                print(f"power_query_failed host={host} error={exc}", flush=True)
            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
