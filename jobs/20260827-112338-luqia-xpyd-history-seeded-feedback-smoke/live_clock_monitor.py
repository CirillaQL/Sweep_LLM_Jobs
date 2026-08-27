#!/usr/bin/env python3
"""Continuously sample actual clocks for the two allocated GPUs on one node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

import pynvml


STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def text_value(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_endpoint(raw: str) -> tuple[str, int]:
    endpoint, separator, gpu_raw = raw.partition(":")
    if not separator or not endpoint or not gpu_raw.isdigit():
        raise argparse.ArgumentTypeError("endpoint mapping must be ENDPOINT:GPU_INDEX")
    return endpoint, int(gpu_raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--endpoint", action="append", type=parse_endpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period-s", type=float, default=0.2)
    args = parser.parse_args()
    if args.period_s <= 0:
        raise ValueError("period must be positive")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    handles = {
        endpoint: pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        for endpoint, gpu_index in args.endpoint
    }
    sequence = 0
    deadline = time.monotonic()
    try:
        with args.output.open("w", encoding="utf-8", buffering=1) as stream:
            while not STOP:
                for endpoint, gpu_index in args.endpoint:
                    sequence += 1
                    started_wall = time.time()
                    started_mono = time.monotonic()
                    row: dict[str, object] = {
                        "sequence": sequence,
                        "timestamp_wall_s": started_wall,
                        "timestamp_monotonic_s": started_mono,
                        "node": args.node,
                        "endpoint_id": endpoint,
                        "gpu_index": gpu_index,
                        "status": "success",
                    }
                    try:
                        handle = handles[endpoint]
                        pci = pynvml.nvmlDeviceGetPciInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        row.update({
                            "gpu_name": text_value(pynvml.nvmlDeviceGetName(handle)),
                            "gpu_uuid": text_value(pynvml.nvmlDeviceGetUUID(handle)),
                            "pci_bus_id": text_value(pci.busId),
                            "graphics_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_GRAPHICS
                            )),
                            "memory_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_MEM
                            )),
                            "gpu_utilization_pct": int(utilization.gpu),
                            "memory_utilization_pct": int(utilization.memory),
                        })
                    except Exception as exc:  # Preserve evidence and fail in post-audit.
                        row.update({
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                deadline += args.period_s
                time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
