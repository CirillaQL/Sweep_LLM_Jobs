#!/usr/bin/env python3
"""Apply a short CUDA load and record the physical GPU's active SM clocks."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch


def sample_gpu(gpu_id, stop, samples):
    query = "clocks.sm,utilization.gpu,power.draw,temperature.gpu"
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            try:
                clock, util, power, temp = [float(value.strip()) for value in result.stdout.split(",")]
                samples.append({
                    "unix_ts": time.time(), "clock_mhz": clock, "util_pct": util,
                    "power_w": power, "temperature_c": temp,
                })
            except ValueError:
                pass
        stop.wait(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smi-index", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.cuda.set_device(0)
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    right = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)

    samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(args.smi_index, stop, samples), daemon=True)
    monitor.start()
    deadline = time.monotonic() + args.seconds
    iterations = 0
    while time.monotonic() < deadline:
        result = left @ right
        left, result = result, left
        torch.cuda.synchronize()
        iterations += 1
    stop.set()
    monitor.join(timeout=2)

    active = [sample for sample in samples if sample["util_pct"] >= 50]
    if not active:
        raise SystemExit(f"no active telemetry samples: {samples}")
    clocks = [sample["clock_mhz"] for sample in active]
    payload = {
        "physical_gpu_id": args.smi_index,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "iterations": iterations,
        "samples": samples,
        "active_samples": len(active),
        "active_clock_min_mhz": min(clocks),
        "active_clock_max_mhz": max(clocks),
        "active_clock_mean_mhz": sum(clocks) / len(clocks),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
