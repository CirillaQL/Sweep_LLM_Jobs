#!/usr/bin/env python3
"""Apply request-level GPU clock commands and record per-GPU telemetry."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUNNING = True


def stop(_signum: int, _frame: Any) -> None:
    global RUNNING
    RUNNING = False


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> Any:
    headers = {}
    token = os.environ.get("CUSTOM_POLICY_ADMIN_TOKEN", "")
    if token:
        headers["X-Admin-Token"] = token
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status == 204:
            return None
        return json.loads(response.read().decode("utf-8"))


def query_gpu(gpu_id: str) -> dict[str, str]:
    fields = (
        "uuid,clocks.current.graphics,power.draw,utilization.gpu,"
        "temperature.gpu,memory.used"
    )
    proc = subprocess.run(
        ["nvidia-smi", "-i", gpu_id, f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    values = [value.strip() for value in proc.stdout.strip().split(",")]
    if proc.returncode != 0 or len(values) != 6:
        return {"uuid": "NA", "clock_mhz": "NA", "power_w": "NA",
                "util_pct": "NA", "temperature_c": "NA", "memory_used_mib": "NA"}
    return dict(zip(
        ("uuid", "clock_mhz", "power_w", "util_pct", "temperature_c", "memory_used_mib"),
        values,
    ))


def main() -> int:
    instance = os.environ["PD_INSTANCE_NAME"]
    node = os.environ.get("PD_NODE_NAME") or os.uname().nodename.split(".")[0]
    proxy = f"http://{os.environ['PROXY_IP']}:{os.environ['PROXY_HTTP_PORT']}"
    gpu_ids = [value.strip() for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    output = Path(os.environ["PD_OUT_DIR"]) / f"gpu_telemetry_{instance}.csv"
    interval = float(os.environ.get("PD_DVFS_TELEMETRY_INTERVAL_SECONDS", "0.5"))
    settle_seconds = float(os.environ.get("PD_DVFS_SETTLE_SECONDS", "0"))
    output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_seq = 0
    target_mhz = 0
    next_sample = 0.0
    fieldnames = [
        "unix_ts", "node", "instance", "gpu_id", "gpu_uuid", "clock_seq",
        "target_freq_mhz", "actual_freq_mhz", "power_w", "util_gpu_pct",
        "temperature_c", "memory_used_mib",
    ]
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
        while RUNNING:
            try:
                command = request_json(
                    f"{proxy}/control/clock-command/{instance}?after_seq={last_seq}"
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                command = None
            if command:
                seq = int(command["seq"])
                target_mhz = int(command["target_mhz"])
                started = time.time()
                errors = []
                for gpu_id in gpu_ids:
                    proc = subprocess.run(
                        ["sudo", "-n", "nvidia-smi", "-i", gpu_id, "-lgc",
                         f"{target_mhz},{target_mhz}"],
                        capture_output=True, text=True, check=False,
                    )
                    if proc.returncode:
                        errors.append(
                            f"gpu={gpu_id} rc={proc.returncode} "
                            f"stderr={proc.stderr.strip()}"
                        )
                lgc_applied = time.time()
                settle_deadline = time.monotonic() + settle_seconds
                while RUNNING and time.monotonic() < settle_deadline:
                    sample_time = time.time()
                    for gpu_id in gpu_ids:
                        probe = query_gpu(gpu_id)
                        writer.writerow({
                            "unix_ts": f"{sample_time:.6f}", "node": node,
                            "instance": instance, "gpu_id": gpu_id,
                            "gpu_uuid": probe["uuid"], "clock_seq": seq,
                            "target_freq_mhz": target_mhz,
                            "actual_freq_mhz": probe["clock_mhz"],
                            "power_w": probe["power_w"],
                            "util_gpu_pct": probe["util_pct"],
                            "temperature_c": probe["temperature_c"],
                            "memory_used_mib": probe["memory_used_mib"],
                        })
                    handle.flush()
                    remaining = settle_deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(interval, remaining))
                probes = {gpu_id: query_gpu(gpu_id) for gpu_id in gpu_ids}
                observed = [probe["clock_mhz"] for probe in probes.values()]
                rc = 0 if not errors else 31
                ack = {
                    "instance": instance, "node": node, "seq": seq,
                    "target_mhz": target_mhz, "rc": rc,
                    "observed_mhz": observed, "gpu_ids": gpu_ids,
                    "apply_ms": round((time.time() - started) * 1000.0, 3),
                    "lgc_apply_ms": round((lgc_applied - started) * 1000.0, 3),
                    "settle_seconds": settle_seconds,
                    "settle_wait_ms": round((time.time() - lgc_applied) * 1000.0, 3),
                    "errors": errors,
                }
                try:
                    request_json(f"{proxy}/control/clock-ack", payload=ack)
                except (urllib.error.URLError, TimeoutError):
                    pass
                last_seq = seq
                next_sample = time.time() + interval
            now = time.time()
            if now >= next_sample:
                for gpu_id in gpu_ids:
                    probe = query_gpu(gpu_id)
                    writer.writerow({
                        "unix_ts": f"{now:.6f}", "node": node,
                        "instance": instance, "gpu_id": gpu_id,
                        "gpu_uuid": probe["uuid"], "clock_seq": last_seq,
                        "target_freq_mhz": target_mhz,
                        "actual_freq_mhz": probe["clock_mhz"],
                        "power_w": probe["power_w"],
                        "util_gpu_pct": probe["util_pct"],
                        "temperature_c": probe["temperature_c"],
                        "memory_used_mib": probe["memory_used_mib"],
                    })
                handle.flush()
                next_sample = now + interval
            time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
