#!/usr/bin/env python3
"""Can this account change GPU clocks through NVML, without sudo? One allocated GPU, by UUID.

Tries nvmlDeviceSetGpuLockedClocks (what `nvidia-smi -lgc` does) and
nvmlDeviceSetApplicationsClocks. A success code alone proves nothing (application clocks on
the L4 returned success and did not hold), so a successful set is checked by reading the SM
clock back, idle and under a short load, and the reset is checked the same way.
Every successful set is undone in a `finally`, also on SIGTERM. The GPU is left alone if
another process is on it. Writes one JSON file and prints one summary line.

  nvml_probe.py GPU_UUID OUT_JSON
"""

import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time

import pynvml as nv

LOAD_S = 12.0
SETTLE_S = 3.0
TOLERANCE_MHZ = 15
LOAD = """
import sys, time, torch
x = torch.rand(2048, 2048, device="cuda")
torch.mm(x, x); print("ready", flush=True)
end = time.time() + float(sys.argv[1])
while time.time() < end:
    x = torch.mm(x, x); x = x / x.norm()
"""


def text(value):
    return value.decode() if isinstance(value, bytes) else value


def call(fn, *args):
    """Timed NVML call: {"ok", "ms"} plus the value or the NVML error."""
    start = time.perf_counter()
    try:
        value = fn(*args)
    except nv.NVMLError as error:
        return {"ok": False, "ms": _ms(start), "code": error.value, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - e.g. a binding too old for this function
        return {"ok": False, "ms": _ms(start), "error": f"{type(error).__name__}: {error}"}
    return {"ok": True, "ms": _ms(start), "value": value}


def _ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def sm(handle):
    return nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)


def sample(handle, seconds, every=0.5):
    readings = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        readings.append(sm(handle))
        time.sleep(every)
    return readings


def under_load(handle, uuid, seconds=LOAD_S):
    """SM clock while this GPU runs matmul in a child process: None if the load did not start."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid}
    child = subprocess.Popen(
        [sys.executable, "-c", LOAD, str(seconds + 5)], env=env, stdout=subprocess.PIPE, text=True
    )
    try:
        if child.stdout.readline().strip() != "ready":
            return None
        return sample(handle, seconds)
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


def median(readings):
    return statistics.median(readings) if readings else None


def holds(readings, target):
    return readings is not None and readings != [] and abs(median(readings) - target) <= TOLERANCE_MHZ


def _terminate(_signum, _frame):
    raise SystemExit(143)  # runs the `finally` that undoes the clocks


def probe(handle, uuid, name, setter, resetter, target, undo):
    """One way of setting clocks: set, read back, reset, read back."""
    result = {"target_mhz": target, "set": call(setter)}
    if not result["set"]["ok"]:
        return result
    undo[name] = resetter
    time.sleep(SETTLE_S)
    result["idle_sm"] = sample(handle, 2.0)
    result["load_sm"] = under_load(handle, uuid)
    result["holds_under_load"] = holds(result["load_sm"], target)
    result["reset"] = call(resetter)
    if result["reset"]["ok"]:
        undo.pop(name)
    time.sleep(SETTLE_S)
    result["idle_sm_after_reset"] = sample(handle, 2.0)
    result["load_sm_after_reset"] = under_load(handle, uuid, seconds=6.0)
    return result


def main():
    uuid, out = sys.argv[1], sys.argv[2]
    signal.signal(signal.SIGTERM, _terminate)
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(uuid.encode())
    max_sm = nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_SM)
    memory = max(nv.nvmlDeviceGetSupportedMemoryClocks(handle))
    clocks = sorted(nv.nvmlDeviceGetSupportedGraphicsClocks(handle, memory))
    target = min(clocks, key=lambda f: abs(f - 0.5 * max_sm))
    processes = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    record = {
        "host": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "driver": text(nv.nvmlSystemGetDriverVersion()),
        "gpu": text(nv.nvmlDeviceGetName(handle)),
        "index": nv.nvmlDeviceGetIndex(handle),
        "uuid": uuid,
        "max_sm_mhz": max_sm,
        "memory_clock_mhz": memory,
        "sm_before": sm(handle),
        "utilization_before": utilization,
        "processes_before": len(processes),
        "restriction_application_clocks": call(
            nv.nvmlDeviceGetAPIRestriction, handle, nv.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS
        ),
    }
    undo = {}
    try:
        if processes or utilization > 5:
            record["result"] = "skipped_gpu_busy"
            return
        record["locked_clocks"] = probe(
            handle, uuid, "locked_clocks",
            lambda: nv.nvmlDeviceSetGpuLockedClocks(handle, target, target),
            lambda: nv.nvmlDeviceResetGpuLockedClocks(handle), target, undo,
        )
        record["application_clocks"] = probe(
            handle, uuid, "application_clocks",
            lambda: nv.nvmlDeviceSetApplicationsClocks(handle, memory, target),
            lambda: nv.nvmlDeviceResetApplicationsClocks(handle), target, undo,
        )
        record["result"] = "done"
    finally:
        for name, resetter in undo.items():  # a reset that failed, or an interrupted probe
            record[f"final_reset_{name}"] = call(resetter)
        record["sm_end"] = sm(handle)
        with open(out, "w") as handle_out:
            json.dump(record, handle_out, indent=1, default=str)
        print(summary(record), flush=True)
        nv.nvmlShutdown()


def summary(record):
    parts = [f"{record['host']} {record['gpu']} gpu{record['index']} euid={record['euid']}",
             f"result={record.get('result', 'interrupted')}"]
    for name in ("locked_clocks", "application_clocks"):
        probe_result = record.get(name)
        if probe_result is None:
            continue
        done = probe_result["set"]
        if not done["ok"]:
            parts.append(f"{name}: set FAILED ({done.get('error')})")
            continue
        parts.append(
            f"{name}: set ok {done['ms']} ms, target {probe_result['target_mhz']}, "
            f"under load {median(probe_result.get('load_sm'))}, "
            f"holds={probe_result.get('holds_under_load')}, "
            f"reset ok={probe_result.get('reset', {}).get('ok')}, "
            f"after reset under load {median(probe_result.get('load_sm_after_reset'))}"
        )
    return "NVML_PROBE " + " | ".join(parts)


if __name__ == "__main__":
    main()
