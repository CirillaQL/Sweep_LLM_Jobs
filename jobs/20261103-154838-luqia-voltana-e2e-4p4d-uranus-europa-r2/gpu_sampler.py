"""NVML sampler for the GPUs of one node: epoch time, GPU index, total energy (mJ), power (mW),
SM clock, throttle reasons.

Energy per stage is the difference of the cumulative counter over the stage's span, so it does not
depend on the sampling interval. Runs until killed; every line is flushed.
"""

import argparse
import time

import pynvml

parser = argparse.ArgumentParser()
parser.add_argument("--indices", required=True, help="NVML (physical) GPU indices, comma separated")
parser.add_argument("--out", required=True)
parser.add_argument("--interval", type=float, default=0.2)
args = parser.parse_args()

pynvml.nvmlInit()
handles = {int(i): pynvml.nvmlDeviceGetHandleByIndex(int(i)) for i in args.indices.split(",")}
with open(args.out, "a", buffering=1) as out:
    out.write("t,gpu,energy_mj,power_mw,sm_mhz,throttle\n")
    while True:
        t = time.time()
        for index, handle in handles.items():
            try:
                out.write(f"{t:.3f},{index},{pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)},"
                          f"{pynvml.nvmlDeviceGetPowerUsage(handle)},"
                          f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)},"
                          f"{pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)}\n")
            except pynvml.NVMLError as error:
                out.write(f"# {t:.3f} {index} {error}\n")
        time.sleep(max(0.0, args.interval - (time.time() - t)))
