#!/usr/bin/env python3
"""Static preflight for the second K4a retry (any allocated GPU) P0(uranus)/D0(ganymede) job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    topology = [(r["endpoint_id"], r["role"], r["node"], r["gpu_ids"]) for r in config["endpoints"]]
    if topology != [("P0", "prefill", "uranus", [0]), ("D0", "decode", "ganymede", [0])]:
        raise ValueError("K4a must use exactly P0 on uranus GPU0 and D0 on ganymede GPU0")
    if config["online_feedback"]["enabled"] is not False:
        raise ValueError("online feedback must stay disabled")
    s = config["burst_overhead"]
    known = {"tier_poisson", "park_idle"}
    if s["phase_order"][0] != "tier_poisson" or not set(s["phase_order"]) <= known:
        raise ValueError("unexpected K4a phase order")
    tp = s["tier_poisson"]
    if sorted(tp["p_mhz"]) != [1305, 1815, 2520] or int(tp["d_mhz"]) != 1050:
        raise ValueError("K4a compares P 1305/1815/2520 at D 1050")
    if float(s["request_timeout_s"]) > 120:
        raise ValueError("PUT_ASYNC needs a short request timeout so a lost KV send cannot hang the job")
    if max(s["burst_sizes"]) > 8 or max(s["background_levels"]) > 16:
        raise ValueError("burst/background sizes exceed the planned K1 range")
    if [list(pair) for pair in s["park_idle"]["pairs"]] != [[900, 1050], [1305, 1050], [1815, 1050]]:
        raise ValueError("park pairs vary only the P clock; D parks at 1050 (K3b)")
    launcher = (args.source_root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
    for marker in ("diagnose_server_start", "VLLM_NO_OUTPUT_TIMEOUT_S", "verify_server_gpu", "expected_uuid_guard",
                   "endpoint_gpu_id", "retrying vLLM start once"):
        if marker not in launcher:
            raise ValueError("launcher marker missing: %s" % marker)
    driver = (args.source_root / "paper/scripts/xpyd/pd_burst_overhead.py").read_text(encoding="utf-8")
    for marker in ("XPYD_DRIVER_DEADLINE_EPOCH", "p0d0_end_to_end_smoke", "pd_requests.csv", "awaiting_first_tokens",
                   "async def tier_poisson", "async def park_idle", "identical for every P clock"):
        if marker not in driver:
            raise ValueError("required driver marker missing: %s" % marker)
    run_script = (args.source_root.parent / "run.sbatch").read_text(encoding="utf-8")
    for marker in ("--partition=short", "--time=00:30:00", "detect_gpu", "XPYD_EXPECTED_GPU_UUID_uranus", "RUNTIME_CONFIG", 'HOME="${WORK_DIR}/home"',
                   "xpyd.pd_burst_overhead", "XPYD_P_SEND_TYPE=PUT_ASYNC", "VLLM_NO_OUTPUT_TIMEOUT_S",
                   "--nodelist=uranus,ganymede"):
        if marker not in run_script:
            raise ValueError("run.sbatch marker missing: %s" % marker)
    result = {"valid": True, "endpoint_pair": ["P0", "D0"], "phases": s["phase_order"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
