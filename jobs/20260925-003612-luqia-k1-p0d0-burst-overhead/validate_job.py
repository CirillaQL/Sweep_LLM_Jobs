#!/usr/bin/env python3
"""Static preflight for the K1 P0/D0 burst-overhead job."""

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
    if topology != [("P0", "prefill", "neptune", [0]), ("D0", "decode", "ganymede", [0])]:
        raise ValueError("K1 must use exactly P0 on neptune GPU0 and D0 on ganymede GPU0")
    if config["online_feedback"]["enabled"] is not False:
        raise ValueError("online feedback must stay disabled")
    s = config["burst_overhead"]
    known = {"bursts_reference", "decode_busy_1500", "decode_busy_1050", "bursts_p1305", "bursts_p1815"}
    if s["phase_order"][0] != "bursts_reference" or not set(s["phase_order"]) <= known:
        raise ValueError("unexpected K1 phase order")
    if max(s["burst_sizes"]) > 8 or max(s["background_levels"]) > 16:
        raise ValueError("burst/background sizes exceed the planned K1 range")
    driver = (args.source_root / "paper/scripts/xpyd/pd_burst_overhead.py").read_text(encoding="utf-8")
    for marker in ("XPYD_DRIVER_DEADLINE_EPOCH", "p0d0_end_to_end_smoke", "pd_requests.csv"):
        if marker not in driver:
            raise ValueError("required driver marker missing: %s" % marker)
    run_script = (args.source_root.parent / "run.sbatch").read_text(encoding="utf-8")
    for marker in ("--partition=short", "--time=00:30:00", "check_gpu0", 'HOME="${WORK_DIR}/home"',
                   "xpyd.pd_burst_overhead"):
        if marker not in run_script:
            raise ValueError("run.sbatch marker missing: %s" % marker)
    result = {"valid": True, "endpoint_pair": ["P0", "D0"], "phases": s["phase_order"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
