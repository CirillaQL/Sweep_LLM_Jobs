#!/usr/bin/env python3
"""Static preflight for the P0/D0 exhaustive joint-grid validation job."""

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
    endpoints = config["endpoints"]
    topology = [(row["endpoint_id"], row["role"], row["node"], row["gpu_ids"]) for row in endpoints]
    expected_topology = [
        ("P0", "prefill", "neptune", [0]),
        ("D0", "decode", "ganymede", [0]),
    ]
    if topology != expected_topology:
        raise ValueError("job must request exactly P0 on neptune and D0 on ganymede")
    settings = config["joint_grid_validation"]
    if settings["design"] != "full_17x15_joint_grid_against_coordinate_canary_strategy":
        raise ValueError("job must compare the full joint grid with the prior Canary strategy")
    if settings["samples_per_pair"] != 3 or config["online_feedback"]["enabled"] is not False:
        raise ValueError("joint grid must make exactly three direct P0/D0 measurements per pair")
    grids = config["online_feedback"]["frequency_grids"]
    if grids["prefill"]["levels"] != 17 or grids["decode"]["levels"] != 15:
        raise ValueError("frequency grid must be 17 by 15")
    workload_ids = [row["id"] for row in config["workloads"]]
    if workload_ids != ["small_light", "prefill_medium", "decode_medium", "decode_heavy", "balanced_medium"]:
        raise ValueError("job must use the five successfully learned low-load workload classes")
    if config["compatible_pairs"] != [{
        "prefill_endpoint_id": "P0", "decode_endpoint_id": "D0",
        "connector": "P2pNcclConnector", "prefill_tp": 1, "decode_tp": 1,
        "supported": True, "reason": "sole joint-grid experiment pair",
    }]:
        raise ValueError("P0/D0 must be the sole compatible pair")
    required = {
        "paper/scripts/xpyd/joint_grid_validation.py": (
            "canary_coordinate_reconstruction",
            "joint_grid_request_measurements.csv",
            "p0d0_end_to_end_smoke",
            "proxy_diagnostics.jsonl",
        ),
        "paper/scripts/xpyd/online_feedback_controller.py": (
            "XPYD_ENERGY_GPU_COUNT",
            "XPYD_ENERGY_CUDA_VISIBLE_DEVICES",
        ),
    }
    for relative, markers in required.items():
        text = (args.source_root / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise ValueError("required marker missing: %s in %s" % (marker, relative))
    launcher = (args.source_root / "run_disagg_benchmark.sh").read_text(encoding="utf-8")
    if 'XPYD_ENDPOINTS_PER_ROLE}" =~ ^[1-4]$' not in launcher:
        raise ValueError("P0/D0-only retry requires the launcher to accept one endpoint per role")
    run_script = (args.source_root.parent / "run.sbatch").read_text(encoding="utf-8")
    for artifact in ("smoke_failure.json", "prefill_server.log", "decode_server.log"):
        if artifact not in run_script:
            raise ValueError("retry must collect %s when the smoke gate fails" % artifact)
    result = {
        "valid": True,
        "endpoint_pair": ["P0", "D0"],
        "workload_count": len(workload_ids),
        "frequency_pairs_per_workload": 17 * 15,
        "samples_per_pair": 3,
        "expected_measurements": len(workload_ids) * 17 * 15 * 3,
        "p0d0_end_to_end_smoke_gate": True,
        "coordinate_strategy_comparison": settings["coordinate_reconstruction"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
