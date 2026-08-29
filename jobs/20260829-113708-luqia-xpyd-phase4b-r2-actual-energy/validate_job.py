#!/usr/bin/env python3
"""Fail-closed preflight for the Uranus/Ganymede Phase 4B-r2 job."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = (
        args.source_root / "run_disagg_benchmark.sh",
        args.source_root / "gpu_monitor.py",
        args.source_root / "paper/scripts/replay_synthetic_trace.py",
        args.source_root / "paper/scripts/xpyd/phase3c_substrate.py",
        args.source_root / "paper/scripts/xpyd/phase3d_control.py",
        args.source_root / "paper/scripts/xpyd/phase4b_evaluation.py",
        args.config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"vendored XPYD files missing: {missing}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    endpoints = {
        row["endpoint_id"]: (row["role"], row["node"], row["gpu_ids"])
        for row in config["endpoints"]
    }
    expected = {
        "P0": ("prefill", "uranus", [0]), "P1": ("prefill", "uranus", [1]),
        "D0": ("decode", "ganymede", [0]), "D1": ("decode", "ganymede", [1]),
    }
    if endpoints != expected:
        raise ValueError(f"unexpected endpoint topology: {endpoints!r}")
    phase4b = config["phase4b"]
    if phase4b["policies"] != [
        "STATIC", "FEEDBACK_ROUTING_ONLY", "FEEDBACK_DVFS_ONLY", "FULL_FEEDBACK"
    ]:
        raise ValueError("Phase 4B must compare exactly the four policies")
    if int(phase4b["repeats"]) != 5 or int(phase4b["requests_per_repeat"]) != 4:
        raise ValueError("Phase 4B-r2 requires five repeats of four requests")
    if phase4b["slo"] != {"ttft_ms": 500.0, "tpot_ms": 200.0}:
        raise ValueError("Phase 4B-r2 must use TTFT=500 ms and TPOT=200 ms")
    if phase4b.get("phase4a_summary") is not None:
        raise ValueError("incompatible Neptune/IO oracle must not be used")
    if phase4b.get("output_root") != "$XPYD_PHASE4B_OUTPUT_ROOT":
        raise ValueError("Phase 4B output must be job-isolated")

    source = required[5].read_text(encoding="utf-8")
    for marker in (
        "phase4b_feedback_decision_trace.csv", "feedback_snapshot_json",
        "selected_active_endpoint_ids_json", "energy_savings_vs_static",
        "for endpoint_id in selected_endpoint_ids",
    ):
        if marker not in source:
            raise RuntimeError(f"required Phase 4B-r2 decision evidence is missing: {marker}")
    actuator = required[4].read_text(encoding="utf-8")
    launcher = required[0].read_text(encoding="utf-8")
    if 'sudo: str = "/usr/bin/sudo"' not in actuator or "-lgc" not in actuator:
        raise RuntimeError("feedback actuator must use sudo nvidia-smi -lgc")
    if "${SUDO_BIN} ${NVIDIA_SMI_BIN}" not in launcher or "-lmc" not in launcher:
        raise RuntimeError("launcher must use sudo for fixed memory clocks")

    job_script = (Path(__file__).parent / "run.sbatch").read_text(encoding="utf-8")
    if job_script.count("--gpus-per-node=2 --gpu-bind=none") != 2:
        raise RuntimeError("both nodes require independent 0.2-second clock monitors")
    if "audit_phase4b_results.py" not in job_script:
        raise RuntimeError("post-run Phase 4B evidence audit is missing")

    model = Path(os.environ.get("XPYD_LOCAL_MODEL_PATH", ""))
    if not model.is_absolute() or not model.is_dir():
        raise ValueError(f"local model snapshot is unavailable: {model}")
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError(f"local model snapshot is incomplete: {model}")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    cache_root = Path("/data/users/chjing/vllm_job_work") / job_id
    for name in ("HOME", "HF_HUB_CACHE", "XDG_RUNTIME_DIR", "TMPDIR"):
        path = Path(os.environ.get(name, "")).resolve()
        if cache_root.resolve() != path and cache_root.resolve() not in path.parents:
            raise ValueError(f"{name} is not isolated under {cache_root}: {path}")

    sys.path.insert(0, str(args.source_root / "paper/scripts"))
    importlib.import_module("xpyd.phase4b_evaluation")
    importlib.import_module("aiohttp")
    importlib.import_module("transformers")
    importlib.import_module("pynvml")
    vllm = importlib.import_module("vllm")
    if getattr(vllm, "__version__", None) != "0.15.1":
        raise RuntimeError(f"expected vLLM 0.15.1, got {getattr(vllm, '__version__', None)!r}")

    result = {
        "valid": True, "phase": "4B-r2", "topology": endpoints,
        "slo": phase4b["slo"], "policies": phase4b["policies"],
        "energy_baseline": "same-job STATIC", "model_path": str(model),
        "decision_evidence": "feedback+candidates+selected route+target/readback+actual route+energy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
