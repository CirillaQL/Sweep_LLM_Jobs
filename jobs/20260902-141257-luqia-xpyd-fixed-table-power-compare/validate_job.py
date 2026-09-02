#!/usr/bin/env python3
"""Fail-closed preflight for the fixed-table power comparison job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "5f8fd6cb80681e24f9948e7fe137769e6a8a831fe306992f0694883ccc95116d"
)
WORKLOADS = [
    ("small_light", 128, 64, 48, 0.04),
    ("prefill_medium", 1024, 64, 48, 0.04),
    ("prefill_heavy", 2048, 64, 48, 0.04),
    ("decode_medium", 128, 128, 48, 0.04),
    ("decode_heavy", 128, 256, 48, 0.04),
    ("balanced_medium", 512, 128, 48, 0.04),
]
OPTIMAL_FREQUENCIES = {
    "small_light": (900, 720),
    "prefill_medium": (900, 975),
    "prefill_heavy": (1980, 975),
    "decode_medium": (900, 975),
    "decode_heavy": (900, 975),
    "balanced_medium": (900, 975),
}
CACHE_ENV_VARS = (
    "HF_HOME", "HF_TOKEN_PATH", "HF_HUB_CACHE", "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR", "VLLM_CACHE_ROOT",
    "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR",
)


def bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source(root: Path) -> dict[str, Any]:
    required = [
        root / "run_disagg_benchmark.sh",
        root / "gpu_monitor.py",
        root / "paper/scripts/replay_synthetic_trace.py",
        root / "paper/scripts/xpyd/disagg_proxy.py",
        root / "paper/scripts/xpyd/online_feedback_controller.py",
        root / "paper/scripts/xpyd/phase3c_substrate.py",
        root / "paper/scripts/xpyd/compare_fixed_table_power.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files: %s" % missing)
    observed = bundle_digest(root)
    if observed != EXPECTED_SOURCE_BUNDLE_SHA256:
        raise ValueError(
            "source bundle digest mismatch expected=%s observed=%s"
            % (EXPECTED_SOURCE_BUNDLE_SHA256, observed)
        )
    proxy = required[3].read_text(encoding="utf-8")
    phase3c = required[5].read_text(encoding="utf-8")
    analyzer = required[6].read_text(encoding="utf-8")
    for marker in (
        "fixed_frequency_table must cover every configured workload",
        "frequency_table.write(", "service_workload_energy_windows",
        "workload_energy_summary.csv", "route_timestamp_wall_s",
        "decode_completion_wall_s",
    ):
        if marker not in proxy + phase3c:
            raise ValueError("fixed-table measurement marker missing: %s" % marker)
    for marker in (
        "baseline and optimized request traces differ", "P1+D1 only",
        "both_heavy must be excluded", "power_comparison.json",
    ):
        if marker not in analyzer:
            raise ValueError("comparison analyzer marker missing: %s" % marker)
    return {"source_bundle_sha256": observed, "required_files": len(required)}


def validate_config(path: Path, mode: str) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    workloads = [
        (row["id"], row["input_len"], row["output_len"], row["count"], row["rate_rps"])
        for row in config["workloads"]
    ]
    if workloads != WORKLOADS:
        raise ValueError("%s workload matrix differs: %r" % (mode, workloads))
    if "both_heavy" in {row[0] for row in workloads}:
        raise ValueError("both_heavy must be excluded")
    if config.get("workload_ordering") != "windowed":
        raise ValueError("workloads must run as serial windows")
    if config.get("background_experiment_traffic") is not False:
        raise ValueError("background exploration traffic must be disabled")
    feedback = config.get("online_feedback", {})
    if feedback.get("enabled") is not True or feedback.get("service_pair") != ["P1", "D1"]:
        raise ValueError("measurement traffic must use P1-D1 feedback service path")
    if feedback.get("service_frequency_settle_s") != 10.0:
        raise ValueError("service must wait 10 seconds after a frequency change")
    if config.get("coverage_policy") != {
        "required_endpoint_ids": ["P1", "D1"], "required_pairs": [["P1", "D1"]]
    }:
        raise ValueError("coverage must require only P1-D1")
    endpoints = [(row["endpoint_id"], row["node"], row["gpu_ids"], row["kv_connector"])
                 for row in config["endpoints"]]
    expected_endpoints = [
        ("P0", "uranus", [0], "P2pNcclConnector"),
        ("P1", "uranus", [1], "P2pNcclConnector"),
        ("D0", "ganymede", [0], "P2pNcclConnector"),
        ("D1", "ganymede", [1], "P2pNcclConnector"),
    ]
    if endpoints != expected_endpoints:
        raise ValueError("unexpected endpoint topology: %r" % endpoints)
    table = config.get("fixed_frequency_table", {})
    if set(table) != set(OPTIMAL_FREQUENCIES):
        raise ValueError("fixed table must exactly cover the six workloads")
    observed = {
        key: (int(value["prefill_frequency_mhz"]), int(value["decode_frequency_mhz"]))
        for key, value in table.items()
    }
    expected = (
        {key: (2520, 1500) for key in OPTIMAL_FREQUENCIES}
        if mode == "baseline" else OPTIMAL_FREQUENCIES
    )
    if observed != expected:
        raise ValueError("%s frequencies differ: %r" % (mode, observed))
    return {"mode": mode, "requests": sum(row[3] for row in workloads), "frequencies": observed}


def validate_job_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for marker in (
        "#SBATCH --nodelist=uranus,ganymede", "#SBATCH --gpus-per-node=2",
        'WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"',
        "run_variant baseline", "run_variant optimized",
        "expected exactly 288 fixed-table service dispatches",
        "unexpectedly performed exploration", "compare_fixed_table_power.py",
    ):
        if marker not in text:
            raise ValueError("job marker missing: %s" % marker)
    for name in CACHE_ENV_VARS:
        if "export %s=" % name not in text and "%s=\"" % name not in text:
            raise ValueError("cache variable not isolated: %s" % name)
    syntax = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    if syntax.returncode:
        raise ValueError("bash syntax error: %s" % syntax.stderr)
    return {"slurm_syntax": "ok", "cache_root": "/data/users/chjing/vllm_job_work/<job_id>"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--optimized-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "valid": True,
        "source": validate_source(args.source_root),
        "baseline": validate_config(args.baseline_config, "baseline"),
        "optimized": validate_config(args.optimized_config, "optimized"),
        "job": validate_job_script(Path(__file__).with_name("run.sbatch")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
