#!/usr/bin/env python3
"""Preflight validation for the vendored XPYD Phase 3C reproduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


EXPECTED_SOURCE_COMMIT = "0eb8926f965cfd550f5bcee0095b563b1bb4e41e"
EXPECTED_ROUTES = (("P0", "D0"), ("P0", "D1"), ("P1", "D0"), ("P1", "D1"))
CACHE_ENV_VARS = (
    "HOME",
    "HF_HOME",
    "HF_TOKEN_PATH",
    "HF_HUB_CACHE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "VLLM_CACHE_ROOT",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TMPDIR",
)


def command_output(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(argv)}")
    return proc.stdout.strip()


def bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_config(path: Path, output_root: str) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    endpoints = config.get("endpoints", [])
    compact = [
        (item.get("endpoint_id"), item.get("role"), item.get("node"),
         item.get("gpu_type"), item.get("gpu_ids"), item.get("tp_degree"),
         item.get("kv_connector"))
        for item in endpoints
    ]
    expected = [
        ("P0", "prefill", "neptune", "NVIDIA L40S", [0], 1, "P2pNcclConnector"),
        ("P1", "prefill", "neptune", "NVIDIA L40S", [1], 1, "P2pNcclConnector"),
        ("D0", "decode", "io", "NVIDIA L4", [0], 1, "P2pNcclConnector"),
        ("D1", "decode", "io", "NVIDIA L4", [1], 1, "P2pNcclConnector"),
    ]
    if compact != expected:
        raise ValueError(f"unexpected Phase 3C endpoint topology: {compact!r}")
    routes = tuple(
        (row.get("prefill_endpoint_id"), row.get("decode_endpoint_id"))
        for row in config.get("compatible_pairs", []) if row.get("supported") is True
    )
    if routes != EXPECTED_ROUTES:
        raise ValueError(f"unexpected compatible routes: {routes!r}")
    workloads = [
        (row.get("id"), row.get("input_len"), row.get("output_len"), row.get("count"))
        for row in config.get("workloads", [])
    ]
    if workloads != [("small", 128, 128, 4), ("prefill_heavy", 2048, 128, 4)]:
        raise ValueError(f"unexpected workload matrix: {workloads!r}")
    if config.get("vllm_version") != "0.15.1":
        raise ValueError("Phase 3C requires vLLM 0.15.1")
    if config.get("output_root") != "$XPYD_PHASE3C_OUTPUT_ROOT":
        raise ValueError("job config must use the isolated output-root variable")
    if not output_root:
        raise ValueError("XPYD_PHASE3C_OUTPUT_ROOT is missing")
    return {"endpoints": compact, "routes": routes, "workloads": workloads}


def validate_cache_environment() -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError(f"invalid SLURM_JOB_ID for isolated cache: {job_id!r}")
    expected_root = Path("/data/users/chjing/vllm_job_work") / job_id
    resolved_root = expected_root.resolve()
    values: dict[str, str] = {}
    for name in CACHE_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            raise ValueError(f"required cache environment variable is missing: {name}")
        resolved = Path(raw).resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(
                f"{name} must be isolated under {expected_root}, got {raw}"
            )
        values[name] = raw
    for name in ("HOME", "HF_HUB_CACHE", "XDG_RUNTIME_DIR", "TMPDIR"):
        path = Path(values[name])
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            raise ValueError(f"cache directory is not writable: {name}={path}")
    return values


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
        args.source_root / "paper/scripts/xpyd/disagg_proxy.py",
        args.config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"vendored XPYD files missing: {missing}")

    sys.path.insert(0, str(args.source_root / "paper/scripts"))
    importlib.import_module("xpyd.phase3c_substrate")
    importlib.import_module("xpyd.disagg_proxy")
    importlib.import_module("aiohttp")
    importlib.import_module("transformers")
    importlib.import_module("pynvml")
    vllm = importlib.import_module("vllm")
    if getattr(vllm, "__version__", None) != "0.15.1":
        raise RuntimeError(f"expected vLLM 0.15.1, got {getattr(vllm, '__version__', None)!r}")

    config_summary = validate_config(
        args.config, os.environ.get("XPYD_PHASE3C_OUTPUT_ROOT", "")
    )
    cache_environment = validate_cache_environment()
    report = {
        "ok": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_bundle_sha256": bundle_digest(args.source_root),
        "python": sys.version,
        "platform": platform.platform(),
        "vllm_version": vllm.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,index", "--format=csv,noheader"]),
        "cache_environment": cache_environment,
        "config": config_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("xpyd_preflight=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
