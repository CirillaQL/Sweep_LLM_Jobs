#!/usr/bin/env python3
"""Fail-closed preflight for the P17/D15 binary-SLO energy-feedback job."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


EXPECTED_SOURCE_COMMIT = "0eb8926f965cfd550f5bcee0095b563b1bb4e41e"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "ecade7d9ec85286553be5606bd4bd574184828f76ec4f2a9805f2aa3083e1aa0"
)
EXPECTED_PAIRS = tuple(
    (("P0", "D0"), ("P1", "D1"))
)
CACHE_ENV_VARS = (
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
    process = subprocess.run(argv, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(
            f"command failed rc={process.returncode}: {' '.join(argv)}"
        )
    return process.stdout.strip()


def bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source(source_root: Path) -> dict[str, Any]:
    required = (
        source_root / "run_disagg_benchmark.sh",
        source_root / "gpu_monitor.py",
        source_root / "paper/scripts/replay_synthetic_trace.py",
        source_root / "paper/scripts/xpyd/phase3c_substrate.py",
        source_root / "paper/scripts/xpyd/disagg_proxy.py",
        source_root / "paper/scripts/xpyd/workload_frequency_table.py",
        source_root / "paper/scripts/xpyd/online_feedback_controller.py",
        source_root / "paper/scripts/xpyd/summarize_service_steady_state.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"vendored XPYD files missing: {missing}")
    observed_digest = bundle_digest(source_root)
    if observed_digest != EXPECTED_SOURCE_BUNDLE_SHA256:
        raise ValueError(
            "source bundle digest mismatch: "
            f"expected={EXPECTED_SOURCE_BUNDLE_SHA256} observed={observed_digest}"
        )
    launcher = required[0].read_text(encoding="utf-8")
    phase3c = required[3].read_text(encoding="utf-8")
    proxy = required[4].read_text(encoding="utf-8")
    for marker in (
        'XPYD_ENDPOINTS_PER_ROLE="${XPYD_ENDPOINTS_PER_ROLE:-2}"',
        "endpoint_index<XPYD_ENDPOINTS_PER_ROLE",
        '"kv_connector":"P2pNcclConnector"',
        '"send_type":"PUT"',
    ):
        if marker not in launcher:
            raise ValueError(f"2P2D launcher marker missing: {marker}")
    table = required[5].read_text(encoding="utf-8")
    controller = required[6].read_text(encoding="utf-8")
    steady = required[7].read_text(encoding="utf-8")
    for marker in (
        "class WorkloadFrequencyTable",
        "threading.RLock()",
        "expected_revision",
        "candidate.measured_energy_j >= current.measured_energy_j",
        "os.replace(temporary, self._persistence_path)",
    ):
        if marker not in table:
            raise ValueError(f"frequency-table marker missing: {marker}")
    for marker in (
        "frequency_table=WorkloadFrequencyTable",
        '"pd_inference_ttft": duration_ms(',
        '@app.get("/xpyd/frequency-table")',
        '@app.put("/xpyd/frequency-table/{workload_id}")',
    ):
        if marker not in proxy:
            raise ValueError(f"proxy frequency-table marker missing: {marker}")
    for marker in (
        "class OnlineFeedbackController",
        "class PhysicalFeedbackRuntime",
        "frequency_changed = await self.actuate(",
        'await self._search_axis(\n            "P"',
        'await self._search_axis(\n            "D"',
        "self._pending.add(workload_id)",
        "await self._queue.put",
        "probe_interval_s",
        'request_received = stamps.get("request_received")',
        "service_frequency_settle",
        "service_warmup_completed",
        "experiment_frequency_settle_s",
        "experiment_warmup_requests",
        "experiment_warmup_completed",
        "probe_samples_per_candidate",
        "minimum_probe_samples",
        "probe_candidate_early_stop",
        "frequency_actuation",
        "measured_energy_j",
        "probe_candidate_aggregated",
        "binary-slo",
        "energy-refine",
        "minimum_mean_request_energy_j",
        "self._queue.join()",
        "and self.ttft_ms < ttft_slo_ms",
        "service_request_log",
        "_record_service_request",
    ):
        if marker not in controller:
            raise ValueError(f"online feedback marker missing: {marker}")
    for marker in (
        "discard_first", "configuration_window_id", "steady_state_included",
        "ttft_ms", "tpot_ms", "stable_window_count",
        "selected_steady_state_by_workload",
    ):
        if marker not in steady:
            raise ValueError(f"steady-state analyzer marker missing: {marker}")
    for marker in (
        "def _online_inference_latency(",
        '"client_observed_ttft_ms": request.get("ttft_ms")',
        "service_slo_violations",
        '"enforcement": "record_only"',
        "for item in request_rows",
    ):
        if marker not in phase3c:
            raise ValueError(f"adjusted SLO-audit marker missing: {marker}")
    return {"source_bundle_sha256": observed_digest, "required_files": len(required)}


def validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    endpoints = config.get("endpoints", [])
    compact = [
        (
            item.get("endpoint_id"),
            item.get("role"),
            item.get("node"),
            item.get("gpu_ids"),
            item.get("tp_degree"),
            item.get("http_port"),
            item.get("kv_port"),
            item.get("kv_connector"),
        )
        for item in endpoints
    ]
    expected = []
    for index in range(2):
        expected.append(
            (f"P{index}", "prefill", "uranus", [index], 1,
             8100 + index, 14579 + index, "P2pNcclConnector")
        )
    for index in range(2):
        expected.append(
            (f"D{index}", "decode", "ganymede", [index], 1,
             8200 + index, 14579 + index, "P2pNcclConnector")
        )
    if compact != expected:
        raise ValueError(f"unexpected 2P2D endpoint topology: {compact!r}")
    pairs = tuple(
        (row.get("prefill_endpoint_id"), row.get("decode_endpoint_id"))
        for row in config.get("compatible_pairs", [])
        if row.get("supported") is True
    )
    if pairs != EXPECTED_PAIRS:
        raise ValueError(f"unexpected compatible pairs: {pairs!r}")
    workloads = [
        (row.get("id"), row.get("input_len"), row.get("output_len"),
         row.get("count"), row.get("rate_rps"))
        for row in config.get("workloads", [])
    ]
    if workloads != [
        ("small_light", 128, 64, 48, 0.20),
        ("prefill_medium", 1024, 64, 48, 0.20),
        ("prefill_heavy", 2048, 64, 48, 0.20),
        ("decode_medium", 128, 128, 48, 0.20),
        ("decode_heavy", 128, 256, 48, 0.20),
        ("balanced_medium", 512, 128, 48, 0.20),
        ("both_heavy", 2048, 256, 48, 0.20),
    ]:
        raise ValueError(f"unexpected workload matrix: {workloads!r}")
    if config.get("routing_policy") != "round_robin":
        raise ValueError("single selected pair must use deterministic routing")
    if config.get("workload_ordering") != "windowed":
        raise ValueError("seven workloads must run in serial windows")
    if config.get("frequency_table_path") != "$XPYD_FREQUENCY_TABLE_PATH":
        raise ValueError("job must persist the frequency table")
    if config.get("vllm_version") != "0.15.1":
        raise ValueError("2P2D job requires vLLM 0.15.1")
    if config.get("model") != "$XPYD_LOCAL_MODEL_PATH":
        raise ValueError("job config must use the validated local model")
    if config.get("tokenizer_model") != "$XPYD_LOCAL_MODEL_PATH":
        raise ValueError("job config must use the validated local tokenizer")
    if config.get("output_root") != "$XPYD_FEEDBACK_TABLE_OUTPUT_ROOT":
        raise ValueError("job config must use the isolated output-root variable")
    if config.get("client", {}).get("max_concurrency") != 1:
        raise ValueError("connector validation boundary requires max_concurrency=1")
    table = config.get("online_feedback", {})
    if table.get("enabled") is not True:
        raise ValueError("online feedback must be enabled")
    if table.get("service_pair") != ["P1", "D1"]:
        raise ValueError("service pair must be P1->D1")
    if table.get("experiment_pair") != ["P0", "D0"]:
        raise ValueError("experiment pair must be P0->D0")
    if table.get("service_request_log") != "$XPYD_SERVICE_REQUEST_LOG":
        raise ValueError("formal P1/D1 requests must use the isolated dispatch log")
    if table.get("slo") != {"ttft_ms": 500.0, "tpot_ms": 200.0}:
        raise ValueError("service SLO must be TTFT=500ms, TPOT=200ms")
    if table.get("exploration_slo") != {
        "ttft_ms": 500.0, "tpot_ms": 200.0,
    }:
        raise ValueError("exploration SLO must be TTFT P95<500ms, TPOT P95<=200ms")
    if int(table.get("probe_samples_per_candidate", 0)) != 3:
        raise ValueError("every candidate must use exactly three requests")
    if int(table.get("minimum_probe_samples", 0)) != 3:
        raise ValueError("adaptive candidates must retain at least three requests")
    if float(table.get("candidate_stability_cv", 0)) != 0.05:
        raise ValueError("adaptive candidate CV threshold must be five percent")
    if float(table.get("candidate_slo_headroom_ratio", 0)) != 0.90:
        raise ValueError("adaptive early acceptance requires ten percent SLO headroom")
    if int(table.get("experiment_warmup_requests", 0)) != 1:
        raise ValueError("experiment pair must run one one-time warmup request")
    if int(table.get("service_warmup_requests", 0)) != 1:
        raise ValueError("service pair must run one one-time warmup request")
    if float(table.get("service_request_interval_s", -1)) != 5.0:
        raise ValueError("service request interval must be five seconds")
    if float(table.get("probe_request_interval_s", -1)) != 0.0:
        raise ValueError("fixed experiment probe interval must be disabled")
    if float(table.get("service_frequency_settle_s", 0)) != 0.5:
        raise ValueError("service post-readback settle must be 0.5 seconds")
    if float(table.get("experiment_frequency_settle_s", 0)) != 0.5:
        raise ValueError("experiment post-readback settle must be 0.5 seconds")
    if table.get("search_order") != ["prefill", "decode"]:
        raise ValueError("energy search must run P before D")
    if table.get("algorithm") != (
        "binary_slo_boundary_then_feasible_suffix_minimum_mean_request_energy"
    ):
        raise ValueError("unexpected feedback search algorithm")
    if table.get("axis_search_levels") != {"prefill": 17, "decode": 15}:
        raise ValueError("axis search must use 17 P levels and 15 D levels")
    grids = table.get("frequency_grids", {})
    if grids.get("prefill") != {
        "minimum_mhz": 900, "maximum_mhz": 2520, "levels": 17,
    }:
        raise ValueError("unexpected prefill frequency grid")
    if grids.get("decode") != {
        "minimum_mhz": 450, "maximum_mhz": 1500, "levels": 15,
    }:
        raise ValueError("unexpected decode frequency grid")
    if float(table.get("exploration_shutdown_timeout_s", 0)) != 36000.0:
        raise ValueError("exploration shutdown timeout must be ten hours")
    if config.get("background_experiment_traffic") is not True:
        raise ValueError("background experiment traffic must be declared")
    if config.get("steady_state") != {
        "discard_first_per_configuration_window": 3,
        "minimum_samples": 8,
        "maximum_cv": 0.10,
    }:
        raise ValueError("unexpected steady-state policy")
    coverage = config.get("coverage_policy", {})
    if coverage != {
        "required_endpoint_ids": ["P1", "D1"],
        "required_pairs": [["P1", "D1"]],
    }:
        raise ValueError("physical scaffold must audit only the selected pair")
    return {
        "endpoint_count": len(compact),
        "pair_count": len(pairs),
        "routing_policy": config["routing_policy"],
        "workload_ordering": config["workload_ordering"],
        "request_count": sum(row[3] for row in workloads),
        "workloads": workloads,
        "service_pair": table["service_pair"],
        "experiment_pair": table["experiment_pair"],
        "frequency_search_enabled": True,
        "dvfs_actuation_enabled": True,
        "service_request_interval_s": table["service_request_interval_s"],
        "probe_request_interval_s": table["probe_request_interval_s"],
        "service_frequency_settle_s": table["service_frequency_settle_s"],
        "experiment_frequency_settle_s": table["experiment_frequency_settle_s"],
        "service_warmup_requests": table["service_warmup_requests"],
        "experiment_warmup_requests": table["experiment_warmup_requests"],
        "exploration_slo": table["exploration_slo"],
        "steady_state": config["steady_state"],
    }


def validate_job_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    required = (
        "#SBATCH --partition=long",
        "#SBATCH --time=12:00:00",
        "#SBATCH --nodes=2",
        "#SBATCH --ntasks=8",
        "#SBATCH --ntasks-per-node=4",
        "#SBATCH --nodelist=uranus,ganymede",
        "#SBATCH --gpus-per-node=2",
        "#SBATCH --cpus-per-gpu=8",
        "export L40S_GPU_IDS=0,1 L4_GPU_IDS=0,1",
        "export XPYD_ENDPOINTS_PER_ROLE=2",
        'WORK_DIR="/data/users/chjing/vllm_job_work/${JOB_ID}"',
        'trap collect_compact_results EXIT',
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError(f"Slurm resource/result marker missing: {missing}")
    return {
        "partition": "long",
        "walltime": "12:00:00",
        "nodes": ["uranus", "ganymede"],
        "gpus_per_node": 2,
        "total_gpus": 4,
        "tasks": 8,
        "cache_root": "/data/users/chjing/vllm_job_work/<job_id>",
    }


def validate_concurrent_table(source_root: Path) -> dict[str, Any]:
    module_path = source_root / "paper/scripts/xpyd/workload_frequency_table.py"
    spec = importlib.util.spec_from_file_location("xpyd_frequency_table_preflight", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load workload frequency table")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    table = module.WorkloadFrequencyTable()
    if tuple(table.keys) != tuple(module.WORKLOAD_KEYS) or len(table.keys) != 7:
        raise RuntimeError("frequency table must expose the exact seven keys")

    def write(energy: float) -> None:
        table.write("both_heavy", {
            "prefill_frequency_mhz": 1500,
            "decode_frequency_mhz": 900,
            "measured_power_w": 120.0,
            "measured_energy_j": energy,
            "ttft_ms": 400.0,
            "tpot_ms": 150.0,
            "prefill_endpoint_id": "P0",
            "decode_endpoint_id": "D0",
            "sample_count": 16,
            "source": "static_preflight",
            "slo_met": True,
        })

    energies = [180.0, 140.0, 175.0, 95.0, 120.0, 101.0, 160.0]
    with ThreadPoolExecutor(max_workers=len(energies)) as executor:
        list(executor.map(write, energies))
    entry = table.read("both_heavy")
    if entry.value is None or entry.value.measured_energy_j != min(energies):
        raise RuntimeError("concurrent writes did not retain lowest energy")
    stale_conflict = False
    try:
        table.write("both_heavy", entry.value, expected_revision=0)
    except module.FrequencyTableError:
        stale_conflict = True
    if not stale_conflict:
        raise RuntimeError("stale revision was not rejected")
    return {
        "key_count": len(table.keys),
        "concurrent_lowest_energy_j": entry.value.measured_energy_j,
        "revision": entry.revision,
        "stale_revision_rejected": stale_conflict,
    }


def validate_runtime_environment() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError(f"invalid SLURM_JOB_ID: {job_id!r}")
    expected_root = (Path("/data/users/chjing/vllm_job_work") / job_id).resolve()
    cache_values: dict[str, str] = {}
    for name in CACHE_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            raise ValueError(f"required cache variable missing: {name}")
        resolved = Path(raw).resolve()
        if resolved != expected_root and expected_root not in resolved.parents:
            raise ValueError(f"{name} is outside the job cache: {raw}")
        cache_values[name] = raw
    for name in ("HF_HUB_CACHE", "XDG_RUNTIME_DIR", "TMPDIR"):
        path = Path(cache_values[name])
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            raise ValueError(f"cache directory is not writable: {name}={path}")
    model = Path(os.environ.get("XPYD_LOCAL_MODEL_PATH", ""))
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError(f"validated local model is unavailable: {model}")
    sys.path.insert(0, os.environ["SOURCE_ROOT"] + "/paper/scripts")
    for module_name in (
        "xpyd.phase3c_substrate",
        "xpyd.disagg_proxy",
        "xpyd.online_feedback_controller",
        "aiohttp",
        "transformers",
        "pynvml",
    ):
        importlib.import_module(module_name)
    vllm = importlib.import_module("vllm")
    if getattr(vllm, "__version__", None) != "0.15.1":
        raise RuntimeError(
            f"expected vLLM 0.15.1, got {getattr(vllm, '__version__', None)!r}"
        )
    return {
        "slurm_job_id": job_id,
        "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "cache_environment": cache_values,
        "local_model": str(model),
        "vllm_version": vllm.__version__,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,index", "--format=csv,noheader"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "ok": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source": validate_source(args.source_root),
        "config": validate_config(args.config),
        "job": validate_job_script(Path(__file__).with_name("run.sbatch")),
        "frequency_table": validate_concurrent_table(args.source_root),
        "python": sys.version,
        "platform": platform.platform(),
        "static_only": args.static_only,
    }
    if not args.static_only:
        os.environ["SOURCE_ROOT"] = str(args.source_root)
        report["runtime"] = validate_runtime_environment()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("xpyd_preflight=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
