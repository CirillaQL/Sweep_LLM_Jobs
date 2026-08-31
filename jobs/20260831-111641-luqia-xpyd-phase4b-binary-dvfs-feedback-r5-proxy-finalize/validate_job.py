#!/usr/bin/env python3
"""Fail-closed preflight for the fine-grained binary-feedback DVFS job."""

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
    job_dir = Path(__file__).parent
    required = (
        args.source_root / "run_disagg_benchmark.sh",
        args.source_root / "gpu_monitor.py",
        args.source_root / "paper/scripts/replay_synthetic_trace.py",
        args.source_root / "paper/scripts/xpyd/phase3c_substrate.py",
        args.source_root / "paper/scripts/xpyd/phase3d_control.py",
        args.source_root / "paper/scripts/xpyd/phase4b_evaluation.py",
        args.source_root / "paper/scripts/xpyd/binary_dvfs_evaluation.py",
        args.config,
        job_dir / "summarize_binary_dvfs.py",
        job_dir / "live_clock_monitor.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("vendored XPYD files missing: %r" % missing)

    config = json.loads(os.path.expandvars(args.config.read_text(encoding="utf-8")))
    endpoints = {
        row["endpoint_id"]: (row["role"], row["node"], row["gpu_ids"])
        for row in config["endpoints"]
    }
    expected = {
        "P0": ("prefill", "uranus", [0]), "P1": ("prefill", "uranus", [1]),
        "D0": ("decode", "ganymede", [0]), "D1": ("decode", "ganymede", [1]),
    }
    if endpoints != expected:
        raise ValueError("unexpected endpoint topology: %r" % endpoints)
    settings = config["binary_dvfs"]
    if settings["routes"] != [["P0", "D0"], ["P1", "D1"]]:
        raise ValueError("routes must be frozen to P0->D0 and P1->D1")
    if settings["slo"] != {"ttft_ms": 500.0, "tpot_ms": 200.0}:
        raise ValueError("SLO must be TTFT=500ms and TPOT=200ms")
    if int(settings["frequency_grids"]["prefill"]["levels"]) < 10:
        raise ValueError("L40S grid has fewer than ten levels")
    if int(settings["frequency_grids"]["decode"]["levels"]) < 10:
        raise ValueError("L4 grid has fewer than ten levels")
    if int(settings["probe_repeats"]) < 2 or int(settings["confirmation_repeats"]) < 5:
        raise ValueError("binary probes or final confirmation are under-replicated")
    if int(settings["max_concurrency"]) != 2:
        raise ValueError("this no-queue job freezes max concurrency at two")
    if int(settings["closed_loop_batch_size"]) != 2:
        raise ValueError("closed-loop batch size must be two")
    if (int(settings["minimum_observed_concurrency"]) != 2
            or int(settings["maximum_observed_concurrency"]) != 2):
        raise ValueError("actual concurrency must be exactly two")
    if float(settings["maximum_client_queue_delay_ms"]) <= 0:
        raise ValueError("client queue delay requires a positive hard gate")
    if int(settings["requests_per_window"]) < 16:
        raise ValueError("concurrent windows require at least sixteen requests")
    if not str(settings["output_root"]).startswith("/data/users/chjing/vllm_job_work/"):
        raise ValueError("raw output must be under the chjing job cache")

    controller_source = required[6].read_text(encoding="utf-8")
    for marker in (
        "select_frequency_grid", "binary_update", "SEARCH_LOWER_HALF",
        "SEARCH_UPPER_HALF", "ESCALATE_VIOLATING_AXIS",
        "p99_ttft_ms", "p99_tpot_ms", "PerEndpointClockActuator",
        "binary_feedback_frozen_balanced_disjoint_routes",
        "peak_observed_concurrency", "concurrency_valid",
        "load_client_request_results", "client\" / \"requests.jsonl",
        "client_queue_valid", "MAX_FREQ_INFEASIBLE",
        "RECORD_MAX_FREQ_INFEASIBLE_CONTINUE",
    ):
        if marker not in controller_source:
            raise RuntimeError("binary controller evidence is missing: %s" % marker)
    actuator_source = required[4].read_text(encoding="utf-8")
    if 'sudo: str = "/usr/bin/sudo"' not in actuator_source or "-lgc" not in actuator_source:
        raise RuntimeError("feedback actuator must use sudo nvidia-smi -lgc")
    launcher = required[0].read_text(encoding="utf-8")
    if "xpyd.binary_dvfs_evaluation" not in launcher:
        raise RuntimeError("launcher does not dispatch the binary-DVFS module")
    proxy_source = (
        args.source_root / "paper/scripts/xpyd/disagg_proxy.py"
    ).read_text(encoding="utf-8")
    if "self.pairs[self._index % len(self.pairs)]" not in proxy_source:
        raise RuntimeError("proxy no longer provides deterministic round-robin routing")
    if '"processing_ttft"' not in proxy_source:
        raise RuntimeError("proxy processing-side TTFT timing is missing")
    stream_finalize = proxy_source.split("async def _real_decode_stream", 1)[1].split(
        "async def prepare", 1
    )[0]
    if b"data: [DONE]".decode() not in stream_finalize:
        raise RuntimeError("proxy does not recognize terminal SSE completion")
    if stream_finalize.index("self._emit(diagnostics)") > stream_finalize.index(
        "await _close_upstream(response, session)"
    ):
        raise RuntimeError("proxy diagnostic must be emitted before async cleanup")
    replay_source = required[2].read_text(encoding="utf-8")
    for marker in ("closed_loop_batches", "closed_loop_batch_size", "client_queue_delay_ms"):
        if marker not in replay_source:
            raise RuntimeError("no-queue replay evidence is missing: %s" % marker)
    phase3c_source = required[3].read_text(encoding="utf-8")
    for marker in (
        "_wait_for_proxy_diagnostics", "expected_request_ids",
        "proxy_diagnostic_sync.json", "proxy_diagnostics_wait_timeout_s",
    ):
        if marker not in phase3c_source:
            raise RuntimeError("proxy diagnostic synchronization is missing: %s" % marker)
    job_script = (job_dir / "run.sbatch").read_text(encoding="utf-8")
    if job_script.count("--gpus-per-node=2 --gpu-bind=none") != 2:
        raise RuntimeError("both nodes require independent live-clock monitors")
    if "summarize_binary_dvfs.py" not in job_script:
        raise RuntimeError("final compact summarization is missing")
    if "${JOB_DIR}/results/${JOB_ID}" not in job_script:
        raise RuntimeError("Git output must be limited to the final result directory")
    if "${CACHE_ROOT}/live_clock" not in job_script:
        raise RuntimeError("live clocks must remain in the job cache")

    model = Path(os.environ.get("XPYD_LOCAL_MODEL_PATH", ""))
    if not model.is_absolute() or not model.is_dir():
        raise ValueError("local model snapshot is unavailable: %s" % model)
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError("local model snapshot is incomplete: %s" % model)
    cache_root = Path("/data/users/chjing/vllm_job_work") / os.environ.get("SLURM_JOB_ID", "")
    for name in ("HF_HUB_CACHE", "XDG_RUNTIME_DIR", "TMPDIR"):
        path = Path(os.environ.get(name, "")).resolve()
        if cache_root.resolve() != path and cache_root.resolve() not in path.parents:
            raise ValueError("%s is not isolated under %s: %s" % (name, cache_root, path))

    sys.path.insert(0, str(args.source_root / "paper/scripts"))
    module = importlib.import_module("xpyd.binary_dvfs_evaluation")
    loaded = module.load_config(args.config)
    p_grid = settings["frequency_grids"]["prefill"]
    d_grid = settings["frequency_grids"]["decode"]
    if module.select_frequency_grid(
        range(int(p_grid["minimum_mhz"]), int(p_grid["maximum_mhz"]) + 1, 15),
        int(p_grid["minimum_mhz"]), int(p_grid["maximum_mhz"]),
        int(p_grid["levels"]),
    )[-1] != int(p_grid["maximum_mhz"]):
        raise RuntimeError("L40S grid selection self-check failed")
    if module.binary_update(0, 16, 8, True) != (0, 8):
        raise RuntimeError("binary feasible-branch self-check failed")
    if module.binary_update(0, 16, 8, False) != (9, 16):
        raise RuntimeError("binary infeasible-branch self-check failed")
    if loaded["binary_dvfs"] != settings:
        raise RuntimeError("loaded config differs from validated config")

    for dependency in ("aiohttp", "transformers", "pynvml", "vllm"):
        imported = importlib.import_module(dependency)
        if dependency == "vllm" and getattr(imported, "__version__", None) != "0.15.1":
            raise RuntimeError("expected vLLM 0.15.1")

    result = {
        "valid": True,
        "phase": "4B_fine_grained_binary_feedback_DVFS",
        "topology": endpoints, "routes": settings["routes"],
        "routing": "frozen deterministic round-robin over disjoint routes",
        "max_concurrency": settings["max_concurrency"],
        "minimum_observed_concurrency": settings["minimum_observed_concurrency"],
        "maximum_observed_concurrency": settings["maximum_observed_concurrency"],
        "dispatch_mode": "closed_loop_batches",
        "ttft_measurement_scope": "proxy request_received to first real decode chunk",
        "slo": settings["slo"],
        "frequency_grid_specs": settings["frequency_grids"],
        "probe_repeats": settings["probe_repeats"],
        "confirmation_repeats": settings["confirmation_repeats"],
        "raw_output_root": settings["output_root"],
        "model_path": str(model),
        "controller": "coordinate lower-bound binary search from completed windows only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
