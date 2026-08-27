#!/usr/bin/env python3
"""Static and environment preflight for the history-seeded physical Job."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


ENDPOINTS = {"P0": ("neptune", "prefill"), "P1": ("neptune", "prefill"),
             "D0": ("io", "decode"), "D1": ("io", "decode")}


def digest(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actuator-audit", type=Path, required=True)
    parser.add_argument("--history-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observed = {item["endpoint_id"]: (item["node"], item["role"])
                for item in config["endpoints"]}
    if observed != ENDPOINTS:
        raise SystemExit(f"unexpected endpoint topology: {observed}")
    if config["model"] != "$XPYD_LOCAL_MODEL_PATH" or config["tokenizer_model"] != "$XPYD_LOCAL_MODEL_PATH":
        raise SystemExit("Job must use the validated local model for server and tokenizer")
    history = json.loads(args.history_summary.read_text(encoding="utf-8"))
    actuator = json.loads(args.actuator_audit.read_text(encoding="utf-8"))
    if not actuator.get("valid"):
        raise SystemExit("Phase 3D-A prerequisite is invalid")
    if not history.get("valid") or not history.get("ready_for_phase4b"):
        raise SystemExit("Phase 4A history is invalid")
    if history.get("models_trained_or_used"):
        raise SystemExit("history must not contain predictive-model decisions")
    model = Path(os.environ["XPYD_LOCAL_MODEL_PATH"])
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise SystemExit(f"local Mistral snapshot is incomplete: {model}")
    sys.path.insert(0, str(args.source_root / "paper/scripts"))
    for module in ("xpyd.history_seeded_control", "xpyd.phase3d_control", "vllm", "pynvml"):
        importlib.import_module(module)
    report = {
        "ok": True,
        "source_bundle_sha256": digest(args.source_root),
        "actuator_audit": str(args.actuator_audit),
        "history_summary": str(args.history_summary),
        "history_summary_sha256": hashlib.sha256(args.history_summary.read_bytes()).hexdigest(),
        "local_model": str(model),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
