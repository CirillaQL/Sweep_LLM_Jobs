#!/usr/bin/env python3
"""Export Dual_Sweep_LLM saturation classifiers to a portable JSON bundle.

The live Minerva environment deliberately does not need pandas, joblib, or
scikit-learn.  This exporter runs before submission and serializes the five
strict-split, per-GPU ``local_retrained_75`` classifiers into the same small
tree representation used by the portable SWEEP scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np


SEEDS = (3, 7, 11, 19, 23)
METHOD = "local_retrained_75"


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_tree(tree) -> dict:
    spec = tree.tree_
    return {
        "children_left": native(spec.children_left),
        "children_right": native(spec.children_right),
        "feature": native(spec.feature),
        "threshold": native(spec.threshold),
        "value": native(spec.value[:, 0, 0]),
    }


def export_probability_model(wrapper) -> dict:
    model = wrapper.model
    classes = [int(value) for value in model.classes_]
    if len(classes) == 1:
        probability = 1.0 if classes[0] == 1 else 0.0
        model_spec = {"kind": "constant", "probability": probability}
    elif hasattr(model, "estimators_"):
        if classes != [0, 1]:
            raise ValueError(f"unsupported classifier classes: {classes}")
        zeros = np.zeros((1, int(model.n_features_in_)), dtype=float)
        model_spec = {
            "kind": "gbdt_binary",
            "init_raw": float(model._raw_predict_init(zeros)[0, 0]),
            "learning_rate": float(model.learning_rate),
            "trees": [export_tree(tree) for tree in model.estimators_.ravel()],
        }
    else:
        raise TypeError(f"unsupported classifier: {type(model).__name__}")
    return {
        "features": list(wrapper.features),
        "model": model_spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    code_dir = args.shared_model_root / "code"
    models_dir = args.shared_model_root / "results" / "per_gpu_base_finetune" / "models"
    sys.path.insert(0, str(code_dir))
    # Importing defines the wrapper classes referenced by the joblib payloads.
    __import__("shared_residual_adapter_experiment")

    pools = {"l4": [], "l40s": []}
    sources = []
    for seed in SEEDS:
        path = models_dir / f"per_gpu_base_finetune__seed={seed}.joblib"
        artifact = joblib.load(path)
        metadata = artifact["metadata"]
        if int(metadata["seed"]) != seed:
            raise ValueError(f"seed mismatch in {path}")
        models = artifact["payload"]["saturation"][METHOD]
        for gpu_type in pools:
            pools[gpu_type].append({
                "seed": seed,
                **export_probability_model(models[gpu_type]),
            })
        sources.append({
            "seed": seed,
            "file": path.name,
            "sha256": sha256(path),
        })

    bundle = {
        "format": "dual-sweep-saturation-ensemble-v1",
        "source_experiment": "per_gpu_base_finetune",
        "source_method": METHOD,
        "split": "strict_config_50_25_25",
        "aggregation": "mean_probability_across_seeds",
        "threshold": float(args.threshold),
        "target_definition": "request_throughput_rps / request_rate < 0.95",
        "seeds": list(SEEDS),
        "sources": sources,
        "pools": pools,
    }
    args.output.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"exported={args.output} bytes={args.output.stat().st_size} models={len(SEEDS) * 2}")


if __name__ == "__main__":
    main()
