#!/usr/bin/env python3
"""Check portable saturation predictions against all source sklearn models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-model-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--samples-per-gpu", type=int, default=100)
    args = parser.parse_args()

    sys.path.insert(0, str(args.shared_model_root / "code"))
    import shared_residual_adapter_experiment as base_exp

    sys.path.insert(0, str(args.scheduler.parent))
    from scheduler import SaturationEnsemble

    portable = SaturationEnsemble(args.bundle)
    prepared = base_exp.load_prepared_data()
    errors = []
    checked = 0
    for gpu_type, frame in prepared.groupby("gpu_type", sort=True):
        sample = frame.sample(n=min(args.samples_per_gpu, len(frame)), random_state=20260722)
        source_models = {}
        for seed in portable.bundle["seeds"]:
            path = (
                args.shared_model_root / "results" / "per_gpu_base_finetune" / "models"
                / f"per_gpu_base_finetune__seed={seed}.joblib"
            )
            artifact = joblib.load(path)
            source_models[int(seed)] = artifact["payload"]["saturation"]["local_retrained_75"][gpu_type]

        for _, row in sample.iterrows():
            one = row.to_frame().T
            prediction = portable.predict(
                gpu_type, int(row.il), int(row.ol), int(row.tp), int(row.freq), float(row.rate)
            )
            by_seed = {item["seed"]: item["probability"] for item in prediction["per_seed"]}
            for seed, source in source_models.items():
                expected = float(source.predict_proba(one)[0])
                error = abs(expected - by_seed[seed])
                errors.append(error)
                checked += 1

    result = {
        "checked_predictions": checked,
        "max_absolute_error": max(errors),
        "mean_absolute_error": float(np.mean(errors)),
        "tolerance": 1e-6,
        "passed": bool(max(errors) <= 1e-6),
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
