#!/usr/bin/env python3
"""Export the fitted scikit-learn models to a portable JSON runtime bundle."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def export_tree(tree):
    spec = tree.tree_
    return {
        "children_left": native(spec.children_left),
        "children_right": native(spec.children_right),
        "feature": native(spec.feature),
        "threshold": native(spec.threshold),
        "value": native(spec.value[:, 0, 0]),
    }


def export_gbdt(model):
    zeros = np.zeros((1, int(model.n_features_in_)), dtype=float)
    return {
        "init_raw": float(model._raw_predict_init(zeros)[0, 0]),
        "learning_rate": float(model.learning_rate),
        "trees": [export_tree(tree) for tree in model.estimators_.ravel()],
    }


def export_classifier(model):
    if hasattr(model, "estimators_"):
        return {"kind": "gbdt", **export_gbdt(model)}
    classes = [int(value) for value in model.classes_]
    if len(classes) == 1:
        probability = 1.0 if classes[0] == 1 else 0.0
    else:
        probability = float(model.class_prior_[classes.index(1)])
    return {"kind": "constant", "p_violate": probability}


def export_scaler(model):
    return {
        "mean": native(model.mean_),
        "scale": native(model.scale_),
    }


def export_ridge(model):
    return {
        "coef": native(model.coef_),
        "intercept": float(model.intercept_),
    }


def load(path):
    return joblib.load(path)


def export_pool(model_dir):
    config = load(model_dir / "config.pkl")
    cap_groups = load(model_dir / "cap_groups.pkl")
    plateau = {}
    for _, row in cap_groups.iterrows():
        key = f"{int(row['il'])}:{int(row['ol'])}:{int(row['tp'])}:{int(row['freq'])}"
        plateau[key] = bool(row["plateau_confirmed"])

    pool = {
        "config": native(config),
        "capacity": {
            "features": native(load(model_dir / "capacity_features.pkl")),
            "model": export_gbdt(load(model_dir / "capacity_model.pkl")),
            "plateau": plateau,
        },
        "phases": {},
    }

    for phase in ("prefill", "decode"):
        phase_spec = {
            "clf_features": native(load(model_dir / f"{phase}_clf_features.pkl")),
            "reg_features": native(load(model_dir / f"{phase}_reg_features.pkl")),
            "power_features": native(load(model_dir / f"{phase}_power_features.pkl")),
            "classifiers": {},
            "scalers": {},
            "latency_model": export_ridge(load(model_dir / f"{phase}_latency_regressor_p99.pkl")),
            "latency_scaler": export_scaler(load(model_dir / f"{phase}_latency_scaler.pkl")),
            "latency_poly_powers": native(load(model_dir / f"{phase}_latency_poly.pkl").powers_),
            "power_model": export_gbdt(load(model_dir / f"{phase}_power_model.pkl")),
        }
        for slo in config["SLO_THRESHOLDS"]:
            phase_spec["classifiers"][str(slo)] = export_classifier(
                load(model_dir / f"{phase}_slo_classifier_{slo}.pkl")
            )
            phase_spec["scalers"][str(slo)] = export_scaler(
                load(model_dir / f"{phase}_slo_scaler_{slo}.pkl")
            )
        pool["phases"][phase] = phase_spec

    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = {
        "format": "sweep-llm-portable-model-v1",
        "pools": {
            gpu: export_pool(args.models_root / f"models_{gpu}")
            for gpu in ("l4", "l40s")
        },
    }
    args.output.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"exported={args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
