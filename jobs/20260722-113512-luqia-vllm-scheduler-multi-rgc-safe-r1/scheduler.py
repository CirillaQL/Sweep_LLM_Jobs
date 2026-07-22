#!/usr/bin/env python3
"""Portable two-pool SWEEP scheduler for the live vLLM PD job.

The runtime is standard-library only. Fitted scikit-learn estimators are
exported ahead of time to model_bundle.json so the Minerva vLLM environment
does not need pandas, joblib, or scikit-learn.
"""

import argparse
import itertools
import json
import math
import struct
from pathlib import Path


LOG_LATENCY_CAP = 15.0
NODE_GROUPS = {"l40s": "neptune", "l4": "ganymede"}


def tree_predict(tree, row):
    node = 0
    while tree["children_left"][node] != tree["children_right"][node]:
        feature = tree["feature"][node]
        # sklearn's tree runtime evaluates features as float32. Preserving that
        # cast matters for values that sit exactly on a learned threshold.
        value = struct.unpack("f", struct.pack("f", row[feature]))[0]
        if value <= tree["threshold"][node]:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]
    return tree["value"][node]


def gbdt_raw(model, row):
    return model["init_raw"] + model["learning_rate"] * sum(
        tree_predict(tree, row) for tree in model["trees"]
    )


def scaled(row, scaler):
    return [
        (value - mean) / scale
        for value, mean, scale in zip(row, scaler["mean"], scaler["scale"])
    ]


def polynomial(row, powers):
    return [
        math.prod(value ** exponent for value, exponent in zip(row, term))
        for term in powers
    ]


def ridge_predict(model, row):
    return model["intercept"] + sum(
        coefficient * value for coefficient, value in zip(model["coef"], row)
    )


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class PoolScheduler:
    def __init__(self, gpu_type, bundle):
        self.gpu_type = gpu_type
        self.pool = bundle["pools"][gpu_type]
        self.config = self.pool["config"]

    @property
    def frequencies(self):
        return [int(freq) for freq in self.config["FREQUENCIES"]]

    @property
    def slos(self):
        return [int(slo) for slo in self.config["SLO_THRESHOLDS"]]

    def features(self, il, ol, tp, freq, rate):
        log_il = math.log1p(il)
        log_ol = math.log1p(ol)
        freq_norm = freq / float(self.config["MAX_FREQ"])
        decode_frac = ol / float(il + ol)
        capacity = self.pool["capacity"]
        cap_values = {
            "log_il": log_il,
            "log_ol": log_ol,
            "tp": tp,
            "freq_norm": freq_norm,
            "decode_frac": decode_frac,
        }
        cap_row = [cap_values[name] for name in capacity["features"]]
        raw_capacity = math.exp(gbdt_raw(capacity["model"], cap_row))
        plateau_key = f"{il}:{ol}:{tp}:{freq}"
        is_plateau = bool(capacity["plateau"].get(plateau_key, False))
        cap_hat = raw_capacity if is_plateau else float(self.config["ETA"]) * raw_capacity

        demand_prefill = rate * il
        demand_decode = rate * ol
        gamma = float(self.config["GAMMA"])
        total_demand = demand_prefill + gamma * demand_decode
        rho = total_demand / cap_hat
        rho_prefill = demand_prefill / cap_hat
        rho_decode = gamma * demand_decode / cap_hat
        prefill_frac = demand_prefill / total_demand
        return {
            "log_il": log_il,
            "log_ol": log_ol,
            "tp": tp,
            "freq_norm": freq_norm,
            "decode_frac": decode_frac,
            "log_rate": math.log1p(rate),
            "log_rho": math.log1p(rho),
            "log_rho_total": math.log1p(rho),
            "rho_sq": rho ** 2,
            "rho_overflow": max(0.0, rho - 1.0),
            "rho_total_overflow": max(0.0, rho - 1.0),
            "log_d_prefill": math.log1p(demand_prefill),
            "log_d_decode": math.log1p(demand_decode),
            "log_kv_pressure": math.log1p(rate * il * ol),
            "prefill_frac": prefill_frac,
            "rho_prefill": rho_prefill,
            "rho_decode": rho_decode,
            "log_rho_prefill": math.log1p(rho_prefill),
            "log_rho_decode": math.log1p(rho_decode),
            "rho_prefill_sq": rho_prefill ** 2,
            "rho_decode_sq": rho_decode ** 2,
            "rho_prefill_overflow": max(0.0, rho_prefill - 1.0),
            "rho_decode_overflow": max(0.0, rho_decode - 1.0),
            "rho": rho,
            "is_plateau": is_plateau,
        }

    def predict(self, phase, il, ol, tp, freq, rate, slo):
        if slo not in self.slos:
            raise ValueError(f"SLO {slo} is unsupported by {self.gpu_type}: {self.slos}")
        feats = self.features(il, ol, tp, freq, rate)
        spec = self.pool["phases"][phase]
        key = str(slo)

        classifier = spec["classifiers"][key]
        if classifier["kind"] == "constant":
            p_violate = classifier["p_violate"]
        else:
            clf_row = [feats[name] for name in spec["clf_features"]]
            p_violate = sigmoid(gbdt_raw(classifier, scaled(clf_row, spec["scalers"][key])))
        phase_targets = self.config.get("PHASE_MODEL_TARGETS", {}).get(phase, {})
        rho_name = phase_targets.get("rho_col", "rho_prefill" if phase == "prefill" else "rho_decode")
        rho_phase = feats[rho_name]
        phase_guards = self.config.get("PHASE_GUARD_SETTINGS", {}).get(phase, {})
        guard = phase_guards.get(key, self.config["GUARD_SETTINGS"][key])
        is_safe = p_violate < float(guard["p_th"]) and rho_phase < float(guard["rho_th"])

        latency_row = [feats[name] for name in spec["reg_features"]]
        latency_scaled = scaled(latency_row, spec["latency_scaler"])
        latency_poly = polynomial(latency_scaled, spec["latency_poly_powers"])
        latency_log = min(ridge_predict(spec["latency_model"], latency_poly), LOG_LATENCY_CAP)
        latency = math.expm1(latency_log)

        power_row = [feats[name] for name in spec["power_features"]]
        power_per_gpu = gbdt_raw(spec["power_model"], power_row)
        result = {
            "phase": phase,
            "gpu_type": self.gpu_type,
            "node_group": NODE_GROUPS[self.gpu_type],
            "tp": tp,
            "freq_mhz": freq,
            "rec_freq_mhz": freq,
            "is_safe": bool(is_safe),
            "p_violate": round(p_violate, 4),
            "rho_phase": round(rho_phase, 4),
            "rho_total": round(feats["rho"], 4),
            "latency_ms": round(latency, 1),
            "power_per_gpu_w": round(power_per_gpu, 1),
            "total_power_w": round(power_per_gpu * tp, 1),
        }
        result["p99_ttft_ms" if phase == "prefill" else "p99_tpot_ms"] = round(latency, 1)
        return result


class PDPlacementScheduler:
    def __init__(self, bundle_path):
        bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
        if bundle.get("format") != "sweep-llm-portable-model-v1":
            raise ValueError("Unsupported model bundle")
        self.pools = {gpu: PoolScheduler(gpu, bundle) for gpu in ("l4", "l40s")}
        self.common_slos = sorted(set(self.pools["l4"].slos) & set(self.pools["l40s"].slos))

    def recommend(self, il, ol, rate, slo_ttft, slo_tpot, tp=1):
        if tp != 1:
            raise ValueError("This job allocates one GPU per node, so TP must be 1")
        if slo_ttft not in self.common_slos or slo_tpot not in self.common_slos:
            raise ValueError(f"Cross-pool SLO must be one of {self.common_slos}")

        predictions = {}
        for gpu, pool in self.pools.items():
            predictions[(gpu, "prefill")] = [
                pool.predict("prefill", il, ol, tp, freq, rate, slo_ttft)
                for freq in pool.frequencies
            ]
            predictions[(gpu, "decode")] = [
                pool.predict("decode", il, ol, tp, freq, rate, slo_tpot)
                for freq in pool.frequencies
            ]

        candidates = []
        for prefill_gpu, decode_gpu in (("l40s", "l4"), ("l4", "l40s")):
            for prefill, decode in itertools.product(
                predictions[(prefill_gpu, "prefill")], predictions[(decode_gpu, "decode")]
            ):
                cluster_power = prefill["total_power_w"] + decode["total_power_w"]
                candidates.append({
                    "prefill": prefill,
                    "decode": decode,
                    "is_safe": prefill["is_safe"] and decode["is_safe"],
                    "predicted_cluster_power_w": round(cluster_power, 1),
                    "predicted_energy_per_request_j": round(cluster_power / rate, 3),
                })

        safe = [candidate for candidate in candidates if candidate["is_safe"]]
        if not safe:
            return {
                "status": "NO_SAFE_CONFIG",
                "workload": {"il": il, "ol": ol, "rate": rate},
                "slos": {"ttft_ms": slo_ttft, "tpot_ms": slo_tpot},
                "num_candidates": len(candidates),
                "num_safe": 0,
            }
        safe.sort(key=lambda item: (item["predicted_cluster_power_w"],
                                    item["prefill"]["latency_ms"] + item["decode"]["latency_ms"]))
        return {
            "status": "OK",
            "workload": {"il": il, "ol": ol, "rate": rate},
            "slos": {"ttft_ms": slo_ttft, "tpot_ms": slo_tpot},
            "tp_per_role": tp,
            "recommended": safe[0],
            "alternatives": safe[1:4],
            "num_candidates": len(candidates),
            "num_safe": len(safe),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path(__file__).with_name("model_bundle.json"))
    parser.add_argument("--il", type=int, required=True)
    parser.add_argument("--ol", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--slo-ttft", type=int, required=True)
    parser.add_argument("--slo-tpot", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scheduler = PDPlacementScheduler(args.bundle)
    result = scheduler.recommend(
        args.il, args.ol, args.rate, args.slo_ttft, args.slo_tpot, args.tp
    )
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["status"] == "OK" else 2)


if __name__ == "__main__":
    main()
