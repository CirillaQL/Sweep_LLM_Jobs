#!/usr/bin/env python3
"""Per-route SLO-aware DVFS prediction for the custom PD proxy.

The fitted model implementation is loaded from the earlier portable SWEEP
scheduler.  Prefill and Decode are evaluated with their own TP degree so NIXL
routes with asymmetric TP are modeled correctly.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any


class RequestDVFSPredictor:
    def __init__(
        self,
        scheduler_script: str,
        model_bundle: str,
        saturation_bundle: str,
        *,
        saturation_threshold: float | None = None,
        kv_effective_bandwidth_gbps: float | None = None,
        dispatch_ms: float = 0.0,
    ) -> None:
        spec = importlib.util.spec_from_file_location(
            "portable_sweep_scheduler", scheduler_script
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load scheduler module: {scheduler_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bundle = json.loads(Path(model_bundle).read_text(encoding="utf-8"))
        self.module = module
        self.pools = {
            gpu: module.PoolScheduler(gpu, bundle) for gpu in ("l40s", "l4")
        }
        self.saturation = module.SaturationEnsemble(
            saturation_bundle, saturation_threshold
        )
        self.kv_effective_bandwidth_gbps = kv_effective_bandwidth_gbps
        self.dispatch_ms = dispatch_ms

    @staticmethod
    def _model_slo(pool: Any, requested_ms: int) -> int:
        # A model trained for a tighter SLO is conservative for the requested
        # SLO.  Reject values below the smallest calibrated threshold.
        eligible = [value for value in pool.slos if value <= requested_ms]
        if not eligible:
            raise ValueError(
                f"requested SLO {requested_ms} ms is below calibrated values "
                f"{pool.slos}"
            )
        return max(eligible)

    def _phase_predictions(
        self,
        *,
        phase: str,
        gpu_type: str,
        tp: int,
        il: int,
        ol: int,
        rate: float,
        requested_slo_ms: int,
        kv_transfer: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pool = self.pools[gpu_type]
        if tp not in pool.config["TP_DEGREES"]:
            raise ValueError(
                f"TP={tp} is outside the {gpu_type} model grid "
                f"{pool.config['TP_DEGREES']}"
            )
        model_slo = self._model_slo(pool, requested_slo_ms)
        predictions = []
        for freq in pool.frequencies:
            item = pool.predict(phase, il, ol, tp, freq, rate, model_slo)
            item.update(self.saturation.predict(gpu_type, il, ol, tp, freq, rate))
            item["requested_slo_ms"] = requested_slo_ms
            item["model_slo_ms"] = model_slo
            item["effective_request_rate_per_instance"] = round(rate, 6)
            if phase == "prefill":
                base = item["p99_ttft_ms"]
                item["p99_queue_plus_prefill_ms"] = base
                item["kv_transfer_ms"] = kv_transfer["kv_transfer_ms"]
                item["dispatch_ms"] = kv_transfer["dispatch_ms"]
                item["p99_ttft_ms"] = round(
                    base + kv_transfer["kv_transfer_ms"] + kv_transfer["dispatch_ms"],
                    1,
                )
                item["is_safe"] = bool(
                    item["is_safe"] and item["p99_ttft_ms"] <= requested_slo_ms
                )
            else:
                item["is_safe"] = bool(
                    item["is_safe"] and item["p99_tpot_ms"] <= requested_slo_ms
                )
            predictions.append(item)
        return predictions

    def recommend(
        self,
        *,
        il: int,
        ol: int,
        prefill_rate: float,
        decode_rate: float,
        slo_ttft_ms: int,
        slo_tpot_ms: int,
        prefill_gpu: str,
        decode_gpu: str,
        prefill_tp: int,
        decode_tp: int,
        overload_action: str = "reject",
    ) -> dict[str, Any]:
        if il <= 0 or ol <= 0:
            raise ValueError(f"token lengths must be positive, got il={il} ol={ol}")
        if prefill_rate <= 0 or decode_rate <= 0:
            raise ValueError("request rates must be positive")
        if overload_action not in {"reject", "min-slo-violation"}:
            raise ValueError(f"unsupported overload action: {overload_action}")
        kv = self.module.kv_transfer_estimate(
            il,
            self.kv_effective_bandwidth_gbps,
            self.module.DEFAULT_KV_NUM_LAYERS,
            self.module.DEFAULT_KV_NUM_HEADS,
            self.module.DEFAULT_KV_HEAD_DIM,
            self.module.DEFAULT_KV_BYTES_PER_ELEMENT,
            self.dispatch_ms,
        )
        prefill = self._phase_predictions(
            phase="prefill", gpu_type=prefill_gpu, tp=prefill_tp,
            il=il, ol=ol, rate=prefill_rate,
            requested_slo_ms=slo_ttft_ms, kv_transfer=kv,
        )
        decode = self._phase_predictions(
            phase="decode", gpu_type=decode_gpu, tp=decode_tp,
            il=il, ol=ol, rate=decode_rate,
            requested_slo_ms=slo_tpot_ms, kv_transfer=kv,
        )
        candidates = []
        effective_rate = max(prefill_rate, decode_rate)
        for p_item, d_item in itertools.product(prefill, decode):
            latency_safe = bool(p_item["is_safe"] and d_item["is_safe"])
            saturation_safe = bool(
                p_item["saturation_safe"] and d_item["saturation_safe"]
            )
            joint_latency = self.module.union_probability(
                p_item["p_violate"], d_item["p_violate"]
            )
            joint_saturation = self.module.union_probability(
                p_item["p_saturated"], d_item["p_saturated"]
            )
            joint_overload = self.module.union_probability(
                joint_latency, joint_saturation
            )
            ttft_excess = max(
                0.0, p_item["p99_ttft_ms"] / float(slo_ttft_ms) - 1.0
            )
            tpot_excess = max(
                0.0, d_item["p99_tpot_ms"] / float(slo_tpot_ms) - 1.0
            )
            power = p_item["total_power_w"] + d_item["total_power_w"]
            candidates.append({
                "prefill": p_item,
                "decode": d_item,
                "latency_safe": latency_safe,
                "saturation_safe": saturation_safe,
                "is_safe": latency_safe and saturation_safe,
                "predicted_joint_latency_violation_probability": round(joint_latency, 6),
                "predicted_joint_saturation_probability": round(joint_saturation, 6),
                "predicted_overload_violation_probability": round(joint_overload, 6),
                "predicted_max_slo_excess_ratio": round(max(ttft_excess, tpot_excess), 6),
                "predicted_cluster_power_w": round(power, 1),
                "predicted_energy_per_request_j": round(power / effective_rate, 3),
            })
        safe = [item for item in candidates if item["is_safe"]]
        common = {
            "workload": {
                "il": il, "ol": ol,
                "prefill_rate": prefill_rate, "decode_rate": decode_rate,
            },
            "slos": {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms},
            "topology": {
                "prefill_gpu": prefill_gpu, "prefill_tp": prefill_tp,
                "decode_gpu": decode_gpu, "decode_tp": decode_tp,
            },
            "kv_transfer_model": kv,
            "saturation_threshold": self.saturation.threshold,
            "num_candidates": len(candidates),
            "num_latency_safe": sum(item["latency_safe"] for item in candidates),
            "num_saturation_safe": sum(item["saturation_safe"] for item in candidates),
            "num_safe": len(safe),
        }
        if safe:
            safe.sort(key=lambda item: (
                item["predicted_cluster_power_w"],
                item["prefill"]["latency_ms"] + item["decode"]["latency_ms"],
            ))
            return {
                "status": "OK",
                "decision_mode": "safe_min_power",
                "recommended": safe[0],
                "alternatives": safe[1:4],
                **common,
            }
        if overload_action == "reject":
            return {"status": "NO_SAFE_CONFIG", "decision_mode": "reject", **common}
        candidates.sort(key=lambda item: (
            item["predicted_overload_violation_probability"],
            item["predicted_max_slo_excess_ratio"],
            item["predicted_cluster_power_w"],
        ))
        return {
            "status": "OVERLOAD_FALLBACK",
            "decision_mode": "overload_min_slo_violation",
            "recommended": candidates[0],
            "alternatives": candidates[1:4],
            **common,
        }
