#!/usr/bin/env python3
"""Choose one L40S clock for a TP=4 single-pool vLLM server per workload."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path


def load_scheduler(path: Path):
    spec = importlib.util.spec_from_file_location("sweep_portable_scheduler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scheduler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-type", choices=("l4", "l40s"), default="l40s")
    parser.add_argument("--node", default="uranus")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--slo-ttft-ms", type=int, default=500)
    parser.add_argument("--slo-tpot-ms", type=int, default=200)
    args = parser.parse_args()

    module = load_scheduler(args.scheduler)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    pool = module.PoolScheduler(args.gpu_type, bundle)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    decisions = []
    with args.workloads.open(newline="", encoding="utf-8") as handle:
        workloads = list(csv.DictReader(handle))
    for seq, workload in enumerate(workloads, 1):
        il = int(workload["input_len"])
        ol = int(workload["output_len"])
        rate = float(workload["request_rate"])
        candidates = []
        for frequency in pool.frequencies:
            prefill = pool.predict(
                "prefill", il, ol, args.tp, frequency, rate, args.slo_ttft_ms
            )
            decode = pool.predict(
                "decode", il, ol, args.tp, frequency, rate, args.slo_tpot_ms
            )
            point_prediction_slo_ok = (
                prefill["p99_ttft_ms"] <= args.slo_ttft_ms
                and decode["p99_tpot_ms"] <= args.slo_tpot_ms
            )
            latency_policy_safe = (
                point_prediction_slo_ok
                and prefill["is_safe"]
                and decode["is_safe"]
            )
            candidates.append(
                {
                    "frequency_mhz": frequency,
                    "predicted_p99_ttft_ms": prefill["p99_ttft_ms"],
                    "predicted_p99_tpot_ms": decode["p99_tpot_ms"],
                    "point_prediction_slo_ok": point_prediction_slo_ok,
                    "latency_policy_safe": latency_policy_safe,
                    "prefill_model_is_safe": prefill["is_safe"],
                    "decode_model_is_safe": decode["is_safe"],
                    "predicted_power_proxy_w": max(
                        prefill["total_power_w"], decode["total_power_w"]
                    ),
                    "prefill": prefill,
                    "decode": decode,
                }
            )
        safe = [item for item in candidates if item["latency_policy_safe"]]
        if safe:
            selected = min(safe, key=lambda item: item["frequency_mhz"])
            mode = "lowest_frequency_meeting_predicted_slo"
        else:
            selected = min(
                candidates,
                key=lambda item: (
                    max(
                        item["predicted_p99_ttft_ms"] / args.slo_ttft_ms,
                        item["predicted_p99_tpot_ms"] / args.slo_tpot_ms,
                    ),
                    item["frequency_mhz"],
                ),
            )
            mode = "minimum_predicted_slo_violation"
        decision = {
            "seq": seq,
            "workload_id": workload["workload_id"],
            "topology": "single_pool",
            "gpu_type": args.gpu_type,
            "node": args.node,
            "tensor_parallel_size": args.tp,
            "input_len": il,
            "output_len": ol,
            "request_rate_rps": rate,
            "slo_ttft_ms": args.slo_ttft_ms,
            "slo_tpot_ms": args.slo_tpot_ms,
            "decision_mode": mode,
            "selected_frequency_mhz": selected["frequency_mhz"],
            "predicted_p99_ttft_ms": selected["predicted_p99_ttft_ms"],
            "predicted_p99_tpot_ms": selected["predicted_p99_tpot_ms"],
            "point_prediction_slo_ok": selected["point_prediction_slo_ok"],
            "predicted_slo_ok": selected["latency_policy_safe"],
            "predicted_power_proxy_w": selected["predicted_power_proxy_w"],
            "selection_note": (
                "TTFT and TPOT use the same TP=4 L40S frequency. The latency-only "
                "policy also applies the fitted phase feasibility guards. No PD "
                "KV-transfer term is added. Power is a ranking proxy because both "
                "phase models refer to the same physical four-GPU pool."
            ),
            "selected_candidate": selected,
            "candidates": candidates,
        }
        decisions.append(decision)
        (args.out_dir / f"decision_{seq}_{workload['workload_id']}.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    fields = [
        "seq", "workload_id", "input_len", "output_len", "request_rate_rps",
        "selected_frequency_mhz", "predicted_p99_ttft_ms",
        "predicted_p99_tpot_ms", "predicted_slo_ok",
        "predicted_power_proxy_w", "decision_mode",
    ]
    with (args.out_dir / "decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in decisions)
    (args.out_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
