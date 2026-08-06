#!/usr/bin/env python3
"""Validate and summarize the fixed Uranus/Ganymede TP2 auto-DVFS run."""

import argparse
import csv
import json
import re
from pathlib import Path


PATTERNS = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "request_throughput_rps": (
        r"Request throughput \(req/s\):\s+([0-9.]+)",
        float,
    ),
    "output_token_throughput_tps": (
        r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        float,
    ),
    "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([0-9.]+)", float),
    "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
    "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_itl_ms": (r"P99 ITL \(ms\):\s+([0-9.]+)", float),
}


def parse_bench(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, (pattern, cast) in PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            result[name] = cast(match.group(1))
    return result


def allocation_gpu_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return len(
            {
                row.get("gpu_uuid", "").strip()
                for row in csv.DictReader(handle)
                if row.get("gpu_uuid", "").strip()
            }
        )


def telemetry_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--expected-workloads", type=int, default=4)
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=200.0)
    args = parser.parse_args()

    with args.workloads.open(newline="", encoding="utf-8") as handle:
        workloads = list(csv.DictReader(handle))
    registry = json.loads(
        (args.out_dir / "registry.json").read_text(encoding="utf-8")
    )
    prefill_log = (args.out_dir / "prefill_primary_step.log").read_text(
        encoding="utf-8", errors="replace"
    )
    decode_log = (args.out_dir / "decode_primary_step.log").read_text(
        encoding="utf-8", errors="replace"
    )

    prefill_telemetry = args.out_dir / "prefill_uranus_telemetry.csv"
    decode_telemetry = args.out_dir / "decode_ganymede_telemetry.csv"
    uranus_power = args.out_dir / "allocation_uranus_power.csv"
    ganymede_power = args.out_dir / "allocation_ganymede_power.csv"

    rows = []
    for seq, workload in enumerate(workloads, start=1):
        workload_id = workload["workload_id"]
        path = args.out_dir / f"bench_{seq}_{workload_id}.txt"
        metrics = parse_bench(path) if path.is_file() else {}
        row = {
            "seq": seq,
            "workload_id": workload_id,
            "input_len": int(workload["input_len"]),
            "output_len": int(workload["output_len"]),
            "configured_request_rate_rps": float(workload["request_rate"]),
            **metrics,
        }
        row["metrics_complete"] = all(name in metrics for name in PATTERNS)
        row["ttft_slo_ok"] = metrics.get("p99_ttft_ms", float("inf")) <= args.slo_ttft_ms
        row["tpot_slo_ok"] = metrics.get("p99_tpot_ms", float("inf")) <= args.slo_tpot_ms
        rows.append(row)

    checks = {
        "workload_count_ok": len(workloads) == args.expected_workloads,
        "requests_ok": all(
            row.get("successful_requests", 0) > 0
            and row.get("failed_requests") == 0
            for row in rows
        ),
        "metrics_complete": all(row["metrics_complete"] for row in rows),
        "registry_ok": (
            len(registry.get("prefill", [])) == 1
            and len(registry.get("decode", [])) == 1
            and registry.get("prefill_tp_sizes") == [2]
            and registry.get("decode_tp_sizes") == [2]
        ),
        "placement_ok": (
            "host=uranus node_group=uranus role=prefill" in prefill_log
            and "tensor_parallel_size=2" in prefill_log
            and "gpu_clock_control_mode=auto" in prefill_log
            and "host=ganymede node_group=ganymede role=decode" in decode_log
            and "tensor_parallel_size=2" in decode_log
            and "gpu_clock_control_mode=auto" in decode_log
        ),
        "telemetry_ok": (
            prefill_telemetry.is_file()
            and decode_telemetry.is_file()
            and telemetry_rows(prefill_telemetry) > 0
            and telemetry_rows(decode_telemetry) > 0
        ),
        "allocation_power_ok": (
            uranus_power.is_file()
            and ganymede_power.is_file()
            and allocation_gpu_count(uranus_power) == 2
            and allocation_gpu_count(ganymede_power) == 2
        ),
    }
    checks["integrity_ok"] = all(checks.values())
    checks["all_slo_ok"] = all(
        row["ttft_slo_ok"] and row["tpot_slo_ok"] for row in rows
    )

    report = {
        "topology": {
            "prefill": {"node": "uranus", "gpu": "L40S", "count": 2, "tp": 2},
            "decode": {"node": "ganymede", "gpu": "L4", "count": 2, "tp": 2},
            "frequency_mode": "auto_dvfs",
        },
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
        "checks": checks,
        "workloads": rows,
    }
    (args.out_dir / "tp2_auto_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fields = sorted({field for row in rows for field in row})
    with (args.out_dir / "tp2_auto_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"result workload={row['workload_id']} success={row.get('successful_requests', 'NA')} "
            f"failed={row.get('failed_requests', 'NA')} "
            f"p99_ttft_ms={row.get('p99_ttft_ms', 'NA')} "
            f"p99_tpot_ms={row.get('p99_tpot_ms', 'NA')}"
        )
    print(
        f"integrity={'PASS' if checks['integrity_ok'] else 'FAIL'} "
        f"all_slo={'PASS' if checks['all_slo_ok'] else 'VIOLATION'}"
    )
    return 0 if checks["integrity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
