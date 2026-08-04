#!/usr/bin/env python3
"""Validate a fixed TP=4/TP=4 PD run and summarize latency and throughput."""

import argparse
import csv
import json
import re
from pathlib import Path


PATTERNS = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "benchmark_duration_s": (r"Benchmark duration \(s\):\s+([0-9.]+)", float),
    "request_throughput_rps": (r"Request throughput \(req/s\):\s+([0-9.]+)", float),
    "output_token_throughput_tps": (
        r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        float,
    ),
    "total_token_throughput_tps": (
        r"Total token throughput \(tok/s\):\s+([0-9.]+)",
        float,
    ),
    "peak_concurrent_requests": (r"Peak concurrent requests:\s+([0-9.]+)", float),
    "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([0-9.]+)", float),
    "median_ttft_ms": (r"Median TTFT \(ms\):\s+([0-9.]+)", float),
    "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
    "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([0-9.]+)", float),
    "median_tpot_ms": (r"Median TPOT \(ms\):\s+([0-9.]+)", float),
    "p99_tpot_ms": (r"P99 TPOT \(ms\):\s+([0-9.]+)", float),
    "mean_itl_ms": (r"Mean ITL \(ms\):\s+([0-9.]+)", float),
    "median_itl_ms": (r"Median ITL \(ms\):\s+([0-9.]+)", float),
    "p99_itl_ms": (r"P99 ITL \(ms\):\s+([0-9.]+)", float),
}

CSV_FIELDS = [
    "seq",
    "workload_id",
    "input_len",
    "output_len",
    "configured_request_rate_rps",
    *PATTERNS,
    "ttft_slo_ok",
    "tpot_slo_ok",
    "metrics_complete",
]


def parse_bench(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, (pattern, cast) in PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            result[name] = cast(match.group(1))
    return result


def load_workloads(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_registry(path: Path) -> tuple[int, int, list[int], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        len(data.get("prefill", [])),
        len(data.get("decode", [])),
        [int(value) for value in data.get("prefill_tp_sizes", [])],
        [int(value) for value in data.get("decode_tp_sizes", [])],
    )


def telemetry_data_rows(paths: list[Path]) -> list[int]:
    counts = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            counts.append(sum(1 for _ in csv.DictReader(handle)))
    return counts


def allocation_gpu_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return len(
            {
                row.get("gpu_uuid", "").strip()
                for row in csv.DictReader(handle)
                if row.get("gpu_uuid", "").strip()
            }
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(path: Path, report: dict) -> None:
    check = report["checks"]
    lines = [
        "# 8192-token fixed TP=4/TP=4 latency report",
        "",
        (
            f"Integrity: **{'PASS' if check['integrity_ok'] else 'FAIL'}**; "
            f"all latency SLOs: **{'PASS' if check['all_slo_ok'] else 'VIOLATION'}**."
        ),
        "",
        (
            f"Observed topology: {report['observed_topology']['registry_prefill_instances']} "
            f"Prefill at TP={report['observed_topology']['prefill_tp']} / "
            f"{report['observed_topology']['registry_decode_instances']} Decode at "
            f"TP={report['observed_topology']['decode_tp']}; allocation GPUs: "
            f"{report['observed_topology']['neptune_allocation_gpus']} / "
            f"{report['observed_topology']['ganymede_allocation_gpus']}."
        ),
        "",
        "| workload | in/out | rate | success/fail | mean TTFT | p99 TTFT | mean TPOT | p99 TPOT | p99 ITL | req/s | SLO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["workloads"]:
        slo = "PASS" if row["ttft_slo_ok"] and row["tpot_slo_ok"] else "VIOLATION"
        lines.append(
            f"| {row['workload_id']} | {row['input_len']}/{row['output_len']} | "
            f"{cell(row['configured_request_rate_rps'])} | "
            f"{cell(row.get('successful_requests', 'NA'))}/"
            f"{cell(row.get('failed_requests', 'NA'))} | "
            f"{cell(row.get('mean_ttft_ms', 'NA'))} ms | "
            f"{cell(row.get('p99_ttft_ms', 'NA'))} ms | "
            f"{cell(row.get('mean_tpot_ms', 'NA'))} ms | "
            f"{cell(row.get('p99_tpot_ms', 'NA'))} ms | "
            f"{cell(row.get('p99_itl_ms', 'NA'))} ms | "
            f"{cell(row.get('request_throughput_rps', 'NA'))} | {slo} |"
        )
    lines.extend(
        [
            "",
            f"TTFT SLO: {report['slo']['ttft_ms']:.0f} ms; TPOT SLO: {report['slo']['tpot_ms']:.0f} ms.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--expected-workloads", type=int, required=True)
    parser.add_argument("--expected-input-len", type=int, default=8192)
    parser.add_argument("--expected-prefill-instances", type=int, default=1)
    parser.add_argument("--expected-decode-instances", type=int, default=1)
    parser.add_argument("--expected-prefill-tp", type=int, default=4)
    parser.add_argument("--expected-decode-tp", type=int, default=4)
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=200.0)
    parser.add_argument("--fail-on-slo", action="store_true")
    args = parser.parse_args()

    workloads = load_workloads(args.workloads)
    live_summary_path = args.out_dir / "live_summary.json"
    registry_path = args.out_dir / "registry.json"
    live_summary = json.loads(live_summary_path.read_text(encoding="utf-8"))
    live_ids = {
        item.get("workload", {}).get("workload_id")
        for item in live_summary.get("workloads", [])
    }
    (
        registry_prefill,
        registry_decode,
        registry_prefill_tp_sizes,
        registry_decode_tp_sizes,
    ) = load_registry(registry_path)
    prefill_paths = list(args.out_dir.glob("prefill_neptune*_telemetry.csv"))
    decode_paths = list(args.out_dir.glob("decode_ganymede*_telemetry.csv"))
    prefill_samples = telemetry_data_rows(prefill_paths)
    decode_samples = telemetry_data_rows(decode_paths)
    prefill_telemetry = len(prefill_paths)
    decode_telemetry = len(decode_paths)
    neptune_allocation_gpus = allocation_gpu_count(
        args.out_dir / "allocation_neptune_power.csv"
    )
    ganymede_allocation_gpus = allocation_gpu_count(
        args.out_dir / "allocation_ganymede_power.csv"
    )
    prefill_step = (args.out_dir / "prefill_primary_step.log").read_text(
        encoding="utf-8", errors="replace"
    )
    decode_step = (args.out_dir / "decode_primary_step.log").read_text(
        encoding="utf-8", errors="replace"
    )
    observed_prefill_tp = args.expected_prefill_tp if re.search(
        rf"tensor_parallel_size={args.expected_prefill_tp}\b", prefill_step
    ) else None
    observed_decode_tp = args.expected_decode_tp if re.search(
        rf"tensor_parallel_size={args.expected_decode_tp}\b", decode_step
    ) else None

    rows = []
    for seq, workload in enumerate(workloads, start=1):
        workload_id = workload["workload_id"]
        bench_path = args.out_dir / f"bench_{seq}_{workload_id}.txt"
        metrics = parse_bench(bench_path) if bench_path.is_file() else {}
        row = {
            "seq": seq,
            "workload_id": workload_id,
            "input_len": int(workload["input_len"]),
            "output_len": int(workload["output_len"]),
            "configured_request_rate_rps": float(workload["request_rate"]),
            **metrics,
        }
        row["metrics_complete"] = all(name in row for name in PATTERNS)
        row["ttft_slo_ok"] = (
            row.get("p99_ttft_ms", float("inf")) <= args.slo_ttft_ms
        )
        row["tpot_slo_ok"] = (
            row.get("p99_tpot_ms", float("inf")) <= args.slo_tpot_ms
        )
        rows.append(row)

    workload_count_ok = len(workloads) == args.expected_workloads
    input_lengths_ok = all(
        row["input_len"] == args.expected_input_len for row in rows
    )
    live_summary_complete = len(live_ids) == len(rows) and all(
        row["workload_id"] in live_ids for row in rows
    )
    requests_ok = all(
        row.get("successful_requests", 0) > 0 and row.get("failed_requests") == 0
        for row in rows
    )
    metrics_complete = all(row["metrics_complete"] for row in rows)
    topology_ok = (
        registry_prefill == args.expected_prefill_instances
        and registry_decode == args.expected_decode_instances
        and registry_prefill_tp_sizes == [args.expected_prefill_tp]
        and registry_decode_tp_sizes == [args.expected_decode_tp]
        and observed_prefill_tp == args.expected_prefill_tp
        and observed_decode_tp == args.expected_decode_tp
        and neptune_allocation_gpus == args.expected_prefill_tp
        and ganymede_allocation_gpus == args.expected_decode_tp
        and prefill_telemetry == 1
        and decode_telemetry == 1
        and min(prefill_samples, default=0) > 0
        and min(decode_samples, default=0) > 0
    )
    all_slo_ok = all(row["ttft_slo_ok"] and row["tpot_slo_ok"] for row in rows)
    integrity_ok = all(
        (
            workload_count_ok,
            input_lengths_ok,
            live_summary_complete,
            requests_ok,
            metrics_complete,
            topology_ok,
        )
    )

    report = {
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
        "expected_topology": {
            "prefill_node": "neptune",
            "prefill_gpu": "L40S",
            "prefill_instances": args.expected_prefill_instances,
            "prefill_tp": args.expected_prefill_tp,
            "decode_node": "ganymede",
            "decode_gpu": "L4",
            "decode_instances": args.expected_decode_instances,
            "decode_tp": args.expected_decode_tp,
        },
        "observed_topology": {
            "registry_prefill_instances": registry_prefill,
            "registry_decode_instances": registry_decode,
            "registry_prefill_tp_sizes": registry_prefill_tp_sizes,
            "registry_decode_tp_sizes": registry_decode_tp_sizes,
            "prefill_tp": observed_prefill_tp,
            "decode_tp": observed_decode_tp,
            "neptune_allocation_gpus": neptune_allocation_gpus,
            "ganymede_allocation_gpus": ganymede_allocation_gpus,
            "prefill_telemetry_files": prefill_telemetry,
            "decode_telemetry_files": decode_telemetry,
            "min_prefill_telemetry_samples": min(prefill_samples, default=0),
            "min_decode_telemetry_samples": min(decode_samples, default=0),
        },
        "checks": {
            "workload_count_ok": workload_count_ok,
            "input_lengths_ok": input_lengths_ok,
            "live_summary_complete": live_summary_complete,
            "requests_ok": requests_ok,
            "metrics_complete": metrics_complete,
            "topology_ok": topology_ok,
            "all_slo_ok": all_slo_ok,
            "integrity_ok": integrity_ok,
        },
        "workloads": rows,
    }

    json_path = args.out_dir / "latency_metrics.json"
    csv_path = args.out_dir / "latency_metrics.csv"
    markdown_path = args.out_dir / "latency_metrics.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(markdown_path, report)

    for row in rows:
        print(
            f"latency workload={row['workload_id']} input={row['input_len']} "
            f"output={row['output_len']} success={row.get('successful_requests', 'NA')} "
            f"failed={row.get('failed_requests', 'NA')} "
            f"p99_ttft_ms={row.get('p99_ttft_ms', 'NA')} "
            f"p99_tpot_ms={row.get('p99_tpot_ms', 'NA')} "
            f"p99_itl_ms={row.get('p99_itl_ms', 'NA')}"
        )
    print(
        f"latency_metrics_integrity={'PASS' if integrity_ok else 'FAIL'} "
        f"latency_slo={'PASS' if all_slo_ok else 'VIOLATION'} "
        f"report={markdown_path}"
    )
    if not integrity_ok:
        return 2
    if args.fail_on_slo and not all_slo_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
