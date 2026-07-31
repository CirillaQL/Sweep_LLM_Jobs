#!/usr/bin/env python3
"""Build deterministic DVFS calibration manifests from the Phase2 master CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_FIELDS = (
    "gpu_freq_mhz",
    "mem_freq_mhz",
    "tp_degree",
    "input_len",
    "output_len",
    "request_rate",
)

EXPECTED = {
    "l40s": {
        "filename": "Phase2_Results_L40S_master_results.csv",
        "shards": 7,
        "gpus_per_node": 4,
        "frequencies": [210, 480, 735, 990, 1245, 1500, 1755, 2010, 2265, 2520],
        "tp": [1, 2, 4],
        "unique_configs": 509,
        "mem_freq_mhz": 9001,
    },
    "l4": {
        "filename": "Phase2_Results_L4_master_results.csv",
        "shards": 12,
        "gpus_per_node": 8,
        "frequencies": [210, 360, 570, 780, 990, 1200, 1410, 1620, 1830, 2040],
        "tp": [1, 2, 4, 8],
        "unique_configs": 566,
        "mem_freq_mhz": 6251,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rate(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def load_configs(path: Path, gpu_type: str) -> list[dict]:
    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in CONFIG_FIELDS)
            grouped[key].append(row)

    configs = []
    for key, rows in grouped.items():
        freq, mem, tp, input_len, output_len, request_rate = key
        durations = [float(row["benchmark_duration_s"]) for row in rows if row["benchmark_duration_s"]]
        num_prompts = max(int(float(row["num_prompts"])) for row in rows)
        rate = canonical_rate(request_rate)
        config_id = (
            f"{gpu_type}-tp{int(tp)}-f{int(float(freq))}-"
            f"i{int(input_len)}-o{int(output_len)}-r{str(rate).replace('.', 'p')}"
        )
        median_s = statistics.median(durations)
        configs.append(
            {
                "config_id": config_id,
                "gpu_type": gpu_type,
                "gpu_freq_mhz": int(float(freq)),
                "mem_freq_mhz": int(float(mem)),
                "tp_degree": int(tp),
                "input_len": int(input_len),
                "output_len": int(output_len),
                "request_rate": rate,
                "num_prompts": num_prompts,
                "repeats": 3,
                "source_steps": sorted({row["step"] for row in rows}),
                "source_rows": len(rows),
                "observed_num_prompts": sorted({int(float(row["num_prompts"])) for row in rows}),
                "historical_duration_median_s": round(median_s, 6),
                "historical_duration_max_s": round(max(durations), 6),
                "estimated_total_duration_s": round(3 * median_s, 6),
            }
        )
    return configs


def assign_lpt(configs: list[dict], shard_count: int) -> list[list[dict]]:
    shards: list[list[dict]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    for config in sorted(
        configs,
        key=lambda item: (-item["estimated_total_duration_s"], item["config_id"]),
    ):
        shard = min(range(shard_count), key=lambda idx: (loads[idx], idx))
        config["shard_id"] = shard
        shards[shard].append(config)
        loads[shard] += config["estimated_total_duration_s"]
    for shard in shards:
        shard.sort(
            key=lambda item: (
                item["tp_degree"],
                -item["gpu_freq_mhz"],
                item["input_len"],
                item["output_len"],
                item["request_rate"],
            )
        )
    return shards


def add_segments(configs: list[dict], target_segment_s: float) -> None:
    """Split long logical repetitions without changing total planned prompts."""
    for config in configs:
        count = max(1, math.ceil(config["historical_duration_median_s"] / target_segment_s))
        count = min(count, config["num_prompts"])
        base, extra = divmod(config["num_prompts"], count)
        config["segments"] = [
            {
                "segment_no": index + 1,
                "segment_count": count,
                "num_prompts": base + (1 if index < extra else 0),
                "historical_estimated_duration_s": round(
                    config["historical_duration_median_s"] / count, 6
                ),
            }
            for index in range(count)
        ]


def validate(gpu_type: str, configs: list[dict]) -> None:
    expected = EXPECTED[gpu_type]
    assert len(configs) == expected["unique_configs"], (gpu_type, len(configs))
    assert sorted({item["tp_degree"] for item in configs}) == expected["tp"]
    assert sorted({item["gpu_freq_mhz"] for item in configs}) == expected["frequencies"]
    assert {item["mem_freq_mhz"] for item in configs} == {expected["mem_freq_mhz"]}
    if gpu_type == "l4":
        tp1 = sorted({item["gpu_freq_mhz"] for item in configs if item["tp_degree"] == 1})
        assert tp1 == [210, 360, 570, 1200, 2040]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", default="phase2-full-grid-rerun-20260731")
    parser.add_argument("--target-segment-s", type=float, default=1200.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for gpu_type, expected in EXPECTED.items():
        source = args.source_dir / expected["filename"]
        configs = load_configs(source, gpu_type)
        validate(gpu_type, configs)
        add_segments(configs, args.target_segment_s)
        shards = assign_lpt(configs, expected["shards"])
        shard_summaries = [
            {
                "shard_id": idx,
                "config_count": len(shard),
                "logical_repeat_count": sum(item["repeats"] for item in shard),
                "benchmark_segment_count": sum(
                    item["repeats"] * len(item["segments"]) for item in shard
                ),
                "num_prompts": sum(item["num_prompts"] * item["repeats"] for item in shard),
                "historical_estimated_hours": round(
                    sum(item["estimated_total_duration_s"] for item in shard) / 3600, 4
                ),
            }
            for idx, shard in enumerate(shards)
        ]
        manifest = {
            "schema_version": 1,
            "campaign_id": args.campaign_id,
            "gpu_type": gpu_type,
            "source": {
                "filename": source.name,
                "sha256": sha256(source),
                "selection": "unique hardware/workload key; num_prompts=max observed; three repeats",
            },
            "clock_control": {
                "method": "nvidia-smi -lgc <target>,<target>",
                "memory_clock_policy": "not scanned; record observed clocks.mem only",
                "recorded_historical_mem_freq_mhz": expected["mem_freq_mhz"],
                "reset_method": "nvidia-smi -rgc",
            },
            "gpus_per_node": expected["gpus_per_node"],
            "shard_count": expected["shards"],
            "configuration_count": len(configs),
            "logical_repeat_count": sum(item["repeats"] for item in configs),
            "benchmark_segment_count": sum(
                item["repeats"] * len(item["segments"]) for item in configs
            ),
            "target_segment_s": args.target_segment_s,
            "shards": shards,
            "shard_summaries": shard_summaries,
        }
        output = args.output_dir / f"{gpu_type}_manifest.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries[gpu_type] = {
            "manifest": str(output),
            "configuration_count": len(configs),
            "logical_repeat_count": manifest["logical_repeat_count"],
            "benchmark_segment_count": manifest["benchmark_segment_count"],
            "historical_estimated_hours": round(
                sum(item["estimated_total_duration_s"] for item in configs) / 3600, 4
            ),
            "shards": shard_summaries,
        }
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
