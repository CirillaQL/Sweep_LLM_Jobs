"""Drive and audit a P0/D0-only exhaustive frequency exploration."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from xpyd.phase3c_substrate import load_config


class ExplorationOnlyError(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json_request_sync(
    method: str, url: str,
    *, payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **dict(headers or {})}
    data = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=60.0) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise ExplorationOnlyError("%s %s failed (%d): %s" % (
            method, url, exc.code, text[:500]
        )) from exc
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ExplorationOnlyError("expected JSON object from %s" % url)
    return value


async def _json_request(
    method: str, url: str, *, payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _json_request_sync, method, url, payload=payload, headers=headers
    )


def _summarize(
    config: Mapping[str, Any], run_dir: Path, event_log: Path, table_path: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    events = _read_events(event_log)
    service_log = Path(os.path.expandvars(str(
        config["online_feedback"]["service_request_log"]
    )))
    if service_log.is_file() and service_log.stat().st_size > 0:
        raise ExplorationOnlyError("formal service dispatch log must remain empty")
    if any(str(row.get("event", "")).startswith("service_") for row in events):
        raise ExplorationOnlyError("service event observed in exploration-only run")
    failures = [row for row in events if row.get("event") == "exploration_failed"]
    if failures:
        raise ExplorationOnlyError("exploration failures: %s" % failures)
    if not table_path.is_file():
        raise ExplorationOnlyError("frequency table was not persisted")
    table = json.loads(table_path.read_text(encoding="utf-8"))["entries"]
    workload_ids = [str(row["id"]) for row in config["workloads"]]
    missing = [key for key in workload_ids if table.get(key, {}).get("value") is None]
    if missing:
        raise ExplorationOnlyError("frequency table is incomplete: %s" % missing)

    pattern = re.compile(r"-explore-([PD])-\d+$")
    candidate_rows = []
    for row in events:
        if row.get("event") != "probe_candidate_aggregated":
            continue
        match = pattern.search(str(row.get("probe_id", "")))
        if match is None:
            continue
        candidate_rows.append({
            "workload_id": row["workload_id"],
            "axis": match.group(1),
            "prefill_frequency_mhz": row["prefill_frequency_mhz"],
            "decode_frequency_mhz": row["decode_frequency_mhz"],
            "sample_count": row["sample_count"],
            "slo_met": row["slo_met"],
            "ttft_p95_ms": row["ttft_ms"],
            "tpot_p95_ms": row["tpot_ms"],
            "prefill_power_w": row["prefill_power_w"],
            "decode_power_w": row["decode_power_w"],
            "joint_power_w": row["measured_power_w"],
            "prefill_energy_j": row["prefill_energy_j"],
            "decode_energy_j": row["decode_energy_j"],
            "joint_energy_j": row["measured_energy_j"],
        })

    expected = {"P": 17, "D": 15}
    counts: dict[str, dict[str, int]] = {}
    for workload_id in workload_ids:
        counts[workload_id] = {}
        for axis, expected_count in expected.items():
            rows = [row for row in candidate_rows if row["workload_id"] == workload_id and row["axis"] == axis]
            if len(rows) != expected_count:
                raise ExplorationOnlyError(
                    "%s axis %s has %d candidates, expected %d"
                    % (workload_id, axis, len(rows), expected_count)
                )
            if any(int(row["sample_count"]) != 3 for row in rows):
                raise ExplorationOnlyError("every candidate must have exactly three samples")
            frequency_field = (
                "prefill_frequency_mhz" if axis == "P"
                else "decode_frequency_mhz"
            )
            if len({int(row[frequency_field]) for row in rows}) != expected_count:
                raise ExplorationOnlyError(
                    "%s axis %s candidates are not distinct" % (workload_id, axis)
                )
            counts[workload_id][axis] = len(rows)

    selected_events = [row for row in events if row.get("event") == "axis_selected"]
    best_rows = []
    for workload_id in workload_ids:
        selected = {
            str(row["axis"]): row for row in selected_events
            if row.get("workload_id") == workload_id
        }
        if set(selected) != {"P", "D"}:
            raise ExplorationOnlyError("missing axis selection for %s" % workload_id)
        value = table[workload_id]["value"]
        p_rows = [row for row in candidate_rows if row["workload_id"] == workload_id and row["axis"] == "P"]
        d_rows = [row for row in candidate_rows if row["workload_id"] == workload_id and row["axis"] == "D"]
        if {int(row["decode_frequency_mhz"]) for row in p_rows} != {1500}:
            raise ExplorationOnlyError("P sweep must hold D at safe-high")
        if {int(row["prefill_frequency_mhz"]) for row in d_rows} != {
            int(value["prefill_frequency_mhz"])
        }:
            raise ExplorationOnlyError("D sweep must hold the selected P frequency")
        expected_p = min(
            (row for row in p_rows if row["slo_met"] is True),
            key=lambda row: (float(row["prefill_power_w"]), int(row["prefill_frequency_mhz"])),
        )
        expected_d = min(
            (row for row in d_rows if row["slo_met"] is True),
            key=lambda row: (float(row["decode_power_w"]), int(row["decode_frequency_mhz"])),
        )
        if int(value["prefill_frequency_mhz"]) != int(expected_p["prefill_frequency_mhz"]):
            raise ExplorationOnlyError("persisted P frequency is not the feasible power minimum")
        if int(value["decode_frequency_mhz"]) != int(expected_d["decode_frequency_mhz"]):
            raise ExplorationOnlyError("persisted D frequency is not the feasible power minimum")
        best_rows.append({
            "workload_id": workload_id,
            "prefill_frequency_mhz": value["prefill_frequency_mhz"],
            "decode_frequency_mhz": value["decode_frequency_mhz"],
            "prefill_power_w": selected["P"]["objective_power_w"],
            "decode_power_w": selected["D"]["objective_power_w"],
            "additive_axis_power_w": (
                float(selected["P"]["objective_power_w"])
                + float(selected["D"]["objective_power_w"])
            ),
            "joint_measured_power_w": value["measured_power_w"],
            "joint_measured_energy_j": value["measured_energy_j"],
            "joint_ttft_p95_ms": value["ttft_ms"],
            "joint_tpot_p95_ms": value["tpot_ms"],
            "joint_sample_count": value["sample_count"],
            "slo_met": value["slo_met"],
        })

    with (run_dir / "candidate_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (run_dir / "best_power_configurations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)

    summary = {
        "schema_version": 1,
        "mode": "P0_D0_exploration_only",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workload_count": len(workload_ids),
        "candidate_counts_per_workload": expected,
        "samples_per_candidate": 3,
        "probes_per_workload": 96,
        "total_measured_probes": len(candidate_rows) * 3,
        "slo": config["online_feedback"]["exploration_slo"],
        "slo_percentile": config["online_feedback"]["exploration_slo_percentile"],
        "selection_rule": "min_P0_mean_power_plus_min_D0_mean_power_with_joint_D_sweep_SLO",
        "candidate_counts": counts,
        "best_configurations": best_rows,
        "formal_service_requests": 0,
    }
    _write_json(run_dir / "best_power_configurations.json", summary)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "audit.json", {
        "valid": True,
        "formal_service_requests": 0,
        "candidate_rows": len(candidate_rows),
        "measured_probes": len(candidate_rows) * 3,
        "all_table_entries_slo_safe": all(row["slo_met"] is True for row in best_rows),
    })
    lines = [
        "# P0-D0 exhaustive power exploration", "",
        "Verdict: **PASS**", "",
        "No P1-D1 formal service requests were issued.", "",
        "| Workload | P MHz | D MHz | P W | D W | P+D W | Joint W | TTFT p95 ms | TPOT p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {workload_id} | {prefill_frequency_mhz} | {decode_frequency_mhz} | "
            "{prefill_power_w:.3f} | {decode_power_w:.3f} | {additive_axis_power_w:.3f} | "
            "{joint_measured_power_w:.3f} | {joint_ttft_p95_ms:.3f} | {joint_tpot_p95_ms:.3f} |".format(**row)
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def run(config: Mapping[str, Any], run_id: str) -> Path:
    from replay_synthetic_trace import build_prompt_cache
    from transformers import AutoTokenizer

    run_dir = Path(str(config["output_root"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = config["online_feedback"]
    event_log = Path(os.path.expandvars(str(settings["event_log"])))
    table_path = Path(os.path.expandvars(str(config["frequency_table_path"])))
    tokenizer = AutoTokenizer.from_pretrained(str(config["tokenizer_model"]))
    lengths = {int(row["input_len"]) for row in config["workloads"]}
    prompts = build_prompt_cache(tokenizer, lengths)
    base_url = str(config["proxy_uri"]).rstrip("/")
    timeout_s = float(settings.get("exploration_wait_timeout_s", 25200.0))
    for index, workload in enumerate(config["workloads"]):
        workload_id = str(workload["id"])
        payload = {
                "model": str(config["model"]),
                "prompt": prompts[int(workload["input_len"])],
                "max_tokens": int(workload["output_len"]),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": True,
                "xpyd_input_len": int(workload["input_len"]),
                "xpyd_output_len": int(workload["output_len"]),
                "xpyd_workload_id": workload_id,
        }
        await _json_request(
            "POST", base_url + "/xpyd/explore", payload=payload,
            headers={"X-Xpyd-Logical-Request-Id": "explore-only-%02d-%s" % (index, workload_id)},
        )

    deadline = time.monotonic() + timeout_s
    while True:
        status = await _json_request("GET", base_url + "/xpyd/controller/status")
        entries = status.get("table", {})
        complete = all(entries.get(str(row["id"]), {}).get("value") is not None for row in config["workloads"])
        if complete and not status.get("active_workload") and not status.get("pending_workloads") and int(status.get("queue_depth", -1)) == 0:
            break
        failures = [row for row in _read_events(event_log) if row.get("event") == "exploration_failed"]
        if failures:
            raise ExplorationOnlyError("exploration failed: %s" % failures[-1])
        if time.monotonic() >= deadline:
            raise ExplorationOnlyError("exploration timeout after %.0f seconds; status=%s" % (timeout_s, status))
        await asyncio.sleep(5.0)
    _summarize(config, run_dir, event_log, table_path)
    print(run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(run(config, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
