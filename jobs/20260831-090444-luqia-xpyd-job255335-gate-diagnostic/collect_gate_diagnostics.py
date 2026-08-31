#!/usr/bin/env python3
"""Collect compact per-window evidence for a failed binary-DVFS gate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def run(run_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("window_results.csv", "binary_search_decisions.csv"):
        source = run_dir / name
        if source.is_file():
            shutil.copyfile(source, output / name)

    rows = []
    window_table = run_dir / "window_results.csv"
    with window_table.open(newline="", encoding="utf-8") as stream:
        controller_rows = list(csv.DictReader(stream))

    windows_root = run_dir / "raw" / "windows"
    for controller in controller_rows:
        window_id = controller["window_id"]
        window = windows_root / window_id
        audit = read_json(window / "audit.json")
        summary = read_json(window / "summary.json")
        client = read_json(window / "client" / "summary.json")
        rows.append({
            "window_id": window_id,
            "controller_row": controller,
            "substrate_audit": audit,
            "route_matrix": summary.get("route_matrix"),
            "endpoint_clock_targets": {
                endpoint: {
                    "graphics": values.get("graphics"),
                    "memory": values.get("memory"),
                }
                for endpoint, values in summary.get("endpoint_clocks", {}).items()
            },
            "client": client,
        })

    with (output / "window_gate_details.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "valid": True,
            "read_only": True,
            "source_run": str(run_dir),
            "window_count": len(rows),
            "windows": rows,
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (args.run_dir / "window_results.csv").is_file():
        parser.error("window_results.csv is missing")
    run(args.run_dir.resolve(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
