#!/usr/bin/env python3
"""Clocks and energy before / after the forced re-calibration -> recal_summary.{json,txt}.

Segments (segment_s of replay time) are grouped by the adaptive run's own timeline:
  A         phase A (default-length mix), cold-start table
  S_before  phase S before the calibration was requested (cold-start table, short mix)
  S_during  the Canary group is out, three groups serve
  S_after   segments that start after the new table was published
The frozen and baseline runs replay the same requests, so the same segments are compared.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
segment_s = float(sys.argv[2])
report = json.loads((out / "shift_replay.report.json").read_text())
shift = report["S"]["start_s"]

events = []
log = out / "adapt_r1" / "forced_recal.jsonl"
if log.is_file():
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
requested = next((e["replay_s"] for e in events if e["event"] == "requested"), None)
published = next((e for e in events if e["event"] == "published"), None)
published_s = None if published is None else published["replay_s"]


def group(k: int) -> str:
    a, b = k * segment_s, (k + 1) * segment_s
    if b <= shift:
        return "A"
    if requested is None or b <= requested:
        return "S_before"
    if published_s is None or a < published_s:
        return "S_during"
    return "S_after"


summary: dict = {"shift_at_s": shift, "requested_at_s": requested, "published_at_s": published_s,
                 "forced": published, "events": [
                     {k: e.get(k) for k in ("replay_s", "event", "reply", "outcome", "error")}
                     for e in events if e["event"] in ("refused", "requested", "outcome", "gave_up")
                 ], "runs": {}}
lines = [f"shift at {shift:.0f} s, full calibration requested at {requested} s, "
         f"published at {published_s} s (replay time)"]
if published:
    lines.append(f"H   {published['h_before']} -> {published['h_after']} MHz (P/D)")
    lines.append(f"park {published['park_before']} -> {published['park_after']}")
    lines.append(f"alpha {published['alpha_before']} -> {published['alpha_after']} tokens, "
                 f"capacity_h {published['capacity_h_before']} -> {published['capacity_h_after']}")
    lines.append(f"calibration {published['duration_s']} s, {published['windows']} windows, "
                 f"{published['reused_windows']} reused")
for run in ("adapt_r1", "frozen_r1", "baseline_r1"):
    path = out / run / "bench_summary.json"
    if not path.is_file():
        continue
    stages = json.loads(path.read_text())["stages"]
    groups: dict[str, dict] = {}
    rows = []
    for name, s in sorted(stages.items()):
        if not name.rsplit("_w", 1)[-1].isdigit():
            continue
        k = int(name.rsplit("_w", 1)[-1])
        g = groups.setdefault(group(k), {"segments": 0, "energy_j": 0.0, "output_tokens": 0,
                                         "offered": 0, "good": 0, "seconds": 0.0})
        span = s.get("window_s") or [k * segment_s, (k + 1) * segment_s]
        g["segments"] += 1
        g["energy_j"] += s.get("energy_j") or 0.0
        g["output_tokens"] += s.get("total_output_tokens") or 0
        g["offered"] += s.get("offered") or 0
        g["good"] += s.get("good") or 0
        g["seconds"] += span[1] - span[0]
        rows.append({"segment": name, "group": group(k), "energy_j": s.get("energy_j"),
                     "power_w": s.get("power_w"), "j_per_output_token": s.get("j_per_output_token"),
                     "goodput": s.get("goodput"), "ttft_p95_ms": s.get("ttft_p95_ms"),
                     "tpot_p95_ms": s.get("tpot_p95_ms"), "modes": s.get("modes"),
                     "energy_by_gpu_j": s.get("energy_by_gpu_j")})
    for g in groups.values():
        g["power_w"] = g["energy_j"] / g["seconds"] if g["seconds"] else None
        g["j_per_output_token"] = g["energy_j"] / g["output_tokens"] if g["output_tokens"] else None
        g["goodput"] = g["good"] / g["offered"] if g["offered"] else None
    summary["runs"][run] = {"groups": groups, "segments": rows}
    for name in ("A", "S_before", "S_during", "S_after"):
        g = groups.get(name)
        if g:
            lines.append(f"{run:12s} {name:9s} n={g['segments']:2d} power={g['power_w']:.1f} W "
                         f"J/token={g['j_per_output_token']:.3f} goodput={g['goodput']:.3f}")
runs = summary["runs"]
for other in ("frozen_r1", "baseline_r1"):
    for name in ("A", "S_before", "S_during", "S_after"):
        a = runs.get("adapt_r1", {}).get("groups", {}).get(name)
        b = runs.get(other, {}).get("groups", {}).get(name)
        if a and b and b["energy_j"]:
            change = a["energy_j"] / b["energy_j"] - 1
            summary.setdefault("energy_change", {})[f"adapt_vs_{other}:{name}"] = change
            lines.append(f"energy adapt_r1 vs {other} in {name}: {100 * change:+.1f} %")
(out / "recal_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "recal_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
