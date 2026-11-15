#!/usr/bin/env python3
"""Per variant and segment: TTFT and its parts, vLLM's KV transfer metrics, UCX transports.

  python summarize_xfer.py <results dir> <segment_s> <variant> [<variant> ...]
  -> <results dir>/xfer_summary.{json,txt}

Per segment (w00 big, w01 conc, w02 short; see make_xfer_replay.py):
  bench          ok / offered, goodput, TTFT p50 / p95 (bench_summary.json)
  parts          medians and p95 of to_prefill_ms, prefill_ms, decode_first_ms per request
                 (requests.jsonl; decode_first = D call to first token: KV read + D queue +
                 first step); a request's segment is its arrival estimated from its record
  kv             vLLM's "KV Transfer metrics" lines of the decode logs: transfers, mean and
                 worst-interval P90 transfer time, MB per transfer, MB / mean time
  ucx            transports named in UCX_PROTO_INFO lines of all logs (counts)
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
import sys
import time
from pathlib import Path

out, segment_s, variants = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
METRIC = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[metrics\.py:\d+\] KV Transfer metrics: (.*)")
FIELD = re.compile(r"([A-Za-z0-9 ()/]+?)=([0-9.]+)")
TRANSPORT = re.compile(r"\|\s*((?:tcp|cuda_ipc|cuda_copy|posix|sysv|cma|xpmem|knem|rc_mlx5|dc_mlx5|ud_mlx5|rc_verbs|self)[/\w:.-]*)\s*\|")
YEAR = time.localtime().tm_year


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 1)


def log_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        yield from handle


summary: dict = {"segment_s": segment_s, "variants": {}}
lines = []
for variant in variants:
    vdir = out / variant
    res: dict = {"segments": {}}
    summary["variants"][variant] = res
    if not vdir.is_dir():
        res["missing"] = True
        lines.append(f"{variant}: not run")
        continue
    # bench stages
    bench = {}
    path = vdir / "bench_summary.json"
    if path.is_file():
        for name, s in json.loads(path.read_text())["stages"].items():
            if name.rsplit("_w", 1)[-1].isdigit():
                bench[f"w{int(name.rsplit('_w', 1)[-1]):02d}"] = {
                    k: s.get(k) for k in ("ok", "offered", "goodput", "ttft_p50_ms", "ttft_p95_ms",
                                          "tpot_p95_ms", "failed")}
    # per-request parts, segment by estimated arrival
    records = []
    path = vdir / "requests.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "request" and r.get("route") == "production":
                records.append(r)
    t0 = None
    for r in records:
        if r.get("ttft_ms") is None:
            continue
        arrival = r["wall_time"] - (r["ttft_ms"] + max(0, (r.get("output_tokens") or 1) - 1)
                                    * (r.get("tpot_ms") or 0.0)) / 1000.0
        r["_arrival"] = arrival
        t0 = arrival if t0 is None else min(t0, arrival)
    status = {}
    parts: dict[str, dict[str, list]] = {}
    for r in records:
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
        if "_arrival" not in r or t0 is None:
            continue
        seg = f"w{int((r['_arrival'] - t0) // segment_s):02d}"
        timing = r.get("timing") or {}
        p = parts.setdefault(seg, {"to_prefill_ms": [], "prefill_ms": [], "decode_first_ms": []})
        for key in p:
            if timing.get(key) is not None:
                p[key].append(timing[key])
    res["request_status"] = status
    # KV transfer metrics of the decode logs
    kv: dict[str, dict] = {}
    ucx: dict[str, int] = {}
    for path in sorted((vdir / "logs").glob("*.log*")):
        for line in log_lines(path):
            m = TRANSPORT.search(line) if "|" in line else None
            if m:
                key = f"{path.name.split('.')[0].rsplit('_', 1)[0]}:{m.group(1)}"
                ucx[key] = ucx.get(key, 0) + 1
            if "decode" not in path.name or "KV Transfer metrics" not in line:
                continue
            m = METRIC.search(line)
            if not m:
                continue
            fields = {k.strip(): float(v) for k, v in FIELD.findall(m.group(2))}
            n = fields.get("Num successful transfers", 0.0)
            when = time.mktime(time.strptime(f"{YEAR}-{m.group(1)}", "%Y-%m-%d %H:%M:%S"))
            seg = "all" if t0 is None else f"w{max(0, int((when - t0) // segment_s)):02d}"
            for key in (seg, "total"):
                acc = kv.setdefault(key, {"transfers": 0.0, "xfer_ms": 0.0, "mb": 0.0,
                                          "p90_max_ms": 0.0, "failed": 0.0})
                acc["transfers"] += n
                acc["xfer_ms"] += n * fields.get("Avg xfer time (ms)", 0.0)
                acc["mb"] += n * fields.get("Avg MB per transfer", 0.0)
                acc["p90_max_ms"] = max(acc["p90_max_ms"], fields.get("P90 xfer time (ms)", 0.0))
                acc["failed"] += fields.get("Num failed transfers", 0.0)
    for acc in kv.values():
        n = acc["transfers"]
        acc["mean_xfer_ms"] = round(acc["xfer_ms"] / n, 1) if n else None
        acc["mb_per_transfer"] = round(acc["mb"] / n, 1) if n else None
        acc["gb_per_s"] = round(acc["mb"] / acc["xfer_ms"], 2) if acc["xfer_ms"] else None
    res["kv"] = kv
    res["ucx_transports"] = dict(sorted(ucx.items(), key=lambda kv_: -kv_[1]))
    lines.append(f"== {variant}  request status {status}")
    for seg in sorted(set(bench) | set(parts) | {k for k in kv if k.startswith("w")}):
        b, p, k = bench.get(seg, {}), parts.get(seg, {}), kv.get(seg, {})
        row = {"bench": b, "kv": k, "parts": {
            key: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "n": len(v)} for key, v in p.items()}}
        res["segments"][seg] = row
        dfirst = row["parts"].get("decode_first_ms", {})
        pre = row["parts"].get("prefill_ms", {})
        lines.append(
            f"  {seg} ok={b.get('ok')}/{b.get('offered')} TTFT p50/p95="
            f"{b.get('ttft_p50_ms') and round(b['ttft_p50_ms'])}/{b.get('ttft_p95_ms') and round(b['ttft_p95_ms'])} ms"
            f" | prefill p50={pre.get('p50')} decode_first p50/p95={dfirst.get('p50')}/{dfirst.get('p95')} ms"
            f" | kv n={k.get('transfers') and int(k['transfers'])} xfer={k.get('mean_xfer_ms')} ms"
            f" (P90 max {k.get('p90_max_ms') and round(k['p90_max_ms'], 1)}) {k.get('mb_per_transfer')} MB"
            f" -> {k.get('gb_per_s')} GB/s failed={k.get('failed') and int(k['failed'])}")
    tot = kv.get("total", {})
    lines.append(f"  total kv: n={tot.get('transfers') and int(tot['transfers'])} "
                 f"xfer={tot.get('mean_xfer_ms')} ms {tot.get('mb_per_transfer')} MB -> {tot.get('gb_per_s')} GB/s")
    top = list(res["ucx_transports"].items())[:6]
    lines.append(f"  ucx: {top}")
(out / "xfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "xfer_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
