"""Summarise the VoltanaLLM 4P4D stages the way CanaTune's bench_production.py does, and set them
beside CanaTune's 275725.

Inputs (under --work): stages.csv from run.sbatch, <config>/stages/<stage>/bench.json from
`vllm bench serve --save-detailed`, {prefill,decode}_nvml.csv from gpu_sampler.py (all four GPUs of
a node), {prefill,decode}_gpu_indices and the server logs logs/{prefill,decode}<i>.log
(freq_setter_nvsmi lines).
Per stage, as bench_production.py (CanaTune 13b6052): a request is good when it has a first token,
TTFT <= the TTFT SLO and vLLM's TPOT (sum of its ITLs over output tokens - 1) <= the TPOT SLO;
goodput = good / offered; energy is the sum over all 8 GPUs over the benchmark's own span
[max(launch, end - duration), end].
Writes <config>/bench_summary.json (CanaTune's schema), summary.csv / summary.txt and, when
--canatune names CanaTune results (label=results/<jobid>), compare.csv / compare.txt: per SLO both
sides scored, goodput and energy per stage beside each reference's baseline and CanaTune runs.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--work", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--score-slos", default="600:100,1000:100,600:150,1000:150")
parser.add_argument("--canatune", nargs="*", default=[],
                    help="label=results/<jobid> of CanaTune runs to compare with")
args = parser.parse_args()
score_slos = [tuple(float(x) for x in s.split(":")) for s in args.score_slos.split(",")]
primary = score_slos[0]


def slo_key(slo):
    return f"{slo[0]:g}/{slo[1]:g}"


def quantile(values, q):  # bench_production._quantile
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def request_outcomes(result, ttft_slo_ms, tpot_slo_ms):  # bench_production.request_outcomes
    ttfts = result.get("ttfts") or []
    itls = result.get("itls") or [[] for _ in ttfts]
    errors = result.get("errors") or ["" for _ in ttfts]
    outputs = result.get("output_lens") or [len(i) + 1 for i in itls]
    ok, good, ttft_ms, tpot_ms = 0, 0, [], []
    for ttft, itl, error, output in zip(ttfts, itls, errors, outputs):
        if error or not ttft:
            continue
        ok += 1
        ttft_ms.append(ttft * 1000.0)
        tpot = sum(itl) / (output - 1) * 1000.0 if itl and output and output > 1 else None
        if tpot is not None:
            tpot_ms.append(tpot)
        if ttft * 1000.0 <= ttft_slo_ms and (tpot is None or tpot <= tpot_slo_ms):
            good += 1
    offered = len(ttfts)
    return {
        "offered": offered, "ok": ok, "good": good,
        "goodput": good / offered if offered else None,
        "ttft_p50_ms": quantile(ttft_ms, 0.5), "ttft_p95_ms": quantile(ttft_ms, 0.95),
        "ttft_p99_ms": quantile(ttft_ms, 0.99),
        "tpot_p50_ms": quantile(tpot_ms, 0.5), "tpot_p95_ms": quantile(tpot_ms, 0.95),
    }


def load_nvml(role):
    """-> {"P0": DataFrame, ...} in slot order of <role>_gpu_indices."""
    path = os.path.join(args.work, f"{role}_nvml.csv")
    idx_path = os.path.join(args.work, f"{role}_gpu_indices")
    if not (os.path.exists(path) and os.path.exists(idx_path)):
        return {}
    indices = [int(x) for x in open(idx_path).read().strip().split(",")]
    df = pd.read_csv(path, comment="#")
    df = df[pd.to_numeric(df["t"], errors="coerce").notna()].astype(float)
    return {f"{role[0].upper()}{slot}": df[df["gpu"] == idx].sort_values("t").reset_index(drop=True)
            for slot, idx in enumerate(indices)}


def load_setter(role):
    """-> {"P0": DataFrame(t, freq, ok, took_ms)} from the server logs."""
    pattern = re.compile(r"\[freq_setter_nvsmi\] t=([\d.]+) set=(\d+) ok=(\w+) took_ms=(\d+)")
    out = {}
    for slot in range(4):
        rows = []
        path = os.path.join(args.work, "logs", f"{role}{slot}.log")
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        rows.append((float(m[1]), int(m[2]), m[3] == "True", int(m[4])))
        out[f"{role[0].upper()}{slot}"] = pd.DataFrame(rows, columns=["t", "freq", "ok", "took_ms"])
    return out


nvml = {**load_nvml("prefill"), **load_nvml("decode")}
setter = {**load_setter("prefill"), **load_setter("decode")}


def energy_between(df, t0, t1):
    if df is None or len(df) < 2 or t0 < df["t"].iloc[0] or t1 > df["t"].iloc[-1] + 1.0:
        return None
    e0, e1 = np.interp([t0, t1], df["t"], df["energy_mj"])
    return (e1 - e0) / 1000.0


def clock_stats(names, t0, t1):
    out = {}
    for prefix in ("P", "D"):
        sel = [nvml[n][(nvml[n]["t"] >= t0) & (nvml[n]["t"] <= t1)] for n in names if n.startswith(prefix)]
        sel = pd.concat(sel) if sel else pd.DataFrame()
        if len(sel):
            out[f"{prefix.lower()}_sm_mhz_mean"] = sel["sm_mhz"].mean()
            out[f"{prefix.lower()}_power_capped_frac"] = ((sel["throttle"].astype(int) & 0x4) > 0).mean()
        changes = [s[(s["t"] >= t0) & (s["t"] <= t1)] for n, s in setter.items() if n.startswith(prefix)]
        changes = pd.concat(changes) if changes else pd.DataFrame()
        out[f"{prefix.lower()}_freq_changes"] = len(changes)
        out[f"{prefix.lower()}_setter_ms_p50"] = changes["took_ms"].median() if len(changes) else None
    return out


stages = pd.read_csv(os.path.join(args.work, "stages.csv"))
stages = stages[stages["config"] != "smoke"]
rows = []
for config, group in stages.groupby("config", sort=False):
    summary = {"stages": {}}
    for st in group.itertuples():
        path = os.path.join(args.work, config, "stages", st.stage, "bench.json")
        result = json.load(open(path)) if st.rc == 0 and os.path.exists(path) else None
        duration = float((result or {}).get("duration") or 0.0)
        start = max(st.t_launch, st.t_end - duration) if duration > 0 else st.t_launch
        per_gpu = {name: energy_between(df, start, st.t_end) for name, df in sorted(nvml.items())}
        known = [e for e in per_gpu.values() if e is not None]
        energy = sum(known) if known and len(known) == len(per_gpu) == 8 else None
        out = {"workload": st.workload, "rate_rps": st.rate, "num_prompts": st.num_prompts,
               "window_s": [round(start, 2), round(st.t_end, 2)], "energy_j": energy,
               "power_w": energy / (st.t_end - start) if energy is not None and st.t_end > start else None,
               "energy_by_gpu_j": per_gpu, "exit": int(st.rc), "policy": st.policy, "idle_s": st.idle_s}
        if result is not None:
            outcomes = request_outcomes(result, *primary)
            output_tokens = result.get("total_output_tokens")
            out.update({
                "completed": result.get("completed"), "failed": result.get("failed"),
                "duration_s": result.get("duration"),
                "request_throughput": result.get("request_throughput"),
                "output_throughput": result.get("output_throughput"),
                "total_input_tokens": result.get("total_input_tokens"),
                "total_output_tokens": output_tokens,
                "vllm_request_goodput": result.get("request_goodput"), **outcomes,
                "j_per_good_request": energy / outcomes["good"] if energy is not None and outcomes["good"] else None,
                "j_per_output_token": energy / output_tokens if energy is not None and output_tokens else None,
                "by_slo": {slo_key(s): request_outcomes(result, *s) for s in score_slos},
            })
        out["clocks"] = clock_stats(list(nvml), start, st.t_end)
        summary["stages"][st.stage] = out
        row = {"config": config, "stage": st.stage, "workload": st.workload, "rate": st.rate,
               "rc": st.rc, "policy": st.policy, "idle_s": st.idle_s}
        for k in ("completed", "failed", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
                  "energy_j", "j_per_good_request", "j_per_output_token", "total_output_tokens"):
            row[k] = out.get(k)
        for s in score_slos:
            row[f"goodput_{slo_key(s)}"] = (out.get("by_slo") or {}).get(slo_key(s), {}).get("goodput")
        row.update(out["clocks"])
        rows.append(row)
    os.makedirs(os.path.join(args.out, config), exist_ok=True)
    with open(os.path.join(args.out, config, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(args.out, "summary.csv"), index=False)
cols = [c for c in ["config", "stage", "rc", "completed", "ttft_p50_ms", "ttft_p95_ms", "tpot_p95_ms"]
        + [f"goodput_{slo_key(s)}" for s in score_slos]
        + ["energy_j", "j_per_good_request", "p_sm_mhz_mean", "d_sm_mhz_mean", "p_freq_changes",
           "d_freq_changes", "p_setter_ms_p50", "idle_s"] if c in table]
with open(os.path.join(args.out, "summary.txt"), "w") as f:
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    f.write(table[cols].round(3).to_string(index=False) + "\n")
print(open(os.path.join(args.out, "summary.txt")).read())

# beside CanaTune: goodput and energy per stage, for every SLO both sides scored; <label>_fill is
# the mean of the reference's CanaTune runs (fill_r*), <label>_base its round-robin baseline
refs = {}  # (label, kind, stage) -> [stage summary]
for item in args.canatune:
    label, _, path = item.partition("=")
    if not os.path.isdir(path):
        print(f"reference {label}: {path} not found")
        continue
    for run in sorted(os.listdir(path)):
        summary_path = os.path.join(path, run, "bench_summary.json")
        if os.path.exists(summary_path):
            kind = "base" if run.startswith("baseline") else "fill"
            for stage, s in json.load(open(summary_path))["stages"].items():
                refs.setdefault((label, kind, stage), []).append(s)
if refs and len(table):
    labels = list(dict.fromkeys(k[0] for k in refs))
    comp, lines = [], []
    for s in score_slos:
        key = slo_key(s)
        for stage in table["stage"].unique():
            row = {"slo": key, "stage": stage}
            for label in labels:
                for kind in ("base", "fill"):
                    vals = refs.get((label, kind, stage), [])
                    g = [(v.get("by_slo") or {}).get(key, {}).get("goodput") for v in vals]
                    g = [x for x in g if x is not None]
                    e = [v.get("energy_j") for v in vals if v.get("energy_j") is not None]
                    row[f"{label}_{kind}_goodput"] = float(np.mean(g)) if g else None
                    row[f"{label}_{kind}_kj"] = float(np.mean(e)) / 1000 if e else None
            for config in table["config"].unique():
                sel = table[(table["config"] == config) & (table["stage"] == stage)]
                if len(sel):
                    row[f"{config}_goodput"] = sel[f"goodput_{key}"].iloc[0]
                    energy = sel["energy_j"].iloc[0]
                    row[f"{config}_kj"] = energy / 1000 if pd.notna(energy) else None
            comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(args.out, "compare.csv"), index=False)
    with open(os.path.join(args.out, "compare.txt"), "w") as f:
        f.write("goodput and energy (kJ, 8 GPUs) per stage; <ref>_base = its round-robin baseline, "
                "<ref>_fill = mean of its CanaTune runs (empty: the reference did not score this SLO)\n")
        for key, part in comp.groupby("slo", sort=False):
            part = part.dropna(axis=1, how="all")
            f.write(f"\n== SLO {key}\n")
            f.write(part.drop(columns="slo").round(3).to_string(index=False) + "\n")
    print(open(os.path.join(args.out, "compare.txt")).read())
