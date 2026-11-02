"""Summarise the VoltanaLLM 1P1D end-to-end stages: SLO attainment, latency, energy, clocks.

Inputs (under --work): stages.csv from run.sbatch, bench/<stage>.jsonl-detail from
bench_serving_voltanallm --output-details, {prefill,decode}_nvml.csv from gpu_sampler.py and the
server logs {prefill,decode}.log (freq_setter_nvsmi lines). Writes summary.csv / summary.txt,
requests.csv.gz (one row per request) and transfer_fit.txt to --out.
A request meets the SLO when it completed, TTFT <= ttft_slo and its mean ITL <= itl_slo (paper:
"requests that satisfy the SLO thresholds for both TTFT and ITL"); failed requests count as misses.
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
parser.add_argument("--ttft-slo", type=float, default=600.0)
parser.add_argument("--itl-slos", default="60,80,100,120,150")
args = parser.parse_args()
itl_slos = [float(x) for x in args.itl_slos.split(",")]

stages = pd.read_csv(os.path.join(args.work, "stages.csv"))


def load_nvml(role):
    path = os.path.join(args.work, f"{role}_nvml.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#")
    df = df[pd.to_numeric(df["t"], errors="coerce").notna()].astype(float)
    return df.sort_values("t").reset_index(drop=True)


def load_setter(role):
    path = os.path.join(args.work, f"{role}.log")
    rows = []
    if os.path.exists(path):
        pattern = re.compile(r"\[freq_setter_nvsmi\] t=([\d.]+) set=(\d+) ok=(\w+) took_ms=(\d+)")
        with open(path, errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    rows.append((float(m[1]), int(m[2]), m[3] == "True", int(m[4])))
    return pd.DataFrame(rows, columns=["t", "freq", "ok", "took_ms"])


nvml = {role: load_nvml(role) for role in ("prefill", "decode")}
setter = {role: load_setter(role) for role in ("prefill", "decode")}


def window_stats(df, t0, t1):
    """Energy (J) from the cumulative counter, mean power, SM clock distribution."""
    if df is None or len(df) < 2:
        return {}
    e0, e1 = np.interp([t0, t1], df["t"], df["energy_mj"])
    sel = df[(df["t"] >= t0) & (df["t"] <= t1)]
    out = {"energy_j": (e1 - e0) / 1000.0}
    if len(sel):
        out["power_w"] = sel["power_mw"].mean() / 1000.0
        out["sm_mhz_mean"] = sel["sm_mhz"].mean()
        out["sm_mhz_p10"] = sel["sm_mhz"].quantile(0.1)
        out["sm_mhz_p90"] = sel["sm_mhz"].quantile(0.9)
        out["power_capped_frac"] = ((sel["throttle"].astype(int) & 0x4) > 0).mean()
    return out


rows, requests = [], []
for st in stages.itertuples():
    detail_path = os.path.join(args.work, "bench", f"{st.stage}.jsonl-detail")
    row = {"stage": st.stage, "config": st.config, "rate": st.rate, "rc": st.rc}
    t_end = float(st.t_end)
    t_begin = float(st.t_start)
    if os.path.exists(detail_path):
        with open(detail_path) as f:
            lines = [l for l in f if l.strip()]
        res = json.loads(lines[-1])
        if res.get("duration"):
            t_begin = max(t_begin, t_end - float(res["duration"]))
        n = len(res["ttfts"])
        errors = np.array([bool(e) for e in res["errors"]])
        ttft = np.array([t if t else np.nan for t in res["ttfts"]], dtype=float) * 1000
        itl_mean = np.array([np.mean(x) * 1000 if x else np.nan for x in res["itls"]], dtype=float)
        itl_p90 = np.array([np.percentile(x, 90) * 1000 if x else np.nan for x in res["itls"]], dtype=float)
        in_len = np.array(res["input_lens"], dtype=float)
        out_len = np.array(res["output_lens"], dtype=float)
        ok = ~errors & (ttft <= args.ttft_slo)
        row.update(n=n, failed=int(errors.sum()), duration_s=res.get("duration"),
                   out_tokens=float(np.nansum(out_len[~errors])),
                   ttft_p50=np.nanpercentile(ttft, 50), ttft_p90=np.nanpercentile(ttft, 90),
                   ttft_p99=np.nanpercentile(ttft, 99),
                   itl_mean_p50=np.nanpercentile(itl_mean, 50), itl_mean_p90=np.nanpercentile(itl_mean, 90),
                   tsar=float(ok.mean()))
        for slo in itl_slos:
            itl_ok = np.isnan(itl_mean) | (itl_mean <= slo)
            row[f"isar_{slo:g}"] = float((~errors & itl_ok).mean())
            row[f"sar_{args.ttft_slo:g}_{slo:g}"] = float((ok & itl_ok).mean())
        requests.append(pd.DataFrame({"stage": st.stage, "input_len": in_len, "output_len": out_len,
                                      "failed": errors, "ttft_ms": ttft, "itl_mean_ms": itl_mean,
                                      "itl_p90_ms": itl_p90}))
    row["window_s"] = t_end - t_begin
    for role in ("prefill", "decode"):
        for k, v in window_stats(nvml[role], t_begin, t_end).items():
            row[f"{role[0]}_{k}"] = v
        s = setter[role]
        if len(s):
            sel = s[(s["t"] >= t_begin) & (s["t"] <= t_end)]
            row[f"{role[0]}_freq_changes"] = len(sel)
            row[f"{role[0]}_setter_ms_p50"] = sel["took_ms"].median() if len(sel) else np.nan
    if "p_energy_j" in row and "d_energy_j" in row:
        row["energy_j"] = row["p_energy_j"] + row["d_energy_j"]
        if row.get("out_tokens"):
            row["j_per_out_token"] = row["energy_j"] / row["out_tokens"]
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
cols = [c for c in ["config", "rate", "n", "failed", "ttft_p50", "ttft_p90", "itl_mean_p50", "itl_mean_p90",
                    "tsar", f"sar_{args.ttft_slo:g}_100", f"sar_{args.ttft_slo:g}_150",
                    "p_energy_j", "d_energy_j", "energy_j", "j_per_out_token",
                    "p_sm_mhz_mean", "d_sm_mhz_mean", "p_freq_changes", "d_freq_changes"] if c in summary]
with open(os.path.join(args.out, "summary.txt"), "w") as f:
    pd.set_option("display.width", 250)
    f.write(summary[cols].round(3).to_string(index=False) + "\n")
if requests:
    req = pd.concat(requests)
    req.to_csv(os.path.join(args.out, "requests.csv.gz"), index=False)
    # TTFT = a + b * input_len at the lowest rate with static max clocks: b bundles prefill compute
    # and KV transfer per token, a the fixed overheads (NIXL handshake, D-side scheduling)
    with open(os.path.join(args.out, "transfer_fit.txt"), "w") as f:
        for stage in summary.loc[summary["config"] == "static_max"].sort_values("rate")["stage"].head(2):
            d = req[(req["stage"] == stage) & ~req["failed"]].dropna(subset=["ttft_ms"])
            if len(d) > 10:
                b, a = np.polyfit(d["input_len"], d["ttft_ms"], 1)
                f.write(f"{stage}: ttft_ms = {a:.1f} + {b:.4f} * input_len (n={len(d)}, "
                        f"input_len p50/p90 {d['input_len'].median():.0f}/{d['input_len'].quantile(0.9):.0f})\n")
print(open(os.path.join(args.out, "summary.txt")).read())
