"""Fit VoltanaLLM v3's XGBoost EcoPred on our profiling points and compare it with the v0.1 linear tables.

v3 (arXiv 2509.04827v3, ISC'26), Sec. V-D and Appx. C:
  prefill  T_P = XGBoost(f, N_tok)         booster gblinear, n_estimators 100000, lr 0.5, early stop 1000
  decode   T_D = XGBoost(f, N_req, N_kv)   booster gbtree,   n_estimators 1000,   lr 0.1, early stop 200
  both     objective reg:absoluteerror, max_depth 6, subsample 0.8, colsample_bytree 0.8, tree_method hist
(max_depth / subsample / colsample / tree_method do nothing for gblinear; passed anyway, as in the paper.)

Inputs are the points fit_latency.py kept (points-<phase>.csv.gz): prefill one row per prefill iteration
(freq, tokens = batched tokens, ttft_exec ms), decode one row per decode iteration (freq, batch_size,
tokens = KV tokens, itl ms, already filtered to 3-300 ms). Rows are split 80/10/10 into train /
validation (early stopping) / test with a fixed seed; every number below is on the test rows.

Reported per phase:
  - MAE, MAPE and absolute-error percentiles of XGBoost and of the linear table, overall and per clock.
    The linear table's prefill fit only used 2000 < tokens < 20000; that range is reported separately.
  - Leave-one-clock-out: refit without one interior clock and predict it, against the linear table's
    interpolation between the neighbouring clocks (what lookup_table.py does for an unprofiled clock).
  - Prediction latency of one booster call with 1 row and with one row per clock (EcoFreq's query).
Models: prefill and decode as in the paper, plus prefill_gbtree (prefill points, decode's gbtree settings;
not in the paper, see PARAMS). Writes xgb-<model>.json (the booster), metrics.json, summary.txt and
parity-<model>.png to --out.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

COMMON = dict(objective="reg:absoluteerror", max_depth=6, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", eval_metric="mae")
PARAMS = {
    "prefill": dict(booster="gblinear", n_estimators=100000, learning_rate=0.5, early_stopping_rounds=1000, **COMMON),
    "decode": dict(booster="gbtree", n_estimators=1000, learning_rate=0.1, early_stopping_rounds=200, **COMMON),
}
# not in the paper: gblinear on raw (f, N_tok) cannot express the f x N_tok interaction, so prefill is
# also fit with the decode model's gbtree settings to show what a tree model gets on the same points.
# colsample_bytree 0.8 of 2 features leaves each tree a single feature (no interaction either), so 1.0
PARAMS["prefill_gbtree"] = dict(PARAMS["decode"], colsample_bytree=1.0)
PHASE = {"prefill": "prefill", "prefill_gbtree": "prefill", "decode": "decode"}
FEATURES = {"prefill": ["freq", "tokens"], "decode": ["freq", "batch_size", "tokens"]}
TARGET = {"prefill": "ttft_exec", "decode": "itl"}


def load_points(phase, path):
    df = pd.read_csv(path)
    df = df[FEATURES[phase] + [TARGET[phase]]].dropna()
    df = df[(df[FEATURES[phase][1]] > 0) & (df[TARGET[phase]] > 0)]
    return df.astype(np.float32).reset_index(drop=True)


def split(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    n_test, n_val = n // 10, n // 10
    return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]


class LinearTable:
    """Vectorised copy of lookup_table.py: per-clock rows, bs buckets for decode, latency <= 0 -> 20,
    linear interpolation between the neighbouring profiled clocks (extrapolation from the outer two)."""

    def __init__(self, path, drop_freq=None):
        table = pd.read_csv(path)
        if drop_freq is not None:
            table = table[table["freq"] != drop_freq]
        self.decode = "bs_start" in table.columns
        self.tables = {int(f): t.sort_values("bs_start" if self.decode else "freq").reset_index(drop=True)
                       for f, t in table.groupby("freq")}
        self.freqs = sorted(self.tables)

    def _at(self, freq, bs, tokens):
        t = self.tables[freq]
        if not self.decode:
            out = t["coef_tokens"].iloc[0] * tokens + t["coef_intercept"].iloc[0]
        else:
            idx = np.searchsorted(t["bs_start"].to_numpy(), bs, side="right") - 1
            ok = (idx >= 0) & (bs <= t["bs_end"].to_numpy()[np.clip(idx, 0, None)])
            idx = np.clip(idx, 0, None)
            out = (t["coef_bs"].to_numpy()[idx] * bs + t["coef_tokens"].to_numpy()[idx] * tokens
                   + t["coef_intercept"].to_numpy()[idx])
            out = np.where(ok, out, np.nan)   # bs outside every bucket: lookup returns None
        return np.where(out <= 0, 20.0, out)

    def predict(self, freq, bs, tokens):
        out = np.full(len(freq), np.nan)
        for f in np.unique(freq):
            sel = freq == f
            f = int(f)
            if f in self.tables:
                out[sel] = self._at(f, bs[sel], tokens[sel])
                continue
            i = np.searchsorted(self.freqs, f)
            lo, hi = (self.freqs[0], self.freqs[1]) if i == 0 else \
                (self.freqs[-2], self.freqs[-1]) if i == len(self.freqs) else (self.freqs[i - 1], self.freqs[i])
            v_lo, v_hi = self._at(lo, bs[sel], tokens[sel]), self._at(hi, bs[sel], tokens[sel])
            out[sel] = v_lo + (v_hi - v_lo) / (hi - lo) * (f - lo)
        return out


def errors(y, pred):
    ok = ~np.isnan(pred)
    abs_err = np.abs(pred[ok] - y[ok])
    if not ok.any():
        return {"n": 0}
    return {"n": int(ok.sum()), "coverage": float(ok.mean()), "mae_ms": float(abs_err.mean()),
            "mape_pct": float((abs_err / y[ok]).mean() * 100),
            "abs_err_p50_ms": float(np.percentile(abs_err, 50)), "abs_err_p90_ms": float(np.percentile(abs_err, 90)),
            "abs_err_p99_ms": float(np.percentile(abs_err, 99)), "bias_ms": float((pred[ok] - y[ok]).mean())}


def columns(phase, df):
    bs = df["batch_size"].to_numpy() if phase == "decode" else np.ones(len(df))
    return df["freq"].to_numpy(), bs, df["tokens"].to_numpy()


def fit(name, train, val, threads, seed):
    phase = PHASE[name]
    model = xgb.XGBRegressor(n_jobs=threads, random_state=seed, **PARAMS[name])
    t0 = time.time()
    model.fit(train[FEATURES[phase]], train[TARGET[phase]],
              eval_set=[(val[FEATURES[phase]], val[TARGET[phase]])], verbose=False)
    return model, time.time() - t0


def compare(phase, model, table, test):
    y = test[TARGET[phase]].to_numpy().astype(np.float64)
    p_xgb = model.predict(test[FEATURES[phase]]).astype(np.float64)
    p_lin = table.predict(*columns(phase, test))
    out = {"all": {"xgb": errors(y, p_xgb), "linear": errors(y, p_lin)}, "per_freq": {}}
    for f in sorted(test["freq"].unique()):
        sel = (test["freq"] == f).to_numpy()
        out["per_freq"][str(int(f))] = {"xgb": errors(y[sel], p_xgb[sel]), "linear": errors(y[sel], p_lin[sel])}
    if phase == "prefill":
        sel = ((test["tokens"] > 2000) & (test["tokens"] < 20000)).to_numpy()
        out["tokens_2000_20000"] = {"xgb": errors(y[sel], p_xgb[sel]), "linear": errors(y[sel], p_lin[sel])}
    else:
        for lo, hi in ((1, 32), (33, 128), (129, 512)):
            sel = test["batch_size"].between(lo, hi).to_numpy()
            out[f"bs_{lo}_{hi}"] = {"xgb": errors(y[sel], p_xgb[sel]), "linear": errors(y[sel], p_lin[sel])}
    return out, p_xgb, p_lin


def predict_latency_us(model, name, freqs):
    """Median wall time of one predict call, as EcoFreq would issue it per iteration."""
    booster = model.get_booster()
    booster.set_param({"nthread": 1})
    # gblinear has no inplace_predict; it goes through a DMatrix
    phase = PHASE[name]
    call = (lambda x: booster.predict(xgb.DMatrix(x, feature_names=FEATURES[phase]))) \
        if PARAMS[name]["booster"] == "gblinear" else booster.inplace_predict
    row = {"prefill": [4096.0], "decode": [64.0, 30000.0]}[phase]
    out = {}
    for name, n in (("1_row", 1), (f"{len(freqs)}_rows", len(freqs))):
        x = np.array([[float(freqs[i % len(freqs)])] + row for i in range(n)], dtype=np.float32)
        for _ in range(50):
            call(x)
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter()
            call(x)
            ts.append(time.perf_counter() - t0)
        out[name] = {"p50_us": float(np.percentile(ts, 50) * 1e6), "p99_us": float(np.percentile(ts, 99) * 1e6)}
    return out


def parity_plot(name, test, p_xgb, p_lin, png, seed):
    phase = PHASE[name]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    y = test[TARGET[phase]].to_numpy()
    for ax, pred, label in zip(axes, (p_xgb, p_lin), (f"XGBoost ({PARAMS[name]['booster']})", "linear table (v0.1)")):
        for i, f in enumerate(sorted(test["freq"].unique())):
            idx = np.flatnonzero((test["freq"] == f).to_numpy())
            idx = rng.choice(idx, min(3000, len(idx)), replace=False)
            ax.scatter(y[idx], pred[idx], s=3, alpha=0.3, c=f"C{i}", label=f"{int(f)} MHz")
        hi = np.nanpercentile(y, 99.5)
        ax.plot([0, hi], [0, hi], "k--", linewidth=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(f"{phase}: {label}")
        ax.set_xlabel("measured (ms)")
        ax.grid(True)
    axes[0].set_ylabel("predicted (ms)")
    axes[0].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def run_model(name, args, data, metrics, lines):
    phase = PHASE[name]
    if phase not in data:
        data[phase] = load_points(phase, getattr(args, f"{phase}_points"))
    df = data[phase]
    tr, va, te = split(len(df), args.seed)
    train, val, test = df.iloc[tr], df.iloc[va], df.iloc[te]
    freqs = sorted(int(f) for f in df["freq"].unique())
    print(f"[{name}] {len(df)} points, clocks {freqs}, train/val/test {len(tr)}/{len(va)}/{len(te)}", flush=True)

    model, fit_s = fit(name, train, val, args.threads, args.seed)
    model.get_booster().save_model(os.path.join(args.out, f"xgb-{name}.json"))  # booster only: the sklearn wrapper's save fails with sklearn >= 1.6
    table = LinearTable(getattr(args, f"{phase}_table"))
    res, p_xgb, p_lin = compare(phase, model, table, test)
    res.update(n_points=len(df), freqs=freqs, fit_s=fit_s, best_iteration=int(model.best_iteration),
               params=dict(PARAMS[name]))
    if PARAMS[name]["booster"] == "gblinear":
        # gblinear is linear in (f, N_tok): no f x N_tok interaction, unlike the per-clock tables
        res["gblinear_weights"] = {"bias": float(np.atleast_1d(model.intercept_)[0]),
                                   "coef": dict(zip(FEATURES[phase], (float(v) for v in model.coef_)))}
    parity_plot(name, test, p_xgb, p_lin, os.path.join(args.out, f"parity-{name}.png"), args.seed)
    res["predict_latency"] = predict_latency_us(model, name, freqs)
    print(f"[{name}] fit {fit_s:.0f}s best_iteration {model.best_iteration}", flush=True)

    # leave one interior clock out: XGBoost extrapolates over f, the table interpolates neighbours
    res["leave_one_clock_out"] = {}
    if not args.skip_lofo:
        for f in freqs[1:-1]:
            keep_tr, keep_va = (train["freq"] != f), (val["freq"] != f)
            m, s = fit(name, train[keep_tr], val[keep_va], args.threads, args.seed)
            held = test[test["freq"] == f]
            y = held[TARGET[phase]].to_numpy().astype(np.float64)
            res["leave_one_clock_out"][str(f)] = {
                "xgb": errors(y, m.predict(held[FEATURES[phase]]).astype(np.float64)),
                "linear_interp": errors(y, LinearTable(getattr(args, f"{phase}_table"), drop_freq=f).predict(*columns(phase, held))),
                "fit_s": s}
            print(f"[{name}] leave out {f} MHz: fit {s:.0f}s", flush=True)
    metrics[name] = res

    def row(name, d):
        x, l = d.get("xgb", {}), d.get("linear", d.get("linear_interp", {}))
        return (f"  {name:<18} xgb MAE {x.get('mae_ms', np.nan):7.2f} ms  MAPE {x.get('mape_pct', np.nan):6.2f}%  p90 {x.get('abs_err_p90_ms', np.nan):7.2f}"
                f" | linear MAE {l.get('mae_ms', np.nan):7.2f} ms  MAPE {l.get('mape_pct', np.nan):6.2f}%  p90 {l.get('abs_err_p90_ms', np.nan):7.2f}"
                f"  cov {l.get('coverage', np.nan):.3f}  (n {x.get('n', 0)})")
    lines.append(f"== {name} ({PARAMS[name]['booster']}): {len(df)} points, best_iteration {model.best_iteration}, fit {fit_s:.0f} s")
    lines.append(row("all", res["all"]))
    for k in ("tokens_2000_20000", "bs_1_32", "bs_33_128", "bs_129_512"):
        if k in res:
            lines.append(row(k, res[k]))
    for f, d in res["per_freq"].items():
        lines.append(row(f"{f} MHz", d))
    for f, d in res["leave_one_clock_out"].items():
        lines.append(row(f"LOCO {f} MHz", d))
    lat = res["predict_latency"]
    lines.append("  predict latency (1 thread): " + ", ".join(f"{k} p50 {v['p50_us']:.0f} us p99 {v['p99_us']:.0f} us" for k, v in lat.items()))
    if "gblinear_weights" in res:
        lines.append(f"  gblinear weights: {res['gblinear_weights']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for phase in ("prefill", "decode"):
        parser.add_argument(f"--{phase}-points", required=True)
        parser.add_argument(f"--{phase}-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="prefill,prefill_gbtree,decode", help=f"subset of {list(PARAMS)}")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-lofo", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metrics, lines, data = {"xgboost_version": xgb.__version__, "seed": args.seed}, [], {}
    for name in args.models.split(","):
        run_model(name, args, data, metrics, lines)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
