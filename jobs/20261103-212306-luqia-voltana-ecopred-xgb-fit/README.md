# VoltanaLLM v3 EcoPred (XGBoost), offline fit

CPU only, no GPU. It fits the XGBoost latency predictors from VoltanaLLM v3 (arXiv 2509.04827v3, ISC'26; Sec. V-D and Appx. C) on the profiling points we already have. It then compares them on held-out rows with the v0.1 linear tables, which are what the open-source code and our end-to-end jobs use.

## Inputs

No new profiling. The points are the ones `fit_latency.py` kept in the profiling jobs:

| Phase | Job | GPU | Clocks (MHz) | Points file |
|---|---|---|---|---|
| prefill | 280101 | L40S | 720, 1080, 1440, 1800, 2520 | `runs/280101/prefill/fit/points-prefill.csv.gz` (one row per prefill iteration) |
| decode | 280104 | L4 | 735, 1020, 1170, 1455, 2040 | `runs/280104/decode/fit/points-decode.csv.gz` (one row per decode iteration, ITL 3–300 ms) |

The linear tables come from the same jobs (`latency-prefill.csv`, `latency-decode.csv`). `inputs.txt` records the sha256 of each input.

## Models

| Model | Features | Settings | Source |
|---|---|---|---|
| `prefill` | (f, N_tok) | gblinear, 100000 rounds, lr 0.5, early stop 1000 | paper |
| `decode` | (f, N_req, N_kv) | gbtree, 1000 rounds, lr 0.1, early stop 200 | paper |
| `prefill_gbtree` | (f, N_tok) | the decode settings, with colsample_bytree 1.0 | ours |

- **Shared settings (paper).** `reg:absoluteerror`, max_depth 6, subsample 0.8, colsample_bytree 0.8, `hist`. Early stopping uses validation MAE.
- **Why `prefill_gbtree` exists.** gblinear on raw (f, N_tok) is a single plane, `w_f·f + w_N·N_tok + b`. It cannot express the per-clock slope a_f, which the per-clock tables capture.
  - On synthetic data shaped like 280101, gblinear's MAE was about 130 ms; the table's was about 4 ms.
  - With 2 features, colsample 0.8 also leaves each tree a single feature, which is why `prefill_gbtree` uses 1.0.
  - It shows what a tree model gets on the same prefill points; it is not a reproduction of the paper.
- **Online fine-tuning: not done.** The paper retrains in the background every "XX" new samples and never gives the number. This job measures the offline model only.

## Evaluation

All on the 10% held-out test rows (fixed seed; 80/10/10 train/validation/test).

- **Errors.** MAE, MAPE, absolute-error p50/p90/p99 and bias, for XGBoost and for the linear table, on the same rows.
- **Breakdowns.**
  - Overall and per clock.
  - Prefill: also 2000–20000 tokens, the range the linear fit used.
  - Decode: also by batch size (1–32, 33–128, 129–512).
- **Leave one clock out.** Refit without one interior clock, then predict that clock.
  - XGBoost has to generalise over f.
  - The table interpolates between the neighbouring clocks, as `lookup_table.py` does for a clock it was not profiled at. In serving the measured clock often differs from the profiled ones.
- **Prediction latency.** One booster call with 1 row, and with one row per clock (an EcoFreq query), single thread. The tables take microseconds.
- **Caveat.** Rows are split at random, and neighbouring iterations of one run are correlated, so the in-sample-clock errors are optimistic for both models alike. The leave-one-clock-out numbers are not affected.

## Outputs (`results/<jobid>/`)

- **Results.**
  - `summary.txt`: the comparison table.
  - `metrics.json`: all numbers plus the parameters, best iteration and fit time.
  - `parity-<model>.png`: measured vs predicted.
  - `xgb-<model>.json.gz`: the boosters, loadable with `xgboost.Booster(model_file=...)`.
- **Run records.** `versions.txt`, `inputs.txt`, `tables/`, `fit.log` and `job_console.log`.

## Environment

- **Python.** The `voltanallm` env's Python.
- **xgboost.** 2.1.4, installed once with `pip install --no-deps --target` into `/data/users/chjing/voltana_work/pylib-xgboost-2.1.4`. The env itself is not modified, and a later serving job can reuse that copy through `PYTHONPATH`.
- **Run time.** Expected 30–60 min. Most of it is the decode fits: 8.2M points, the main fit plus three leave-one-clock-out fits.
