# KV-cache transfer term in saturation-gated TTFT scheduling

## Result

Minerva Slurm Job **250048** completed all 12 online workload windows with
`latency_plus_saturation`, zero failed requests, six overload fallbacks, and
verified GPU clock cleanup.

Adding the modeled KV-cache transfer time increased predicted TTFT, but did not
change any operating point in this trace:

- 12/12 windows selected the same L40S prefill and L4 decode frequencies as the
  previous saturation-gated Job 250014.
- The number of latency-safe, saturation-safe, and jointly safe candidates was
  unchanged in every window.
- Both runs had six `OVERLOAD_FALLBACK` windows.

Consequently, the measured energy difference between the two live executions
cannot be attributed to the KV term. Total two-GPU board energy was
51,050.33 J without the modeled term and 51,282.13 J with it, a difference of
+231.79 J (+0.45%). Because the operating points were identical and there is
only one execution per condition, this is run-to-run variation.

## Scheduler model

The scheduler now uses:

```text
predicted P99 TTFT
  = predicted P99(queue + prefill)
  + T_kv
  + T_dispatch

T_kv
  = input_tokens * KV_bytes_per_token * 8
    / (effective_bandwidth_gbps * 10^9) * 1000 ms

KV_bytes_per_token
  = 2 (K and V)
    * num_layers
    * num_kv_heads
    * head_dim
    * bytes_per_element
```

For Mistral-7B-v0.1, the experiment used 32 layers, 8 KV heads, head dimension
128, and 2-byte elements:

```text
KV_bytes_per_token = 2 * 32 * 8 * 128 * 2
                   = 131072 bytes
                   = 128 KiB
```

`T_dispatch` was set to zero. The effective bandwidth was set to
**24.076 Gbit/s**, the 256 MiB result from prior Minerva Job 249400 on the same
Neptune-Ganymede NCCL Socket/TCP path. This is an empirical large-message proxy,
not the physical 100 Gbit/s link rate.

The total TTFT, its queue/prefill component, KV transfer time, payload size, and
bandwidth are recorded in each scheduler decision and in `live_summary.csv`.
The adjusted total is used in the explicit TTFT SLO admission check and TTFT
excess calculation. The pre-existing learned latency-violation probability is
not recalibrated by this deterministic term.

TPOT is unchanged because the KV-cache transfer is a one-time handoff between
prefill and decode before the first output token, rather than a per-output-token
decode cost. The live benchmark's measured TTFT already includes the real
network handoff; this change corrects the scheduler's prediction, not the
runtime request path.

## Controlled setup

The new run replayed the exact `request_trace.csv` from Job 250014:

- fixed placement: L40S prefill on Neptune, L4 decode on Ganymede
- TP=1
- TTFT SLO 500 ms; TPOT SLO 200 ms
- identical frequency candidates and caps: L40S up to 2520 MHz, L4 up to
  780 MHz
- identical saturation model, threshold, overload fallback, vLLM settings,
  request counts, and finite-rate Poisson arrival mode
- two allocated GPU board-power telemetry streams sampled every 0.5 seconds

Job 250048 requested both nodes exclusively. Job 250014 did not request
exclusive nodes, so live latency and energy differences are not a strict
one-variable causal ablation even though scheduler inputs and selected GPU
operating points are identical.

The scheduler uses the configured input lengths below. vLLM's random-data
sampler generated one fewer actual input token (31, 127, 511, or 1023), making
the KV estimate conservative by one token, approximately 0.0436 ms at the
configured bandwidth.

## Per-window comparison

`old pred` is Job 250014's predicted P99 TTFT without the modeled KV term.
`new pred` is Job 250048's total predicted P99 TTFT. Frequencies are
L40S-prefill/L4-decode MHz.

| # | Workload (IL/OL @ req/s) | T_kv ms | old → new pred ms | safe configs old → new | selected MHz | actual P99 TTFT old → new ms | energy old → new J |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | steady_start (32/32 @ 1) | 1.394 | 112.4 → 113.8 | 30 → 30 | 480/360 | 280.99 → 262.79 | 3078.20 → 2830.64 |
| 2 | short_ramp (32/32 @ 4) | 1.394 | 142.7 → 144.1 | 30 → 30 | 210/360 | 412.03 → 433.38 | 3463.21 → 3460.25 |
| 3 | prefill_shift (128/32 @ 3) | 5.575 | 181.0 → 186.6 | 18 → 18 | 480/570 | 433.72 → 460.06 | 3201.04 → 3247.44 |
| 4 | decode_shift (32/128 @ 2) | 1.394 | 75.0 → 76.4 | 0 → 0 | 2265/780 | 292.17 → 327.83 | 5157.15 → 5154.23 |
| 5 | sudden_burst (128/32 @ 8) | 5.575 | 90.6 → 96.2 | 0 → 0 | 1755/780 | 405.65 → 483.70 | 3446.40 → 3492.82 |
| 6 | short_recovery (32/32 @ 2) | 1.394 | 129.2 → 130.6 | 30 → 30 | 210/360 | 362.87 → 371.59 | 3321.68 → 3372.42 |
| 7 | long_prefill (512/32 @ 2) | 22.299 | 168.2 → 190.5 | 0 → 0 | 1755/570 | 485.07 → 602.79 | 3359.31 → 3494.77 |
| 8 | mixed_pressure (128/128 @ 3) | 5.575 | 80.7 → 86.3 | 0 → 0 | 1755/780 | 413.70 → 404.33 | 5189.10 → 5236.47 |
| 9 | quiet_window (32/32 @ 1) | 1.394 | 112.4 → 113.8 | 30 → 30 | 480/360 | 262.33 → 284.30 | 3058.66 → 3241.07 |
| 10 | prefill_peak (1024/32 @ 2) | 44.598 | 309.9 → 354.5 | 0 → 0 | 2520/780 | 612.33 → 710.61 | 3898.28 → 3829.28 |
| 11 | decode_peak (32/512 @ 1) | 1.394 | 88.9 → 90.3 | 0 → 0 | 2265/780 | 248.92 → 253.87 | 10671.67 → 10687.83 |
| 12 | final_recovery (128/32 @ 1) | 5.575 | 177.4 → 183.0 | 20 → 20 | 210/570 | 277.75 → 292.17 | 3205.63 → 3234.92 |

## Aggregate observations

| Metric | Job 250014, no modeled T_kv | Job 250048, modeled T_kv | Difference |
|---|---:|---:|---:|
| Successful / failed requests | 332 / 0 | 332 / 0 | unchanged |
| TTFT-SLO-passing windows | 11/12 | 10/12 | -1 window |
| TPOT-SLO-passing windows | 12/12 | 12/12 | unchanged |
| Mean of window P99 TTFT | 373.96 ms | 407.28 ms | +33.32 ms |
| Median of window P99 TTFT | 384.26 ms | 387.96 ms | +3.70 ms |
| Total two-GPU board energy | 51,050.33 J | 51,282.13 J | +0.45% |
| Energy per successful request | 153.77 J | 154.46 J | +0.45% |
| Integrated average board power | 108.29 W | 108.51 W | +0.20% |
| Measured interval duration | 471.42 s | 472.61 s | +0.25% |

The extra TTFT-SLO violation in Job 250048 was `long_prefill`: 602.79 ms versus
485.07 ms in Job 250014. `prefill_peak` violated TTFT in both runs. Since the
selected configurations were identical, this observed change is not evidence
that adding `T_kv` degraded performance; it reflects live-run variance and the
different node-sharing condition.

At `prefill_peak`, both runs requested 2520 MHz on L40S, but the active mean
clock did not sustain that target (2239 MHz in Job 250014 and 2225 MHz in Job
250048). The runner continued by design and recorded the mismatch.

## Interpretation

The formula correction is valid and now exposes the missing transfer component,
but this particular 500 ms TTFT-SLO trace does not contain a boundary case where
the additional 1.4–44.6 ms changes admission or frequency selection:

- windows already safe retained enough TTFT margin;
- unsafe windows were already rejected by latency or saturation constraints;
- fallback rankings selected the same configurations.

The practical effect in this trace is therefore improved prediction accounting,
not a changed schedule. To measure a causal scheduling/energy effect, a follow-up
trace should place at least one candidate within approximately `T_kv` of the
TTFT boundary and rerun both formula variants under the same exclusive-node
condition with repeated seeds.
