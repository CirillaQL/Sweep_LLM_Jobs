# Saturation-gated online non-uniform DVFS experiment

Date: 2026-07-23  
Slurm job: 250014 (`COMPLETED`, `ExitCode=0:0`, elapsed 686 s)  
Job directory: `jobs/20260723-231809-luqia-vllm-online-nonuniform-dvfs-r2`

## What was executed

The request trace was executed directly inside a two-node Slurm allocation.
No workload operating points were locally precomputed or validated. Each
control window invoked the scheduler only when that window arrived, applied
the selected L40S/L4 clocks, waited for active-clock telemetry, and then sent
the requests through the running cross-node PD vLLM service.

The 12 windows varied request rate from 1 to 8 req/s, input length from 32 to
1024 tokens, output length from 32 to 512 tokens, and duration from 10 to 12 s.
The installed vLLM 0.15.1 did not expose its `--burstiness` option, so the
configured per-window gamma burstiness values were not applied. Requests
inside every finite-rate window still used random Poisson arrivals, while
rate, token shape, and duration changed sharply between windows.

SLOs were p99 TTFT <= 500 ms and p99 TPOT <= 200 ms.

## Results

| # | Workload (in/out, req/s) | Decision | L40S/L4 MHz | Actual p99 TTFT/TPOT ms | Avg GPU power W | SLO |
|---:|---|---|---:|---:|---:|---|
| 1 | steady_start (32/32, 1) | OK | 480/360 | 280.99/64.80 | 96.40 | pass |
| 2 | short_ramp (32/32, 4) | OK | 210/360 | 412.03/69.31 | 98.67 | pass |
| 3 | prefill_shift (128/32, 3) | OK | 480/570 | 433.72/65.08 | 102.67 | pass |
| 4 | decode_shift (32/128, 2) | minimum-violation fallback | 2265/780 | 292.17/64.39 | 114.66 | pass |
| 5 | sudden_burst (128/32, 8) | minimum-violation fallback | 1755/780 | 405.65/66.63 | 115.99 | pass |
| 6 | short_recovery (32/32, 2) | OK | 210/360 | 362.87/66.30 | 96.52 | pass |
| 7 | long_prefill (512/32, 2) | minimum-violation fallback | 1755/570 | 485.07/68.33 | 108.44 | pass |
| 8 | mixed_pressure (128/128, 3) | minimum-violation fallback | 1755/780 | 413.70/66.51 | 110.09 | pass |
| 9 | quiet_window (32/32, 1) | OK | 480/360 | 262.33/64.75 | 98.25 | pass |
| 10 | prefill_peak (1024/32, 2) | minimum-violation fallback | 2520/780 | 612.33/68.73 | 121.42 | **TTFT fail** |
| 11 | decode_peak (32/512, 1) | minimum-violation fallback | 2265/780 | 248.92/64.63 | 119.78 | pass |
| 12 | final_recovery (128/32, 1) | OK | 210/570 | 277.75/62.53 | 95.23 | pass |

All 332 requests succeeded and none failed. Six windows had at least one safe
candidate and used an `OK` operating point. Six had zero safe candidates and
used the configured minimum-predicted-SLO-violation fallback. All six `OK`
windows met both latency SLOs. Five of the six fallback windows also met both
SLOs in this run; the 1024-token prefill peak violated TTFT.

This is the intended overload behavior: an unsafe candidate is not labeled or
admitted as safe. When no safe point exists, the scheduler explicitly enters
`OVERLOAD_FALLBACK` and selects the least predicted violation instead of
silently using an infeasible normal configuration.

## Frequency, power, and thermal observations

- The selected clocks changed with load and token shape, including a drop from
  1755/780 MHz under mixed pressure to 480/360 MHz in the quiet window, then a
  rise to 2520/780 MHz for the prefill peak.
- The L40S active-clock probe initially observed only 2239 MHz for the 2520 MHz
  request and emitted `clock_target_not_sustained`. The request window
  continued and workload telemetry averaged 2516.5 MHz during the actual vLLM
  run. The synthetic acknowledgement probe was therefore conservative for
  this case and should not be treated as the workload's sustained clock.
- Combined average GPU board power ranged from 95.23 W to 121.42 W. Peak board
  samples were 137.69 W on L40S and 59.42 W on L4.
- Maximum observed temperatures were 35 C on L40S and 60 C on L4.
- The 12 measured request windows consumed 51.05 kJ of GPU board energy.
  Energy per request is dominated by trace length and tail drain for sparse,
  long-output windows; the 512-token decode peak used 1067.17 J/request.

## Interpretation limits

The generated `actual_throughput_ratio` is not a valid saturation verdict for
these fixed-count finite Poisson windows. Benchmark wall time includes random
arrival spacing, warm-tail completion, and queue drain, so even low-load
windows report ratios near 0.85 despite zero failed requests. The accompanying
`actual_measured_saturated` field must therefore not be used to claim that all
windows saturated. Saturation conclusions require arrival/completion
timestamps or queue/backlog telemetry; this report uses request success and
p99 TTFT/TPOT SLO outcomes.

The calibration model is also conservative and not quantitatively calibrated
to this runtime in several fallback windows: it predicted no safe operating
point, while five such windows met both measured latency SLOs. Conversely, the
1024-token prefill peak showed that the fallback label was warranted. The gate
is behaving safely, but the probability/latency models need runtime
recalibration before their predicted violation probabilities are interpreted
as calibrated probabilities.

## Slurm and system safety

The run allocated one GPU on each of Ganymede and Neptune and only controlled
those assigned devices. It launched and stopped processes within its own
allocation, locked application clocks for each control window, and reset both
GPU clocks after stopping the servers.

It did not cancel or modify other jobs, change Slurm node state, change GPU
power limits, use power gating, change MIG or persistence mode, or alter system
configuration. Cleanup completed with `parent_reset_verified=true`. A Slurm
NVGPUFREQ plugin job-ID lookup warning appeared, but the explicit clock
requests, telemetry, benchmark execution, and final reset all completed.

